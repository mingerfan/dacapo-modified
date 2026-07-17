#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <vector>

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_ASSIGNPLACEMENT
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

using namespace mlir;
namespace ckks = hecate::ckks;

namespace {

using Json = nlohmann::json;

struct Place {
  int64_t rank = 0;
  int64_t device = -1;

  bool operator==(const Place &other) const {
    return rank == other.rank && device == other.device;
  }
  bool operator<(const Place &other) const {
    return std::tie(rank, device) < std::tie(other.rank, other.device);
  }
};

struct Interval {
  int64_t start = 0;
  int64_t finish = 0;
};

struct Node {
  Operation *op = nullptr;
  size_t originalIndex = 0;
  int64_t cost = 0;
  int64_t priority = -1;
  int64_t start = 0;
  int64_t finish = 0;
  Place place;
  SmallVector<size_t> predecessors;
  SmallVector<size_t> successors;
  size_t remainingPredecessors = 0;
  bool scheduled = false;
};

FailureOr<std::vector<int64_t>> parseDeviceCounts(llvm::StringRef encoded,
                                                   Operation *op) {
  if (encoded.empty()) {
    op->emitError("assign-ckks-placement requires --device-counts");
    return failure();
  }
  std::vector<int64_t> result;
  SmallVector<llvm::StringRef> parts;
  encoded.split(parts, 'x', -1, false);
  for (llvm::StringRef part : parts) {
    int64_t count = 0;
    if (part.empty() || part.getAsInteger(10, count) || count < 0 ||
        count > std::numeric_limits<int32_t>::max()) {
      op->emitError("device-counts must contain nonnegative int32 values "
                    "separated by x");
      return failure();
    }
    result.push_back(count);
  }
  const bool cpuTopology =
      std::all_of(result.begin(), result.end(),
                  [](int64_t count) { return count == 0; });
  const bool deviceTopology =
      std::all_of(result.begin(), result.end(),
                  [](int64_t count) { return count > 0; });
  if (!cpuTopology && !deviceTopology) {
    op->emitError(
        "device-counts must be either all zero for CPU ranks or all positive");
    return failure();
  }
  return result;
}

FailureOr<Json> readOperatorSpec(llvm::StringRef path, Operation *op) {
  if (path.empty()) {
    op->emitError("assign-ckks-placement requires --operator-spec");
    return failure();
  }
  std::ifstream input(path.str(), std::ios::binary);
  if (!input) {
    op->emitError("cannot open placement OperatorSpec: ") << path;
    return failure();
  }
  Json spec = Json::parse(input, nullptr, false);
  if (spec.is_discarded() || !spec.is_object()) {
    op->emitError("placement OperatorSpec is not valid JSON");
    return failure();
  }
  auto version = spec.find("spec_format_version");
  if (version == spec.end() || !version->is_number_integer() ||
      version->get<int64_t>() != 2) {
    op->emitError("placement requires OperatorSpec V2");
    return failure();
  }
  return spec;
}

llvm::StringRef operatorName(Operation *op) {
  if (isa<ckks::AddCCOp>(op))
    return "add_cc";
  if (isa<ckks::AddCPOp>(op))
    return "add_cp";
  if (isa<ckks::MulCCOp>(op))
    return "mul_cc";
  if (isa<ckks::MulCPOp>(op))
    return "mul_cp";
  if (isa<ckks::NegateCOp>(op))
    return "negate";
  if (isa<ckks::RotateCOp>(op))
    return "rotate";
  if (isa<ckks::RescaleCOp>(op))
    return "rescale";
  if (isa<ckks::ModswitchCOp>(op))
    return "mod_switch";
  if (isa<ckks::RelinearizeOp>(op))
    return "relinearize";
  if (isa<ckks::BootstrapCOp>(op))
    return "boot";
  return {};
}

FailureOr<int64_t> operationCost(Operation *op, const Json &spec,
                                 llvm::StringRef bootProfile) {
  llvm::StringRef name = operatorName(op);
  if (name.empty()) {
    op->emitError("unsupported operation in CKKS placement");
    return failure();
  }
  if (op->getNumOperands() == 0 || !ckks::getPolyType(op->getOperand(0))) {
    op->emitError("placed CKKS operation requires a polynomial operand");
    return failure();
  }
  const uint64_t level = ckks::getPolyType(op->getOperand(0)).getLevel();
  auto operators = spec.find("operators");
  if (operators == spec.end() || !operators->is_object()) {
    op->emitError("OperatorSpec is missing operators");
    return failure();
  }
  auto entry = operators->find(name.str());
  if (entry == operators->end() || !entry->is_object()) {
    op->emitError("OperatorSpec is missing operator ") << name;
    return failure();
  }
  auto supported = entry->find("supported");
  if (supported == entry->end() || !supported->is_boolean() ||
      !supported->get<bool>()) {
    op->emitError("OperatorSpec does not support operator ") << name;
    return failure();
  }
  if (name == "boot") {
    if (bootProfile.empty()) {
      op->emitError("CKKS Boot placement requires --boot-profile");
      return failure();
    }
    auto profiles = spec.find("boot_profiles");
    if (profiles == spec.end() || !profiles->is_array()) {
      op->emitError("OperatorSpec is missing boot_profiles");
      return failure();
    }
    const Json *selected = nullptr;
    for (const Json &profile : *profiles) {
      if (!profile.is_object())
        continue;
      auto profileId = profile.find("profile_id");
      if (profileId != profile.end() && profileId->is_string() &&
          profileId->get<std::string>() == bootProfile.str()) {
        if (selected != nullptr) {
          op->emitError("OperatorSpec contains duplicate Boot profile ")
              << bootProfile;
          return failure();
        }
        selected = &profile;
      }
    }
    if (selected == nullptr) {
      op->emitError("OperatorSpec is missing Boot profile ") << bootProfile;
      return failure();
    }
    auto latency = selected->find("latency_us_by_input_level");
    if (latency == selected->end() || !latency->is_array() ||
        level >= latency->size() || !(*latency)[level].is_number_integer()) {
      op->emitError("Boot profile has no integer latency at input level ")
          << level;
      return failure();
    }
    const int64_t cost = (*latency)[level].get<int64_t>();
    if (cost <= 0) {
      op->emitError("Boot profile latency must be positive at input level ")
          << level;
      return failure();
    }
    return cost;
  }
  auto latency = entry->find("latency_us_by_level");
  if (latency == entry->end() || !latency->is_array() ||
      level >= latency->size() || !(*latency)[level].is_number_integer()) {
    op->emitError("OperatorSpec has no integer latency for operator ")
        << name << " at level " << level;
    return failure();
  }
  const int64_t cost = (*latency)[level].get<int64_t>();
  if (cost <= 0) {
    op->emitError("OperatorSpec latency must be positive for operator ")
        << name << " at level " << level;
    return failure();
  }
  return cost;
}

FailureOr<int64_t> checkedAdd(int64_t lhs, int64_t rhs, Operation *op) {
  if (lhs < 0 || rhs < 0 || lhs > std::numeric_limits<int64_t>::max() - rhs) {
    op->emitError("placement cost overflow");
    return failure();
  }
  return lhs + rhs;
}

int64_t earliestSlot(const std::vector<Interval> &intervals, int64_t ready,
                     int64_t duration) {
  int64_t start = ready;
  for (const Interval &interval : intervals) {
    if (start <= interval.start && duration <= interval.start - start)
      return start;
    if (start < interval.finish)
      start = interval.finish;
  }
  return start;
}

void insertInterval(std::vector<Interval> &intervals, Interval added) {
  auto position = std::lower_bound(
      intervals.begin(), intervals.end(), added.start,
      [](const Interval &interval, int64_t start) {
        return interval.start < start;
      });
  intervals.insert(position, added);
}

class PlacementScheduler {
public:
  PlacementScheduler(func::FuncOp func, const Json &spec,
                     std::vector<int64_t> deviceCounts,
                     llvm::StringRef bootProfile, int64_t intraRankCost,
                     int64_t interRankCost)
      : func(func), spec(spec), deviceCounts(std::move(deviceCounts)),
        bootProfile(bootProfile.str()), intraRankCost(intraRankCost),
        interRankCost(interRankCost) {
    const bool cpuTopology = this->deviceCounts.front() == 0;
    for (size_t rank = 0; rank < this->deviceCounts.size(); ++rank) {
      if (cpuTopology) {
        candidates.push_back(Place{static_cast<int64_t>(rank), -1});
        continue;
      }
      for (int64_t device = 0; device < this->deviceCounts[rank]; ++device)
        candidates.push_back(Place{static_cast<int64_t>(rank), device});
    }
    averageCommCost = computeAverageCommunicationCost();
  }

  LogicalResult run() {
    if (!func.getBody().hasOneBlock())
      return func.emitError("placement requires a single-block function");
    if (failed(buildGraph()) || failed(computePriorities()) ||
        failed(schedule()) || failed(verifySchedule()))
      return failure();
    applyPlacement();
    return success();
  }

private:
  LogicalResult buildGraph() {
    size_t logicalId = func.getNumArguments();
    size_t originalIndex = 0;
    for (Operation &op : func.getBody().front()) {
      if (isa<func::ReturnOp>(op))
        continue;
      if (op.getName().getDialectNamespace() != "ckks" ||
          op.getNumResults() != 1 || op.getNumRegions() != 0)
        return op.emitError("placement expects linear single-result CKKS ops");
      op.setAttr("dist.logical_id",
                 IntegerAttr::get(IntegerType::get(func.getContext(), 64),
                                  logicalId++));
      if (isa<ckks::EncodeOp>(op)) {
        setPlace(&op, Place{0, -1});
        origins[op.getResult(0)] = {Place{0, -1}, 0};
        continue;
      }
      FailureOr<int64_t> cost = operationCost(&op, spec, bootProfile);
      if (failed(cost))
        return failure();
      opToNode[&op] = nodes.size();
      Node node;
      node.op = &op;
      node.originalIndex = originalIndex++;
      node.cost = *cost;
      nodes.push_back(std::move(node));
    }
    for (BlockArgument argument : func.getArguments())
      origins[argument] = {Place{0, -1}, 0};

    for (size_t index = 0; index < nodes.size(); ++index) {
      llvm::SmallDenseSet<size_t> uniquePredecessors;
      for (Value operand : nodes[index].op->getOperands()) {
        Operation *definition = operand.getDefiningOp();
        auto found = opToNode.find(definition);
        if (found != opToNode.end())
          uniquePredecessors.insert(found->second);
      }
      for (size_t predecessor : uniquePredecessors) {
        nodes[index].predecessors.push_back(predecessor);
        nodes[predecessor].successors.push_back(index);
      }
      nodes[index].remainingPredecessors = nodes[index].predecessors.size();
    }
    return success();
  }

  int64_t communicationCost(const Place &source,
                            const Place &destination) const {
    if (source == destination)
      return 0;
    return source.rank == destination.rank ? intraRankCost : interRankCost;
  }

  int64_t computeAverageCommunicationCost() const {
    __int128 total = 0;
    for (const Place &source : candidates)
      for (const Place &destination : candidates)
        total += communicationCost(source, destination);
    const __int128 count = static_cast<__int128>(candidates.size());
    return static_cast<int64_t>(total / (count * count));
  }

  FailureOr<int64_t> priorityOf(size_t index,
                                std::vector<unsigned char> &state) {
    if (state[index] == 2)
      return nodes[index].priority;
    if (state[index] == 1) {
      nodes[index].op->emitError("cycle detected in placement graph");
      return failure();
    }
    state[index] = 1;
    int64_t tail = 0;
    for (size_t successor : nodes[index].successors) {
      FailureOr<int64_t> successorPriority = priorityOf(successor, state);
      if (failed(successorPriority))
        return failure();
      FailureOr<int64_t> candidate =
          checkedAdd(averageCommCost, *successorPriority,
                     nodes[index].op);
      if (failed(candidate))
        return failure();
      tail = std::max(tail, *candidate);
    }
    FailureOr<int64_t> result =
        checkedAdd(nodes[index].cost, tail, nodes[index].op);
    if (failed(result))
      return failure();
    nodes[index].priority = *result;
    state[index] = 2;
    return *result;
  }

  LogicalResult computePriorities() {
    std::vector<unsigned char> state(nodes.size(), 0);
    for (size_t index = 0; index < nodes.size(); ++index)
      if (failed(priorityOf(index, state)))
        return failure();
    return success();
  }

  FailureOr<int64_t> arrivalTime(Value operand, const Place &destination,
                                 Operation *consumer) const {
    auto origin = origins.find(operand);
    if (origin == origins.end()) {
      consumer->emitError("operand origin has not been scheduled");
      return failure();
    }
    auto copiesForValue = copies.find(operand);
    if (copiesForValue != copies.end()) {
      auto localCopy = copiesForValue->second.find(destination);
      if (localCopy != copiesForValue->second.end())
        return localCopy->second;
    }
    return checkedAdd(origin->second.second,
                      communicationCost(origin->second.first, destination),
                      consumer);
  }

  LogicalResult scheduleNode(size_t index) {
    Node &node = nodes[index];
    bool found = false;
    Place bestPlace;
    int64_t bestStart = 0;
    int64_t bestFinish = 0;
    for (const Place &candidatePlace : candidates) {
      int64_t ready = 0;
      for (Value operand : node.op->getOperands()) {
        FailureOr<int64_t> arrival =
            arrivalTime(operand, candidatePlace, node.op);
        if (failed(arrival))
          return failure();
        ready = std::max(ready, *arrival);
      }
      int64_t start = earliestSlot(schedules[candidatePlace], ready, node.cost);
      FailureOr<int64_t> finish = checkedAdd(start, node.cost, node.op);
      if (failed(finish))
        return failure();
      if (!found || std::tie(*finish, start, candidatePlace.rank,
                             candidatePlace.device) <
                        std::tie(bestFinish, bestStart, bestPlace.rank,
                                 bestPlace.device)) {
        found = true;
        bestPlace = candidatePlace;
        bestStart = start;
        bestFinish = *finish;
      }
    }
    if (!found)
      return node.op->emitError("no feasible placement candidate");

    node.place = bestPlace;
    node.start = bestStart;
    node.finish = bestFinish;
    node.scheduled = true;
    insertInterval(schedules[bestPlace], Interval{bestStart, bestFinish});
    for (Value operand : node.op->getOperands()) {
      if (!copies[operand].count(bestPlace)) {
        FailureOr<int64_t> arrival = arrivalTime(operand, bestPlace, node.op);
        if (failed(arrival))
          return failure();
        copies[operand][bestPlace] = *arrival;
      }
    }
    origins[node.op->getResult(0)] = {bestPlace, bestFinish};
    copies[node.op->getResult(0)][bestPlace] = bestFinish;
    return success();
  }

  LogicalResult schedule() {
    size_t scheduledCount = 0;
    while (scheduledCount != nodes.size()) {
      size_t selected = nodes.size();
      for (size_t index = 0; index < nodes.size(); ++index) {
        const Node &candidate = nodes[index];
        if (candidate.scheduled || candidate.remainingPredecessors != 0)
          continue;
        if (selected == nodes.size() ||
            std::tie(candidate.priority,
                     nodes[selected].originalIndex) >
                std::tie(nodes[selected].priority, candidate.originalIndex))
          selected = index;
      }
      if (selected == nodes.size())
        return func.emitError("placement graph has no ready operation");
      if (failed(scheduleNode(selected)))
        return failure();
      ++scheduledCount;
      for (size_t successor : nodes[selected].successors) {
        if (nodes[successor].remainingPredecessors == 0)
          return nodes[successor].op->emitError(
              "invalid placement predecessor count");
        --nodes[successor].remainingPredecessors;
      }
    }
    return success();
  }

  LogicalResult verifySchedule() {
    for (const Node &node : nodes) {
      if (!node.scheduled || node.start < 0 || node.finish <= node.start ||
          node.finish - node.start != node.cost)
        return node.op->emitError("placement produced an invalid time interval");
      for (Value operand : node.op->getOperands()) {
        FailureOr<int64_t> arrival =
            arrivalTime(operand, node.place, node.op);
        if (failed(arrival))
          return failure();
        if (node.start < *arrival)
          return node.op->emitError(
              "placement starts before a dependency can arrive");
      }
    }
    for (const auto &[place, intervals] : schedules) {
      static_cast<void>(place);
      for (size_t index = 1; index < intervals.size(); ++index)
        if (intervals[index - 1].finish > intervals[index].start)
          return func.emitError("placement overlaps operations at one place");
    }
    return success();
  }

  void setPlace(Operation *op, const Place &place) {
    Builder builder(op->getContext());
    op->setAttr("dist.rank", builder.getI64IntegerAttr(place.rank));
    op->setAttr("dist.device", builder.getI64IntegerAttr(place.device));
  }

  void applyPlacement() {
    Builder builder(func.getContext());
    SmallVector<int64_t> counts(deviceCounts.begin(), deviceCounts.end());
    func->setAttr("dist.device_counts", builder.getDenseI64ArrayAttr(counts));
    func->setAttr("dist.intra_rank_communication_cost",
                  builder.getI64IntegerAttr(intraRankCost));
    func->setAttr("dist.inter_rank_communication_cost",
                  builder.getI64IntegerAttr(interRankCost));
    for (unsigned index = 0; index < func.getNumArguments(); ++index) {
      func.setArgAttr(index, "dist.rank", builder.getI64IntegerAttr(0));
      func.setArgAttr(index, "dist.device", builder.getI64IntegerAttr(-1));
    }
    for (Node &node : nodes) {
      setPlace(node.op, node.place);
      node.op->setAttr("dist.schedule_start",
                       builder.getI64IntegerAttr(node.start));
      node.op->setAttr("dist.schedule_finish",
                       builder.getI64IntegerAttr(node.finish));
    }

    std::vector<Node *> launchOrder;
    launchOrder.reserve(nodes.size());
    for (Node &node : nodes)
      launchOrder.push_back(&node);
    std::sort(launchOrder.begin(), launchOrder.end(), [](const Node *lhs,
                                                         const Node *rhs) {
      return std::tie(lhs->start, lhs->finish, lhs->place.rank,
                      lhs->place.device, lhs->originalIndex) <
             std::tie(rhs->start, rhs->finish, rhs->place.rank,
                      rhs->place.device, rhs->originalIndex);
    });
    Operation *terminator = func.getBody().front().getTerminator();
    for (Node *node : launchOrder)
      node->op->moveBefore(terminator);
  }

  func::FuncOp func;
  const Json &spec;
  std::vector<int64_t> deviceCounts;
  std::string bootProfile;
  int64_t intraRankCost;
  int64_t interRankCost;
  int64_t averageCommCost = 0;
  std::vector<Place> candidates;
  std::vector<Node> nodes;
  llvm::DenseMap<Operation *, size_t> opToNode;
  llvm::DenseMap<Value, std::pair<Place, int64_t>> origins;
  llvm::DenseMap<Value, std::map<Place, int64_t>> copies;
  std::map<Place, std::vector<Interval>> schedules;
};

struct AssignPlacementPass
    : public hecate::ckks::impl::AssignPlacementBase<AssignPlacementPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (intraRankCommunicationCost <= 0 || interRankCommunicationCost <= 0) {
      func.emitError("placement communication costs must be positive");
      signalPassFailure();
      return;
    }
    FailureOr<std::vector<int64_t>> counts =
        parseDeviceCounts(deviceCounts, func);
    FailureOr<Json> spec = readOperatorSpec(operatorSpecPath, func);
    if (failed(counts) || failed(spec)) {
      signalPassFailure();
      return;
    }
    PlacementScheduler scheduler(func, *spec, std::move(*counts), bootProfile,
                                 intraRankCommunicationCost,
                                 interRankCommunicationCost);
    if (failed(scheduler.run()))
      signalPassFailure();
  }
};

} // namespace
