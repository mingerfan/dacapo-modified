#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "hecate/Dialect/CKKS/Transforms/RuntimePlanUtils.h"
#include "hecate/Dialect/Dist/IR/DistOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>

namespace hecate::ckks {
#define GEN_PASS_DEF_PLANPLAINTEXTSTREAMING
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
}
using namespace mlir;
namespace ckks = hecate::ckks;
namespace dist = hecate::dist;
namespace {
using Json = nlohmann::json;
uint64_t addBytes(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::runtime_error("streaming byte count overflow");
  return a + b;
}
uint64_t multiplyBytes(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::runtime_error("streaming byte count overflow");
  return a * b;
}
struct PlanPlaintextStreamingPass
    : ckks::impl::PlanPlaintextStreamingBase<PlanPlaintextStreamingPass> {
  using Base::Base;
  void runOnOperation() override {
    auto func = getOperation();
    try { schedule(func); }
    catch (const std::exception &error) {
      func.emitError(error.what());
      signalPassFailure();
    }
  }
  void schedule(func::FuncOp func) {
    if (!gpuBudget || !hostBudget || !pinnedBudget || !rawBudget || prefix.empty())
      throw std::runtime_error("streaming requires positive GPU, Host, pinned, raw budgets and report prefix");
    if (func->hasAttr("runtime.memory_planned") || func->hasAttr("runtime.streaming"))
      throw std::runtime_error("streaming must precede memory planning and run once");
    auto counts = func->getAttrOfType<DenseI64ArrayAttr>("dist.device_counts");
    if (!counts || counts.size() != 1 || counts.asArrayRef()[0] < 1)
      throw std::runtime_error("streaming requires one placed rank with local GPUs");
    auto layout = ckks::buildRuntimePlanLayout(func);
    if (failed(layout)) throw std::runtime_error("invalid streaming layout");
    std::ifstream specInput(operatorSpecPath.getValue());
    if (!specInput) throw std::runtime_error("streaming cannot open physical OperatorSpec");
    auto spec = Json::parse(specInput);
    const uint64_t degree = spec.at("context").at("poly_degree").get<uint64_t>();
    if (!degree || degree > (1ULL << 24)) throw std::runtime_error("invalid polynomial degree");
    auto placeOf = [&](Value v) {
      auto place = ckks::runtimePlace(v, func);
      if (failed(place)) throw std::runtime_error("missing streaming value place");
      return place->device + 1; // Host=0, GPU i=i+1.
    };
    auto bytesOf = [&](Value v) -> uint64_t {
      auto type = ckks::getPolyType(v);
      if (!type || !type.getComponents() || type.getLevel() < 0)
        throw std::runtime_error("streaming requires physical CKKS types");
      uint64_t words = uint64_t(type.getLevel()) + 1;
      if (words > (1ULL << 20)) throw std::runtime_error("invalid physical limb count");
      return multiplyBytes(multiplyBytes(multiplyBytes(degree, words),
                                         type.getComponents()), placeOf(v) == 0 ? 8 : 4);
    };
    const size_t places = counts.asArrayRef()[0] + 1;
    std::vector<uint64_t> live(places), batch(places), peak(places);
    llvm::DenseMap<Value, Operation *> lastUse;
    llvm::DenseSet<Value> returned, external;
    auto ret = cast<func::ReturnOp>(func.getBody().front().getTerminator());
    for (Value v : ret.getOperands()) returned.insert(v);
    for (Value v : func.getArguments()) {
      external.insert(v);
      live.at(placeOf(v)) = addBytes(live.at(placeOf(v)), bytesOf(v));
    }
    // Map only Encode and its direct physical copies; other operations keep order.
    llvm::DenseMap<Value, Value> roots;
    SmallVector<Operation *> originals, work;
    uint64_t nextId = layout->values.empty() ? 0 : layout->values.rbegin()->first + 1;
    uint64_t transferId = 0;
    for (const auto &inst : layout->instructions) {
      Operation *op = inst.operation;
      if (isa<dist::ReleaseOp, dist::FenceOp>(op))
        throw std::runtime_error("streaming requires IR without Release/Fence");
      if (auto t = dyn_cast<dist::TransferOp>(op))
        transferId = std::max(transferId, uint64_t(t.getTransferId()) + 1);
      if (isa<ckks::EncodeOp>(op)) {
        roots[op->getResult(0)] = op->getResult(0);
        originals.push_back(op);
      } else if (auto t = dyn_cast<dist::TransferOp>(op); t && roots.count(t.getInput())) {
        if (!isa<ckks::EncodeOp>(t.getInput().getDefiningOp()))
          throw std::runtime_error("streaming supports direct Host Encode uploads only");
        roots[t.getResult()] = roots.lookup(t.getInput());
        originals.push_back(op);
      } else work.push_back(op);
      for (Value input : op->getOperands()) lastUse[input] = op;
      for (Value output : op->getResults())
        if (!lastUse.count(output)) lastUse[output] = op;
    }
    for (Value v : ret.getOperands())
      if (roots.count(v)) throw std::runtime_error("streaming does not support plaintext final outputs");
    OpBuilder builder(func.getContext());
    // Keep IDs of existing computation values, allocate distinct IDs for each copy.
    for (const auto &[id, v] : layout->values)
      if (auto *op = v.getDefiningOp()) op->setAttr("runtime.value_id", builder.getI64IntegerAttr(id));
    llvm::DenseMap<Value, Value> copies;
    llvm::DenseSet<Value> encodedRoots;
    SmallVector<Operation *> preparations;
    Operation *first = nullptr;
    uint64_t pinned = 0, pinnedPeak = 0, rawPeak = 0, prefetched = 0;
    uint64_t batches = 0, encodes = 0, uploads = 0, uploadBytes = 0;
    batch = live;
    auto check = [&](const std::vector<uint64_t> &usage, uint64_t staging) -> std::string {
      for (size_t p = 0; p < places; ++p) {
        uint64_t budget = p == 0 ? hostBudget : gpuBudget;
        if (usage[p] > budget)
          return (p == 0 ? "Host" : "GPU" + std::to_string(p - 1)) +
              " requires " + std::to_string(usage[p]) + " bytes, budget " + std::to_string(budget);
      }
      if (staging > pinnedBudget)
        return "pinned requires " + std::to_string(staging) + " bytes, budget " + std::to_string(pinnedBudget);
      return {};
    };
    auto closeBatch = [&] {
      if (!first) return;
      // Budgeted lookahead: hoist at most prefetchBytes of preparations. Without
      // lookahead each encoding/upload remains immediately before its consumer.
      for (Operation *op : preparations) op->moveBefore(first);
      builder.setInsertionPoint(ret);
      builder.create<dist::FenceOp>(ret.getLoc());
      for (size_t p = 0; p < places; ++p) peak[p] = std::max(peak[p], batch[p]);
      pinnedPeak = std::max(pinnedPeak, pinned);
      ++batches;
      copies.clear(); preparations.clear(); first = nullptr;
      batch = live; pinned = 0; prefetched = 0;
    };
    for (Operation *op : work) {
      // Discover this operation's missing weights once (bounded operand count).
      SmallVector<Value> needed;
      auto addWeight = [&](Value v) {
        if (!copies.count(v) && !llvm::is_contained(needed, v)) needed.push_back(v);
      };
      auto gather = [&] {
        needed.clear();
        for (Value v : op->getOperands()) if (roots.count(v)) {
          addWeight(roots.lookup(v)); addWeight(v);
        }
      };
      std::vector<uint64_t> proposed;
      uint64_t proposedPinned = 0;
      auto estimate = [&] {
        proposed = batch; proposedPinned = pinned;
        for (Value v : needed) {
          proposed.at(placeOf(v)) = addBytes(proposed.at(placeOf(v)), bytesOf(v));
          if (placeOf(v)) proposedPinned = addBytes(proposedPinned, bytesOf(v));
        }
        if (auto transfer = dyn_cast<dist::TransferOp>(op)) {
          const auto source = placeOf(transfer.getInput());
          const auto destination = placeOf(transfer.getResult());
          if ((source == 0) != (destination == 0))
            proposedPinned = addBytes(proposedPinned, bytesOf(
                source == 0 ? transfer.getResult() : transfer.getInput()));
        }
        for (Value v : op->getResults()) {
          auto p = placeOf(v);
          proposed[p] = addBytes(proposed[p], bytesOf(v));
          if (p && op->getName().getDialectNamespace() == "ckks")
            proposed[p] = addBytes(proposed[p], workspaceBytes);
        }
      };
      gather(); estimate();
      if (!check(proposed, proposedPinned).empty() && first) {
        closeBatch(); gather(); estimate();
      }
      auto error = check(proposed, proposedPinned);
      if (!error.empty())
        throw std::runtime_error("indivisible operation " + op->getName().getStringRef().str() +
            " ValueId " + std::to_string(layout->ids.lookup(op->getResult(0))) + ": " + error);
      for (Value v : needed) {
        if (auto encode = dyn_cast<ckks::EncodeOp>(v.getDefiningOp())) {
          auto payload = dyn_cast<DenseElementsAttr>(encode.getPayload());
          if (!payload) throw std::runtime_error("streaming requires dense Encode payload");
          if (payload.getNumElements() <= 0 || uint64_t(payload.getNumElements()) > degree / 2)
            throw std::runtime_error("streaming Encode payload exceeds CKKS slot capacity");
          uint64_t raw = multiplyBytes(uint64_t(payload.getNumElements()), 16); // file + decoded vector
          if (raw > rawBudget) throw std::runtime_error("raw Encode payload requires " +
              std::to_string(raw) + " bytes, budget " + std::to_string(rawBudget.getValue()));
          rawPeak = std::max(rawPeak, raw); ++encodes;
          encodedRoots.insert(v);
        } else { ++uploads; uploadBytes = addBytes(uploadBytes, bytesOf(v)); }
        if (nextId >= uint64_t(std::numeric_limits<int64_t>::max()) ||
            transferId >= uint64_t(std::numeric_limits<int64_t>::max()))
          throw std::runtime_error("streaming ValueId/TransferId space exhausted");
        builder.setInsertionPoint(ret);
        Operation *copy = builder.clone(*v.getDefiningOp());
        if (isa<dist::TransferOp>(copy)) {
          copy->setOperand(0, copies.lookup(roots.lookup(v)));
          copy->setAttr("transfer_id", builder.getI64IntegerAttr(transferId++));
          copy->setAttr("initialization", builder.getBoolAttr(false));
        } else copy->setAttr("runtime.streaming", builder.getUnitAttr());
        copy->setAttr("runtime.value_id", builder.getI64IntegerAttr(nextId));
        if (isa<ckks::EncodeOp>(copy)) copy->setAttr("dist.logical_id", builder.getI64IntegerAttr(nextId));
        ++nextId;
        copies[v] = copy->getResult(0);
        // Hoist an Encode together with its dependent uploads only when the
        // whole accumulated prefix fits the byte lookahead allowance.
        prefetched = addBytes(prefetched, bytesOf(v));
        if (prefetched <= prefetchBytes) preparations.push_back(copy);
      }
      llvm::DenseSet<Value> retired;
      for (OpOperand &operand : op->getOpOperands()) {
        Value old = operand.get();
        if (roots.count(old)) operand.set(copies.lookup(old));
        else if (!external.count(old) && !returned.count(old) && lastUse.lookup(old) == op && retired.insert(old).second)
          live.at(placeOf(old)) -= bytesOf(old);
      }
      for (Value v : op->getResults())
        if (returned.count(v) || lastUse.lookup(v) != op)
          live.at(placeOf(v)) = addBytes(live.at(placeOf(v)), bytesOf(v));
      if (auto t = dyn_cast<dist::TransferOp>(op))
        t->setAttr("initialization", builder.getBoolAttr(false));
      op->moveBefore(ret);
      if (!first) first = op;
      batch = std::move(proposed); pinned = proposedPinned;
    }
    closeBatch();
    for (Operation *op : llvm::reverse(originals)) {
      if (!op->use_empty()) throw std::runtime_error("unhandled streamed plaintext use");
      op->erase();
    }
    func->setAttr("runtime.streaming", builder.getUnitAttr());
    Json report = {{"format_version", 1}, {"batches", batches}, {"encodes", encodes},
      {"distinct_encode_definitions", encodedRoots.size()},
      {"reencodes", encodes - encodedRoots.size()},
      {"uploads", uploads}, {"upload_bytes", uploadBytes}, {"object_peak_bytes_host_then_gpu", peak},
      {"pinned_peak_bytes", pinnedPeak}, {"raw_peak_bytes", rawPeak},
      {"gpu_budget_bytes", gpuBudget.getValue()}, {"host_budget_bytes", hostBudget.getValue()},
      {"pinned_budget_bytes", pinnedBudget.getValue()}, {"raw_budget_bytes", rawBudget.getValue()},
      {"prefetch_bytes", prefetchBytes.getValue()}, {"workspace_bytes_per_op", workspaceBytes.getValue()},
      {"assumption", "Batch-start live RNS objects plus all batch allocations; Release returns no credit before Fence. External inputs remain retained. No raw cache."},
      {"excluded", {"keys", "parameter tables", "unmeasured operator workspace", "allocator pools and fragmentation", "encoder workspace", "object metadata"}},
      {"total_device_memory_bound", false}};
    std::ofstream out(prefix.getValue() + "." + func.getName().str() + ".streaming.json");
    out << report.dump(2) << '\n';
    if (!out) throw std::runtime_error("cannot write streaming report");
  }
};
}
