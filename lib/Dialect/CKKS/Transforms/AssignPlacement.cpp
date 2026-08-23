#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
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
  bool requiresHost = false;
};

struct RatePoint {
  int64_t payloadBytes = 0;
  double rateBytesPerMicrosecond = 0;
};

struct LinkModel {
  int64_t startupLatencyMicroseconds = 0;
  double maxRateBytesPerMicrosecond = 0;
  int64_t saturationBytes = 0;
  std::vector<RatePoint> ratePoints;
};

enum class EndpointKind { Host, Device };

struct EndpointSelector {
  std::optional<EndpointKind> kind;
  std::optional<int64_t> rank;
  std::optional<int64_t> device;
  std::optional<int64_t> node;
};

struct CommunicationRule {
  std::string id;
  EndpointSelector source;
  EndpointSelector destination;
  bool bidirectional = false;
  std::string transport = "auto";
  LinkModel model;
};

struct ResolvedLink {
  const LinkModel *model = nullptr;
  std::string ruleId;
  std::string transport = "auto";
};

struct CommunicationProfile {
  int64_t coefficientBytes = 0;
  int64_t polyDegree = 0;
  int64_t modulusCount = 0;
  int64_t formatVersion = 1;
  std::vector<int64_t> rankToNode;
  std::optional<LinkModel> hostDevice;
  std::optional<LinkModel> intraRank;
  std::optional<LinkModel> interRank;
  std::vector<CommunicationRule> rules;

  FailureOr<ResolvedLink> resolve(const Place &source,
                                   const Place &destination,
                                   Operation *diagnostic) const;
  LogicalResult validateTopology(llvm::ArrayRef<int64_t> deviceCounts,
                                 Operation *diagnostic) const;
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
  const bool cpuTopology = std::all_of(
      result.begin(), result.end(), [](int64_t count) { return count == 0; });
  const bool deviceTopology = std::all_of(
      result.begin(), result.end(), [](int64_t count) { return count > 0; });
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

FailureOr<LinkModel> readLinkModelObject(const Json &entry,
                                         llvm::StringRef name,
                                         Operation *op) {
  if (!entry.is_object()) {
    op->emitError("communication profile is missing link model ") << name;
    return failure();
  }
  auto startup = entry.find("startup_latency_us");
  auto maxRate = entry.find("max_rate_bytes_per_us");
  auto saturation = entry.find("saturation_bytes");
  if (startup == entry.end() || !startup->is_number_integer() ||
      startup->get<int64_t>() < 0) {
    op->emitError("communication profile ")
        << name << ".startup_latency_us must be a nonnegative integer";
    return failure();
  }
  if (maxRate == entry.end() || !maxRate->is_number()) {
    op->emitError("communication profile ")
        << name << ".max_rate_bytes_per_us must be a positive number";
    return failure();
  }
  const double maxRateValue = maxRate->get<double>();
  if (!std::isfinite(maxRateValue) || maxRateValue <= 0) {
    op->emitError("communication profile ")
        << name << ".max_rate_bytes_per_us must be a positive finite number";
    return failure();
  }
  if (saturation == entry.end() || !saturation->is_number_integer() ||
      saturation->get<int64_t>() <= 0) {
    op->emitError("communication profile ")
        << name << ".saturation_bytes must be a positive integer";
    return failure();
  }

  LinkModel model;
  model.startupLatencyMicroseconds = startup->get<int64_t>();
  model.maxRateBytesPerMicrosecond = maxRateValue;
  model.saturationBytes = saturation->get<int64_t>();
  auto points = entry.find("rate_points");
  if (points == entry.end())
    return model;
  if (!points->is_array() || points->empty()) {
    op->emitError("communication profile ")
        << name << ".rate_points must be a nonempty array when present";
    return failure();
  }
  for (size_t index = 0; index < points->size(); ++index) {
    const Json &point = (*points)[index];
    if (!point.is_object()) {
      op->emitError("communication profile ")
          << name << ".rate_points entries must be objects";
      return failure();
    }
    auto payload = point.find("payload_bytes");
    auto rate = point.find("rate_bytes_per_us");
    if (payload == point.end() || !payload->is_number_integer() ||
        payload->get<int64_t>() <= 0 || rate == point.end() ||
        !rate->is_number()) {
      op->emitError("communication profile ")
          << name
          << ".rate_points require positive payload_bytes and "
             "rate_bytes_per_us";
      return failure();
    }
    const int64_t payloadValue = payload->get<int64_t>();
    const double rateValue = rate->get<double>();
    if (!std::isfinite(rateValue) || rateValue <= 0 ||
        rateValue > model.maxRateBytesPerMicrosecond) {
      op->emitError("communication profile ")
          << name
          << ".rate_points rates must be positive, finite and no greater than "
             "max_rate_bytes_per_us";
      return failure();
    }
    if (!model.ratePoints.empty() &&
        (payloadValue <= model.ratePoints.back().payloadBytes ||
         rateValue < model.ratePoints.back().rateBytesPerMicrosecond)) {
      op->emitError("communication profile ")
          << name
          << ".rate_points must have increasing payloads and nondecreasing "
             "rates";
      return failure();
    }
    model.ratePoints.push_back({payloadValue, rateValue});
  }
  return model;
}

FailureOr<LinkModel> readLinkModel(const Json &links, llvm::StringRef name,
                                   Operation *op) {
  auto entry = links.find(name.str());
  if (entry == links.end() || !entry->is_object()) {
    op->emitError("communication profile is missing link model ") << name;
    return failure();
  }
  return readLinkModelObject(*entry, name, op);
}

LogicalResult readSelectorInteger(const Json &object, llvm::StringRef field,
                                  std::optional<int64_t> &result,
                                  Operation *diagnostic) {
  auto value = object.find(field.str());
  if (value == object.end())
    return success();
  if (value->is_string() && value->get<std::string>() == "*") {
    result.reset();
    return success();
  }
  if (!value->is_number_integer() || value->get<int64_t>() < 0) {
    diagnostic->emitError("communication rule selector ")
        << field << " must be a nonnegative integer or '*';";
    return failure();
  }
  result = value->get<int64_t>();
  return success();
}

FailureOr<EndpointSelector> readEndpointSelector(const Json &value,
                                                  llvm::StringRef path,
                                                  Operation *diagnostic) {
  EndpointSelector result;
  if (value.is_string() && value.get<std::string>() == "*")
    return result;
  if (!value.is_object()) {
    diagnostic->emitError("communication rule ")
        << path << " must be an object or '*';";
    return failure();
  }

  auto kind = value.find("kind");
  if (kind != value.end()) {
    if (!kind->is_string()) {
      diagnostic->emitError("communication rule ")
          << path << ".kind must be 'host' or 'device';";
      return failure();
    }
    const std::string kindName = kind->get<std::string>();
    if (kindName == "host")
      result.kind = EndpointKind::Host;
    else if (kindName == "device")
      result.kind = EndpointKind::Device;
    else {
      diagnostic->emitError("communication rule ")
          << path << ".kind must be 'host' or 'device';";
      return failure();
    }
  }

  if (failed(readSelectorInteger(value, "rank", result.rank, diagnostic)) ||
      failed(readSelectorInteger(value, "device", result.device, diagnostic)) ||
      failed(readSelectorInteger(value, "node", result.node, diagnostic)))
    return failure();

  if (result.kind == EndpointKind::Host && result.device.has_value()) {
    diagnostic->emitError("communication rule ")
        << path << " cannot constrain device for a host endpoint;";
    return failure();
  }
  return result;
}

FailureOr<CommunicationRule> readCommunicationRule(const Json &value,
                                                    std::size_t index,
                                                    Operation *diagnostic) {
  if (!value.is_object()) {
    diagnostic->emitError("communication profile rules[")
        << index << "] must be an object;";
    return failure();
  }

  CommunicationRule result;
  auto id = value.find("id");
  if (id == value.end()) {
    result.id = "rule_" + std::to_string(index);
  } else if (!id->is_string() || id->get<std::string>().empty()) {
    diagnostic->emitError("communication profile rules[")
        << index << "].id must be a nonempty string;";
    return failure();
  } else {
    result.id = id->get<std::string>();
  }

  auto source = value.find("from");
  auto destination = value.find("to");
  if (source == value.end() || destination == value.end()) {
    diagnostic->emitError("communication profile rules[")
        << index << "] requires 'from' and 'to';";
    return failure();
  }
  FailureOr<EndpointSelector> parsedSource = readEndpointSelector(
      *source, "rules[" + std::to_string(index) + "].from", diagnostic);
  FailureOr<EndpointSelector> parsedDestination = readEndpointSelector(
      *destination, "rules[" + std::to_string(index) + "].to", diagnostic);
  if (failed(parsedSource) || failed(parsedDestination))
    return failure();
  result.source = std::move(*parsedSource);
  result.destination = std::move(*parsedDestination);

  auto direction = value.find("direction");
  if (direction != value.end()) {
    if (!direction->is_string()) {
      diagnostic->emitError("communication profile rules[")
          << index << "].direction must be 'forward' or 'both';";
      return failure();
    }
    const std::string directionName = direction->get<std::string>();
    if (directionName == "both")
      result.bidirectional = true;
    else if (directionName != "forward") {
      diagnostic->emitError("communication profile rules[")
          << index << "].direction must be 'forward' or 'both';";
      return failure();
    }
  }

  auto transport = value.find("transport");
  if (transport != value.end()) {
    if (!transport->is_string() || transport->get<std::string>().empty()) {
      diagnostic->emitError("communication profile rules[")
          << index << "].transport must be a nonempty string;";
      return failure();
    }
    result.transport = transport->get<std::string>();
  }

  auto cost = value.find("cost");
  if (cost == value.end() || !cost->is_object()) {
    diagnostic->emitError("communication profile rules[")
        << index << "].cost must be an object;";
    return failure();
  }
  FailureOr<LinkModel> model = readLinkModelObject(
      *cost, "rules[" + std::to_string(index) + "].cost", diagnostic);
  if (failed(model))
    return failure();
  result.model = std::move(*model);
  return result;
}

bool selectorUsesNode(const EndpointSelector &selector) {
  return selector.node.has_value();
}

std::optional<int64_t> nodeForRank(const std::vector<int64_t> &rankToNode,
                                   int64_t rank) {
  if (rank < 0 || rank >= static_cast<int64_t>(rankToNode.size()))
    return std::nullopt;
  return rankToNode[static_cast<std::size_t>(rank)];
}

bool selectorMatches(const EndpointSelector &selector, const Place &place,
                     std::optional<int64_t> node) {
  const EndpointKind actualKind =
      place.device < 0 ? EndpointKind::Host : EndpointKind::Device;
  if (selector.kind.has_value() && selector.kind.value() != actualKind)
    return false;
  if (selector.rank.has_value() && selector.rank.value() != place.rank)
    return false;
  if (selector.device.has_value() && selector.device.value() != place.device)
    return false;
  if (selector.node.has_value() &&
      (!node.has_value() || selector.node.value() != node.value()))
    return false;
  return true;
}

bool ruleMatches(const CommunicationRule &rule, const Place &source,
                 const Place &destination,
                 std::optional<int64_t> sourceNode,
                 std::optional<int64_t> destinationNode) {
  const bool forward =
      selectorMatches(rule.source, source, sourceNode) &&
      selectorMatches(rule.destination, destination, destinationNode);
  if (forward)
    return true;
  if (!rule.bidirectional)
    return false;
  return selectorMatches(rule.source, destination, destinationNode) &&
         selectorMatches(rule.destination, source, sourceNode);
}

FailureOr<CommunicationProfile>
readCommunicationProfile(llvm::StringRef path, const Json &operatorSpec,
                         Operation *op) {
  std::ifstream input(path.str(), std::ios::binary);
  if (!input) {
    op->emitError("cannot open communication profile: ") << path;
    return failure();
  }
  Json profile = Json::parse(input, nullptr, false);
  if (profile.is_discarded() || !profile.is_object()) {
    op->emitError("communication profile is not valid JSON");
    return failure();
  }
  auto version = profile.find("format_version");
  auto coefficientBytes = profile.find("coefficient_bytes");
  auto links = profile.find("links");
  if (version == profile.end() || !version->is_number_integer() ||
      (version->get<int64_t>() != 1 && version->get<int64_t>() != 2)) {
    op->emitError("communication profile format_version must be 1 or 2");
    return failure();
  }
  const int64_t formatVersion = version->get<int64_t>();
  if (coefficientBytes == profile.end() ||
      !coefficientBytes->is_number_integer() ||
      coefficientBytes->get<int64_t>() <= 0) {
    op->emitError("communication profile coefficient_bytes must be positive");
    return failure();
  }
  if (formatVersion == 1 &&
      (links == profile.end() || !links->is_object())) {
    op->emitError("communication profile links must be an object");
    return failure();
  }

  auto context = operatorSpec.find("context");
  if (context == operatorSpec.end() || !context->is_object()) {
    op->emitError("OperatorSpec context must be an object");
    return failure();
  }
  auto polyDegree = context->find("poly_degree");
  auto moduli = context->find("rns_moduli_log2");
  if (polyDegree == context->end() || !polyDegree->is_number_integer() ||
      polyDegree->get<int64_t>() <= 0 || moduli == context->end() ||
      !moduli->is_array() || moduli->empty() ||
      moduli->size() >
          static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    op->emitError(
        "OperatorSpec context has invalid poly_degree or rns_moduli_log2");
    return failure();
  }

  CommunicationProfile result;
  result.formatVersion = formatVersion;
  result.coefficientBytes = coefficientBytes->get<int64_t>();
  result.polyDegree = polyDegree->get<int64_t>();
  result.modulusCount = static_cast<int64_t>(moduli->size());

  if (links != profile.end()) {
    if (!links->is_object()) {
      op->emitError("communication profile links must be an object");
      return failure();
    }
    auto readOptional = [&](llvm::StringRef name,
                            std::optional<LinkModel> &destination) {
      auto entry = links->find(name.str());
      if (entry == links->end())
        return success();
      FailureOr<LinkModel> model = readLinkModel(*links, name, op);
      if (failed(model))
        return failure();
      destination = std::move(*model);
      return success();
    };
    if (failed(readOptional("host_device", result.hostDevice)) ||
        failed(readOptional("intra_rank", result.intraRank)) ||
        failed(readOptional("inter_rank", result.interRank)))
      return failure();
  }

  if (formatVersion == 1 &&
      (!result.hostDevice.has_value() || !result.intraRank.has_value() ||
       !result.interRank.has_value())) {
    op->emitError(
        "communication profile V1 requires host_device, intra_rank and "
        "inter_rank link models");
    return failure();
  }

  auto topology = profile.find("topology");
  if (topology != profile.end()) {
    if (!topology->is_object()) {
      op->emitError("communication profile topology must be an object");
      return failure();
    }
    auto rankToNode = topology->find("rank_to_node");
    if (rankToNode == topology->end() || !rankToNode->is_array() ||
        rankToNode->empty()) {
      op->emitError(
          "communication profile topology.rank_to_node must be a nonempty "
          "array");
      return failure();
    }
    for (const Json &node : *rankToNode) {
      if (!node.is_number_integer() || node.get<int64_t>() < 0) {
        op->emitError(
            "communication profile topology.rank_to_node values must be "
            "nonnegative integers");
        return failure();
      }
      result.rankToNode.push_back(node.get<int64_t>());
    }
  }

  auto rules = profile.find("rules");
  if (rules != profile.end()) {
    if (!rules->is_array()) {
      op->emitError("communication profile rules must be an array");
      return failure();
    }
    for (std::size_t index = 0; index < rules->size(); ++index) {
      FailureOr<CommunicationRule> rule =
          readCommunicationRule((*rules)[index], index, op);
      if (failed(rule))
        return failure();
      result.rules.push_back(std::move(*rule));
    }
  }

  if (formatVersion == 2 && result.rules.empty() &&
      !result.hostDevice.has_value() && !result.intraRank.has_value() &&
      !result.interRank.has_value()) {
    op->emitError(
        "communication profile V2 requires rules or legacy link models");
    return failure();
  }
  return result;
}

FailureOr<ResolvedLink> CommunicationProfile::resolve(
    const Place &source, const Place &destination, Operation *diagnostic) const {
  const std::optional<int64_t> sourceNode = nodeForRank(rankToNode, source.rank);
  const std::optional<int64_t> destinationNode =
      nodeForRank(rankToNode, destination.rank);

  for (const CommunicationRule &rule : rules) {
    if (!ruleMatches(rule, source, destination, sourceNode, destinationNode))
      continue;
    return ResolvedLink{&rule.model, rule.id, rule.transport};
  }

  const LinkModel *fallback = nullptr;
  std::string fallbackId;
  if (source.rank == destination.rank) {
    if (source.device < 0 || destination.device < 0) {
      if (hostDevice.has_value()) {
        fallback = &*hostDevice;
        fallbackId = "legacy.host_device";
      }
    } else if (intraRank.has_value()) {
      fallback = &*intraRank;
      fallbackId = "legacy.intra_rank";
    }
  } else if (interRank.has_value()) {
    fallback = &*interRank;
    fallbackId = "legacy.inter_rank";
  }

  if (fallback != nullptr)
    return ResolvedLink{fallback, std::move(fallbackId), "auto"};

  diagnostic->emitError("communication profile has no matching rule or "
                        "legacy fallback for Place pair ")
      << "(" << source.rank << "," << source.device << ") -> ("
      << destination.rank << "," << destination.device << ")";
  return failure();
}

LogicalResult CommunicationProfile::validateTopology(
    llvm::ArrayRef<int64_t> deviceCounts, Operation *diagnostic) const {
  bool usesNodeSelector = false;
  for (const CommunicationRule &rule : rules)
    usesNodeSelector = usesNodeSelector || selectorUsesNode(rule.source) ||
                       selectorUsesNode(rule.destination);

  if (rankToNode.empty()) {
    if (usesNodeSelector) {
      diagnostic->emitError(
          "communication rules use node selectors but topology.rank_to_node "
          "is missing");
      return failure();
    }
    return success();
  }

  if (rankToNode.size() != deviceCounts.size()) {
    diagnostic->emitError(
        "communication topology rank_to_node length does not match "
        "device-counts rank count");
    return failure();
  }
  return success();
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

FailureOr<int64_t> rotateDecompositionTermCount(ckks::RotateCOp op,
                                                const Json &spec) {
  auto context = spec.find("context");
  if (context == spec.end() || !context->is_object()) {
    op.emitError("OperatorSpec context must be an object");
    return failure();
  }
  auto polyDegree = context->find("poly_degree");
  if (polyDegree == context->end() || !polyDegree->is_number_integer()) {
    op.emitError("OperatorSpec context poly_degree must be an integer");
    return failure();
  }
  const int64_t degree = polyDegree->get<int64_t>();
  const int64_t slotCount = degree / 2;
  if (degree < 2 ||
      slotCount > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    op.emitError("OperatorSpec slot count is outside the supported rotation "
                 "range");
    return failure();
  }

  if (op.getOffset().size() != 1 || op.getOffset()[0] == 0 ||
      op.getOffset()[0] < std::numeric_limits<int32_t>::min() ||
      op.getOffset()[0] > std::numeric_limits<int32_t>::max()) {
    op.emitError("placement Rotate requires exactly one nonzero int32 step");
    return failure();
  }
  int64_t normalized = op.getOffset()[0] % slotCount;
  const int64_t half = slotCount / 2;
  if (normalized > half)
    normalized -= slotCount;
  else if (normalized <= -half)
    normalized += slotCount;
  if (normalized == 0) {
    op.emitError("Rotate step becomes zero after slot-count normalization");
    return failure();
  }

  uint64_t magnitude =
      static_cast<uint64_t>(normalized < 0 ? -normalized : normalized);
  int64_t terms = 0;
  while (magnitude != 0) {
    terms += static_cast<int64_t>(magnitude & 1);
    magnitude >>= 1;
  }
  return terms;
}

FailureOr<int64_t> operationCost(Operation *op, const Json &spec,
                                 llvm::StringRef bootProfile,
                                 bool &requiresHost) {
  requiresHost = false;
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
    auto implementation = selected->find("implementation");
    if (implementation == selected->end() || !implementation->is_string()) {
      op->emitError("Boot profile is missing implementation");
      return failure();
    }
    const std::string implementationName = implementation->get<std::string>();
    if (implementationName != "native" &&
        implementationName != "decrypt_reencrypt") {
      op->emitError("Boot profile implementation must be native or "
                    "decrypt_reencrypt");
      return failure();
    }
    requiresHost = implementationName == "decrypt_reencrypt";
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
  if (auto rotate = dyn_cast<ckks::RotateCOp>(op)) {
    FailureOr<int64_t> terms = rotateDecompositionTermCount(rotate, spec);
    if (failed(terms))
      return failure();
    if (cost > std::numeric_limits<int64_t>::max() / *terms) {
      op->emitError("placement cost overflow");
      return failure();
    }
    return cost * *terms;
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
  auto position =
      std::lower_bound(intervals.begin(), intervals.end(), added.start,
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
                     int64_t interRankCost,
                     std::optional<CommunicationProfile> communicationProfile)
      : func(func), spec(spec), deviceCounts(std::move(deviceCounts)),
        bootProfile(bootProfile.str()), intraRankCost(intraRankCost),
        interRankCost(interRankCost),
        communicationProfile(std::move(communicationProfile)) {
    const bool cpuTopology = this->deviceCounts.front() == 0;
    for (size_t rank = 0; rank < this->deviceCounts.size(); ++rank) {
      hostCandidates.push_back(Place{static_cast<int64_t>(rank), -1});
      if (cpuTopology) {
        candidates.push_back(Place{static_cast<int64_t>(rank), -1});
        continue;
      }
      for (int64_t device = 0; device < this->deviceCounts[rank]; ++device)
        candidates.push_back(Place{static_cast<int64_t>(rank), device});
    }
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
      bool requiresHost = false;
      FailureOr<int64_t> cost =
          operationCost(&op, spec, bootProfile, requiresHost);
      if (failed(cost))
        return failure();
      opToNode[&op] = nodes.size();
      Node node;
      node.op = &op;
      node.originalIndex = originalIndex++;
      node.cost = *cost;
      node.requiresHost = requiresHost;
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

  FailureOr<int64_t> estimatePayloadBytes(Value value,
                                          Operation *consumer) const {
    auto poly = ckks::getPolyType(value);
    if (!poly) {
      consumer->emitError(
          "communication cost requires a CKKS polynomial value");
      return failure();
    }
    if (!communicationProfile) {
      consumer->emitError(
          "payload size estimation requires a communication profile");
      return failure();
    }
    const int64_t limbs = static_cast<int64_t>(poly.getLevel()) + 1;
    if (limbs <= 0 || limbs > communicationProfile->modulusCount) {
      consumer->emitError(
          "CKKS level is outside the OperatorSpec modulus chain");
      return failure();
    }
    int64_t elements = 1;
    if (auto shaped = value.getType().dyn_cast<ShapedType>()) {
      if (!shaped.hasStaticShape()) {
        consumer->emitError(
            "communication cost requires statically shaped CKKS values");
        return failure();
      }
      elements = shaped.getNumElements();
    }
    const __int128 bytes = static_cast<__int128>(poly.getComponents()) * limbs *
                           elements * communicationProfile->polyDegree *
                           communicationProfile->coefficientBytes;
    if (bytes <= 0 || bytes > std::numeric_limits<int64_t>::max()) {
      consumer->emitError(
          "estimated communication payload size is out of range");
      return failure();
    }
    return static_cast<int64_t>(bytes);
  }

  FailureOr<int64_t> modeledCommunicationCost(int64_t payloadBytes,
                                              const LinkModel &model,
                                              Operation *consumer) const {
    const double bytes = static_cast<double>(payloadBytes);
    double rate = 0;
    if (model.ratePoints.empty()) {
      rate = model.maxRateBytesPerMicrosecond * bytes /
             (bytes + static_cast<double>(model.saturationBytes));
    } else if (payloadBytes <= model.ratePoints.front().payloadBytes) {
      const RatePoint &first = model.ratePoints.front();
      rate = first.rateBytesPerMicrosecond * bytes /
             static_cast<double>(first.payloadBytes);
    } else if (payloadBytes >= model.ratePoints.back().payloadBytes) {
      rate = model.ratePoints.back().rateBytesPerMicrosecond;
    } else {
      auto upper = std::upper_bound(
          model.ratePoints.begin(), model.ratePoints.end(), payloadBytes,
          [](int64_t payload, const RatePoint &point) {
            return payload < point.payloadBytes;
          });
      const RatePoint &right = *upper;
      const RatePoint &left = *(upper - 1);
      const double fraction =
          static_cast<double>(payloadBytes - left.payloadBytes) /
          static_cast<double>(right.payloadBytes - left.payloadBytes);
      rate = left.rateBytesPerMicrosecond +
             fraction *
                 (right.rateBytesPerMicrosecond - left.rateBytesPerMicrosecond);
    }
    rate = std::min(rate, model.maxRateBytesPerMicrosecond);
    if (!std::isfinite(rate) || rate <= 0) {
      consumer->emitError("communication profile produced an invalid rate");
      return failure();
    }
    const double transfer = std::ceil(bytes / rate);
    if (!std::isfinite(transfer) || transfer < 0 ||
        transfer > static_cast<double>(std::numeric_limits<int64_t>::max() -
                                       model.startupLatencyMicroseconds)) {
      consumer->emitError("modeled communication cost is out of range");
      return failure();
    }
    return model.startupLatencyMicroseconds + static_cast<int64_t>(transfer);
  }

  FailureOr<int64_t> communicationCost(Value value, const Place &source,
                                       const Place &destination,
                                       Operation *consumer) const {
    if (source == destination)
      return 0;
    if (!communicationProfile)
      return source.rank == destination.rank ? intraRankCost : interRankCost;
    FailureOr<int64_t> payloadBytes = estimatePayloadBytes(value, consumer);
    if (failed(payloadBytes))
      return failure();
    FailureOr<ResolvedLink> link =
        communicationProfile->resolve(source, destination, consumer);
    if (failed(link))
      return failure();
    if (link->model == nullptr) {
      consumer->emitError("communication profile resolved a null link model");
      return failure();
    }
    return modeledCommunicationCost(*payloadBytes, *link->model, consumer);
  }

  FailureOr<int64_t>
  computeAverageCommunicationCost(Value value, Operation *consumer) const {
    __int128 total = 0;
    for (const Place &source : candidates) {
      for (const Place &destination : candidates) {
        FailureOr<int64_t> cost =
            communicationCost(value, source, destination, consumer);
        if (failed(cost))
          return failure();
        total += *cost;
      }
    }
    const __int128 count = static_cast<__int128>(candidates.size());
    const __int128 average = total / (count * count);
    if (average > std::numeric_limits<int64_t>::max()) {
      consumer->emitError("average communication cost is out of range");
      return failure();
    }
    return static_cast<int64_t>(average);
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
    FailureOr<int64_t> averageCommCost = computeAverageCommunicationCost(
        nodes[index].op->getResult(0), nodes[index].op);
    if (failed(averageCommCost))
      return failure();
    for (size_t successor : nodes[index].successors) {
      FailureOr<int64_t> successorPriority = priorityOf(successor, state);
      if (failed(successorPriority))
        return failure();
      FailureOr<int64_t> candidate =
          checkedAdd(*averageCommCost, *successorPriority, nodes[index].op);
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
    FailureOr<int64_t> cost =
        communicationCost(operand, origin->second.first, destination, consumer);
    if (failed(cost))
      return failure();
    return checkedAdd(origin->second.second, *cost, consumer);
  }

  LogicalResult scheduleNode(size_t index) {
    Node &node = nodes[index];
    bool found = false;
    Place bestPlace;
    int64_t bestStart = 0;
    int64_t bestFinish = 0;
    const std::vector<Place> &nodeCandidates =
        node.requiresHost ? hostCandidates : candidates;
    for (const Place &candidatePlace : nodeCandidates) {
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
      if (!found ||
          std::tie(*finish, start, candidatePlace.rank, candidatePlace.device) <
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
            std::tie(candidate.priority, nodes[selected].originalIndex) >
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
        return node.op->emitError(
            "placement produced an invalid time interval");
      for (Value operand : node.op->getOperands()) {
        FailureOr<int64_t> arrival = arrivalTime(operand, node.place, node.op);
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
    std::sort(launchOrder.begin(), launchOrder.end(),
              [](const Node *lhs, const Node *rhs) {
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
  std::optional<CommunicationProfile> communicationProfile;
  std::vector<Place> candidates;
  std::vector<Place> hostCandidates;
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
    if (communicationProfilePath.empty() &&
        (intraRankCommunicationCost <= 0 || interRankCommunicationCost <= 0)) {
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
    std::optional<CommunicationProfile> communicationProfile;
    if (!communicationProfilePath.empty()) {
      FailureOr<CommunicationProfile> parsed =
          readCommunicationProfile(communicationProfilePath, *spec, func);
      if (failed(parsed)) {
        signalPassFailure();
        return;
      }
      if (failed(parsed->validateTopology(*counts, func))) {
        signalPassFailure();
        return;
      }
      communicationProfile = std::move(*parsed);
    }
    PlacementScheduler scheduler(func, *spec, std::move(*counts), bootProfile,
                                 intraRankCommunicationCost,
                                 interRankCommunicationCost,
                                 std::move(communicationProfile));
    if (failed(scheduler.run()))
      signalPassFailure();
  }
};

} // namespace
