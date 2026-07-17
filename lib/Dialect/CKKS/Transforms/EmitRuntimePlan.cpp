#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "nlohmann/json.hpp"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/SHA256.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <string>

namespace hecate {
namespace ckks {
#define GEN_PASS_DEF_EMITRUNTIMEPLAN
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace ckks
} // namespace hecate

using namespace mlir;
namespace ckks = hecate::ckks;

namespace {

using Json = nlohmann::json;

bool isCanonicalId(llvm::StringRef value) {
  if (value.empty() || (value.size() > 1 && value.front() == '0'))
    return false;
  for (char c : value)
    if (c < '0' || c > '9')
      return false;
  uint64_t parsed = 0;
  return !value.getAsInteger(10, parsed);
}

bool isSha256(llvm::StringRef value) {
  if (!value.consume_front("sha256:") || value.size() != 64)
    return false;
  for (char c : value)
    if (!llvm::isDigit(c) && (c < 'a' || c > 'f'))
      return false;
  return true;
}

Json hostPlace() { return {{"kind", "host"}, {"rank", 0}}; }

std::string sha256(llvm::StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  const auto digest = hasher.final();
  return "sha256:" + llvm::toHex(llvm::ArrayRef<uint8_t>(digest), true);
}

LogicalResult writeFile(const std::filesystem::path &path,
                        llvm::StringRef bytes, Operation *op) {
  std::ofstream output(path, std::ios::binary);
  if (!output)
    return op->emitError("cannot open output file: ") << path.string();
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output)
    return op->emitError("failed to write output file: ") << path.string();
  return success();
}

class PlaintextBundleWriter {
public:
  PlaintextBundleWriter(std::filesystem::path directory, std::string id)
      : directory(std::move(directory)), id(std::move(id)) {}

  bool used() const { return !blobs.empty(); }

  FailureOr<Json> externalize(DenseElementsAttr dense, Operation *op) {
    if (!dense.getElementType().isa<FloatType>()) {
      op->emitError(
          "RuntimePlan bundle externalization requires a float payload");
      return failure();
    }

    std::string bytes;
    const int64_t elements = dense.getNumElements();
    constexpr uint64_t maxBundleBytes = (1ULL << 53) - 1;
    if (elements <= 0 || static_cast<uint64_t>(elements) > maxBundleBytes / 8) {
      op->emitError("RuntimePlan bundle payload byte length is out of range");
      return failure();
    }
    bytes.reserve(static_cast<size_t>(elements) * 8);
    for (const APFloat &value : dense.getValues<APFloat>()) {
      const double converted = value.convertToDouble();
      if (!std::isfinite(converted)) {
        op->emitError("RuntimePlan Encode payload must contain finite values");
        return failure();
      }
      uint64_t bits = 0;
      static_assert(sizeof(bits) == sizeof(converted), "float64 is required");
      std::memcpy(&bits, &converted, sizeof(bits));
      for (int byte = 0; byte < 8; ++byte)
        bytes.push_back(static_cast<char>((bits >> (byte * 8)) & 0xff));
    }

    const std::string content = sha256(bytes);
    const auto [it, inserted] = blobs.emplace(content, bytes.size());
    if (inserted) {
      std::error_code error;
      std::filesystem::create_directories(directory / "data", error);
      if (error) {
        op->emitError("cannot create RuntimePlan bundle directory: ")
            << directory.string() << ": " << error.message();
        return failure();
      }
      const std::filesystem::path blobPath =
          directory / "data" / (content.substr(7) + ".bin");
      if (failed(writeFile(blobPath, bytes, op)))
        return failure();
    } else if (it->second != bytes.size()) {
      op->emitError("RuntimePlan bundle SHA-256 collision");
      return failure();
    }
    return Json{{"kind", "bundle"}, {"content", content}};
  }

  FailureOr<Json> writeManifest(Operation *op) const {
    Json entries = Json::array();
    for (const auto &[content, byteLength] : blobs)
      entries.push_back(
          {{"content", content}, {"byte_length", byteLength}});
    const Json manifest = {{"bundle_format_version", 1},
                           {"bundle_id", id},
                           {"version", 1},
                           {"blobs", std::move(entries)}};
    const std::string manifestBytes = manifest.dump(2) + '\n';
    if (failed(writeFile(directory / "manifest.json", manifestBytes, op)))
      return failure();
    return Json{{"id", id},
                {"version", 1},
                {"manifest_sha256", sha256(manifestBytes)}};
  }

private:
  std::filesystem::path directory;
  std::string id;
  std::map<std::string, uint64_t> blobs;
};

FailureOr<Json> encodePayload(Attribute payload, Operation *op,
                              PlaintextBundleWriter &bundle,
                              uint64_t inlinePayloadMaxBytes) {
  auto dense = payload.dyn_cast<DenseElementsAttr>();
  if (!dense) {
    op->emitError("RuntimePlan Encode requires a DenseElementsAttr; "
                  "legacy .cst indices are not supported");
    return failure();
  }

  const int64_t elements = dense.getNumElements();
  if (elements <= 0 ||
      static_cast<uint64_t>(elements) >
          std::numeric_limits<uint64_t>::max() / 8) {
    op->emitError("RuntimePlan Encode payload size is out of range");
    return failure();
  }
  if (static_cast<uint64_t>(elements) * 8 > inlinePayloadMaxBytes)
    return bundle.externalize(dense, op);

  Json values = Json::array();
  auto elementType = dense.getElementType();
  if (elementType.isa<FloatType>()) {
    for (const APFloat &value : dense.getValues<APFloat>()) {
      double converted = value.convertToDouble();
      if (!std::isfinite(converted)) {
        op->emitError("RuntimePlan Encode payload must contain finite values");
        return failure();
      }
      values.push_back(converted);
    }
  } else if (elementType.isa<IntegerType>()) {
    for (const APInt &value : dense.getValues<APInt>()) {
      if (value.getBitWidth() > 64) {
        op->emitError("RuntimePlan Encode integer payload exceeds int64");
        return failure();
      }
      values.push_back(value.getSExtValue());
    }
  } else {
    op->emitError("RuntimePlan Encode payload must contain integers or floats");
    return failure();
  }

  if (values.empty()) {
    op->emitError("RuntimePlan Encode payload cannot be empty");
    return failure();
  }
  return Json{{"kind", "inline"}, {"values", std::move(values)}};
}

struct RuntimePlanBuilder {
  RuntimePlanBuilder(func::FuncOp func, llvm::StringRef contextId, bool ntt,
                     llvm::StringRef bootProfile,
                     llvm::StringRef bootImplementation,
                     PlaintextBundleWriter &bundle,
                     uint64_t inlinePayloadMaxBytes)
      : func(func), contextId(contextId), ntt(ntt), bootProfile(bootProfile),
        bootImplementation(bootImplementation), bundle(bundle),
        inlinePayloadMaxBytes(inlinePayloadMaxBytes) {}

  LogicalResult build(Json &values, Json &externalInputs, Json &initialization,
                      Json &execution, Json &finalOutputs) {
    if (!func.getBody().hasOneBlock())
      return func.emitError(
          "RuntimePlan V1 export currently requires a single-block function");

    for (BlockArgument argument : func.getArguments()) {
      FailureOr<uint64_t> id = addValue(argument, values);
      if (failed(id))
        return failure();
      externalInputs.push_back(std::to_string(*id));
    }

    for (Operation &op : func.getBody().front()) {
      if (isa<func::ReturnOp>(op))
        continue;
      if (op.getNumRegions() != 0)
        return op.emitError(
            "RuntimePlan V1 export does not support nested control flow");
      if (op.getNumResults() != 1 ||
          op.getName().getDialectNamespace() != "ckks")
        return op.emitError("unsupported operation in RuntimePlan export");

      FailureOr<uint64_t> outputId = addValue(op.getResult(0), values);
      if (failed(outputId))
        return failure();

      if (auto encode = dyn_cast<ckks::EncodeOp>(op)) {
        FailureOr<Json> payload = encodePayload(
            encode.getPayload(), &op, bundle, inlinePayloadMaxBytes);
        if (failed(payload))
          return failure();
        initialization.push_back({{"kind", "encode"},
                                  {"payload", std::move(*payload)},
                                  {"output", std::to_string(*outputId)}});
        continue;
      }

      FailureOr<Json> instruction = buildCompute(&op, *outputId);
      if (failed(instruction))
        return failure();
      execution.push_back(std::move(*instruction));
    }

    uint64_t ordinal = 0;
    for (Json &instruction : initialization)
      instruction["ordinal"] = ordinal++;
    for (Json &instruction : execution)
      instruction["ordinal"] = ordinal++;

    auto returnOp =
        dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
    if (!returnOp)
      return func.emitError("RuntimePlan export requires func.return");
    if (returnOp.getNumOperands() == 0)
      return func.emitError(
          "RuntimePlan V1 requires at least one final output");
    for (Value output : returnOp.getOperands()) {
      auto it = ids.find(output);
      if (it == ids.end())
        return returnOp.emitError("final output has no RuntimePlan ValueId");
      finalOutputs.push_back(std::to_string(it->second));
    }
    return success();
  }

private:
  FailureOr<uint64_t> addValue(Value value, Json &values) {
    auto type = ckks::getPolyType(value);
    if (!type) {
      if (Operation *definingOp = value.getDefiningOp())
        definingOp->emitError("expected a CKKS polynomial value");
      else
        func.emitError("expected a CKKS polynomial function argument");
      return failure();
    }
    constexpr unsigned maxRuntimeInteger = std::numeric_limits<int32_t>::max();
    if (type.getComponents() == 0 || type.getComponents() > maxRuntimeInteger ||
        type.getLevel() > maxRuntimeInteger ||
        type.getScaleLog2() > maxRuntimeInteger) {
      if (Operation *definingOp = value.getDefiningOp())
        definingOp->emitError("CKKS metadata is outside RuntimePlan V1 range");
      else
        func.emitError(
            "CKKS argument metadata is outside RuntimePlan V1 range");
      return failure();
    }
    uint64_t id = nextValueId++;
    ids.insert({value, id});
    values.push_back(
        {{"id", std::to_string(id)},
         {"kind", type.getComponents() == 1 ? "plaintext" : "ciphertext"},
         {"place", hostPlace()},
         {"context", contextId},
         {"level", type.getLevel()},
         {"scale_log2", type.getScaleLog2()},
         {"ntt", ntt},
         {"components", type.getComponents()}});
    return id;
  }

  FailureOr<std::string> valueId(Value value, Operation *user) const {
    auto it = ids.find(value);
    if (it == ids.end()) {
      user->emitError("RuntimePlan input is not defined before use");
      return failure();
    }
    return std::to_string(it->second);
  }

  FailureOr<Json> makeCompute(Operation *op, uint64_t outputId,
                              llvm::StringRef name, ValueRange inputs) const {
    Json inputIds = Json::array();
    for (Value input : inputs) {
      FailureOr<std::string> id = valueId(input, op);
      if (failed(id))
        return failure();
      inputIds.push_back(*id);
    }
    return Json{{"kind", "compute"},
                {"op", name.str()},
                {"place", hostPlace()},
                {"inputs", std::move(inputIds)},
                {"output", std::to_string(outputId)}};
  }

  FailureOr<Json> buildCompute(Operation *op, uint64_t outputId) const {
    FailureOr<Json> instruction = failure();
    if (auto value = dyn_cast<ckks::AddCCOp>(op))
      instruction =
          makeCompute(op, outputId, "add_cc", {value.getLhs(), value.getRhs()});
    else if (auto value = dyn_cast<ckks::AddCPOp>(op))
      instruction =
          makeCompute(op, outputId, "add_cp", {value.getLhs(), value.getRhs()});
    else if (auto value = dyn_cast<ckks::MulCCOp>(op))
      instruction =
          makeCompute(op, outputId, "mul_cc", {value.getLhs(), value.getRhs()});
    else if (auto value = dyn_cast<ckks::MulCPOp>(op))
      instruction =
          makeCompute(op, outputId, "mul_cp", {value.getLhs(), value.getRhs()});
    else if (auto value = dyn_cast<ckks::NegateCOp>(op))
      instruction = makeCompute(op, outputId, "negate", {value.getSrc()});
    else if (auto value = dyn_cast<ckks::RelinearizeOp>(op))
      instruction = makeCompute(op, outputId, "relinearize", {value.getSrc()});
    else if (auto value = dyn_cast<ckks::RotateCOp>(op)) {
      if (value.getOffset().size() != 1 || value.getOffset()[0] == 0 ||
          value.getOffset()[0] < std::numeric_limits<int32_t>::min() ||
          value.getOffset()[0] > std::numeric_limits<int32_t>::max()) {
        op->emitError("RuntimePlan Rotate requires exactly one nonzero step");
        return failure();
      }
      instruction = makeCompute(op, outputId, "rotate", {value.getSrc()});
      if (succeeded(instruction))
        (*instruction)["attrs"] = {{"steps", value.getOffset()[0]}};
    } else if (auto value = dyn_cast<ckks::RescaleCOp>(op)) {
      instruction = makeCompute(op, outputId, "rescale", {value.getSrc()});
      if (succeeded(instruction)) {
        auto outputType = ckks::getPolyType(op->getResult(0));
        (*instruction)["attrs"] = {
            {"target_level", outputType.getLevel()},
            {"target_scale_log2", outputType.getScaleLog2()}};
      }
    } else if (auto value = dyn_cast<ckks::ModswitchCOp>(op)) {
      instruction = makeCompute(op, outputId, "mod_switch", {value.getSrc()});
      if (succeeded(instruction)) {
        auto outputType = ckks::getPolyType(op->getResult(0));
        (*instruction)["attrs"] = {{"target_level", outputType.getLevel()}};
      }
    } else if (auto value = dyn_cast<ckks::BootstrapCOp>(op)) {
      if (bootProfile.empty()) {
        op->emitError("RuntimePlan Boot requires --boot-profile");
        return failure();
      }
      instruction = makeCompute(op, outputId, "boot", {value.getSrc()});
      if (succeeded(instruction)) {
        auto outputType = ckks::getPolyType(op->getResult(0));
        (*instruction)["attrs"] = {
            {"target_level", outputType.getLevel()},
            {"target_scale_log2", outputType.getScaleLog2()},
            {"target_components", outputType.getComponents()},
            {"operator_profile", bootProfile},
            {"implementation", bootImplementation}};
      }
    } else if (isa<ckks::UpscaleCOp>(op)) {
      op->emitError(
          "ckks.upscalec must be eliminated before RuntimePlan export");
      return failure();
    } else {
      op->emitError("unsupported CKKS operation in RuntimePlan export");
      return failure();
    }
    return instruction;
  }

  func::FuncOp func;
  std::string contextId;
  bool ntt;
  std::string bootProfile;
  std::string bootImplementation;
  PlaintextBundleWriter &bundle;
  uint64_t inlinePayloadMaxBytes;
  llvm::DenseMap<Value, uint64_t> ids;
  uint64_t nextValueId = 0;
};

struct EmitRuntimePlanPass
    : public hecate::ckks::impl::EmitRuntimePlanBase<EmitRuntimePlanPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (prefix.empty() || targetId.empty() || operatorSpecId.empty() ||
        operatorSpecSha256.empty() || contextId.empty()) {
      func.emitError("RuntimePlan export requires prefix, target-id, "
                     "operator-spec-id, operator-spec-sha256 and context-id");
      signalPassFailure();
      return;
    }
    if (!isCanonicalId(planId)) {
      func.emitError("RuntimePlan plan-id must be a canonical uint64 string");
      signalPassFailure();
      return;
    }
    if (!isSha256(operatorSpecSha256)) {
      func.emitError("operator-spec-sha256 must be sha256: followed by 64 "
                     "lowercase hexadecimal digits");
      signalPassFailure();
      return;
    }
    if (capabilityVersion < 1 ||
        capabilityVersion > std::numeric_limits<int32_t>::max() ||
        operatorSpecVersion < 1 ||
        operatorSpecVersion > std::numeric_limits<int32_t>::max() ||
        deviceCount < 0 || deviceCount > std::numeric_limits<int32_t>::max() ||
        inlinePayloadMaxBytes < 0 ||
        inlinePayloadMaxBytes > static_cast<int64_t>((1ULL << 53) - 1)) {
      func.emitError("RuntimePlan versions and device-count must fit the V1 "
                     "ranges; versions cannot be zero and the inline payload "
                     "limit must be a nonnegative safe JSON integer");
      signalPassFailure();
      return;
    }
    if (bootImplementation != "native" &&
        bootImplementation != "decrypt_reencrypt") {
      func.emitError("boot-implementation must be native or decrypt_reencrypt");
      signalPassFailure();
      return;
    }

    Json values = Json::array();
    Json externalInputs = Json::array();
    Json initialization = Json::array();
    Json execution = Json::array();
    Json finalOutputs = Json::array();
    const std::string artifactStem =
        prefix.getValue() + "." + func.getName().str();
    PlaintextBundleWriter bundle(
        artifactStem + ".bundle",
        "runtime-plan-" + planId.getValue() + "-" + func.getName().str() +
            "-plaintexts");
    RuntimePlanBuilder builder(func, contextId.getValue(), ntt.getValue(),
                               bootProfile.getValue(),
                               bootImplementation.getValue(), bundle,
                               static_cast<uint64_t>(
                                   inlinePayloadMaxBytes.getValue()));
    if (failed(builder.build(values, externalInputs, initialization, execution,
                             finalOutputs))) {
      signalPassFailure();
      return;
    }

    Json plan = {{"format_version", 1},
                 {"plan_id", planId.getValue()},
                 {"target",
                  {{"target_id", targetId.getValue()},
                   {"capability_version", capabilityVersion.getValue()},
                   {"operator_spec",
                    {{"id", operatorSpecId.getValue()},
                     {"version", operatorSpecVersion.getValue()},
                     {"source_sha256", operatorSpecSha256.getValue()}}},
                   {"world_size", 1},
                   {"device_counts", Json::array({deviceCount.getValue()})}}},
                 {"values", std::move(values)},
                 {"external_inputs", std::move(externalInputs)},
                 {"initialization", std::move(initialization)},
                 {"execution", std::move(execution)},
                 {"finalization", Json::array()},
                 {"final_outputs", std::move(finalOutputs)}};

    if (bundle.used()) {
      FailureOr<Json> reference = bundle.writeManifest(func);
      if (failed(reference)) {
        signalPassFailure();
        return;
      }
      plan["plaintext_bundle"] = std::move(*reference);
    }

    const std::filesystem::path outputPath =
        artifactStem + ".runtime-plan.json";
    const std::string planBytes = plan.dump(2) + '\n';
    if (failed(writeFile(outputPath, planBytes, func))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace
