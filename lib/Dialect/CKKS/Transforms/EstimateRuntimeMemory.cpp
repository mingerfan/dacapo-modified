#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "hecate/Dialect/CKKS/Transforms/RuntimePlanUtils.h"
#include "hecate/Dialect/Dist/IR/DistOps.h"
#include "nlohmann/json.hpp"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace hecate::ckks {
#define GEN_PASS_DEF_ESTIMATERUNTIMEMEMORY
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
} // namespace hecate::ckks
using namespace mlir;
namespace ckks = hecate::ckks;
namespace dist = hecate::dist;
namespace {
using Json = nlohmann::json;
uint64_t add(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a)
    throw std::overflow_error("memory byte sum overflow");
  return a + b;
}
uint64_t multiply(uint64_t a, uint64_t b) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    throw std::overflow_error("memory size overflow");
  return a * b;
}
struct Block {
  uint64_t valueId, bytes;
  ckks::RuntimePlace place;
  bool plaintext, external;
  uint64_t born, retired = std::numeric_limits<uint64_t>::max();
};
struct PlaceStats {
  uint64_t plaintext = 0, ciphertext = 0, peak = 0, allocated = 0;
  uint64_t peakEventIndex = 0;
  Json peakEvent;
};
struct EstimateRuntimeMemoryPass
    : ckks::impl::EstimateRuntimeMemoryBase<EstimateRuntimeMemoryPass> {
  using Base::Base;
  void runOnOperation() override {
    auto func = getOperation();
    if (prefix.empty() || operatorSpecPath.empty()) {
      func.emitError("memory estimation requires prefix and operator-spec");
      return signalPassFailure();
    }
    auto layout = ckks::buildRuntimePlanLayout(func);
    if (failed(layout))
      return signalPassFailure();
    try {
      std::ifstream input(operatorSpecPath.getValue());
      if (!input)
        throw std::runtime_error("cannot open OperatorSpec: " +
                                 operatorSpecPath.getValue());
      Json spec = Json::parse(input, nullptr, false);
      if (spec.is_discarded() || !spec.is_object() ||
          !spec.contains("context") || !spec["context"].is_object() ||
          !spec["context"].contains("poly_degree"))
        throw std::runtime_error("OperatorSpec requires context.poly_degree");
      const auto &degreeJson = spec["context"]["poly_degree"];
      if (!degreeJson.is_number_integer() || degreeJson.get<int64_t>() <= 0)
        throw std::runtime_error(
            "OperatorSpec poly_degree must be a positive integer");
      const uint64_t degree = degreeJson.get<uint64_t>();
      std::vector<Block> blocks;
      uint64_t eventIndex = 0;
      std::map<uint64_t, size_t> aliases;
      std::map<ckks::RuntimePlace, PlaceStats> places;
      if (auto counts =
              func->getAttrOfType<DenseI64ArrayAttr>("dist.device_counts")) {
        for (size_t rank = 0; rank < counts.size(); ++rank) {
          if (counts.asArrayRef()[rank] < 0)
            throw std::runtime_error("negative device count");
          places[{static_cast<int64_t>(rank), -1}];
          for (int64_t device = 0; device < counts.asArrayRef()[rank]; ++device)
            places[{static_cast<int64_t>(rank), device}];
        }
      }

      auto allocate = [&](Value value, bool external) {
        auto type = ckks::getPolyType(value);
        auto place = ckks::runtimePlace(value, func);
        if (!type || !type.getComponents() || failed(place))
          throw std::runtime_error("memory estimation requires CKKS polynomial "
                                   "values with valid places");
        uint64_t bytes =
            multiply(multiply(multiply(degree, add(type.getLevel(), 1)),
                              type.getComponents()),
                     place->device < 0 ? 8 : 4);
        uint64_t id = layout->ids.lookup(value);
        if (aliases.count(id))
          throw std::runtime_error("value allocated twice");
        aliases.emplace(id, blocks.size());
        blocks.push_back(
            {id, bytes, *place, type.getComponents() == 1, external, eventIndex});
        auto &stats = places[*place];
        (type.getComponents() == 1 ? stats.plaintext : stats.ciphertext) =
            add(type.getComponents() == 1 ? stats.plaintext : stats.ciphertext,
                bytes);
        stats.allocated = add(stats.allocated, bytes);
      };
      auto sample = [&](const char *phase, Json ordinal,
                        const char *operation) {
        for (auto &[place, stats] : places) {
          uint64_t current = add(stats.plaintext, stats.ciphertext);
          if (current <= stats.peak && !stats.peakEvent.is_null())
            continue;
          stats.peak = current;
          stats.peakEventIndex = eventIndex;
          stats.peakEvent = {{"phase", phase},
                             {"instruction_ordinal", ordinal},
                             {"operation", operation},
                             {"plaintext_bytes", stats.plaintext},
                             {"ciphertext_bytes", stats.ciphertext}};
        }
      };
      for (Value argument : func.getArguments())
        allocate(argument, true);
      sample("external_inputs", nullptr, "external_inputs");
      auto returnOp =
          dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
      if (!returnOp)
        throw std::runtime_error("memory estimation requires func.return");
      for (const auto &instruction : layout->instructions) {
        ++eventIndex;
        Operation *op = instruction.operation;
        if (auto release = dyn_cast<dist::ReleaseOp>(op)) {
          auto id = layout->ids.lookup(release.getInput());
          auto alias = aliases.find(id);
          if (alias == aliases.end())
            throw std::runtime_error(
                "Release references undefined or retired value " +
                std::to_string(id));
          if (llvm::is_contained(returnOp.getOperands(), release.getInput()))
            throw std::runtime_error("Release references a final output");
          auto &block = blocks[alias->second];
          if (!block.external) {
            auto &stats = places[block.place];
            (block.plaintext ? stats.plaintext : stats.ciphertext) -=
                block.bytes;
            block.retired = eventIndex;
          }
          aliases.erase(alias);
        } else if (isa<dist::FenceOp>(op)) {
          // Explicit completion boundary; object lifetimes are unchanged.
        } else {
          for (Value operand : op->getOperands())
            if (!aliases.count(layout->ids.lookup(operand)))
              throw std::runtime_error(
                  "instruction uses undefined or retired value");
          if (Attribute attr = op->getAttr("runtime.reuse_input")) {
            auto reuse = dyn_cast<IntegerAttr>(attr);
            if (!reuse || reuse.getInt() != 0 ||
                !ckks::canReuseRuntimeInput(op, func) ||
                llvm::is_contained(returnOp.getOperands(), op->getOperand(0)))
              throw std::runtime_error(
                  "invalid runtime.reuse_input in memory estimation");
            auto oldId = layout->ids.lookup(op->getOperand(0)),
                 newId = layout->ids.lookup(op->getResult(0));
            auto index = aliases.at(oldId);
            // Record alias lifetimes so a later reuse does not rewrite the
            // ValueId reported for an earlier peak. The allocation is shared.
            Block replacement = blocks[index];
            blocks[index].retired = eventIndex;
            replacement.valueId = newId;
            replacement.born = eventIndex;
            replacement.retired = std::numeric_limits<uint64_t>::max();
            aliases.erase(oldId);
            aliases.emplace(newId, blocks.size());
            blocks.push_back(replacement);
          } else
            allocate(op->getResult(0), false);
        }
        sample(instruction.initialization ? "initialization" : "execution",
               instruction.ordinal, op->getName().getStringRef().data());
      }
      for (Value output : returnOp.getOperands())
        if (!aliases.count(layout->ids.lookup(output)))
          throw std::runtime_error("final output is undefined or retired");
      Json rows = Json::array();
      for (auto &[place, stats] : places) {
        // Reconstruct only the final peak, rather than scanning and sorting
        // every live object whenever the running peak rises.
        std::vector<const Block *> largest;
        for (const Block &block : blocks) {
          if (!(block.place == place) || block.born > stats.peakEventIndex ||
              block.retired <= stats.peakEventIndex)
            continue;
          auto position = std::lower_bound(
              largest.begin(), largest.end(), &block,
              [](const Block *a, const Block *b) {
                return a->bytes != b->bytes ? a->bytes > b->bytes
                                            : a->valueId < b->valueId;
              });
          largest.insert(position, &block);
          if (largest.size() > 5)
            largest.pop_back();
        }
        if (!stats.peakEvent.is_null()) {
          Json objects = Json::array();
          for (const Block *block : largest)
            objects.push_back({{"value_id", std::to_string(block->valueId)},
                               {"bytes", block->bytes},
                               {"kind", block->plaintext ? "plaintext" : "ciphertext"},
                               {"external", block->external}});
          stats.peakEvent["largest_objects"] = std::move(objects);
        }
        Json location = {{"kind", place.device < 0 ? "host" : "device"},
                         {"rank", place.rank}};
        if (place.device >= 0)
          location["index"] = place.device;
        rows.push_back(
            {{"place", std::move(location)},
             {"peak_bytes", stats.peak},
             {"peak", stats.peakEvent},
             {"distinct_allocation_bytes", stats.allocated},
             {"retained_bytes", add(stats.plaintext, stats.ciphertext)}});
        llvm::errs() << "runtime memory " << func.getName()
                     << " rank=" << place.rank << " device=" << place.device
                     << " peak="
                     << static_cast<double>(stats.peak) / (1024 * 1024)
                     << " MiB\n";
      }
      Json report = {
          {"format_version", 1},
          {"function", func.getName().str()},
          {"poly_degree", degree},
          {"memory_planned", func->hasAttr("runtime.memory_planned")},
          {"assumption",
           "Each instruction completes before the next; external inputs are "
           "held by the caller throughout the run. This is not measured "
           "process memory or a bound for parallel execution."},
          {"word_bytes", {{"host", 8}, {"device", 4}}},
          {"distinct_allocation_meaning",
           "Sum of distinct planned object blocks, counting a reuse chain "
           "once; excludes unmodeled memory and is not a process memory upper "
           "bound."},
          {"excluded",
           {"keys", "parameter tables", "Rotate and Boot workspaces",
            "communication staging", "encoding caches",
            "allocator alignment and reserved pool memory", "object metadata"}},
          {"places", std::move(rows)}};
      std::string path =
          prefix.getValue() + "." + func.getName().str() + ".memory" +
          (reportTag.empty() ? "" : "." + reportTag.getValue()) + ".json";
      std::ofstream output(path);
      if (!output)
        throw std::runtime_error("cannot open memory report: " + path);
      output << report.dump(2) << '\n';
      if (!output)
        throw std::runtime_error("cannot write memory report: " + path);
    } catch (const std::exception &error) {
      func.emitError(error.what());
      signalPassFailure();
    }
  }
};
} // namespace
