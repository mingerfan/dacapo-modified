#include "hecate/Dialect/CKKS/Transforms/Passes.h"
#include "hecate/Dialect/CKKS/Transforms/RuntimePlanUtils.h"
#include "hecate/Dialect/CKKS/IR/CKKSOps.h"
#include "hecate/Dialect/Dist/IR/DistOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseSet.h"
#include "nlohmann/json.hpp"
#include <fstream>
#include <limits>
#include <stdexcept>
namespace hecate::ckks {
#define GEN_PASS_DEF_PLANPLAINTEXTWATERLINE
#include "hecate/Dialect/CKKS/Transforms/Passes.h.inc"
}
using namespace mlir;
namespace ckks = hecate::ckks;
namespace dist = hecate::dist;
namespace {
using Json = nlohmann::json;
uint64_t add(uint64_t a, uint64_t b) {
  if (b > std::numeric_limits<uint64_t>::max() - a) throw std::runtime_error("Waterline byte overflow");
  return a + b;
}
struct PlanPlaintextWaterlinePass : ckks::impl::PlanPlaintextWaterlineBase<PlanPlaintextWaterlinePass> {
  using Base::Base;
  void runOnOperation() override {
    try { rewrite(); } catch (const std::exception &e) {
      getOperation().emitError(e.what()); signalPassFailure();
    }
  }
  void rewrite() {
    auto func = getOperation();
    auto counts = func->getAttrOfType<DenseI64ArrayAttr>("dist.device_counts");
    if (!counts || counts.size() != 1 || counts.asArrayRef()[0] < 1)
      throw std::runtime_error("Waterline requires one rank with local GPUs");
    if (!gpuBudget || reserveBytes >= gpuBudget || !hostBudget || !pinnedBudget || !rawBudget ||
        !(fraction > 0 && fraction < 1) || !lookaheadGroups ||
        (backend != "cpu" && backend != "gpu" && backend != "auto"))
      throw std::runtime_error("invalid Waterline configuration");
    if (func->hasAttr("runtime.memory_planned") || func->hasAttr("runtime.waterline"))
      throw std::runtime_error("Waterline must run once, before memory planning");
    auto original = ckks::buildRuntimePlanLayout(func);
    if (failed(original)) throw std::runtime_error("invalid Waterline input layout");
    OpBuilder builder(func.getContext());
    for (const auto &[id, v] : original->values)
      if (auto *op = v.getDefiningOp()) op->setAttr("runtime.value_id", builder.getI64IntegerAttr(id));
    SmallVector<dist::TransferOp> uploads;
    for (const auto &i : original->instructions)
      if (auto t = dyn_cast<dist::TransferOp>(i.operation))
        if (isa_and_nonnull<ckks::EncodeOp>(t.getInput().getDefiningOp()) &&
            t.getSourceDevice() == -1 && t.getDestinationDevice() >= 0 &&
            t.getSourceRank() == t.getDestinationRank()) uploads.push_back(t);
    uint64_t prepared = 0;
    for (auto t : uploads) {
      auto encode = cast<ckks::EncodeOp>(t.getInput().getDefiningOp());
      builder.setInsertionPoint(t);
      OperationState state(t.getLoc(), dist::PreparePlainOp::getOperationName());
      state.addTypes(t.getResult().getType());
      state.addAttribute("payload", encode.getPayload());
      for (auto attr : t->getAttrs())
        if (attr.getName().strref().startswith("dist.") || attr.getName() == "runtime.value_id")
          state.addAttribute(attr.getName(), attr.getValue());
      auto *p = builder.create(state);
      t.getResult().replaceAllUsesWith(p->getResult(0));
      t.erase(); ++prepared;
      if (encode->use_empty()) encode.erase();
    }
    std::ifstream input(operatorSpecPath.getValue());
    if (!input) throw std::runtime_error("cannot open physical OperatorSpec");
    const auto degree = Json::parse(input).at("context").at("poly_degree").get<uint64_t>();
    if (!degree || degree > (1ULL << 24)) throw std::runtime_error("invalid CKKS degree");
    auto layout = ckks::buildRuntimePlanLayout(func);
    if (failed(layout)) throw std::runtime_error("invalid Prepare layout");
    const auto place = [&](Value v) {
      auto p = ckks::runtimePlace(v, func);
      if (failed(p)) throw std::runtime_error("missing Waterline place");
      return p->device + 1;
    };
    const auto bytes = [&](Value v) -> uint64_t {
      auto t = ckks::getPolyType(v);
      if (!t || t.getLevel() < 0 || t.getLevel() > (1 << 20)) throw std::runtime_error("invalid physical CKKS type");
      return degree * (uint64_t(t.getLevel()) + 1) * t.getComponents() * (place(v) ? 4 : 8);
    };
    llvm::DenseMap<Value, uint64_t> remaining;
    llvm::DenseSet<Value> external, returned, liveValues;
    for (Value v : func.getArguments()) external.insert(v);
    auto ret = cast<func::ReturnOp>(func.getBody().front().getTerminator());
    for (Value v : ret.getOperands()) returned.insert(v);
    for (const auto &i : layout->instructions)
      for (Value v : i.operation->getOperands()) ++remaining[v];
    std::vector<uint64_t> front(counts.asArrayRef()[0]+1), plain(front.size()), peaks(front.size());
    for (Value v : func.getArguments()) { front[place(v)] = add(front[place(v)], bytes(v)); liveValues.insert(v); }
    uint64_t groups=0, plainPeak=0, hostPeak=0, pinnedPeak=0;
    for (const auto &i : layout->instructions) {
      auto *op = i.operation;
      if (isa<dist::PreparePlainOp>(op)) continue; // registered; admitted at first direct use
      uint64_t hostTemp=0, pinnedTemp=0;
      std::vector<uint64_t> foregroundWorkspace(front.size());
      std::vector<uint64_t> temporary(front.size());
      for (Value v : op->getOperands()) if (isa_and_nonnull<dist::PreparePlainOp>(v.getDefiningOp()) && liveValues.insert(v).second) {
        auto payload = cast<dist::PreparePlainOp>(v.getDefiningOp()).getPayload();
        if (uint64_t(payload.getNumElements()) > degree/2) throw std::runtime_error("Prepare payload exceeds slot capacity");
        const auto b=bytes(v); plain[place(v)] = add(plain[place(v)], b);
        const uint64_t raw=degree*8; // bounded file/decode/padded input buffers
        if (raw > rawBudget) throw std::runtime_error("raw budget below single Prepare baseline");
        hostTemp=add(hostTemp, raw + (backend == "gpu" ? 0 : b*2+degree*32));
        pinnedTemp=add(pinnedTemp, backend == "gpu" ? degree*4 : b);
        if (backend == "gpu") temporary[place(v)]=add(temporary[place(v)], degree*64+b);
      }
      for (Value v : op->getResults()) { front[place(v)] = add(front[place(v)], bytes(v)); liveValues.insert(v); }
      for (Value v : op->getResults()) if (place(v) && op->getName().getDialectNamespace() == "ckks")
        foregroundWorkspace[place(v)] = add(foregroundWorkspace[place(v)], workspaceBytes);
      for (size_t p=1;p<front.size();++p) {
        const uint64_t required=add(plain[p], temporary[p]);
        const uint64_t foreground=add(front[p],foregroundWorkspace[p]);
        const uint64_t capacity=gpuBudget-reserveBytes;
        const uint64_t water=uint64_t(fraction*gpuBudget);
        if (required>water || foreground>capacity || required>capacity-foreground)
          throw std::runtime_error("single-group baseline instruction #"+std::to_string(i.ordinal)+" GPU"+std::to_string(p-1)+
              " front="+std::to_string(front[p])+" plain/workspace="+std::to_string(required)+
              " capacity="+std::to_string(capacity)+" waterline="+std::to_string(water));
        peaks[p]=std::max(peaks[p],add(foreground,required)); plainPeak=std::max(plainPeak,plain[p]);
      }
      hostPeak=std::max(hostPeak,add(front[0],hostTemp)); pinnedPeak=std::max(pinnedPeak,pinnedTemp);
      if (hostPeak>hostBudget || pinnedPeak>pinnedBudget) throw std::runtime_error("single-group shared Host/pinned budget exceeded");
      ++groups;
      for (Value v : op->getOperands()) if (--remaining[v]==0 && !external.count(v) && !returned.count(v)) {
        auto &used=isa_and_nonnull<dist::PreparePlainOp>(v.getDefiningOp())?plain[place(v)]:front[place(v)];
        used-=bytes(v);liveValues.erase(v);
      }
      for (Value v : op->getResults()) if (!remaining[v] && !returned.count(v)) {front[place(v)]-=bytes(v);liveValues.erase(v);}
    }
    Json c={{"device_capacity_bytes",std::vector<uint64_t>(counts.asArrayRef()[0],gpuBudget)},
      {"device_reserve_bytes",std::vector<uint64_t>(counts.asArrayRef()[0],reserveBytes)}, {"device_fractions",std::vector<double>(counts.asArrayRef()[0],fraction.getValue())},
      {"host_budget_bytes",hostBudget.getValue()}, {"pinned_budget_bytes",pinnedBudget.getValue()},
      {"raw_budget_bytes",rawBudget.getValue()}, {"backend",backend.getValue()}, {"lookahead_groups",lookaheadGroups.getValue()}};
    func->setAttr("runtime.waterline",builder.getStringAttr(c.dump()));
    if (!prefix.getValue().empty()) {
      std::ofstream out(prefix.getValue()+"."+func.getName().str()+".waterline.json");
      out << Json{{"prepare_count",prepared},{"groups",groups},{"single_group_object_and_workspace_peak",peaks},
        {"plain_peak",plainPeak},{"host_peak",hostPeak},{"pinned_peak",pinnedPeak},
        {"workspace_upper_bound_per_op",workspaceBytes.getValue()}, {"config",c}}.dump(2) << '\n';
      if (!out) throw std::runtime_error("cannot write Waterline capacity report");
    }
  }
};
}
