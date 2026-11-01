#include "hecate/Dialect/Earth/Analysis/CandidateAnalysis.h"
#include "hecate/Dialect/Earth/Analysis/ScaleManagementUnit.h"
#include "hecate/Dialect/Earth/IR/EarthOps.h"
#include "hecate/Support/Support.h"
#include "mlir/IR/BuiltinTypes.h"
#include "hecate/Dialect/Earth/Transforms/Common.h"
#include "hecate/Support/CompilationProgress.h"

using namespace hecate;

hecate::CandidateAnalysis::CandidateAnalysis(mlir::Operation *op)
    : _op(op) {
  CompilationProgress progress("candidate-analysis");
  if (!earth::usesRegionalScalePlanning()) {
    _l = std::make_unique<mlir::Liveness>(op);
    smu = std::make_unique<ScaleManagementUnit>(op);
  }
  build();
}

void hecate::CandidateAnalysis::build() {
  llvm::SmallVector<int64_t, 4> liveIn, liveOut;
  if (smu)
    smu->attach();
  values.emplace_back(0, nullptr);
  edges.push_back(0);
  for (auto argument : mlir::cast<mlir::func::FuncOp>(_op).getArguments()) {
    auto id = static_cast<int64_t>(values.size());
    hecate::setIntegerAttr("opid", argument, id);
    values.emplace_back(id, argument);
    if (!argument.use_empty() && earth::getScaleType(argument).isCipher())
      liveOut.push_back(id);
  }
  liveIn = liveOut;
  values.front().setLiveOuts(liveOut);
  auto &&func = dyn_cast<mlir::func::FuncOp>(_op);
  auto &&bb = func.getBody().getBlocks().front();
  if (earth::usesRegionalScalePlanning()) {
    // Regional cuts already define the analysis units. Do not construct the
    // unrelated SMU fixed point or retain live vectors at every raw operation.
    for (auto &op : bb)
      for (auto value : op.getResults()) {
        auto id = static_cast<int64_t>(values.size());
        hecate::setIntegerAttr("opid", value, id);
        values.emplace_back(id, value);
      }
    retOpid = values.size();
    values.emplace_back(retOpid, nullptr);
    for (auto &op : bb) {
      auto user = op.getNumResults()
                      ? hecate::getIntegerAttr("opid", op.getResult(0))
                      : retOpid;
      for (auto input : op.getOperands())
        if (earth::getScaleType(input).isCipher())
          values[hecate::getIntegerAttr("opid", input)].setDeadOpid(user);
    }
    toFromMap[0] = {};
    return;
  }
  for (auto iter = bb.begin(); iter != bb.end(); ++iter) {
    mlir::Operation *op = &*iter;
    if ((llvm::isa<hecate::earth::UpscaleOp>(op) ||
         llvm::isa<hecate::earth::RescaleOp>(op) ||
         llvm::isa<hecate::earth::BootstrapOp>(op) ||
         llvm::isa<hecate::earth::ModswitchOp>(op))) {
      assert(0 && "Currently not supported");
    }
    doLiveAnalysis(op, liveIn, liveOut);
  }
  retOpid = values.size();
  values.emplace_back(retOpid, nullptr);
  toFromMap[0] = {};
  }

void hecate::CandidateAnalysis::doLiveAnalysis(
    mlir::Operation *op, mlir::SmallVector<int64_t, 4> &liveIn,
    mlir::SmallVector<int64_t, 4> &liveOut) {
  int64_t opid;
  
  for (auto &&val : op->getResults()) {
    opid = values.size();
    auto &&v = hecate::getIntegerAttr("opid", val);
    if (v != -1) {
      idToIdMap[opid] = v;
    }
    hecate::setIntegerAttr("opid", val, opid);
    values.emplace_back(opid, val);
 }

    if (auto sop = dyn_cast<hecate::earth::HEScaleOpInterface>(op)) {
      if (!sop.isCipher())
        return;
      for (auto &&oper : sop->getOperands()) {
        if (!hecate::earth::getScaleType(oper).isCipher())
          continue;
        if (_l->isDeadAfter(oper, sop)) {
          auto operID = hecate::getIntegerAttr("opid", oper);
          auto dead = std::find(liveOut.begin(), liveOut.end(), operID);
          if (dead != liveOut.end()) {
            liveOut.erase(dead);
            values[operID].setDeadOpid(opid);
          }
        }
      }

      for (auto &&val : op->getResults()) {
        auto &&v = hecate::getIntegerAttr("opid", val);
        liveOut.push_back(v);
        for (auto &&user : val.getUsers()) {
          if (smu->getID(user) != smu->getID(sop)) {
            values[v].setLiveOuts(liveOut);
            values[v].setLiveIns(liveIn);
            edges.push_back(v);
            break;
          }
        }
        liveIn = liveOut;
      }
    }
    return;
}

size_t hecate::CandidateAnalysis::getNumValues() const { return values.size(); }
int64_t hecate::CandidateAnalysis::getRetOpid() const { return retOpid; }
size_t hecate::CandidateAnalysis::getNumEdges() const { return edges.size(); }

mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::getInitialTargets() const {
  mlir::SmallVector<int64_t, 4> targets;
  for (auto id : values.front().getLiveOuts())
    if (earth::getScaleType(values[id].getValue()).getLevel() > 0)
      targets.push_back(id);
  return targets;
}

ScaleManagementUnit hecate::CandidateAnalysis::getSMU() const { return *smu; };
size_t hecate::CandidateAnalysis::getMaxNumOuts() const {
  size_t ret = 0;
  for (auto a : candidateSet) {
    if (a.first > ret)
      ret = a.first;
  }
  return ret;
}
mlir::SmallVector<int64_t, 4> hecate::CandidateAnalysis::getEdges() const {
  return edges;
}
mlir::SmallVector<int64_t, 4> hecate::CandidateAnalysis::getCandidates() const {
  return candidates;
}

mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::getCandidateSet(size_t size) const {
  auto &&set = candidateSet.find(size);
  if (set != candidateSet.end())
    return set->second;
  else
    return {};
}
mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::getTargets(int64_t opid) const {
  if (opid == 0 || opid == retOpid)
    return {};
  if (!values[opid].getValidLiveOuts().size())
    return values[opid].getLiveOuts();
  else
    return values[opid].getValidLiveOuts();
}
mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::getActiveLiveOuts(int64_t opid) const {
  auto active = values[opid].getLiveOuts();
  if (regionalBoundaries)
    llvm::erase_if(active, [&](int64_t id) {
      return llvm::all_of(values[id].getValue().getUsers(), [](mlir::Operation *user) {
        return mlir::isa<mlir::func::ReturnOp>(user);
      });
    });
  return active;
}

mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::getTargets(int64_t opid, int64_t setNum) const {
  if (opid == 0 || opid == retOpid)
    return {};
  if (regionalBoundaries)
    return values[opid].getValidLiveOuts();
  /* if (values[opid].getValidLiveOuts().size() == setNum) */
  /*   return values[opid].getValidLiveOuts(); */
  /* else */
  /*   return values[opid].getLiveOuts(); */
  if (values[opid].getLiveOuts().size() <= setNum)
    return values[opid].getLiveOuts();
  else
    return values[opid].getValidLiveOuts();
}
mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::getTargets(int64_t from, int64_t to,
                                      int64_t setNum) const {
  if (from == retOpid)
    return {};
  if (values[from].getValidLiveOuts().size() == setNum) {
    mlir::SmallVector<int64_t, 4> segLiveOuts = values[from].getValidLiveOuts();
    for (auto tt : values[from].getLiveOuts()) {
      if (!values[tt].isBypassEdge(from))
        continue;
      for (auto user : values[tt].getValue().getUsers()) {
        auto useOpid = hecate::getIntegerAttr("opid", user->getResult(0));
        if (useOpid > from && useOpid < to) {
          segLiveOuts.push_back(tt);
          break;
        }
      }
    }
    /* return values[from].getValidLiveOuts(); */
    return segLiveOuts;
  } else
    return values[from].getLiveOuts();
}
mlir::SmallVector<bool, 4>
hecate::CandidateAnalysis::getBypassTypeOfLiveOuts(int64_t opid) {
  mlir::SmallVector<bool, 4> bypassTypeOfLiveOuts;
  for (auto tt : values[opid].getLiveOuts()) {
    bool bb = values[tt].isBypassEdge(opid);
    bypassTypeOfLiveOuts.push_back(bb);
    /* hecate::setIntegerAttr("is_bypassed", getValueInfo(tt)->getValue(), bb);
     */
  }
  return bypassTypeOfLiveOuts;
}

ValueInfo *hecate::CandidateAnalysis::getValueInfo(int64_t opid) {
  return &values[opid];
}

void hecate::CandidateAnalysis::sortValidCandidates(int64_t opid) {
  auto v = values[opid];
  candidateSet[v.getValidLiveOuts().size()].push_back(opid);
  const auto active = getActiveLiveOuts(opid);
  if (active.size() != v.getValidLiveOuts().size())
    candidateSet[active.size()].push_back(opid);
  return;
}
mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::sortTargets(int64_t setNum) {
  mlir::SmallVector<int64_t, 4> sortedTargets;
  for (int i = 1; i <= setNum; i++) {
    for (auto a : candidateSet[setNum]) {
      for (auto b : getTargets(a, setNum)) {
        auto &&itr = std::find(sortedTargets.begin(), sortedTargets.end(), b);
        if (itr == sortedTargets.end()) {
          sortedTargets.push_back(b);
        }
      }
    }
  }
  return sortedTargets;
}

mlir::SmallVector<int64_t, 4>
hecate::CandidateAnalysis::sortTargets(int64_t setNum,
                                       mlir::SmallVector<int64_t, 4> opids) {
  mlir::SmallVector<int64_t, 4> sortedTargets;
  /* llvm::errs() << "final targets\n"; */
  for (auto a : opids) {
    /* llvm::errs() << a << " : "; */
    for (auto b : getTargets(a, setNum)) {
      /* llvm::errs() << b << " "; */
      auto &&itr = std::find(sortedTargets.begin(), sortedTargets.end(), b);
      if (itr == sortedTargets.end()) {
        sortedTargets.push_back(b);
      }
    }
    /* llvm::errs() << '\n'; */
  }
  return sortedTargets;
}

void hecate::CandidateAnalysis::finalizeCandidates(int64_t setNum) {
  candidates.clear();
  toFromMap.clear();
  candidates.push_back(0);
  for (int i = 1; i <= setNum; i++) {
    candidates.append(getCandidateSet(i));
  }
  candidates.push_back(getRetOpid());
  std::sort(candidates.begin(), candidates.end());
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  return;
}

void hecate::CandidateAnalysis::restrictToRegionalBoundaries(
    llvm::ArrayRef<int64_t> boundaries) {
  regionalBoundaries = true;
  std::set<int64_t> allowed(boundaries.begin(), boundaries.end());
  edges.assign(1, 0);
  auto live = values.front().getLiveOuts();
  for (auto &op : cast<mlir::func::FuncOp>(_op).front()) {
    auto scaleOp = dyn_cast<earth::HEScaleOpInterface>(op);
    if (!scaleOp || !scaleOp.isCipher())
      continue;
    auto cut = hecate::getIntegerAttr("opid", op.getResult(0));
    for (auto input : op.getOperands())
      if (earth::getScaleType(input).isCipher()) {
        auto id = hecate::getIntegerAttr("opid", input);
        if (values[id].getDeadOpid() == cut)
          llvm::erase_value(live, id);
      }
    live.push_back(cut);
    // Include every public regional boundary, not just old SMU edges. In
    // particular, a region's output may feed rotations in the same SMU.
    bool savesOutput = llvm::any_of(op.getResult(0).getUsers(), [](mlir::Operation *user) {
      return user->hasAttr("earth.saved_output");
    });
    if (!savesOutput && allowed.count(cut) && llvm::all_of(live, [&](int64_t value) {
          return allowed.count(value);
        })) {
      values[cut].setLiveOuts(live);
      edges.push_back(cut);
    }
  }
}

void hecate::CandidateAnalysis::pushFromCoverage(
    int64_t from, mlir::SmallVector<int64_t, 2> coverages) {
  auto c = coverages.front();
  auto bc = coverages.back();
  getValueInfo(from)->setCoverage(c);
  getValueInfo(from)->setBootCoverage(bc);
  if (c < 0)
    c = getRetOpid();
  if (bc < 0)
    bc = getRetOpid();
  for (auto to : candidates) {
    if (to < bc && from < to) {
      auto &sources = toFromMap[to];
      if (!llvm::is_contained(sources, from))
        sources.push_back(from);
    } else if (to == getRetOpid() && c == getRetOpid()) {
      auto &sources = toFromMap[to];
      if (!llvm::is_contained(sources, from))
        sources.push_back(from);
    }
  }
  return;
}

void hecate::CandidateAnalysis::setBypassEdges(int64_t from, int64_t overThr) {
  auto v = getValueInfo(from);
  auto deadOpid = v->getDeadOpid();
  if (deadOpid > overThr) {
    getValueInfo(from)->setBypassEdge();
  }
  return;
}

hecate::ValueInfo::ValueInfo(int64_t opid, mlir::Value v) : opid(opid), v(v) {}
void hecate::ValueInfo::setLiveOuts(mlir::SmallVector<int64_t, 4> outs) {
  liveOuts = outs;
  return;
}

void hecate::ValueInfo::setLiveIns(mlir::SmallVector<int64_t, 4> ins) {
  liveIns = ins;
  return;
}
void hecate::ValueInfo::setCoverage(int64_t c) {
  coverage = c;
  return;
}
void hecate::ValueInfo::setThresholdOpid(int64_t overThr) {
  thresholdOpid = overThr;
  return;
}
void hecate::ValueInfo::setBootCoverage(int64_t bc) {
  bootCoverage = bc;
  return;
}
void hecate::ValueInfo::setDeadOpid(int64_t dead) {
  deadOpid = dead;
  return;
}
void hecate::ValueInfo::setBypassEdge() {
  if (!v)
    return;
  isBypass = true;
  hecate::setIntegerAttr("is_bypassed", v, 1);
  return;
}
void hecate::ValueInfo::setValidLiveOuts(mlir::SmallVector<int64_t, 4> outs) {
  validLiveOuts = outs;
  return;
}

mlir::Value hecate::ValueInfo::getValue() const { return v; }
mlir::SmallVector<int64_t, 4> hecate::ValueInfo::getLiveIns() const {
  return liveIns;
}
mlir::SmallVector<int64_t, 4> hecate::ValueInfo::getLiveOuts() const {
  return liveOuts;
}
int64_t hecate::ValueInfo::getCoverage() const { return coverage; }
bool hecate::ValueInfo::isBypassEdge(int64_t to) const {
  bool isBypass = true;

  if (thresholdOpid <= to)
    return isBypass;
  if (opid == to)
    return false;
  for (auto user : getValue().getUsers()) {
    if (auto sop = dyn_cast<hecate::earth::HEScaleOpInterface>(user)) {
      auto useOpid = hecate::getIntegerAttr("opid", sop->getResult(0));
      if (useOpid <= thresholdOpid) {
        if (to < useOpid) {
          /* isBypass = false; */
          return false;
        }
      }
    }
  }
  return isBypass;
}
int64_t hecate::ValueInfo::getThresholdOpid() const { return thresholdOpid; }
int64_t hecate::ValueInfo::getBootCoverage() const { return bootCoverage; }
int64_t hecate::ValueInfo::getDeadOpid() const { return deadOpid; }
mlir::SmallVector<int64_t, 4> hecate::ValueInfo::getValidLiveOuts() const {
  return validLiveOuts;
}
