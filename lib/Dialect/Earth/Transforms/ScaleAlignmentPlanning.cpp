#include "hecate/Support/ScaleAlignmentPlanning.h"
#include <algorithm>
#include <cmath>
#include <tuple>

namespace hecate {
namespace {
constexpr double tolerance = 1e-8;
bool close(double a, double b) { return std::abs(a - b) <= tolerance; }
bool isPlain(AlignmentKind kind) {
  return kind == AlignmentKind::MulPlain || kind == AlignmentKind::AddPlain;
}
bool finiteScale(double scale) {
  return std::isfinite(scale) && scale >= 0 && scale <= 1023;
}
bool goodUnit(double scale, double error) {
  if (!finiteScale(scale)) return false;
  double value = std::exp2(scale);
  return std::abs(std::round(value) / value - 1) <= error;
}
double removed(const ModulusChain &chain, unsigned source, unsigned target) {
  double sum = 0;
  for (unsigned at = source; at > target; --at)
    sum += chain.modulusLog2(at);
  return sum;
}
} // namespace

AlignmentResult planScaleAlignment(const ModulusChain &chain, const AlignmentRequest &r) {
  AlignmentResult result;
  auto invalid = [&](const std::string &message) {
    result.status = AlignmentStatus::InvalidInput;
    result.diagnostic = message;
    return result;
  };
  if (!finiteScale(r.minimumCiphertextScale) || !finiteScale(r.minimumPlaintextScale) ||
      !std::isfinite(r.maximumUnitError) || r.maximumUnitError < 0 ||
      !std::isfinite(r.capacityMargin) || r.capacityMargin < 0 ||
      !r.maximumCandidates || !r.maximumChainLength)
    return invalid("invalid precision, capacity, or search limits");
  auto stateError = [&](AlignmentState s, bool checkFloor = true) -> std::string {
    if (s.level >= chain.size() || s.level < r.minimumLevel) return "level_budget";
    if (!finiteScale(s.scale)) return "invalid_scale";
    if (checkFloor && s.scale + tolerance < r.minimumCiphertextScale) return "ciphertext_floor";
    if (s.scale + r.capacityMargin >= chain.capacity(s.level)) return "capacity";
    return {};
  };
  if (!stateError(r.input).empty() || !stateError(r.target).empty())
    return invalid("input or target state violates the configured bounds");
  if (r.operations.size() > r.maximumChainLength) {
    result.status = AlignmentStatus::SearchLimit;
    result.diagnostic = "single-use chain exceeds maximumChainLength";
    return result;
  }
  std::vector<AlignmentState> states{r.input};
  std::vector<AlignmentStep> original;
  for (unsigned i = 0; i < r.operations.size(); ++i) {
    const auto &op = r.operations[i];
    AlignmentState next = states.back();
    // A scheduled RS can need the unit multiplication being planned here to
    // satisfy the scale floor. Validate the skeleton's arithmetic/capacity,
    // then enforce the floor on EVERY state in each completed candidate.
    if (!stateError(op.output, false).empty()) return invalid("invalid original operation state");
    if (isPlain(op.kind) && !finiteScale(op.plaintextScale))
      return invalid("invalid original plaintext scale");
    if (op.kind < AlignmentKind::MulPlain || op.kind > AlignmentKind::ModSwitch)
      return invalid("unsupported chain operation");
    switch (op.kind) {
    case AlignmentKind::MulPlain: next.scale += op.plaintextScale; break;
    case AlignmentKind::AddPlain:
      if (!close(next.scale, op.plaintextScale)) return invalid("original AddCP scale mismatch");
      break;
    case AlignmentKind::Rescale:
    case AlignmentKind::ModSwitch:
      if (op.output.level >= next.level) return invalid("original level drop must be positive");
      if (op.kind == AlignmentKind::Rescale)
        for (unsigned at = next.level; at > op.output.level; --at)
          next.scale = chain.rescale(next.scale, at);
      next.level = op.output.level;
      break;
    case AlignmentKind::Rotate: case AlignmentKind::Negate: break;
    }
    if (next.level != op.output.level || !close(next.scale, op.output.scale))
      return invalid("original state disagrees with actual-prime replay");
    states.push_back(next);
    original.push_back({op, static_cast<int>(i)});
  }
  if (states.back().level != r.target.level)
    return invalid("target must preserve the original exit level");

  // Required scale at every edge; use actual primes and keep the original
  // coefficients while pulling the request across a potential absorption site.
  std::vector<double> required(states.size());
  required.back() = r.target.scale;
  for (unsigned i = r.operations.size(); i > 0; --i) {
    auto &op = r.operations[i - 1];
    double scale = required[i];
    if (op.kind == AlignmentKind::MulPlain) scale -= op.plaintextScale;
    if (op.kind == AlignmentKind::Rescale)
      scale += removed(chain, states[i - 1].level, states[i].level);
    required[i - 1] = scale;
  }
  auto cost = [](const ScaleAlignmentPlan &p) {
    // At equal resource cost prefer a later site. No latency claims are made.
    return std::make_tuple(p.addedRescaleLevels, p.addedMultiplications,
                           p.reencodedPlaintexts, -static_cast<int>(p.site));
  };
  bool limit = false;
  auto consider = [&](std::vector<AlignmentStep> steps, std::string strategy,
                      unsigned site, unsigned drops, unsigned multiplications) {
    if (limit) return;
    if (result.candidates == r.maximumCandidates) { limit = true; return; }
    ++result.candidates;
    AlignmentState current = r.input;
    unsigned encodings = 0;
    auto reject = [&](const std::string &reason) { ++result.rejected[reason]; };
    for (auto &step : steps) {
      auto &op = step.operation;
      if (op.kind == AlignmentKind::AddPlain) op.plaintextScale = current.scale;
      if (isPlain(op.kind)) {
        double plain = op.plaintextScale;
        if (!finiteScale(plain) || plain + r.capacityMargin >= chain.capacity(current.level)) {
          reject("plaintext_capacity"); return;
        }
        if (op.unitPlaintext ? !goodUnit(plain, r.maximumUnitError)
                            : plain + tolerance < r.minimumPlaintextScale) {
          reject("plaintext_precision"); return;
        }
        if (step.origin >= 0 && !close(plain, r.operations[step.origin].plaintextScale)) ++encodings;
      }
      if (op.kind == AlignmentKind::MulPlain) current.scale += op.plaintextScale;
      if (op.kind == AlignmentKind::Rescale || op.kind == AlignmentKind::ModSwitch) {
        if (op.output.level >= current.level) { reject("level_budget"); return; }
        if (op.kind == AlignmentKind::Rescale)
          for (unsigned at = current.level; at > op.output.level; --at) {
            current.scale = chain.rescale(current.scale, at);
            if (auto error = stateError({at - 1, current.scale}); !error.empty()) {
              reject(error); return;
            }
          }
        current.level = op.output.level;
      } else if (op.output.level != current.level) {
        reject("level_budget"); return;
      }
      if (auto error = stateError(current); !error.empty()) { reject(error); return; }
      op.output = current; // Store produced scales, not requested/rounded scales.
    }
    if (current.level != r.target.level || !close(current.scale, r.target.scale)) {
      reject("exit_state"); return;
    }
    ScaleAlignmentPlan candidate{std::move(steps), current, strategy, site, drops, multiplications, encodings};
    if (!result.plan || cost(candidate) < cost(*result.plan)) result.plan = std::move(candidate);
  };
  if (close(states.back().scale, r.target.scale)) {
    consider(original, "unchanged", 0, 0, 0);
    if (result.plan) {
      result.status = AlignmentStatus::Success;
      result.diagnostic = "original chain already satisfies precision and exit state";
      return result; // A verified zero-change plan cannot be improved.
    }
  }
  for (unsigned i = 0; i < original.size(); ++i) {
    if (r.operations[i].kind != AlignmentKind::MulPlain) continue;
    auto candidate = original;
    candidate[i].operation.plaintextScale = required[i + 1] - states[i].scale;
    consider(std::move(candidate), "absorb-mulcp", i, 0, 0);
  }
  // A precision-safe direct unit (including an exact small integer scale) may
  // need no Rescale. Check every edge rather than imposing a universal floor.
  for (unsigned i = 0; i < states.size(); ++i) {
    if (close(required[i], states[i].scale)) continue;
    auto candidate = original;
    AlignmentOperation unit{AlignmentKind::MulPlain, states[i], required[i] - states[i].scale, true};
    candidate.insert(candidate.begin() + i, {unit, -1});
    consider(std::move(candidate), "explicit-unit", i, 0, 1);
  }
  // Replace part of an existing ModSwitch drop with a real unit+RS conversion.
  // Enumerate contiguous prime windows, including windows above the final Q.
  for (unsigned i = 0; i < original.size() && !limit; ++i) {
    if (r.operations[i].kind != AlignmentKind::ModSwitch) continue;
    unsigned source = states[i].level, target = states[i + 1].level;
    for (unsigned drops = 1; drops <= source - target && !limit; ++drops)
      for (unsigned work = target + drops; work <= source && !limit; ++work) {
        auto candidate = std::vector<AlignmentStep>(original.begin(), original.begin() + i);
        auto append = [&](AlignmentKind kind, unsigned level, double plain = 0, bool unit = false) {
          candidate.push_back({{kind, {level, 0}, plain, unit}, -1});
        };
        if (work < source) append(AlignmentKind::ModSwitch, work);
        double plain = required[i + 1] + removed(chain, work, work - drops) - states[i].scale;
        append(AlignmentKind::MulPlain, work, plain, true);
        for (unsigned at = work; at > work - drops; --at) append(AlignmentKind::Rescale, at - 1);
        if (work - drops > target) append(AlignmentKind::ModSwitch, target);
        candidate.insert(candidate.end(), original.begin() + i + 1, original.end());
        consider(std::move(candidate), "unit-rescale", i, drops, 1);
      }
  }
  if (limit) {
    result.plan.reset();
    result.status = AlignmentStatus::SearchLimit;
    result.diagnostic = "candidate enumeration exceeds maximumCandidates";
  } else if (result.plan) {
    result.status = AlignmentStatus::Success;
    result.diagnostic = "fixed exit state verified by actual-prime replay";
  } else {
    result.diagnostic = "no precision-safe candidate in the fixed-level single-chain search; "
                        "no Boot or exit-level relaxation was attempted";
  }
  return result;
}
} // namespace hecate
