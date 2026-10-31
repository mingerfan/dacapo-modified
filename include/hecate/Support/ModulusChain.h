#ifndef HECATE_SUPPORT_MODULUSCHAIN_H
#define HECATE_SUPPORT_MODULUSCHAIN_H

#include <cassert>
#include <cmath>
#include <vector>

namespace hecate {

// Moduli are ordered q0, q1, ... . Physical level l means Q(l+1).
// All scales and capacities here are log2 values, not integer bit widths.
class ModulusChain {
public:
  template <class Moduli> void reset(const Moduli &moduli) {
    logs.clear();
    prefix.assign(1, 0);
    logs.reserve(moduli.size());
    prefix.reserve(moduli.size() + 1);
    for (auto q : moduli) {
      assert(q > 1);
      logs.push_back(std::log2(static_cast<double>(q)));
      prefix.push_back(prefix.back() + logs.back());
    }
  }

  unsigned size() const { return logs.size(); }
  double modulusLog2(unsigned level) const { return logs.at(level); }
  double capacity(unsigned level) const { return prefix.at(level + 1); }

  double removedLog2(unsigned sourceLevel, unsigned drops) const {
    assert(sourceLevel < size() && drops <= sourceLevel + 1);
    // Preserve the exact single-RS calculation used by physical replay.
    if (drops == 1)
      return modulusLog2(sourceLevel);
    return capacity(sourceLevel) - prefix.at(sourceLevel + 1 - drops);
  }

  double rescale(double scale, unsigned sourceLevel, unsigned drops = 1) const {
    return scale - removedLog2(sourceLevel, drops);
  }

  struct Normalization {
    unsigned drops;
    double scale;
  };

  Normalization normalize(double scale, unsigned sourceLevel,
                          double waterline) const {
    assert(sourceLevel < size());
    unsigned drops = 0;
    // Replay each RS in emission order. This is O(1 + number of RS), with
    // no repeated logarithms. Consuming q0 is an infeasible-trial sentinel;
    // callers must reject it before emitting any operations.
    while (drops <= sourceLevel) {
      double next = rescale(scale, sourceLevel - drops);
      if (next < waterline)
        break;
      scale = next;
      ++drops;
    }
    return {drops, scale};
  }

private:
  std::vector<double> logs;
  std::vector<double> prefix{0};
};

} // namespace hecate
#endif
