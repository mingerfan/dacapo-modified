#ifndef HECATE_SUPPORT_BOOTSTRAPINPUTALIGNMENT_H
#define HECATE_SUPPORT_BOOTSTRAPINPUTALIGNMENT_H

#include "hecate/Support/ModulusChain.h"
#include <cmath>
#include <optional>

namespace hecate {
struct BootstrapInputAlignment {
  unsigned drops;
  double plaintextScale;
  double capacity;
};

// ModSwitch to the smallest prefix that permits a >=30-bit encoded-one
// alignment, then Rescale to the Boot input level. The returned plan includes
// the capacity check, so planning and materialization accept the same states.
// Chain preprocessing is O(L); each query is O(log L).
inline std::optional<BootstrapInputAlignment>
planBootstrapInputAlignment(const ModulusChain &chain, unsigned sourceLevel,
                            double scale, unsigned targetLevel,
                            double targetScale) {
  if (sourceLevel < targetLevel || sourceLevel >= chain.size() ||
      !std::isfinite(scale) || scale < 0 || !std::isfinite(targetScale) ||
      targetScale < 0 || scale >= chain.capacity(sourceLevel) ||
      targetScale >= chain.capacity(targetLevel))
    return std::nullopt;
  if (std::abs(targetScale - scale) <= 1e-8) {
    if (scale >= chain.capacity(targetLevel))
      return std::nullopt;
    return BootstrapInputAlignment{0, 0, chain.capacity(targetLevel)};
  }
  auto plaintextScale = [&](unsigned drops) {
    return targetScale - scale + chain.removedLog2(targetLevel + drops, drops);
  };
  unsigned low = 1, high = sourceLevel - targetLevel;
  if (high == 0 || plaintextScale(high) < 30)
    return std::nullopt;
  while (low < high) {
    unsigned mid = low + (high - low) / 2;
    if (plaintextScale(mid) >= 30)
      high = mid;
    else
      low = mid + 1;
  }
  double plain = plaintextScale(low);
  double capacity = chain.capacity(targetLevel + low);
  if (!std::isfinite(plain) || scale + plain >= capacity)
    return std::nullopt;
  return BootstrapInputAlignment{low, plain, capacity};
}
} // namespace hecate
#endif
