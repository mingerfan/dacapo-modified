#ifndef HECATE_SUPPORT_BOOTSTRAPINPUTALIGNMENT_H
#define HECATE_SUPPORT_BOOTSTRAPINPUTALIGNMENT_H

#include <cmath>
#include <optional>

namespace hecate {
struct BootstrapInputAlignment {
  unsigned drops;
  double plaintextScale;
  double capacity;
};

// ModSwitch to the smallest prefix that permits a >=30-bit encoded-one
// alignment, then Rescale to the Boot input level. Scales are log2 values.
template <class Moduli>
std::optional<BootstrapInputAlignment>
planBootstrapInputAlignment(const Moduli &moduli, unsigned sourceLevel,
                            double scale, unsigned targetLevel,
                            double targetScale) {
  if (sourceLevel < targetLevel || sourceLevel >= moduli.size() ||
      !std::isfinite(scale))
    return std::nullopt;
  double plain = targetScale - scale;
  if (std::abs(plain) <= 1e-8)
    return BootstrapInputAlignment{0, plain, 0};
  for (unsigned drops = 1; targetLevel + drops <= sourceLevel; ++drops) {
    plain += std::log2(static_cast<double>(moduli.at(targetLevel + drops)));
    if (plain < 30)
      continue;
    double capacity = 0;
    for (unsigned i = 0; i <= targetLevel + drops; ++i)
      capacity += std::log2(static_cast<double>(moduli.at(i)));
    return BootstrapInputAlignment{drops, plain, capacity};
  }
  return std::nullopt;
}
} // namespace hecate
#endif
