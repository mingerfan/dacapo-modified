#include "hecate/Support/BootstrapInputAlignment.h"
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace {
void check(bool condition, const char *message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

double capacity(const std::vector<uint64_t> &moduli, unsigned level) {
  double result = 0;
  for (unsigned i = 0; i <= level; ++i)
    result += std::log2(static_cast<double>(moduli.at(i)));
  return result;
}

// Exhaustively try every working prefix, independently of the binary search.
std::optional<unsigned> referenceAlignment(const std::vector<uint64_t> &moduli,
                                          unsigned source, double scale,
                                          unsigned target, double targetScale) {
  if (scale >= capacity(moduli, source) || targetScale >= capacity(moduli, target))
    return std::nullopt;
  if (std::abs(scale - targetScale) <= 1e-8)
    return 0;
  for (unsigned work = target + 1; work <= source; ++work) {
    double removed = 0;
    for (unsigned i = target + 1; i <= work; ++i)
      removed += std::log2(static_cast<double>(moduli.at(i)));
    double plain = targetScale - scale + removed;
    if (plain >= 30 && scale + plain < capacity(moduli, work))
      return work - target;
  }
  return std::nullopt;
}

void checkChain(const std::vector<uint64_t> &moduli) {
  hecate::ModulusChain chain;
  chain.reset(moduli);
  for (unsigned source = 0; source < moduli.size(); ++source) {
    check(std::abs(chain.capacity(source) - capacity(moduli, source)) < 1e-10,
          "capacity differs from actual-prime sum");
    for (unsigned drops = 0; drops <= source + 1; ++drops) {
      double expected = 250;
      for (unsigned i = 0; i < drops; ++i)
        expected -= std::log2(static_cast<double>(moduli[source - i]));
      check(std::abs(chain.rescale(250, source, drops) - expected) < 1e-10,
            "multi-RS differs from sequential actual-prime division");
    }
    for (double scale : {39., 40., 60., 70., 80., 100., 120., 250., 900.}) {
      unsigned drops = 0;
      double expected = scale;
      while (drops <= source &&
             expected - std::log2(static_cast<double>(moduli[source - drops])) >= 40) {
        expected -= std::log2(static_cast<double>(moduli[source - drops]));
        ++drops;
      }
      auto normalized = chain.normalize(scale, source, 40);
      check(normalized.drops == drops && normalized.scale == expected,
            "threshold normalization changed sequential RS behavior");
    }
    for (unsigned target = 0; target <= source; ++target)
      for (double scale = 1; scale < 350; scale += 0.73) {
        auto expected = referenceAlignment(moduli, source, scale, target, 40);
        auto plan = hecate::planBootstrapInputAlignment(chain, source, scale, target, 40);
        check(plan.has_value() == expected.has_value(), "Boot feasibility disagrees with enumeration");
        if (!plan)
          continue;
        check(plan->drops == *expected, "Boot plan did not choose the smallest working prefix");
        double replay = scale + plan->plaintextScale;
        for (unsigned i = plan->drops; i > 0; --i)
          replay -= std::log2(static_cast<double>(moduli[target + i]));
        check(std::abs(replay - 40) < 1e-8, "Boot plan replay missed the fixed input scale");
        check(!plan->drops || plan->plaintextScale >= 30, "Boot unit encoding lost precision");
      }
  }
}
} // namespace

int main() {
  // Actual values matter even at the same nominal bit width. Include moduli
  // near both ends of a bit-width interval and a nonuniform 30/31/32-bit chain.
  checkChain({1073479681, 1072496641, 1071513601, 1070727169, 1069219841});
  checkChain({1073479681, 2147352577, 4293918721, 1072496641, 2146041857, 4292870145});
  checkChain({536903681, 536952833, 1073479681, 1072496641, 536903681});

  hecate::ModulusChain chain;
  // Exact powers of two isolate inclusive thresholds from floating-point logs.
  chain.reset(std::vector<uint64_t>{1ULL << 30, 1ULL << 30, 1ULL << 30, 1ULL << 32});
  auto aligned = hecate::planBootstrapInputAlignment(chain, 3, 40, 1, 40);
  check(aligned && aligned->drops == 0 && aligned->plaintextScale == 0,
        "already aligned input should only need ModSwitch");
  auto one = hecate::planBootstrapInputAlignment(chain, 3, 39, 1, 40);
  auto two = hecate::planBootstrapInputAlignment(chain, 3, 41, 1, 40);
  check(one && one->drops == 1 && two && two->drops == 2,
        "a 30-bit removed modulus must not be rounded into enough encoding precision");
  auto boundary = hecate::planBootstrapInputAlignment(chain, 3, 72, 1, 40);
  check(boundary && boundary->drops == 2 && boundary->plaintextScale == 30,
        "exact 30-bit unit-encoding boundary must be accepted");
  check(!hecate::planBootstrapInputAlignment(chain, 2, 41, 1, 40),
        "insufficient Q must be rejected");
  check(!hecate::planBootstrapInputAlignment(chain, 3, 60, 1, 60),
        "already aligned scale at modulus capacity must be rejected");
  check(!hecate::planBootstrapInputAlignment(chain, 3, 60, 1, 60 - 5e-9),
        "alignment tolerance must not hide target capacity overflow");
  check(!hecate::planBootstrapInputAlignment(chain, 3, 60, 1, 61),
        "alignment exceeding target modulus capacity must be rejected");
  check(!hecate::planBootstrapInputAlignment(chain, 0, 40, 1, 40),
        "source below Boot input level must be rejected");
  check(!hecate::planBootstrapInputAlignment(chain, 4, 40, 1, 40),
        "source outside chain must be rejected");
  check(!hecate::planBootstrapInputAlignment(chain, 3, -1, 1, 40) &&
        !hecate::planBootstrapInputAlignment(chain, 3, 40, 1,
                                            std::numeric_limits<double>::quiet_NaN()),
        "invalid scales must be rejected");
  auto exhausted = chain.normalize(100, 0, 40);
  check(exhausted.drops == 1, "exhausted trial must report consumption of terminal q0");
  chain.reset(std::vector<uint64_t>{1ULL << 31, 1ULL << 32});
  check(chain.size() == 2 && chain.capacity(1) == 63,
        "reset must discard the previous chain prefix sums");
  std::cout << "actual-prime transitions and exhaustive Boot alignment checks passed\n";
}
