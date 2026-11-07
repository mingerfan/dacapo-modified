// JSON adapter for exercising the production C++ planner without MLIR/GPU.
#include "hecate/Support/ScaleAlignmentPlanning.h"
#include "nlohmann/json.hpp"
#include <cstdint>
#include <iostream>
#include <stdexcept>

using nlohmann::json;
using namespace hecate;
namespace {
const std::map<std::string, AlignmentKind> kinds{
    {"mul_cp", AlignmentKind::MulPlain}, {"add_cp", AlignmentKind::AddPlain},
    {"rotate", AlignmentKind::Rotate}, {"negate", AlignmentKind::Negate},
    {"rescale", AlignmentKind::Rescale}, {"mod_switch", AlignmentKind::ModSwitch}};
unsigned natural(const json &value) {
  if (!value.is_number_integer() || value.get<int64_t>() < 0 || value.get<int64_t>() > 1000000)
    throw std::invalid_argument("invalid nonnegative integer");
  return value.get<unsigned>();
}
AlignmentState state(const json &j) { return {natural(j.at("level")), j.at("scale_log2").get<double>()}; }
json state(AlignmentState s) { return {{"level", s.level}, {"scale_log2", s.scale}}; }
std::string kind(AlignmentKind k) {
  for (auto &[name, value] : kinds) if (k == value) return name;
  throw std::invalid_argument("unknown operation kind");
}
} // namespace

int main() {
  try {
    json j;
    std::cin >> j;
    auto moduli = j.at("moduli").get<std::vector<int64_t>>();
    if (moduli.empty() || moduli.size() > 1024) throw std::invalid_argument("invalid modulus count");
    for (auto q : moduli)
      if (q <= 1 || q >= (1LL << 53)) throw std::invalid_argument("invalid exact native modulus");
    ModulusChain chain;
    chain.reset(moduli);
    AlignmentRequest r;
    r.input = state(j.at("input"));
    r.target = state(j.at("target"));
    auto &p = j.at("policy");
    r.minimumLevel = natural(p.at("minimum_level"));
    r.minimumCiphertextScale = p.at("minimum_ciphertext_scale");
    r.minimumPlaintextScale = p.at("minimum_plaintext_scale");
    r.maximumUnitError = p.at("maximum_unit_error");
    r.capacityMargin = p.at("capacity_margin");
    if (p.contains("maximum_chain_length")) r.maximumChainLength = natural(p.at("maximum_chain_length"));
    if (p.contains("maximum_candidates")) r.maximumCandidates = natural(p.at("maximum_candidates"));
    for (auto &op : j.at("operations"))
      r.operations.push_back({kinds.at(op.at("op").get<std::string>()), state(op.at("output")),
                             op.value("plaintext_scale", 0.), op.value("unit_plaintext", false)});
    auto result = planScaleAlignment(chain, r);
    const char *status = result.status == AlignmentStatus::Success ? "success" :
                        result.status == AlignmentStatus::InvalidInput ? "invalid-input" :
                        result.status == AlignmentStatus::SearchLimit ? "search-limit" : "no-candidate";
    json output{{"status", status}, {"diagnostic", result.diagnostic},
                {"candidates", result.candidates}, {"rejected", result.rejected}};
    if (result.plan) {
      auto &plan = *result.plan;
      output["plan"] = {{"strategy", plan.strategy}, {"site", plan.site}, {"output", state(plan.output)},
                        {"added_rescale_levels", plan.addedRescaleLevels},
                        {"added_multiplications", plan.addedMultiplications},
                        {"reencoded_plaintexts", plan.reencodedPlaintexts}, {"steps", json::array()}};
      for (auto &step : plan.steps)
        output["plan"]["steps"].push_back({{"origin", step.origin}, {"op", kind(step.operation.kind)},
            {"output", state(step.operation.output)}, {"plaintext_scale", step.operation.plaintextScale},
            {"unit_plaintext", step.operation.unitPlaintext}});
    }
    std::cout << output.dump(2) << '\n';
    return result.status == AlignmentStatus::Success ? 0 : 2;
  } catch (const std::exception &error) {
    std::cout << json({{"status", "invalid-input"}, {"diagnostic", error.what()}}).dump(2) << '\n';
    return 2;
  }
}
