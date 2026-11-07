#ifndef HECATE_SUPPORT_SCALEALIGNMENTPLANNING_H
#define HECATE_SUPPORT_SCALEALIGNMENTPLANNING_H

#include "hecate/Support/ModulusChain.h"
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace hecate {

// Physical levels: L means q0..qL remain. Scales are actual log2 values.
struct AlignmentState {
  unsigned level;
  double scale;
};
enum class AlignmentKind { MulPlain, AddPlain, Rotate, Negate, Rescale, ModSwitch };
struct AlignmentOperation {
  AlignmentKind kind;
  AlignmentState output;
  double plaintextScale = 0;
  bool unitPlaintext = false; // Full-slot +/-1 only; never a mask.
};
struct AlignmentRequest {
  AlignmentState input, target;
  // A private, single-use ciphertext chain. Shared/fixed values terminate it.
  // Plaintext payloads and rotation offsets stay with the caller's operation.
  // A planned RS may temporarily leave scale below the floor in this input
  // skeleton. All states in a successful output plan must satisfy the floor.
  std::vector<AlignmentOperation> operations;
  unsigned minimumLevel = 0;
  double minimumCiphertextScale = 40;
  double minimumPlaintextScale = 40;
  double maximumUnitError = 1e-6;
  double capacityMargin = 1;
  unsigned maximumChainLength = 128, maximumCandidates = 8192;
};
struct AlignmentStep {
  AlignmentOperation operation;
  // Index in the immutable input snapshot; -1 denotes a new unit conversion.
  int origin = -1;
};
struct ScaleAlignmentPlan {
  std::vector<AlignmentStep> steps;
  AlignmentState output;
  std::string strategy;
  unsigned site = 0;
  unsigned addedRescaleLevels = 0, addedMultiplications = 0, reencodedPlaintexts = 0;
};
enum class AlignmentStatus { Success, InvalidInput, NoCandidate, SearchLimit };
struct AlignmentResult {
  AlignmentStatus status = AlignmentStatus::NoCandidate;
  std::string diagnostic;
  unsigned candidates = 0;
  std::map<std::string, unsigned> rejected;
  std::optional<ScaleAlignmentPlan> plan;
};

// No IR access, mutation, Boot insertion, or secret-key/backend dependency.
// Searches one absorbed coefficient change OR one explicit conversion. Other
// MulCP scales and existing Rescale placements stay fixed; AddCP encodings
// follow their ciphertext inputs. NoCandidate is not a global infeasibility
// proof. Every accepted result preserves the requested physical exit state.
AlignmentResult planScaleAlignment(const ModulusChain &chain,
                                   const AlignmentRequest &request);

} // namespace hecate
#endif
