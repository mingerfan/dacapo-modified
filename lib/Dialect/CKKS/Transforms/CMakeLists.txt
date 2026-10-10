
add_mlir_dialect_library(CKKSTransforms
  AssignPlacement.cpp
  MaterializePhysicalLevels.cpp
  MaterializeCommunication.cpp
  EmitRuntimePlan.cpp
  RuntimePlanUtils.cpp
  PlanPlaintextStreaming.cpp
  PlanRuntimeMemory.cpp
  EstimateRuntimeMemory.cpp
  DEPENDS
  HecateCKKSTransformsIncGen
  LINK_LIBS
  PUBLIC
  CKKSDialect
  DistDialect
  MLIRIR
  )

# Convert JSON and arithmetic failures into pass diagnostics.
set_source_files_properties(EstimateRuntimeMemory.cpp PlanPlaintextStreaming.cpp PROPERTIES COMPILE_OPTIONS "-fexceptions")
