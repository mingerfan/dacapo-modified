
add_mlir_dialect_library(CKKSTransforms
  AssignPlacement.cpp
  MaterializePhysicalLevels.cpp
  MaterializeCommunication.cpp
  RemoveLevel.cpp
  EmitHEVM.cpp
  EmitRuntimePlan.cpp
  DEPENDS
  HecateCKKSTransformsIncGen
  LINK_LIBS
  PUBLIC
  CKKSDialect
  DistDialect
  MLIRIR
  )
