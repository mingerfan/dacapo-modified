
add_mlir_dialect_library(CKKSTransforms
  RemoveLevel.cpp
  EmitHEVM.cpp
  EmitRuntimePlan.cpp
  DEPENDS
  HecateCKKSTransformsIncGen
  LINK_LIBS
  PUBLIC
  CKKSDialect
  MLIRIR
  )
