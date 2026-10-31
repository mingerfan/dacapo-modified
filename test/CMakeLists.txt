find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(DacapoModulusChainTest Support/ModulusChainTest.cpp)
add_test(NAME DacapoModulusChain COMMAND DacapoModulusChainTest)
add_test(NAME DacapoRegionalScale
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-regional-scale.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR})

add_test(NAME DacapoReductionRescales
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-reduction-rescales.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR})

add_test(NAME DacapoNativePlanning
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-native-planning.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR})

add_test(NAME DacapoPolynomialArithmetic
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-polynomial-arithmetic.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR})

add_test(NAME DacapoThresholdGraphs
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-threshold-graphs.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR})

add_test(NAME DacapoNativeScale40
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-native-scale40.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR})

add_test(NAME DacapoNativeScale45
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-native-scale40.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR}
          --boot-scale 45)

add_test(NAME DacapoRuntimePlanBinary
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-binary.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR})

add_test(NAME DacapoRuntimePlanIO
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-io.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR})

add_test(
  NAME DacapoUnitPlaintext
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-unit-plaintext.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR}
)

add_test(
  NAME DacapoRuntimeScale
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-runtime-scale.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR}
)

add_test(
  NAME DacapoFrontendLocations
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-frontend-locations.py
          --source-dir ${PROJECT_SOURCE_DIR}
)

add_test(
  NAME DacapoFrontendBytecode
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-frontend-bytecode.py
          --frontend $<TARGET_FILE:HecateFrontend>
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR}
)

add_test(
  NAME DacapoGreedyBootstrap
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-greedy.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR}
)

add_test(
  NAME DacapoManualPrecompile
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-precompile.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR}
)

add_test(
  NAME DacapoRuntimePlanPipeline
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR}
)

add_test(
  NAME DacapoRuntimeMemory
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-memory.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR}
)

add_test(NAME DacapoPlaintextWaterline
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-waterline.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR})

add_test(NAME DacapoOptimizedNativeScale45
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-native-scale40.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR}
          --boot-scale 45 --strategy optimized)

add_test(NAME DacapoOptimizedSchedule
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check-optimized.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR})
