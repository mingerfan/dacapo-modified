find_package(Python3 REQUIRED COMPONENTS Interpreter)

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
