find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(
  NAME DacapoRuntimePlanPipeline
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/runtime-plan/check.py
          --hecate-opt $<TARGET_FILE:hecate-opt>
          --source-dir ${PROJECT_SOURCE_DIR}
)
