module {
  // Q-count is polynomial level + 1. The final prepared callback is an opaque
  // Boot: its Q30 result is authoritative, not inferred from the Q6 input.
  func.func @prepared_hoist_buffer() -> tensor<1x!ckks.poly<2 * 40 * 29>> {
    %stem = "ckks.apicall"() {callee = "prepared_stem", call_id = 10 : i64, latency_us = 20 : i64, runtime.scale_log2 = 40.125 : f64, prepared.label = "stem"} : () -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %hoisted = "ckks.apicall"(%stem) {callee = "prepared_hoist", call_id = 20 : i64, latency_us = 100 : i64, prepared.label = "shared hoist"} : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> !ckks.buffer<1234, "ckks.hoisted.v1">
    %left = "ckks.apicall"(%stem, %hoisted) {callee = "prepared_rotate_left", call_id = 30 : i64, latency_us = 70 : i64, runtime.scale_log2 = 40.125 : f64, prepared.steps = 1 : i64} : (tensor<1x!ckks.poly<2 * 40 * 5>>, !ckks.buffer<1234, "ckks.hoisted.v1">) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %right = "ckks.apicall"(%stem, %hoisted) {callee = "prepared_rotate_right", call_id = 40 : i64, latency_us = 80 : i64, runtime.scale_log2 = 40.125 : f64, prepared.steps = -1 : i64} : (tensor<1x!ckks.poly<2 * 40 * 5>>, !ckks.buffer<1234, "ckks.hoisted.v1">) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %sum = "ckks.addcc"(%left, %right) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %boot = "ckks.apicall"(%sum) {callee = "prepared_boot_q30", call_id = 60 : i64, latency_us = 500 : i64, runtime.scale_log2 = 40.125 : f64, prepared.label = "opaque Boot"} : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 29>>
    return %boot : tensor<1x!ckks.poly<2 * 40 * 29>>
  }
}
