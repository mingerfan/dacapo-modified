module {
  // Native-scale metadata must not make the exporter infer opaque API scales.
  func.func @prepared_branching() -> tensor<1x!ckks.poly<2 * 45 * 1>> attributes {ckks.rns_moduli = array<i64: 1099511627791, 1099511627791, 1099511627791, 1099511627791, 1099511627791, 1099511627791>} {
    %stem = "ckks.apicall"() {callee = "prepared_stem", call_id = 41 : i64, latency_us = 20 : i64, prepared.label = "stem", prepared.persistent = true} : () -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %left = "ckks.apicall"(%stem) {callee = "prepared_left", call_id = 7 : i64, latency_us = 1000 : i64, prepared.label = "left", prepared.batch = 3 : i64} : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 53 * 3>>
    %right = "ckks.apicall"(%stem) {callee = "prepared_right", call_id = 103 : i64, latency_us = 700 : i64, prepared.label = "right", prepared.levels = array<i64: 5, 4, 1>} : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 37 * 4>>
    %join = "ckks.apicall"(%right, %left) {callee = "prepared_join", call_id = 23 : i64, latency_us = 120 : i64, prepared.label = "join", prepared.note = "operand order matters"} : (tensor<1x!ckks.poly<2 * 37 * 4>>, tensor<1x!ckks.poly<2 * 53 * 3>>) -> tensor<1x!ckks.poly<2 * 48 * 2>>
    %head = "ckks.apicall"(%join) {callee = "prepared_head", call_id = 9007199254741009 : i64, latency_us = 80 : i64, prepared.label = "head", prepared.extra = {coefficients = array<i64: 9, 2>, note = "untouched"}} : (tensor<1x!ckks.poly<2 * 48 * 2>>) -> tensor<1x!ckks.poly<2 * 45 * 1>>
    return %head : tensor<1x!ckks.poly<2 * 45 * 1>>
  }
}
