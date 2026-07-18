module {
  func.func @lazy_physical_levels_underflow(
      %input: tensor<4x!ckks.poly<2 * 40 * 5>>)
      -> tensor<4x!ckks.poly<2 * 40 * 1>> attributes {init_level = 5 : i64} {
    %modswitch = "ckks.modswitchc"(%input) {
      downFactor = 4 : i64
    } : (tensor<4x!ckks.poly<2 * 40 * 5>>)
        -> tensor<4x!ckks.poly<2 * 40 * 1>>
    return %modswitch : tensor<4x!ckks.poly<2 * 40 * 1>>
  }
}
