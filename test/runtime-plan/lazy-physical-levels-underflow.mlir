module {
  func.func @lazy_physical_levels_underflow(
      %input: tensor<4x!ckks.poly<2 * 120 * 10>>)
      -> tensor<4x!ckks.poly<2 * 120 * 0>> attributes {init_level = 10 : i64} {
    %modswitch = "ckks.modswitchc"(%input) {
      downFactor = 10 : i64
    } : (tensor<4x!ckks.poly<2 * 120 * 10>>)
        -> tensor<4x!ckks.poly<2 * 120 * 0>>
    return %modswitch : tensor<4x!ckks.poly<2 * 120 * 0>>
  }
}
