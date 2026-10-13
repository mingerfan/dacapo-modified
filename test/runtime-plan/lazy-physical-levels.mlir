module {
  func.func @lazy_physical_levels(
      %input: tensor<4x!ckks.poly<2 * 240 * 9>>)
      -> tensor<4x!ckks.poly<2 * 120 * 0>> attributes {init_level = 9 : i64} {
    %rescale = "ckks.rescalec"(%input) :
        (tensor<4x!ckks.poly<2 * 240 * 9>>)
        -> tensor<4x!ckks.poly<2 * 120 * 8>>
    %modswitch = "ckks.modswitchc"(%rescale) {
      downFactor = 8 : i64
    } : (tensor<4x!ckks.poly<2 * 120 * 8>>)
        -> tensor<4x!ckks.poly<2 * 120 * 0>>
    return %modswitch : tensor<4x!ckks.poly<2 * 120 * 0>>
  }
}
