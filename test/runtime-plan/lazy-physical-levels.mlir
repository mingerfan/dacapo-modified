module {
  func.func @lazy_physical_levels(
      %input: tensor<4x!ckks.poly<2 * 160 * 5>>)
      -> tensor<4x!ckks.poly<2 * 40 * 3>> attributes {init_level = 5 : i64} {
    %rescale = "ckks.rescalec"(%input) :
        (tensor<4x!ckks.poly<2 * 160 * 5>>)
        -> tensor<4x!ckks.poly<2 * 40 * 4>>
    %modswitch = "ckks.modswitchc"(%rescale) {
      downFactor = 1 : i64
    } : (tensor<4x!ckks.poly<2 * 40 * 4>>)
        -> tensor<4x!ckks.poly<2 * 40 * 3>>
    return %modswitch : tensor<4x!ckks.poly<2 * 40 * 3>>
  }
}
