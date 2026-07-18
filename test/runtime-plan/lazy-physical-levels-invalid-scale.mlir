module {
  func.func @lazy_physical_levels_invalid_scale(
      %input: tensor<4x!ckks.poly<2 * 159 * 5>>)
      -> tensor<4x!ckks.poly<2 * 40 * 4>> attributes {init_level = 5 : i64} {
    %rescale = "ckks.rescalec"(%input) :
        (tensor<4x!ckks.poly<2 * 159 * 5>>)
        -> tensor<4x!ckks.poly<2 * 40 * 4>>
    return %rescale : tensor<4x!ckks.poly<2 * 40 * 4>>
  }
}
