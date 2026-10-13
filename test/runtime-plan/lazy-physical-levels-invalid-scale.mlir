module {
  func.func @lazy_physical_levels_invalid_scale(
      %input: tensor<4x!ckks.poly<2 * 241 * 9>>)
      -> tensor<4x!ckks.poly<2 * 120 * 8>> attributes {init_level = 9 : i64} {
    %rescale = "ckks.rescalec"(%input) :
        (tensor<4x!ckks.poly<2 * 241 * 9>>)
        -> tensor<4x!ckks.poly<2 * 120 * 8>>
    return %rescale : tensor<4x!ckks.poly<2 * 120 * 8>>
  }
}
