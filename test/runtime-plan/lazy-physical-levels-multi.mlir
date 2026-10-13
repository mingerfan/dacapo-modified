module {
  func.func @lazy_physical_levels_multi(
      %input: tensor<4x!ckks.poly<2 * 480 * 9>>)
      -> tensor<4x!ckks.poly<2 * 120 * 6>> attributes {init_level = 9 : i64} {
    %rescale = "ckks.rescalec"(%input) :
        (tensor<4x!ckks.poly<2 * 480 * 9>>)
        -> tensor<4x!ckks.poly<2 * 120 * 6>>
    return %rescale : tensor<4x!ckks.poly<2 * 120 * 6>>
  }
}
