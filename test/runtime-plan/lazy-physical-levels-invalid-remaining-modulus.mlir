module {
  func.func @lazy_physical_levels_invalid_remaining_modulus(
      %input: tensor<4x!ckks.poly<2 * 270 * 1>>)
      -> tensor<4x!ckks.poly<2 * 150 * 0>> attributes {init_level = 9 : i64} {
    %rescale = "ckks.rescalec"(%input) :
        (tensor<4x!ckks.poly<2 * 270 * 1>>)
        -> tensor<4x!ckks.poly<2 * 150 * 0>>
    return %rescale : tensor<4x!ckks.poly<2 * 150 * 0>>
  }
}
