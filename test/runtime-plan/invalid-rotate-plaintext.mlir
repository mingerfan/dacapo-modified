module {
  func.func @invalid_rotate_plaintext(
      %plain: tensor<4x!ckks.poly<1 * 40 * 5>>)
      -> tensor<4x!ckks.poly<1 * 40 * 5>> {
    %result = "ckks.rotatec"(%plain) {offset = array<i64: 1>} :
        (tensor<4x!ckks.poly<1 * 40 * 5>>)
        -> tensor<4x!ckks.poly<1 * 40 * 5>>
    return %result : tensor<4x!ckks.poly<1 * 40 * 5>>
  }
}
