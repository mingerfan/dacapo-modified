module {
  func.func @invalid_relinearize(
      %input: tensor<4x!ckks.poly<2 * 40 * 5>>)
      -> tensor<4x!ckks.poly<2 * 40 * 5>> {
    %result = "ckks.relinearize"(%input) :
        (tensor<4x!ckks.poly<2 * 40 * 5>>)
        -> tensor<4x!ckks.poly<2 * 40 * 5>>
    return %result : tensor<4x!ckks.poly<2 * 40 * 5>>
  }
}
