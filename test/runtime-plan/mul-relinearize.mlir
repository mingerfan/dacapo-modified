module {
  func.func @mul_relinearize(
      %lhs: tensor<4x!earth.ci<20 * 0>>,
      %rhs: tensor<4x!earth.ci<20 * 0>>)
      -> tensor<4x!earth.ci<40 * 0>> attributes {init_level = 5 : i64} {
    %result = "earth.mul"(%lhs, %rhs) :
        (tensor<4x!earth.ci<20 * 0>>, tensor<4x!earth.ci<20 * 0>>)
        -> tensor<4x!earth.ci<40 * 0>>
    return %result : tensor<4x!earth.ci<40 * 0>>
  }
}
