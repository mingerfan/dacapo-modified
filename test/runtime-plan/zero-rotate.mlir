module {
  func.func @zero_rotate(%input: tensor<4x!earth.ci<20 * 0>>)
      -> tensor<4x!earth.ci<20 * 0>> attributes {init_level = 5 : i64} {
    %zero = "earth.rotate"(%input) {offset = array<i64: 0>} :
        (tensor<4x!earth.ci<20 * 0>>) -> tensor<4x!earth.ci<20 * 0>>
    %one = "earth.rotate"(%zero) {offset = array<i64: 1>} :
        (tensor<4x!earth.ci<20 * 0>>) -> tensor<4x!earth.ci<20 * 0>>
    return %one : tensor<4x!earth.ci<20 * 0>>
  }
}
