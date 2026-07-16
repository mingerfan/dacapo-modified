module {
  func.func @upscale(%input: tensor<4x!earth.ci<20 * 0>>)
      -> tensor<4x!earth.ci<40 * 0>> attributes {init_level = 5 : i64} {
    %result = "earth.upscale"(%input) {upFactor = 20 : i64} :
        (tensor<4x!earth.ci<20 * 0>>) -> tensor<4x!earth.ci<40 * 0>>
    return %result : tensor<4x!earth.ci<40 * 0>>
  }
}
