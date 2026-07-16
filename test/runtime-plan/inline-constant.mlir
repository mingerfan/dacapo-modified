module {
  func.func @inline_constant(%input: tensor<4x!earth.ci<20 * 0>>)
      -> tensor<4x!earth.ci<20 * 0>> attributes {init_level = 5 : i64} {
    %constant = "earth.constant"() {
      value = dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf64>,
      rms_var = 2.7386127875258306 : f64
    } : () -> tensor<4x!earth.pl<20 * 0>>
    %result = "earth.add"(%input, %constant) :
        (tensor<4x!earth.ci<20 * 0>>, tensor<4x!earth.pl<20 * 0>>)
        -> tensor<4x!earth.ci<20 * 0>>
    return %result : tensor<4x!earth.ci<20 * 0>>
  }
}
