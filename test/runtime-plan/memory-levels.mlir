module {
  func.func @memory_levels(%arg0: tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 1>> {
    %0 = "ckks.modswitchc"(%arg0) <{downFactor = 4 : i64}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    return %0 : tensor<1x!ckks.poly<2 * 40 * 1>>
  }
}
