module {
  func.func @placement_communication_model(%arg0: tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>> {
    %0 = "ckks.rotatec"(%arg0) <{offset = array<i64: 1>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %1 = "ckks.rotatec"(%arg0) <{offset = array<i64: 2>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %2 = "ckks.addcc"(%0, %1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    return %2 : tensor<1x!ckks.poly<2 * 40 * 5>>
  }
}
