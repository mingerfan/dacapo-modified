module {
  func.func @placement_boot(%arg0: tensor<1x!ckks.poly<2 * 40 * 2>>) -> tensor<1x!ckks.poly<2 * 40 * 5>> {
    %0 = "ckks.negatec"(%arg0) : (tensor<1x!ckks.poly<2 * 40 * 2>>) -> tensor<1x!ckks.poly<2 * 40 * 2>>
    %1 = "ckks.bootstrapc"(%0) : (tensor<1x!ckks.poly<2 * 40 * 2>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %2 = "ckks.negatec"(%1) : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    return %2 : tensor<1x!ckks.poly<2 * 40 * 5>>
  }
}
