module {
  func.func @placement_boot(%arg0: tensor<1x!ckks.poly<2 * 40 * 2>>) -> tensor<1x!ckks.poly<2 * 40 * 5>> {
    %0 = "ckks.bootstrapc"(%arg0) : (tensor<1x!ckks.poly<2 * 40 * 2>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    return %0 : tensor<1x!ckks.poly<2 * 40 * 5>>
  }
}
