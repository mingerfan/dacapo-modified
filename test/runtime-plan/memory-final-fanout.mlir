module {
  func.func @memory_final_fanout(%arg0: tensor<1x!ckks.poly<2 * 40 * 1>>) -> (tensor<1x!ckks.poly<2 * 40 * 1>>, tensor<1x!ckks.poly<2 * 40 * 1>>) {
    %0 = "ckks.negatec"(%arg0) : (tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    %1 = "ckks.negatec"(%0) : (tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    %2 = "ckks.negatec"(%0) : (tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    %3 = "ckks.addcc"(%1, %1) : (tensor<1x!ckks.poly<2 * 40 * 1>>, tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    %4 = "ckks.addcc"(%2, %3) : (tensor<1x!ckks.poly<2 * 40 * 1>>, tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    return %0, %4 : tensor<1x!ckks.poly<2 * 40 * 1>>, tensor<1x!ckks.poly<2 * 40 * 1>>
  }
}
