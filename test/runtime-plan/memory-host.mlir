module {
  func.func @memory_host(%arg0: tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>> {
    %0 = "ckks.negatec"(%arg0) : (tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    // Encode is deliberately physically after the first compute, but runs in initialization.
    %w = "ckks.encode"() <{payload = dense<1.0> : tensor<1xf64>}> : () -> tensor<1x!ckks.poly<1 * 40 * 1>>
    %1 = "ckks.negatec"(%0) : (tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    %2 = "ckks.negatec"(%1) : (tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    %3 = "ckks.negatec"(%2) : (tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    %4 = "ckks.addcp"(%3, %w) : (tensor<1x!ckks.poly<2 * 40 * 1>>, tensor<1x!ckks.poly<1 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    return %4 : tensor<1x!ckks.poly<2 * 40 * 1>>
  }
}
