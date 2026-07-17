module {
  func.func @placement_fanout(%arg0: tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>> {
    %0 = "ckks.rotatec"(%arg0) <{offset = array<i64: 1>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %1 = "ckks.rotatec"(%arg0) <{offset = array<i64: 2>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %2 = "ckks.rotatec"(%arg0) <{offset = array<i64: 3>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %3 = "ckks.rotatec"(%arg0) <{offset = array<i64: 4>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %4 = "ckks.rotatec"(%arg0) <{offset = array<i64: 5>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %5 = "ckks.rotatec"(%arg0) <{offset = array<i64: 6>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %6 = "ckks.rotatec"(%arg0) <{offset = array<i64: 7>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %7 = "ckks.rotatec"(%arg0) <{offset = array<i64: -1>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %8 = "ckks.rotatec"(%arg0) <{offset = array<i64: -2>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %9 = "ckks.rotatec"(%arg0) <{offset = array<i64: -3>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %10 = "ckks.rotatec"(%arg0) <{offset = array<i64: -4>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %11 = "ckks.rotatec"(%arg0) <{offset = array<i64: -5>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %12 = "ckks.rotatec"(%arg0) <{offset = array<i64: -6>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %13 = "ckks.rotatec"(%arg0) <{offset = array<i64: -7>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %14 = "ckks.rotatec"(%arg0) <{offset = array<i64: 1>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %15 = "ckks.rotatec"(%arg0) <{offset = array<i64: 2>}> : (tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %16 = "ckks.addcc"(%0, %1) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %17 = "ckks.addcc"(%2, %3) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %18 = "ckks.addcc"(%4, %5) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %19 = "ckks.addcc"(%6, %7) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %20 = "ckks.addcc"(%8, %9) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %21 = "ckks.addcc"(%10, %11) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %22 = "ckks.addcc"(%12, %13) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %23 = "ckks.addcc"(%14, %15) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %24 = "ckks.addcc"(%16, %17) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %25 = "ckks.addcc"(%18, %19) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %26 = "ckks.addcc"(%20, %21) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %27 = "ckks.addcc"(%22, %23) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %28 = "ckks.addcc"(%24, %25) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %29 = "ckks.addcc"(%26, %27) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    %30 = "ckks.addcc"(%28, %29) : (tensor<1x!ckks.poly<2 * 40 * 5>>, tensor<1x!ckks.poly<2 * 40 * 5>>) -> tensor<1x!ckks.poly<2 * 40 * 5>>
    return %30 : tensor<1x!ckks.poly<2 * 40 * 5>>
  }
}
