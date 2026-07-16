module {
  func.func @compute_ops(
      %lhs: tensor<4x!ckks.poly<2 * 40 * 5>>,
      %rhs: tensor<4x!ckks.poly<2 * 40 * 5>>)
      -> tensor<4x!ckks.poly<2 * 20 * 3>> {
    %add_dst = tensor.empty() : tensor<4x!ckks.poly<2 * 40 * 5>>
    %add = "ckks.addcc"(%add_dst, %lhs, %rhs) :
        (tensor<4x!ckks.poly<2 * 40 * 5>>,
         tensor<4x!ckks.poly<2 * 40 * 5>>,
         tensor<4x!ckks.poly<2 * 40 * 5>>)
        -> tensor<4x!ckks.poly<2 * 40 * 5>>
    %negate_dst = tensor.empty() : tensor<4x!ckks.poly<2 * 40 * 5>>
    %negate = "ckks.negatec"(%negate_dst, %add) :
        (tensor<4x!ckks.poly<2 * 40 * 5>>,
         tensor<4x!ckks.poly<2 * 40 * 5>>)
        -> tensor<4x!ckks.poly<2 * 40 * 5>>
    %rotate_dst = tensor.empty() : tensor<4x!ckks.poly<2 * 40 * 5>>
    %rotate = "ckks.rotatec"(%rotate_dst, %negate) {offset = array<i64: 1>} :
        (tensor<4x!ckks.poly<2 * 40 * 5>>,
         tensor<4x!ckks.poly<2 * 40 * 5>>)
        -> tensor<4x!ckks.poly<2 * 40 * 5>>
    %rescale_dst = tensor.empty() : tensor<4x!ckks.poly<2 * 20 * 4>>
    %rescale = "ckks.rescalec"(%rescale_dst, %rotate) :
        (tensor<4x!ckks.poly<2 * 20 * 4>>,
         tensor<4x!ckks.poly<2 * 40 * 5>>)
        -> tensor<4x!ckks.poly<2 * 20 * 4>>
    %modswitch_dst = tensor.empty() : tensor<4x!ckks.poly<2 * 20 * 3>>
    %modswitch = "ckks.modswitchc"(%modswitch_dst, %rescale) {
      downFactor = 1 : i64
    } : (tensor<4x!ckks.poly<2 * 20 * 3>>,
         tensor<4x!ckks.poly<2 * 20 * 4>>)
        -> tensor<4x!ckks.poly<2 * 20 * 3>>
    return %modswitch : tensor<4x!ckks.poly<2 * 20 * 3>>
  }
}
