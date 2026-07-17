module {
  func.func @compute_ops(
      %lhs: tensor<4x!ckks.poly<2 * 40 * 5>>,
      %rhs: tensor<4x!ckks.poly<2 * 40 * 5>>)
      -> tensor<4x!ckks.poly<2 * 20 * 3>> {
    %add = "ckks.addcc"(%lhs, %rhs) :
        (tensor<4x!ckks.poly<2 * 40 * 5>>,
         tensor<4x!ckks.poly<2 * 40 * 5>>)
        -> tensor<4x!ckks.poly<2 * 40 * 5>>
    %negate = "ckks.negatec"(%add) :
        (tensor<4x!ckks.poly<2 * 40 * 5>>)
        -> tensor<4x!ckks.poly<2 * 40 * 5>>
    %rotate = "ckks.rotatec"(%negate) {offset = array<i64: 1>} :
        (tensor<4x!ckks.poly<2 * 40 * 5>>)
        -> tensor<4x!ckks.poly<2 * 40 * 5>>
    %rescale = "ckks.rescalec"(%rotate) :
        (tensor<4x!ckks.poly<2 * 40 * 5>>)
        -> tensor<4x!ckks.poly<2 * 20 * 4>>
    %modswitch = "ckks.modswitchc"(%rescale) {
      downFactor = 1 : i64
    } : (tensor<4x!ckks.poly<2 * 20 * 4>>)
        -> tensor<4x!ckks.poly<2 * 20 * 3>>
    return %modswitch : tensor<4x!ckks.poly<2 * 20 * 3>>
  }
}
