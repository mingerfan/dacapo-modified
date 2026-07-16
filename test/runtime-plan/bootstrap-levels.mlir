module {
  func.func @bootstrap_levels(%input: tensor<4x!earth.ci<40 * 13>>)
      -> tensor<4x!earth.ci<40 * 0>> attributes {init_level = 16 : i64} {
    %result = "earth.bootstrap"(%input) {targetLevel = 0 : i64} :
        (tensor<4x!earth.ci<40 * 13>>) -> tensor<4x!earth.ci<40 * 0>>
    return %result : tensor<4x!earth.ci<40 * 0>>
  }
}
