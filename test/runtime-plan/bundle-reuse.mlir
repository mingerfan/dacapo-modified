module {
  func.func @bundle_reuse()
      -> (tensor<600x!earth.pl<20 * 0>>, tensor<600x!earth.pl<20 * 0>>)
      attributes {init_level = 5 : i64} {
    %first = "earth.constant"() {
      value = dense<1.25> : tensor<600xf64>,
      rms_var = 1.25 : f64
    } : () -> tensor<600x!earth.pl<20 * 0>>
    %second = "earth.constant"() {
      value = dense<1.25> : tensor<600xf64>,
      rms_var = 1.25 : f64
    } : () -> tensor<600x!earth.pl<20 * 0>>
    return %first, %second : tensor<600x!earth.pl<20 * 0>>,
                             tensor<600x!earth.pl<20 * 0>>
  }
}
