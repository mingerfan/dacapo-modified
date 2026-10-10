module {
  func.func @memory_devices(%arg0: tensor<1x!ckks.poly<2 * 40 * 1>> {dist.rank = 0 : i64, dist.device = -1 : i64}) -> tensor<1x!ckks.poly<2 * 40 * 1>> attributes {dist.device_counts = array<i64: 1, 1>} {
    %x = "dist.transfer"(%arg0) <{transfer_id = 0 : i64, source_rank = 0 : i64, source_device = -1 : i64, destination_rank = 0 : i64, destination_device = 0 : i64, initialization = true}> {dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    %w = "ckks.encode"() <{payload = dense<1.0> : tensor<1xf64>}> {dist.logical_id = 1 : i64, dist.rank = 0 : i64, dist.device = -1 : i64} : () -> tensor<1x!ckks.poly<1 * 40 * 1>>
    %p = "dist.transfer"(%w) <{transfer_id = 1 : i64, source_rank = 0 : i64, source_device = -1 : i64, destination_rank = 0 : i64, destination_device = 0 : i64, initialization = true}> {dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<1 * 40 * 1>>) -> tensor<1x!ckks.poly<1 * 40 * 1>>
    %0 = "ckks.addcp"(%x, %p) {dist.logical_id = 2 : i64, dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 1>>, tensor<1x!ckks.poly<1 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    %1 = "ckks.rotatec"(%0) <{offset = array<i64: 1>}> {dist.logical_id = 3 : i64, dist.rank = 0 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    %copy = "dist.transfer"(%1) <{transfer_id = 2 : i64, source_rank = 0 : i64, source_device = 0 : i64, destination_rank = 1 : i64, destination_device = 0 : i64, initialization = false}> {dist.rank = 1 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    %2 = "ckks.negatec"(%copy) {dist.logical_id = 4 : i64, dist.rank = 1 : i64, dist.device = 0 : i64} : (tensor<1x!ckks.poly<2 * 40 * 1>>) -> tensor<1x!ckks.poly<2 * 40 * 1>>
    return %2 : tensor<1x!ckks.poly<2 * 40 * 1>>
  }
}
