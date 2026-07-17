# 单进程 CPU 三方测试计划

| 步骤 | 做什么 | 通过标准 | 状态 |
| --- | --- | --- | --- |
| 1 | 生成 MLP 的 MLIR 和 RuntimePlan | 只有 rank 0 Host，无通信，包含 1 次 Boot | 完成 |
| 2 | 用固定 seed 生成 784 个随机输入并跑 Python | 三条路径共用同一份输入和结果 | 完成 |
| 3 | 用 MockVecApi 跑 RuntimePlan | 521 条检查全通过，与 Python 最大误差 1.24e-13 | 完成 |
| 4 | 用 PoseidonCpuApi 加密执行并解密 | 与 Python、MockVecApi 最大误差 1.43e-6 | 完成 |
| 5 | 保存结果，提交并推送三个仓库 | 源码和 submodule 指针都已更新 | 完成 |

本测试只验证单进程 CPU。多 rank placement 和通信不在这次范围内。
