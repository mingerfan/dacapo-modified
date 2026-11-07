# Scale 对齐回归检查

本目录验证固定物理 level、Boot 和区域出口边界内的 scale 对齐。
`test/runtime-plan/check-hoist-scale-alignment.py` 检查独立 pass，
`check-graph-scale-alignment.py` 检查跨加法、乘法和平方的回溯，
`check-alignment-planner.py` 通过 JSON adapter 检查 C++ 规划核心。

测试保留少量明确的人工方案，独立检查真实模数链的物理状态、固定边界和全槽单位明文量化误差。
共享值、固定值、预算不足和失败后的 IR 保持均有拒绝案例。
这些检查不模拟完整加密噪声，也不代表模型加密准确率。

构建后可通过 CTest 运行。运行依赖 Python 标准库，不需要 GPU 或 Python MLIR binding。
生成的 MLIR、RuntimePlan、报告和绘图放在构建或临时目录。
