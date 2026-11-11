# Scale / Boot 方案校验与比较

这里放算法方案的公共检查工具。人工或 Python 搜索生成的方案与编译器方案走同一个校验器。
`test/runtime-plan/` 继续负责原有导出、协议及 pass 测试；这里不搬动那些测试。

校验、编译比较和小图生成只使用 Python 标准库，不需要 GPU、FHE 库或 Python MLIR binding。
重建大图额外需要 NumPy，用于展开仓库原有的 ResNet 多项式系数树；使用已冻结的大图不需要 NumPy。

## 文件

- `verify.py`：Python API 与命令行入口，验证两份方案并输出 JSON 报告。
- `example.py`：用 Python 构造完整示例，展示如何提交参考方案。
- `check.py`：验证器自身的正向与故意出错的用例。
- `alignment.py`：比较修复前后的固定区域边界和 Boot，不将 Rescale 次数等同于 level 预算。
- `alignment_example.py`：三个对齐案例的明确人工 recipe、验收报告及可选绘图。
- `check-alignment.py`：编码取整、误差传播、边界预算与 Boot 绑定的回归检查。
- `alignment_planner.py`：只读链提取、调用 C++ 规划核心、在 RuntimePlan 副本上物化和验收；不是 MLIR pass。
- `check-alignment-planner.py`：C++ 搜索、跨算子回溯、独立枚举以及生成方案的回归检查。
- `check-compiler.py`：编译一个深链残差图，用独立向量答案检查三种算法的实际导出。
- `reference.py`：独立 Python 参考搜索器，以及按人工 recipe 生成完整方案的工具。
- `build-small.py`：重建默认小图、独立明文答案和明确指定 Boot 位置的参考方案。
- `build-large.py`：从仓库中的 ResNet20/Qwen 源码提取计算模式，生成大图与独立 Python 参考规划。
- `check-large.py`：大图固定参考、Boot 契约、BSGS 数值结果和旋转错误检查。
- `run-references.py`：校验固定参考方案，可选调用三种编译策略进行比较。
- `check-reference.py`：固定参考方案与人工 recipe 的回归测试。
- `draw_graphs.py`：完整计算图并排比较，生成离线 HTML、SVG 和 DOT。
- `check-graphs.py`：绘图完整性、节点对应、失败状态和数据摘要检查。

固定模式与人工基线放在本目录的 `cases/<suite>/<case-id>/`。
生成的 MLIR、报告和搜索中间结果放在构建目录，不提交到源码目录。
完整参考方案 `manual.json` 和逐步搜索记录也不入库。仓库只保留计算图、测试向量、
显式 recipe（使用参考搜索的边界套件保留搜索参数）及简短的 `reference.json`。
测试运行时重建方案，必须与已固定的 `plan_sha256` 逐字节一致，不能自动更新期望值。
`run-references.py` 将展开后的 `manual.json` 写入指定输出目录，供检查和绘图。
参考搜索器可以使用自己的内部数据结构，最后转换成这里接受的格式。

## 默认小图与参数

默认套件为 `cases/strict-small-q50/`，包含 8 个合成计算模式，原始计算数 12～50，
最长乘法路径深度 12～42。这些是手工设计的模式，不宣称已从 ResNet20/Qwen 自动提取。
每个 case 保存 `graph.json`、`case.json`、`recipe.json` 和简短的 `reference.json`。

使用现有 `native-q50` 的真实 50 个模数列表（每个约 30～32 bit），统一参数：

- 输入 level 30，scale 为 2^40。
- Boot 输入必须为 level 5、scale 2^40、2 个组件。
- Boot 输出固定为 level 30、scale 2^40、2 个组件。
- 普通计算最低 level 4；还需要 Boot 的路径必须保留到输入 level 5 的对齐预算。
- 密文和普通明文的 `log2(scale)` 最低 40，单位明文最低 30。
- 最终输出 level ≥ 4，`log2(scale)` 在 [40,70]。

Level 是物理模数链的零基索引，L 表示保留 L+1 个模数。每段从 L30 到 L5 有 25 层
模数消耗预算，包含 scale 对齐成本；不是固定可做 25 次乘法。Q50 表示 50 个模数，
不是每个模数为 50 bit。Degree 缩小到 16 以进行 8-slot 编译和明文重放，未进行加密执行。

| case | 原始计算数 | 乘法深度 | 参考 Boot | 参考 Rescale 总层数 |
|---|---:|---:|---:|---:|
| cp-chain-12 | 12 | 12 | 0 | 16 |
| cp-chain-30 | 30 | 30 | 1 | 42 |
| cp-chain-42 | 42 | 42 | 2 | 60 |
| cc-chain-30 | 30 | 30 | 1 | 42 |
| square-chain-18 | 18 | 18 | 1 | 28 |
| shared-prefix-fork | 40 | 27 | 1 | 54 |
| long-residual | 32 | 30 | 1 | 43 |
| independent-branches | 50 | 25 | 2 | 69 |

Rescale 总层数是整张参考图所有 Rescale 消耗的合计；有分支时不等于单一路径的消耗，
也不包含 ModSwitch。加上边界套件和大图套件，当前共有 32 份固定参考方案。

参考 Boot 位置由 `recipe.json` 明确指定，生成器不读取编译器输出、不自动补 Boot。
例如共享前缀图在 15 次乘法后刷新前缀值，两个分支共享它；独立分支图分别刷新两条链。
所有实际对齐、Rescale、ModSwitch、Relinearize 都物化为指令。非零 Boot 基线未证明全局最优。
每个 case 使用 3 组非均匀输入；平方链选取接近 1 的数，避免连续平方导致溢出或下溢。
期望输出只从原始算术图计算，与任何规划无关。

运行默认小图和三种编译策略，附带图形对比：

```bash
python3 test/scale-boot/run-references.py \
  --hecate-opt build/nix/bin/hecate-opt \
  --output-dir build/scale-boot-comparison --graphs
```

可用 `--cases cc-chain-30 shared-prefix-fork` 选择用例，或用 `--strategies greedy optimized`
选择策略。每次保存 MLIR、完整命令、编译器摘要、日志、自动方案、逐例比较及 `summary.json`。
报告同时包含乘法深度、参考方案统计和所有自动方案的校验结果。
编译失败、超时和非法方案分别记录，返回码 1；不能将它们称为物理不可行，因为存在合法参考。
合法但 Boot 更多只报告 `worse`，加 `--fail-on-worse` 才返回 2。
当前算法可能违反严格单位编码下限，或被剪枝拒绝；这些失败属于应报告的测试结果，不放宽参数。

只校验所有固定参考（不运行编译器）：

```bash
python3 test/scale-boot/check-reference.py
```

重建新小图及其参考方案，先输出到构建目录供审查：

```bash
python3 test/scale-boot/build-small.py --output-dir build/strict-small-candidates
```

生成器会覆写指定目标目录中的对应文件；不会自动替换源码中的固定基线。
修改图、参数或 recipe 后，应在构建目录重新生成并验证方案，再审查、更新 `reference.json`；
不能只修改摘要以绕过不一致检查。

## ResNet20 / Qwen 大图

`cases/model-large-q50/` 包含 5 个大图，默认不绘图。原始指令只统计算术/旋转，不含
参考规划或编译器插入的 Encode、Rescale、ModSwitch、Relinearize、Boot。

| case | 原始指令数 | 乘法深度 | 来源模式 |
|---|---:|---:|---|
| resnet20-residual-1block | 361 | 30 | ConvBN → SiLU → ConvBN → 残差 → SiLU |
| resnet20-residual-2block | 722 | 60 | 连续两个上述残差块 |
| qwen-rmsnorm-projections | 264 | 96 | 六段 RMSNorm＋BSGS 线性投影 |
| qwen-swiglu-residual-2block | 310 | 78 | 两段 post norm＋gate/up＋SiLU 门控＋down＋残差 |
| qwen-attention-cached-2tokens | 277 | 63 | pre norm＋Q/K/V＋Q/K norm＋RoPE＋注意力＋O＋残差 |

模式来源：

- ResNet 的 `python/poly/poly/models/ResNet.py:BasicBlock.forward`、
  `examples/benchmarks/ResNet.py` 和 `python/poly/poly/MPCB.py:MultParConvBN`。
  原样执行 `MPCB.GenPoly` 与 `Func.HE_SiLU` 的函数 AST，读取原有 `treeStr.txt`、
  `coeffStr.txt`，保留完整多项式树和系数。只删除源码局部未使用的算术临时值。
- Poseidon 的 `scripts/qwen3_dsl/qwen3/kernels/qwen_layers.py` 和 `qwen_model.py`。
  原样执行 `qwen_nonlinear.py` 中的 `inverse_sqrt`、`small_exp`、`reciprocal`、
  `normalize_square`、`normalize_square_silu`，通过符号算术记录成 DAG。
- 每个 `pattern.json` 保存源码路径、SHA-256、直接执行的函数行号、计算阶段及简化说明。
  卷积、线性投影和网络连接按源码模式缩小；不是从完整模型 MLIR 无损裁切的子图。

缩小为 8-slot、一个卷积密文包或一个注意力头，使用确定性的合成权重，保留真实的
旋转、向量掩码、公共常量加法、分叉共享和残差。注意力有一个当前 K/V 和一个已经过
归一化/RoPE 的缓存 K/V，二者都可见。线性投影使用 4 个 baby rotation、2 个 giant group。
Qwen 选定 3 次 Newton rsqrt、6 次 exp 多项式、1 次 normalize-square，倒数迭代 4/5 次；
这些是本测试明确指定的近似配置，不代表训练模型的校准结果或完整模型准确率。

**Boot 契约与编译器共用同一来源。** 生成器直接调用仓库
`integrations/dacapo/generate_resnet20_manual_profiles.py` 的 `generate()`，只把
`poly_degree` 改为 16 以便 CPU 重放。参考规划、校验器、编译器都使用输出的同一份
OperatorSpec（含摘要）：`L5/s40 → L30/s40`，普通计算最低 L4，初始 L30/s40。
单位明文编码最低 s30，普通明文/密文最低 s40。未沿用旧 Qwen 脚本中不限制输入 scale
的 legacy profile，也不把那个 profile 的参数混入本比较。Latency 表是占位值，不能用于预测运行时间。

大图参考来自独立 Python 局部搜索：每条指令尝试不刷新或刷新当前输入子集，优先减少
Boot 次数，再保留更多 level；保守预留 L≥8 供后续精确 Boot 对齐。所有决策冻结到
`recipe.json`，复现时不再搜索或自动补 Boot。它是合法回归基线，尤其在多项式分支中
可能明显次优，不作为最优次数 oracle。无效自动方案不会因此被评为更好。

```bash
# 重建候选大图（需要 NumPy；不会自动覆写源码基线）
python3 test/scale-boot/build-large.py --output-dir build/model-large-candidates

# 校验冻结方案，并运行当前三种编译算法；不传 --graphs
python3 test/scale-boot/run-references.py \
  --suite test/scale-boot/cases/model-large-q50 \
  --hecate-opt build/nix/bin/hecate-opt \
  --output-dir build/scale-boot-large --timeout 120

# 不需要 NumPy/GPU 的大图回归检查
python3 test/scale-boot/check-large.py
```

`summary.json` 给出指令数、乘法深度、合法性、Boot/Rescale/ModSwitch 统计和编译耗时。
自动方案的 `diagnostics` 记录未经完整校验认可的原始指令数、Boot 次数及输入/输出状态，
便于检查编译器实际生成了什么；`invalid` 的次数不参与 `boot_delta` 或优劣比较。
超时也保存已有编译日志。`--timeout` 为每次编译的秒数。

对独立 scale 对齐 pass 做大图开关回归，可分别启动以下两个任务并行运行；输出目录相互独立：

```bash
python3 test/scale-boot/run-large-alignment.py \
  --hecate-opt build/nix/bin/hecate-opt --family resnet20 \
  --output-dir build/large-alignment/resnet20 --timeout 120
python3 test/scale-boot/run-large-alignment.py \
  --hecate-opt build/nix/bin/hecate-opt --family qwen \
  --output-dir build/large-alignment/qwen --timeout 120
```

默认覆盖 threshold/regional × greedy/optimized × pass 关闭/开启。`--strategies` 也支持
depth-dp。每份实际导出同时检查原来的单位明文 s≥30 契约，以及前面几步使用的单位量化
契约（局部误差 ≤1e-6，输出绝对/相对容差沿用原 case）；冻结的 case 和人工方案不变。
报告的主要 `status` 使用单位量化契约，旧契约结果单独保存在 `legacy` 中。低于 s30 的
单位明文不等于一定不精确，但不能把量化检查通过写成旧的 scale 下限检查通过。
两种契约下的人工参考都必须有效，才能启动编译比较。

regional 默认可能保留高于案例输出上限 70 的 scale。这种情况仍记为失败；需要让编译器
明确满足输出约束时，可另开一轮，保持 case 不变：

```bash
python3 test/scale-boot/run-large-alignment.py \
  --hecate-opt build/nix/bin/hecate-opt --family qwen --policies regional \
  --regional-output-scale 40 --output-dir build/large-alignment/qwen-output40
```

每个 case 保存输入 MLIR、实际采用的两份校验配置、逐次命令、编译日志和 RuntimePlan；
`index.html` 与 `summary.json` 汇总成对结果。两侧都有导出时，额外绑定公共输入、最终输出
和按执行顺序对应的 Boot 两端，检查物理状态保持；这项检查不声称验证所有内部共享值的
语义对应。非法方案的 raw Boot 只用于诊断，不参与优劣比较。
报告还保留所有单位编码的违规列表，避免只看到验证器遇到的第一个错误。
任意变体超时、编译失败或量化校验失败，runner 返回 1 并继续保存其他变体的结果。

图格式保持原有三元组 `[name, op, inputs]`，旋转可额外提供第四项 `{"steps": -4}`；
常量可以是标量广播或完整 8-slot 数组，`add` 支持密文加明文。MLIR 浮点字面量显式含小数点，
包括 `eps=1e-6` 这种科学计数法。

## 边界测试与单独参考方案

`cases/threshold-q2/` 的 18 个旧案例保留为低预算边界回归，**不是默认算法评估套件**。
它使用合成约 30-bit 模数、初始 L6（耗尽输入为 L4）、最低 L4、Boot 输出 L6/s40，
未限制 Boot 输入 scale，单位明文下限为 0。它允许高 scale 直接 Boot，结果不能代表实际后端收益。
要显式运行它，传 `--suite test/scale-boot/cases/threshold-q2`。

`cases/native-q50/deep-residual/` 保留原来的 18 次平方加残差案例。它与默认套件共享严格参数，
但独立成套，可用 `--suite test/scale-boot/cases/native-q50` 运行。

参考搜索器仍支持单独调用：

```bash
python3 test/scale-boot/reference.py \
  --case-dir test/scale-boot/cases/native-q50/deep-residual \
  --operator-spec test/scale-boot/cases/native-q50/operator-spec.json \
  --recipe test/scale-boot/cases/native-q50/deep-residual/recipe.json \
  --output-dir build/reference-candidate
```

指定 recipe 时只物化人工决策，找不到合法方案就报错。省略 recipe 才进行有限 beam 搜索，
搜索失败不代表不可行。只有合法的零 Boot 方案可由次数非负直接证明最优。

## 计算图对比

本机安装 Graphviz 并在 PATH 中提供 `dot` 后，可以在比较命令后加 `--graphs`：

```bash
python3 test/scale-boot/run-references.py \
  --hecate-opt build/nix/bin/hecate-opt \
  --output-dir build/scale-boot-comparison --graphs
```

打开 `build/scale-boot-comparison/graphs/index.html` 浏览全部 case。
即使存在编译失败或校验未通过，也会先生成对应状态的页面再返回失败码。
绘图是可选功能；普通验证不依赖 Graphviz，也不自动安装任何软件。

已有本版本 runner 生成的结果无需重新编译，可单独绘图：

```bash
python3 test/scale-boot/draw_graphs.py \
  --suite test/scale-boot/cases/strict-small-q50 \
  --results-dir build/scale-boot-comparison \
  --output-dir build/scale-boot-graphs
```

结果记录原始图、公共约束、参考方案和自动方案的 SHA-256。绘图时重新检查摘要，避免
把旧统计配到已修改的图上；缺少摘要的旧结果需要重新运行比较。

每个 case 的页面默认并排显示参考方案与已运行的编译方案，可勾选 Original 查看完整原始图。
所有输入、Encode、计算、Boot、Rescale、ModSwitch、Relinearize 和输出节点都保留。
同一个操作数使用两次时合并为标有 `x2` 的边；它仍表达两次使用。
原始图中不影响输出的分支使用虚线和淡色，便于观察死代码删除。

- 红色 Boot，橙色 Rescale，紫色 ModSwitch/scale 对齐，蓝色原始计算。
- 页面和并排总图显示 Boot 输入/输出契约；页面额外显示初始状态、输出要求和单位明文下限。
- 每个结果标注 `s=log2(scale)`、模数链索引 `L`、`c=components`。
- 全 slot 编码 1 后的 MulCP 标为 scale alignment；实际 Encode/MulCP 节点都保留。
- `[v0]` 等标签通过原始表达式结构匹配，忽略保持明文语义的管理操作，并支持加法/乘法交换操作数。
  它不证明任意代数等价；无法确认时不添加标签。点击节点可跨方案高亮对应值。
- 点击可查看精确状态、原始指令和输入描述。图上的 scale 只保留四位小数，不能用显示舍入值做验证。
- 可缩放、适应列宽、同步纵向滚动和突出管理操作。同步按各图高度比例滚动，不表示同一行就是同一操作。
- 编译失败不显示旧方案；未通过校验的已生成方案可供诊断，明确标为失败，次数不参与合法性比较。

页面完全离线，无 CDN、网络请求或外部 JavaScript。可以单独分享一个 case 的 `index.html`；
该页内嵌完整 SVG，图形交互仍可使用。分享导航和下载链接时应复制整个输出目录。
每份方案还输出独立 SVG/DOT，每个 case 输出 `comparison.svg`/`comparison.dot` 并排总图。
`graph-metadata.json` 保存节点、边和操作数量，用于检查图是否遗漏操作。

图形回归测试：

```bash
python3 test/scale-boot/check-graphs.py
```

CMake 配置时找到 `dot` 才注册 `DacapoScaleBootGraphs`。

## 输入

需要四个文件：

1. `case.json`：公共输入状态、输出约束、编码 scale 下限和独立明文测试向量。
2. `operator-spec.json`：双方共用的实际模数与 Boot profile。
3. `manual.json`：参考方案，RuntimePlan V5 JSON。
4. `automatic.json`：待测方案，同样使用 RuntimePlan V5 JSON。

`manual` 只是角色名，不要求手写，也不代表已证明最优。可以放 Python 搜索结果或旧版本输出。
双方的 ValueId、操作数量和插入位置可以不同；输入、输出按列表位置对应公共用例。

第一版明确支持以下子集：

- `rescale_mode=native`，提供真实整数 `rns_moduli`；level 越大，剩余模数越多。
- Host rank 0 的单块展开方案，无设备分配、通信和内存调度。
- initialization/execution 的 inline Encode，以及 RuntimePlan 的 12 种 compute op。
- 实数 slot 向量；短 Encode payload 按运行时语义补零，不当作广播。
- 不支持 bundle、Release、Prepare、Transfer、Replicate、eager/lazy rescale；遇到这些输入明确报错。

自动方案可以从 `--dacapo` 直接导出，或对 `--dacapo-precompile` 的 CKKS checkpoint
运行 `--ckks-to-runtime-plan`。设置 `--runtime-plan-format=json`，提供一致的
OperatorSpec、Boot profile 和 target/context 元信息；不要启用 placement、memory 或
plaintext waterline。将 `--runtime-plan-inline-payload-max-bytes` 设得足以容纳测试常量。
本工具接收物理状态已经物化的 RuntimePlan，不直接解析 Earth/CKKS 文本。

OperatorSpec 的 `source_sha256` 是文件原始字节摘要，双方方案必须引用同一份文件。
更改空格或重新格式化也需要重新计算摘要。不要只修改引用而保留旧上下文生成的方案。

## 运行示例

在编译器根目录运行：

```bash
python3 test/scale-boot/example.py --output-dir build/scale-boot-example
python3 test/scale-boot/verify.py \
  --case build/scale-boot-example/case.json \
  --operator-spec build/scale-boot-example/operator-spec.json \
  --manual build/scale-boot-example/manual.json \
  --automatic build/scale-boot-example/automatic.json \
  --output build/scale-boot-example/report.json
```

示例计算 `rotate(x*x, 1)`。参考方案不需要 Boot；另一份合法方案故意多放一次 Boot，
所以结果是 `boot_delta=1`、`comparison="worse"`。
示例的两份方案都由 Python 构造，`automatic.json` 不是某次真实编译结果。
小模数只用于工具测试，不是可部署的加密参数。

退出码：0 表示两份方案都合法；1 表示输入或方案非法；加上 `--fail-on-worse` 后，
待测方案 Boot 更多时返回 2。比参考方案更好不会失败。

Python 搜索器也可以直接调用：

```python
report = compare(manual_plan, automatic_plan, spec, case, spec_sha256)
stats = verify(plan, spec, case, spec_sha256)  # 非法时抛出 InvalidPlan
```

`verify()` 的参数是 Python 字典。`spec_sha256` 由调用方从可信的原始 spec 文件计算；
CLI 自动计算它。`compare()` 记录每份方案的错误，不将非法方案参与次数比较。

## 公共用例格式

完整可运行格式由 `example.py` 生成，关键字段为：

```json
{
  "format_version": 1,
  "case_id": "square-rotate-example",
  "inputs": [
    {"kind": "ciphertext", "level": 2, "scale_log2": 10, "components": 2, "ntt": true}
  ],
  "outputs": [
    {"kind": "ciphertext", "min_level": 1, "min_scale_log2": 9,
     "max_scale_log2": 10, "components": 2, "ntt": true}
  ],
  "min_ciphertext_scale_log2": 9,
  "min_plaintext_scale_log2": 9,
  "min_unit_plaintext_scale_log2": 3,
  "numeric_tolerance": {"abs": 1e-12, "rel": 1e-12},
  "test_vectors": [
    {"inputs": [[0.25, -0.5, 0.75, 1]], "outputs": [[0.25, 0.5625, 1, 0.0625]]}
  ]
}
```

每个测试向量必须包含全部 `poly_degree / 2` 个 slot。建议至少提供两组非均匀向量，
覆盖符号变化、分支差异和旋转。期望结果应从原始计算独立生成，不从待测规划的输出反推。
深平方链需要选合适数值，避免所有路径都下溢到零而掩盖错误。

输入状态必须精确匹配；输出通过范围约束比较，若要求固定 scale，将上下界设为相同值。
状态的 scale 比较绝对容差固定为 `1e-8`；明文重放容差由用例指定。

`min_unit_plaintext_scale_log2` 仅适用于全部 slot 都为 1 或全部为 -1 的载荷。
掩码和其他系数使用普通明文下限。这些下限是用例约束，不是工具替用户证明的精度保证。
现有 threshold 策略会生成低 scale 的单位常量；设置更高下限可能正确地拒绝它。
不要为了让算法通过而自动放宽公共约束。

用例可以显式启用基于误差的单位明文检查：

```json
"unit_encoding": {
  "max_abs_error": 1e-6,
  "output_tolerance": {"abs": 1e-9, "rel": 1e-9}
}
```

该策略对全槽 ±1 **替代** `min_unit_plaintext_scale_log2`，普通明文仍使用原下限。
编码 scale 为 `S=2^s` 时，正单位的标量系数取整模型为 `round(S)/S`，负单位取其相反数。
取整采用 C++ `std::round` 的半整数远离零规则；不能用 Python 的偶数舍入代替。
因此整数 s 下可精确表示的 1 不会仅因低于 30 bit 被拒绝。

校验器同时运行两条重放：原始 payload 的理想语义重放，以及包含单位编码取整的重放。
后者检查每个单位的局部误差，并将误差传播到输出，与独立预期向量比较。
普通向量的 CKKS 编码不是逐槽取整，当前没有模拟其量化；Rescale 舍入、Boot 误差和
加密噪声也没有模拟。该结果不是完整加密精度保证。
没有此字段的旧套件保持原有下限及理想重放，并报告 `unit_quantization_checked=false`。

## 固定预算的对齐案例

`cases/alignment-q50/` 单独保存三份人工改写，不是编译器 pass 或自动搜索的输出。
模数和 Boot profile 与 `model-large-q50` 一致，degree=16、8 slots。
每例有三个独立输入向量，包含负数和残差抵消；期望值直接计算 `0.75*x+y` 或 `x+y`。

| case | 人工改写 | 固定出口 | 预期 |
|---|---|---|---|
| absorb-mulcp | 将对齐吸收到已有 0.75 的编码 scale；删除低精度乘 1 | L6/s40.2 | 接受；MulCP 减 1，Rescale 不变 |
| reuse-modswitch | 利用原 L6→L5 的丢弃预算，使用高精度乘 1＋Rescale | L5/s40.2 | 接受；Rescale 增 1，出口 level 不变 |
| insufficient-budget | 输入已是 L5，显式修复将输出降到 L4 | 必须为 L5/s40.2 | 数值校验通过，但预算校验拒绝 |

每例保存 `case.json`、`before.json`、`candidate.json`、`contract.json` 和 `expected.json`。
`before` 是含低精度乘 1 的错误方案，不能作为合法精度参考；它只提供待保持的边界预算。
第三例是一个明确的违规候选，用来测试拒绝机制，不表示已经实现了能证明无解的搜索器。
`case.json` 的通用输出范围故意允许 L4；严格 L5 要求由独立预算检查执行，避免只测试到
原有输出范围校验而没有覆盖新的边界检查。

`contract.json` 用可信的语义边界对应关系连接两份方案：

```json
{
  "format_version": 1,
  "boundaries": [
    {"name": "input-x", "before": "0", "after": "0"},
    {"name": "input-y", "before": "1", "after": "1"},
    {"name": "region-exit", "before": "5", "after": "5"}
  ]
}
```

所有外部输入、最终输出和 Boot 输入输出必须列出；区域内还有共享出口时，调用方必须
把它们一并列出。工具不能从不同的 SSA 编号自动证明语义对应，也不能发现未申报的内部出口。
边界的 level、scale、组件、NTT、context、place 和种类保持一致（scale 容差 `1e-8`）；
Boot 数量、绑定的输入输出、顺序和 profile/attrs 保持一致。
这是一项独立的状态契约检查，必须与 `verify()` 的计算及精度检查同时通过。

```python
from alignment import verify_alignment_budget
from verify import verify

# before/candidate/contract/spec/case 为已加载的 JSON，digest 是 spec 文件摘要。
stats = verify(candidate, spec, case, digest)
budget = verify_alignment_budget(before, candidate, contract)
# 任何一项失败都会抛出 InvalidPlan；不能仅凭 stats 中 Boot 为 0 接受方案。
```

在编译器根目录运行：

```bash
python3 test/scale-boot/check-alignment.py
python3 test/scale-boot/alignment_example.py --output-dir build/alignment-step1 --graphs
```

第二条读取固定案例，检查接受/拒绝状态符合 `expected.json`，输出 `summary.json`；
`--graphs` 额外生成离线 `index.html` 和每例实际完整 DAG 的 SVG、PNG、DOT，需要 Graphviz。
没有 `--graphs` 时只依赖 Python 标准库。按预期拒绝第三例属于测试通过，CLI 返回 0；
若实际结果与预期不符则报错。JSON 中 `accepted` 与 `expectation_met` 分别表示这两个概念。

重建候选案例可运行 `alignment_example.py --write-fixtures --output-dir build/alignment-candidates`；
它不自动替换固定案例。生成器在 `make_plan()` 中明确写出两种人工 recipe，适合作为后续
提交人工或自动规划的代码样例。

## 单链反向对齐规划核心

生产 C++ 接口在 `include/hecate/Support/ScaleAlignmentPlanning.h`，实现位于
`lib/Dialect/Earth/Transforms/ScaleAlignmentPlanning.cpp`，已加入 `EarthTransforms`。
核心只依赖 C++ 标准库和已有 `ModulusChain`；不访问 MLIR，不修改调用方输入，不插入 Boot。
`test/Support/ScaleAlignmentDriver.cpp` 是 JSON 测试入口，同一个 C++ 实现也由
`hoist-scale-alignment` pass 调用。

请求包含：固定入口状态、单一使用链的只读快照、固定出口状态、真实模数链，以及编码精度、
容量余量和搜索上限。支持 MulCP、AddCP、Rotate、Negate、Rescale、ModSwitch。
调用方必须在共享/固定密文边界处截断链。单链接口不直接表示 MulCC、Square 和 AddCC；
独立 pass 的图级规划器负责这些多输入算子的需求拆分，再调用本接口。
跨 Boot 回溯仍被禁止，测试适配器不会把共享输入当成可改写的私有节点。

算法先从出口反推每条边需要的 scale：穿过 Rescale 补回实际模数的对数，穿过 MulCP
减去当前明文编码 scale；其他支持的运算保持需求。然后枚举两类候选：

1. 在某一个已有 MulCP 吸收全部调整，明文新 scale 等于所需输出 scale 减实际输入 scale。
2. 插入一个精度合格的单位乘法；如果需要新增 Rescale，则只使用某个现有 ModSwitch
   丢弃区间里的连续模数窗口。枚举窗口的起点和长度，必要时在其前后保留 ModSwitch。
   可精确编码的小整数 scale 不强制消耗 Rescale。

第二类也包括直接在**已有 Rescale 之前**插入单位乘法，保留这次 Rescale 的原位置和
模数，因此新增 Rescale 层数为 0。输入规划骨架中的 Rescale 结果可以暂时低于 scale 下限；
其模数容量、非负实际 scale 和算术关系仍须正确，完整候选的每个状态都必须满足下限。
例如 L6/s40 经过 q6 后原本只有 s≈8.00994，要求出口 L5/s40.2 时，可以先以
s≈32.19006 编码乘 1，再执行已有 Rescale。相反，若原 Rescale 已将 s≈71.99006 降到
s40，而目标只是 s40.2，则在前后插入都只需要 s0.2 的单位，仍然必须因精度不足被拒绝。

每个候选都完整向前重放。AddCP 的明文 scale 跟随新密文 scale，payload 保持不变；
检查每一步容量、密文下限、普通明文下限、单位编码误差、物理 level 和最后的固定出口。
保留真实算出的浮点 scale，不直接把状态覆盖成请求值。
普通权重精度目前仍按配置的 scale 下限判断，不声称模拟了任意向量编码误差。

候选按新增 Rescale 层数、新增 MulCP 数、重新编码的原明文数排序；相同成本选更靠后的
位置。这里的统计相对于移除错误末端单位乘法后的原链；最终验收报告另给完整方案的操作数差值。
不使用占位 latency 表预测时间。这个版本一次只改变一个乘法系数，或插入一组显式转换；
不是任意多位置联合优化，也不移动原有 Rescale。

状态区分 `success`、`invalid-input`、`no-candidate`、`search-limit`。
默认上限为 128 个链节点、8192 个候选；达到搜索上限会明确拒绝，不冒充无解，也不返回
一个未经完整搜索比较的计划。`no-candidate` 仅指当前固定预算及搜索空间内没有方案，不是全图
不可行的证明。输出保存候选数、淘汰原因、选择位置、全部步骤及其原操作索引，便于审查。

编译器根目录下可执行：

```bash
cmake --build build/nix --target DacapoScaleAlignmentDriver -j2
ctest --test-dir build/nix -R '^DacapoScaleAlignment' --output-on-failure
python3 test/scale-boot/alignment_planner.py \
  --driver build/nix/bin/DacapoScaleAlignmentDriver \
  --output-dir build/alignment-step2/results --graphs
```

不用 MLIR 构建环境也能直接编译同一核心及测试适配器（只使用仓库已有的 JSON 头文件）：

```bash
mkdir -p build/alignment-step2
c++ -std=c++17 -O2 -Wall -Wextra -Werror -Iinclude \
  lib/Dialect/Earth/Transforms/ScaleAlignmentPlanning.cpp \
  test/Support/ScaleAlignmentDriver.cpp -o build/alignment-step2/scale-alignment-driver
python3 test/scale-boot/check-alignment-planner.py --driver build/alignment-step2/scale-alignment-driver
```

验收 runner 不读取人工 `candidate.json` 作为规划输入。它从第一步的 `before.json` 提取问题链，
交给 C++ 搜索，并依据返回的计划在副本中重新生成操作和 Encode；原始明文 payload 和旋转偏移
由原操作绑定保留。自动方案必须同时通过 `verify()` 与固定边界预算检查。
前两例输出 `automatic.json`、`contract.json` 和 `request.json`；预算不足的第三例只有失败诊断，
不输出降级方案。报告含 spec、输入和 driver 摘要；可选绘图直接读取实际自动方案。
独立 MLIR pass 和生产流水线的显式开关见下节。

测试覆盖三个核心案例、跨 Rescale/Rotate/AddCP/Negate、源输入不变、共享边界截断、精确小单位、
普通明文精度、真实模数、不同模数窗口、多次 Rescale、容量和预算拒绝、搜索上限、确定性，
以及 36 组小状态的独立显式转换枚举。合成的 2 的幂模数只用于隔离搜索边界，不是可部署参数。

## 独立 scale 对齐 pass

`lib/Dialect/Earth/Transforms/HoistScaleAlignment.cpp` 实现函数级
`hoist-scale-alignment`。它读取已规划的 Earth IR，提取私有子图，将多输入算子的 scale
需求传到各输入，再沿其中的单链调用上述 C++ 核心，
先在函数副本上完成改写、状态重放和 MLIR verify，全部成功才替换原函数体。
无候选、超过搜索上限或最终校验失败都会报错；不会加 Boot 或降低固定出口 level。

处理顺序是：修复精度不足的全槽位乘 1；修复乘 1 之前低于密文 scale 下限的私有链
（包括将已有乘 1 移到已有 Rescale 前）；再修复 AddCC 两侧的实际 scale 差异。
原乘 1 的替换出口和修复后的 AddCC 结果都必须保留原有物理 level、实际 scale 和 Earth
名义类型。生成 MulCP 时，末端明文的名义 scale 由出口类型扣除输入名义 scale 得到，
实际编码仍使用规划的浮点 scale。类型重建后再次检查出口，避免独立取整改变下游 AddCP 的类型。
被不同算子使用的共享密文、函数输入输出、Boot 两端、`sm_fixed`、`earth.saved_output` 和
`earth.output_scale_source` 是不可穿越的边界。共享明文可以为改写的使用处另建 Encode，
原明文和 payload 保留。不同 level 的 Add 输入应由前面的调度阶段处理。

函数带 `earth.scale_alignment_policy` 标记，`earth.scale_alignment_trace` 记录每次选择的
策略、候选数、图查询数、目标 level/scale、新增 Rescale 层数、新增乘法数和重新编码数。
后续 EarlyModswitch 和系数 scale
吸收跳过此函数，canonicalize 保留带精确编码 scale 的单位明文；CKKS 阶段根据实际 scale
验收 Add，发现残留错配会报错，不再临时插入未经预算检查的单位乘法。

接入位置：

```text
Boot/scale 规划（greedy、depth-dp 或 optimized）
    → hoist-scale-alignment（显式开启）
    → EarlyModswitch / AbsorbCoefficientScale / canonicalize / CSE
    → EarthToCKKS → UpscaleToMulCP → 物理 level/scale 展开 → RuntimePlan
```

在现有 native-Q 编译命令的 `--dacapo` 或 `--dacapo-precompile` 上增加
`--enable-scale-alignment-hoisting` 即可，默认关闭。它要求有效的 native OperatorSpec、Boot
profile 和单基本块函数；当前没有接入 `--pars` 等旧流水线。
默认普通明文/密文下限取流水线的 `--waterline`，单位局部量化误差上限是 `1e-6`。

已有 Earth 输入也可单独运行，pass 的参数用空格分隔：

```bash
build/nix/bin/hecate-opt scheduled.mlir \
  --ckks-config=test/scale-boot/cases/model-large-q50/compiler.json \
  --hoist-scale-alignment="operator-spec-path=test/scale-boot/cases/alignment-q50/operator-spec.json boot-profile=poseidon-resnet20-manual-q50-scale40 waterline=40 unit-error=1e-6" \
  -o aligned.mlir
```

其中 `boot-profile` 必须填写 OperatorSpec 中的实际 `profile_id`；完整可复现命令由下方
runner 保存到每个案例的 `*.command.json`。独立运行时原输入需要先通过 MLIR 解析和类型
校验，`init_level` 必须等于 native level anchor，实际输入 scale 用 `earth.arg_scales` 表达。

验收命令（编译器根目录）：

```bash
cmake --build build/nix --target hecate-opt DacapoScaleAlignmentDriver -j2
ctest --test-dir build/nix \
  -R 'Dacapo(ScaleAlignment|HoistScaleAlignment|RegionalHoistScaleAlignment|OptimizedHoistScaleAlignment)' \
  --output-on-failure
python3 test/runtime-plan/check-hoist-scale-alignment.py \
  --hecate-opt build/nix/bin/hecate-opt \
  --output-dir build/alignment-step3/results --graphs
```

`test/runtime-plan/check-hoist-scale-alignment.py` 的 15 个案例覆盖：三个冻结案例、已有
Rescale、尚未显式乘 1 的 AddCC、共享/固定密文拒绝、共享明文克隆、跨 Rotate/AddCP/Negate、
Boot 边界、先成功修复再失败的事务回滚，以及 threshold/regional × greedy/optimized 的完整流水线。
成功的独立 pass 案例连续运行两次必须产生完全相同的 IR；失败案例逐字比较 pass 前后打印的函数 IR，检查没有留下
部分改写。前两个冻结案例的实际导出操作数、Encode 数须与人工方案一致。
另有两个 CTest 分别在启用开关后运行现有 regional greedy/optimized 图集，覆盖更长的图和 Boot。

输出包括输入和改写后的 MLIR、真实导出的 `compiled.graph.runtime-plan.json`、执行命令、日志、
`summary.json` 和可选对比图 `index.html`。验收对 RuntimePlan 执行独立的物理状态重放、
单位量化及明文向量计算，检查固定边界和 Boot 契约。复用 ModSwitch 可以使 Rescale 数增加，
但不会使固定边界的 level 更低；复用已有 Rescale 的案例连 Rescale 数也保持不变。

独立 pass 不改变上游区域规划的搜索规则。如果原规划器先行拒绝，独立 pass 不会运行；
也不会修改共享密文的原状态或跨 Boot 回溯。`no-candidate` 不等于全图无解。
这些测试尚不包含 GPU/加密执行、普通明文编码噪声、Rescale 舍入噪声或 Boot 近似误差。

## AddCC、MulCC、Square 的图级回溯

图级搜索在同一 `HoistScaleAlignment.cpp` 的 `GraphPlanner` 中实现，不修改纯 C++ 单链
核心的语义。每个修复请求先生成只读的方案树，完整选择后才生成 IR；分支之一失败不会先
改写另一分支。候选按新增 Rescale 层数、新增 MulCP 数、重新编码数依次比较。

记 `s` 为实际 `log2(scale)`，某多输入节点要求输出 scale 为 `T`：

| 节点 | 向输入传递的需求 |
|---|---|
| AddCC(a,b) | 两侧都需要 `T`，两个请求都成功才接受 |
| MulCC(a,b) | 令 `d=T-s(a)-s(b)`；尝试全部调整左侧、全部调整右侧、两侧各分配 `d/2` |
| Square(a) / MulCC(a,a) | 输入需要 `T/2`，只规划、生成并计费一次，然后把同一 SSA 值用两次 |
| AddCC(a,a) | 输入需要 `T`，不使用平方的减半规则 |

经过这些节点前后的 Rescale/MulCP/Rotate/AddCP/Negate/ModSwitch，仍按真实模数反推、
再向前重放每个状态。既可以由已有权重吸收，也可以在某一输入侧使用高精度乘 1 与
已有 Rescale，或复用 ModSwitch 的 level 差额。精确单位乘法若位于一个无效的私有图之后，
也会触发修复：例如 Rescale 先把一侧降到下限以下，后面的 MulCC 和乘 1 又把 scale 提高，
不能只因为最后状态合法就接受。

`a*a` 的两个 operand use 属于同一个 consumer，因此不再被误判为共享边界。
如果 `a` 还被其他算子、返回值或 Boot 使用，则仍保持它原有的状态；允许在它的某个消费边
之后，利用该边已有的 ModSwitch/Rescale 预算转换。不会复制或改写它之前的共享计算。
`(SSA value, 目标实际 scale)` 的请求结果会缓存；生成阶段也缓存同一方案的 SSA 结果，
平方不会复制输入链。后续 CSE 可以进一步共享完全相同的边转换。

这是有界搜索，不是连续 scale 空间的全局优化：不同输入的 MulCC 当前只枚举上述三种
分配方式，不任意移动 Rescale，也不同时重选所有区域出口。默认每次修复最多 1024 个
不同请求、128 层图回溯、8192 个单链候选；超过限制明确报错并回滚，不能把搜索截断
报告成无解或接受一个尚未完成比较的方案。前两个限制可用 pass 参数
`max-graph-queries`、`max-graph-depth` 设置。已经合法的长图检查使用迭代遍历，不消耗回溯深度。
pass 与物理展开对已规划 scale 统一使用 `1e-8` 的比较容差，避免实际模数重放在
`40-约7e-15` 处被误判为低于 40。

图级验收运行：

```bash
cmake --build build/nix --target hecate-opt -j2
ctest --test-dir build/nix -R 'Dacapo(GraphScaleAlignment|HoistScaleAlignment)' --output-on-failure
python3 test/runtime-plan/check-graph-scale-alignment.py \
  --hecate-opt build/nix/bin/hecate-opt \
  --output-dir build/alignment-step4/results --graphs
```

新增 19 个案例包含两侧 AddCC、嵌套 AddCC、MulCC 两侧选择、Square、重复 Add、
从 AddCC 直接传入 Square 的需求、共享前缀上的边转换、MulCC/Square 输入侧显式乘 1、
跨 MulCC 复用已有 Rescale，以及共享冲突、单侧失败、Boot 边界和搜索上限拒绝。
其中 2 个案例走完整 `--dacapo` 的 greedy/optimized 流水线，计算 `(.75*x)^2+y`：
基线会导出一个局部单位量化误差约 0.00539 的明文，开启 pass 后由平方之前的权重吸收
调整，并保持基线的固定出口状态和 Boot 数量。

每个成功的独立案例有显式人工配方 `manual.json` 和实际编译结果
`compiled.graph.runtime-plan.json`；双方分别通过独立验证器和固定边界预算检查。
`contract.json`、`manual-contract.json` 保存绑定，`repaired.mlir` 展示 pass 的直接结果，
`*.command.json` 保存完整执行命令，`index.html` 对比原方案、人工方案和实际导出。
测试覆盖重复运行不变、拒绝时原函数 IR 不变，以及平方输入仍是同一个 SSA 值。
运行后续 canonicalize/CSE 可能共享加法两侧的 Rescale 或相同转换，因此实际指令数允许
少于人工配方，但不允许超过配方中的各类操作数和 Encode 数。固定出口 level/scale、
共享值和 Boot 的检查不因此放宽。

## 检查与统计

检查 SSA 定义/使用、公共输入、算子支持、模数容量、scale/level 转移、组件数、
NTT 状态、明文编码下限、Boot 输入输出契约和最终输出要求。
原生 Rescale 使用实际 `log2(q)` 重放，支持一次操作移除多个模数。
最后对双方分别进行明文向量重放，与公共预期结果比较。

报告包含：

- 每份方案是否合法；非法时给出第一个错误，不自动补指令。
- Boot 数、各算子数、Encode 数、Rescale/ModSwitch 消耗的总层数、value 数。
- 最终输出状态、重放向量数量及最大绝对误差。
- `unit_quantization_checked`、单位编码明细和 `unit_quantized_max_abs_error`；旧模式后者为 null。
- `boot_delta = automatic - manual`，以及 better/equal/worse/invalid。
- CLI 输入文件路径与 SHA-256，便于复现。

一个 Boot compute 记录计一次，共享刷新值不会因多个消费者重复计数；真正重复的 Boot
记录会分别计数。这要求提交的是已展开的 RuntimePlan，而不是含隐式循环的高层算子。
编译耗时应由外部 runner 测量；工具不能从已经生成的方案推断编译耗时。

明文重放将 Boot、Rescale、ModSwitch、Relinearize 视为保持明文值的操作。
这只能检验给定向量上的计算关系，既不是所有输入的等价证明，也不是 CKKS 加密误差测试。
工具不证明参考方案全局最优，不检查密钥/GPU 支持或全部 RuntimePlan 协议字段，
不能替代运行时 reader 与后端 preflight。

## 验证工具本身

```bash
python3 test/scale-boot/check.py
python3 test/scale-boot/check-alignment.py
python3 test/scale-boot/check-reference.py
python3 test/scale-boot/check-large.py
python3 test/scale-boot/check-compiler.py --hecate-opt build/nix/bin/hecate-opt
```

这些测试注册为 CTest 的 `DacapoScaleBootVerifier`、`DacapoScaleBootReferences`、`DacapoScaleBootLarge` 和 `DacapoScaleBootCompiler`。
对齐测试注册为 `DacapoScaleAlignmentVerifier`。
已有构建目录重新配置后可运行：

```bash
ctest --test-dir build/nix -R '^DacapoScale(Boot|Alignment)' --output-on-failure
```
