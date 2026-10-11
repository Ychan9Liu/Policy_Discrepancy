# DL 合成探针检测能力规范

版本 **`DL-probe-assay-r1`**，工程 **`DL-probe-assay-engineering-r1`**，2026-10-11。**科学方案经内部交叉审查冻结，源码已实现；以下为冻结时记录，当时拟合未运行、须先确定提交并完成 clean Linux 工程验收；后续实际结果与独立审计见 [DL_PROBE_ASSAY_RESULTS.md](DL_PROBE_ASSAY_RESULTS.md)。** 这是用户总体前验证授权内的 Agent 决定，不是用户逐项认可的方法。仅允许一次固定四 case 的纯 CPU assay。本规范不授权真实数据重拟合、blind、Dreamer、physics、render、GPU、gate 或训练，不重判已停止的 `DL-action-information-r1`。

事实与动机见 [有限动作结果](DL_ACTION_INFORMATION_RESULTS.md)、[原统计协议](DL_ACTION_INFORMATION.md)、[下一对象建议 r0](DL_CUE_IDENTIFICATION.md)。本稿读取的仓库基准为 `53fc13e01d5554292736a1b9e67851530564d976`。

## 1. 唯一问题与结论范围

本对象只问：**固定 λ1、同容量、group RMS 的原 readout，能否从已知线性可读且有明确动作 margin 的合成接口中学到动作选择；若不能，其阴性检测范围有多窄？** 它检查测量工具，不检查真实 Dreamer 的信息、当前 RGB、物理关键性或 KL 学习机制。

真实有限对象已有外部动作选择依据，但 q/p/constant 全选同一控制、接口增量为零。其收缩几何不是正则为主因的证据。本 assay 不能把真实结果翻案，也不能用合成阳性推断真实 latent 有信息。四个对象必须一次固定、全部报告；不按结果改变 λ、seed、维数、成本、样本、对象或预算。

## 2. 合成数组与已知 oracle

固定八个人工 episode `e=0..7`，每个四 state `s=0..3`、每 state 四 sample `r=0..3`。这些索引只用于生成、聚类和留出，**不作为 readout 特征**。定义 `C[e,s]=1` 当且仅当 `s=e mod 4`，否则为 0；每 episode 一正三负，共八个 class1 与二十四个 class0。

输入契约：

| 数组 | dtype／shape | 含义 |
| --- | --- | --- |
| episodes | int32 `[8]` | 人工索引 0..7 |
| h | float64 `[8,4,4096]` | q/p 共享、四 sample 重复同 state 的 h |
| latent | float64 `[8,4,4,2,32,32]` | branch 顺序 q/p；32 factor，各 32 class 的实际 onehot |
| rewards | float64 `[8,4,7]` | 抽象七控制的确定合成 reward |

七个抽象控制只称 `a0..a6`，不具有 quadruped 动作语义。`R(a2)=1−C`、`R(a3)=C`，其余五项为 0；reward 越大越好。不可把旧 stats 的物理 control 名称或 H20 字段解读为本对象发生了物理动作。

每 state 最优值为 1、span 为 1；closed `Aδ={a:max R−R(a)<=.2}` 为 singleton，class0 为 `{a2}`、class1 为 `{a3}`，存在跨 episode 不相交集合。每个训练 fold 的 class 比例仍为 1/4，train-only 最好 constant 唯一为 a2，全部留出 episode 的 constant reward 均为 `.75`。可读取 C 的 oracle reward 为 1，相对无 cue constant 的差为 `.25`。这些是构造的逻辑事实，不是待运行的统计结果，也不来自真实 cal32 标签。

## 3. 冻结随机流与 dtype

生成 seed 固定为 `20261022`。每个工作进程只执行一次 `children = numpy.random.SeedSequence(20261022).spawn(2)`，按下表固定绑定 `Generator(PCG64(children[i]))`。不增 spawn、不使用其他默认 RNG、不按 case 或 q/p 分支再生成子流：

| child／spawn_key | 一次生成的数组／顺序 |
| --- | --- |
| 0／`(0,)` | `standard_normal((8,4,4096), dtype=float64)`，C order 的 h Gaussian |
| 1／`(1,)` | `integers(0,32,size=(8,4,4,32),dtype=int32)`，共同 base nuisance class index |

两个流按此生成，h 与 base nuisance 相互独立且生成不依赖 C；base 各 state/sample/factor 的 class 为 uniform 0..31。基础 h/base nuisance 数组在四 case 间复制，q/p 先共享同一 base，只作指定 cue 覆盖。这样把分支差异限定为 cue 植入，避免额外随机噪声混入 q−p。共享基础数组不把四 case 变成四次独立实验；这一次有限 nuisance 与 C 可能偶然相关，必须保存完整数组，不人为正交化或按相关性换 seed。

**已采用 h 直接 F64 Gaussian，不作 BF16 或 F32 转换。** history case 的 ±1 也直接 F64。latent 由 class index 构造 F64 0/1，不把类别概率均值当 sample。BF16 非理论识别所必需，去掉它避免增加 dtype 依赖和生产来源的错误暗示；本 assay 不模拟生产 BF16 网络运算图。

实施时记录 NumPy／PCG64／Python 与平台版本、SeedSequence entropy／spawn_key、初始及生成后 bit-generator state。生成／求解非有限或依赖不可用均停止，不替代 RNG、插入 dtype 转换或变换维数。构造采用缓存的只读 canonical 数组，validation 仅读取和精确比较其副本，不再次生成／推进随机流。**本节 child 绑定、跨 case 流复用与直接 F64 均已冻结；尚无拟合运行证据。**

## 4. 四个一次固定的 case

| 固定顺序／名称 | h | q latent | p latent | 已知线性可读性与预期 |
| --- | --- | --- | --- | --- |
| 0 `sparse` | 基础 F64 Gaussian | 仅 factor0 的 class index 覆盖为 C；其余31 factor为 base nuisance | 全部 base nuisance | q 的 factor0/class1 即 C，可线性精确预测 a3；这是稀疏 cue 检测对象，可能被 group RMS／λ1 漏检 |
| 1 `replicated` | 同一基础 F64 Gaussian | 全32 factor 的 class index 均覆盖为 C | 同一 base nuisance | 任一 factor/class1 或其平均即 C；预期比 sparse 更易被固定 probe 利用，不保证拟合一定阳性 |
| 2 `history` | 全4096维复制 `2C−1` | 全部 base nuisance | **逐字节相同 base nuisance** | 两接口同样含 C，`C=(h0+1)/2`；两边可读，要求 q−p 精确零；实际拟合是否利用 h 仍为对照预期 |
| 3 `identity` | 同一基础 F64 Gaussian | 全部 base nuisance | **逐字节相同 base nuisance** | 两接口完全同输入，应得到相同预测、选择、reward 与精确零 q−p；对 constant 的偶然效应不解释为 cue |

**feature-only decoder 必须实际核验**：sparse／replicated 仅读取 q 的 factor0/class1；该项为0选 a2、为1选 a3。history 的 q/p 均仅读取 h0 符号，负选 a2、正选 a3。这些 decoder 不读 C、episode/state 索引或 reward 表来决定动作，且每 sample 都应得到 reward1。identity 没有指定可解码的 cue，不允许借 privileged C oracle 把它判为漏检。

线性可读性证明使用接口的实际单次 onehot／h 坐标，无需多 sample 投票或 nonlinear readout。已知 oracle 和 λ1 拟合算法是两个对象：假设类含有完美线性解，不保证带正则、有限样本与噪声的训练程序恢复它。replicated 提高 cue 在组能量中的占比，改变的是信号结构；不能据其成功推断真实 cue 也被复制。C 的 e mod4 布局是人工平衡，**不构成不可预知物理 cue 或完整历史增量证据**。

## 5. 原 stats 原样调用与验收边界

只复用已冻结 `scripts/dl_action_information_stats.py` 的 `_external()`、`_readouts()`、`_interval()` 数值例程，不调用含 actor 的 `analyze()`，不改原文件、常数或 readout 算法：λ1、mean-row 多输出目标／`nλ`、八个整 episode LOEO、train-only h/z group RMS／centering、零 RMS 规则、sample argmax 后四均、constant train-only、`1e−10` lowest-index tie、closed δ全保留。bootstrap 按原例程的调用方式，固定 `default_rng(20261021).integers(0,8,(2000,8))`；q−p 与 q−constant 共用同 draws、linear 分位数。

wrapper 须将原例程的模板 metadata 重标为 synthetic h/onehot、人工 state index0..3、抽象控制 a0..a6；跨 episode pair 的旧 position 数值只映射回对应 state index，不改变数值裁决。**不生成 fake actor、物理 H20、真实特征来源或原 r1 advance decision。** 两项 lower95 均>0且外部门成立、任一 upper95<=0、其余三个数值分支分别只标 `synthetic_selection_detected`／`synthetic_selection_not_detected`／`synthetic_selection_inconclusive`，不能称真实接口支持或方法有效。

顶层始终标记 `synthetic_only=true`、`actor_applicable=false`、`source_provenance_applicable=false`、`method_evidence=false`、`proceed_short_training=false`、`training_authorized=false`；**最终 decision 永远为 `stop_after_fixed_synthetic_assay`**。完整数值结果保存供核对，语义重标不得改变预测、选择或区间。

需要报告：

- 四 case 全部输入 shape／dtype／字节 SHA256、基础随机流与覆盖索引／C／成本表，脚本及旧 stats 文件 SHA、实际 clean commit／平台依赖／总耗时。
- oracle／constant 的上述逻辑验证、每个 admissible feature-only decoder 的全部动作／reward与证明，onehot 每 factor 恰一项为1；跨 case 及 q/p 应共享的数组精确核对。
- 全部 LOEO fit metadata、预测、选择、selected reward、regret、MSE、q−p／q−constant episode 值及重标的合成名义分类；不只展示最成功 case。
- history／identity 的 q/p 输入、预测、选择和 selected reward精确一致；原 stats 输入调用前后 hash 不变，明确没有 Dreamer 或环境调用。

**预定检测漏检反例仅适用于 sparse／replicated 两个 cue case**：其 admissible q feature-only decoder 实际核验正确、unit 动作 margin、oracle 对 constant `.25` 优势，而固定算法 q 的整体 selected reward 相对 `.75` 无正增量。若 q 全部选择 a2，则是最清楚的精确零实例；不得要求某种预期数值必须出现。其余连续差、CI 和合成名义分类照报。不能只凭 privileged C oracle 和 q<=constant 判定 identity 为漏检。若仅 MSE 改善、选择未改善，也只说明此 assay 的决策检测范围有限，不证明真实 q 有信号或 shrink 是真实失效主因。

replicated 预期为 q 选择增量阳性，history 预期两边都利用 h，history／identity 均要求 q−p 精确零。前两项是对照预期而非保证；失败保存、限制 sparse-vs-replicated 的归因，不调参求阳性。history／identity 不一致、decoder／oracle／class／成本／哈希／onehot／输入未修改验失败属于工程失败，停止且不作检测范围裁决。任何超时／非有限／求解失败只保存失败收据，不发布部分 case 的总裁决。

## 6. 区间、预算与停止

旧 bootstrap 输出仅对这一次固定生成器、人工 episode 聚类及固定 cross-fitted 结果作名义描述。四 sample、四 state 和四 case 都不是新增独立 episode；LOEO fit 重叠、八 episode 人为平衡、未重拟合 bootstrap、固定一个生成 seed，不能据其区间宣称真实任务、生成器总体、多个 seed 或独立盲测功效。oracle `.25` 是已知逻辑差，不需用 CI 证明。

冻结 sv3（hostname `lyg0326`）Linux CPU，一次父进程 **120 秒预算**。入口 `scripts/dl_probe_assay_supervisor.py` 仅导入标准库；入口时钟覆盖启动、工作进程 NumPy import／Git 检查／生成／全部四 case 拟合／写盘及父进程完整 hash 链验收。Linux SIGALRM 与 wait 的剩余预算共同执行；失败或超时仅清理此次创建的 session／process group，工作进程没有第二层 session。预算超出或 worker 未终止不可宣称完成；清理和失败记录仍须保存。实际 CLI 不提供预算或主机更换参数。BLAS／OMP／MKL／NumExpr／vecLib 线程环境均设1；工作进程强制 NumPy1.26.4、GPU **0 张**，无 JAX／Dreamer／physics／render、无真实 NPZ／pickle／blind。本规范没有当前 GPU 占用核查。

实现只允许一个不存在的新独立输出目录；存在即拒绝，不覆盖失败或成功产物。sv3 输出 `/data/Policy_Discrepancy/runs/dl-probe-assay-20261011-<执行SHA短码>/attempt01/`，本机 ignored 镜像 `analysis/outputs/dl-probe-assay/<执行SHA>/attempt01/`。保存 budget、plan（基础流初末 state／基础数组 hash）、四 case 的完整数组及 stats、summary／assay_complete／supervisor_complete 与日志；不提交数组。父进程验收所有四 case、版本／SHA／预算／无真实推进标记及输入／结果／plan／summary 的完整文件 hash 链，成功退出和完整产物同时满足才是此次工程完成。预算不足或依赖失败不得加预算／改维数／改 seed 后把当前对象称通过。工程修复须保留失败、同科学参数另新 attempt，并由 root 核语义。

无论阳性、阴性或不确定，完成后停止。旧真实 r1 阴性与完整目标仍保留，不能进入训练或读取 blind。

## 7. 完整目标的下一分叉与未决项

若 sparse 已知线性 cue 被漏检而 replicated／history 检测符合预期，可有限支持“此固定 probe 会漏掉指定稀疏决策信息”的测量反例。此时**另立**事前 nested probe 规范比立刻施加 torque 更有必要：它先减少 readout 收缩混淆，不同时引入物理 OOD、视觉不可辨和动作需要不明。新的有限 λ 集合、共同 q/p 选参、整 episode inner/outer 隔离、train-only 标准化、唯一主决策量和停止门须另行冻结；当前八 cal 已被观察，不能称新确认，不用24 blind调参。

若 sparse 也被检测到，排除了这一次规定合成稀疏对象的漏检，不排除真实 latent 的更弱信号、非线性或采样差异；不能由此断言真实 q 无信息或自动启动 torque。若主要对照均未检测到，先审工具能力及生成／接口正确性，不能继续加 signal 复制、换 seed 或搜索算法。

torque 路径仍独立需要同 `p−1→p` 合法 transition／actual p carry 对齐、事件独立历史、真实64px可识别及不同有代价动作的外部证据；合成结果不提供这些。最终仍需第二信号对 D 的增量、active normal-loss 下可学信息的保护、替代解释及后续学习／性能证据。当前 frozen actor 收益不是未来可学习路线的普遍必要条件。

内部科学审查已确认第3节固定随机流／F64、第5节 feature-only decoder 和有限检测反例判据，以及第6节纯合成区间与停止边界；root 采用上述唯一方案。工程验收分为构造／拒绝守卫（拟合例程禁止调用）与标准库真实 process fixtures（Linux 包含失败 leader／timeout 的后代清理、无关 sentinel 保留及 worker 完成后验收超时）。fixture 成功只验证相应工程接口，不是实际整组拟合或方法效果。**冻结时尚未执行任何合成拟合，未采用新的真实 probe 或物理对象；实际执行 SHA／全部结果和独立数值审计另记结果文档及更新日志。**
