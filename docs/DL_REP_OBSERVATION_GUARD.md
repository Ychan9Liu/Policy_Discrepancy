# DL-rep-observation-guard-r0：真实 GPU 上的观察图终态负对照

## 状态与唯一目的

**Agent 候选，尚未采用、未执行。** 本文由只读源码审查起草，等待 root 审定；不代表用户逐项确认、已分配新资源或已有真实 GPU 验收。起草时没有读取科学 checkpoint、source 数组或服务器，没有修改现有 r1 代码、规则或结果。

本候选只处理一个工程归因问题：在已冻结的128更新对象中，额外返回 whole-window/nonreset 压力指标，是否改变用于 native actor 评价的完整训练终态。它不新增评价、native episode、训练数据、信号、gate、更新数或统计选择，不改变 [DL_REP_NATIVE_ASSAY.md](DL_REP_NATIVE_ASSAY.md) 的 r1 裁决；原 B/A 和原数据均不可修改或覆盖。

必读：[AGENTS.md](../AGENTS.md)、[DL_REP_NATIVE_ASSAY.md](DL_REP_NATIVE_ASSAY.md)、[DL_REP_NATIVE_RESULTS.md](DL_REP_NATIVE_RESULTS.md)、[DL_RESET_JOINT_RESULTS.md](DL_RESET_JOINT_RESULTS.md)、[DL_VALIDATION_LOG.md](DL_VALIDATION_LOG.md)。源码为 `scripts/dl_reset_learning.py`、`scripts/dl_reset_all_rep.py`、`scripts/dl_rep_native.py`、`dreamerv3/rssm.py`、`dreamerv3/agent.py`、`embodied/jax/agent.py`、`embodied/jax/opt.py`；起草审查基准为 `422ae48e9f1c4728c3e5cfe57a48ce6e2e12b10d`，不是本候选的实际执行版本。

## 必要性、已有事实与假设

**源码事实**：旧 B 的包装返回三个 `dl_reset/*` 指标；当前 A 另外返回五个 `dl_allrep/*` 指标。这些数组作为 `RSSM.loss` 的 aux 经 `Agent.loss → Optimizer(has_aux=True) → _train` 返回。它们没有数学损失系数、没有参数或 NJ 随机调用，但改变了编译函数的输出与中间量存活图。不能从“只读指标”直接推出 XLA 优化器状态逐字节不变。

**已记录的工程反例**：`DL_VALIDATION_LOG.md` 的真实 size50m/1m `55a527c` attempt03 中，logging 的 `opt/state/2/1/rew/head/logits/bias` 与 `opt/state/2/1/val/head/logits/bias` hash 不同，而模型参数及独立总梯度相同；该精确验收失败，后续将只读 logging 与 optimizer 图分离再验。附加输出影响融合舍入是当时的工程解释，不能把它写成当前 A 已发生同类差异的证明。

**已有构造证据**：CPU tiny 证明包装的一次 original loss call、rep 前向保留、指定 rep 梯度停止、其余指定梯度等价；另一个 tiny Optimizer 证明官方 `opt/step/value` int32 递增。它没有运行 size50m 完整 CUDA Agent 的这两个输出图，更没有证明128更新终态等价。

**工作假设**：在同一真实初态、输入、carry 初始化与 key 流下，额外五个指标不会改变128更新终态。任何一位不同都是该有限精确假设的反例；即使差异很小，也不能改容差后称等价。差异可能来自附加图、环境/配置或复演不确定性；未完成定位时只记录“观察图等价性未建立”，不将数值差自动归为 XLA 融合或任务信息压缩。

## 最小对象：两条新路径与两个历史终态 anchor

| 新路径 | 数学 rep 梯度 | 训练图中的压力指标 | 对照 anchor |
| --- | --- | --- | --- |
| `Bwhole` | 正常 rep，不停止、不乘单位权重 | 原三个 reset 指标＋与当前 A 完全相同的五个 whole/nonreset 指标 | 已审旧 B128 完整终态及128条收据 |
| `Amin` | 与当前 A 相同，整个 loss 窗口的当前 rep 梯度停止 | 仅原三个 reset 指标 | 当前 r1 A128 完整终态及128条收据 |

每条新路径从原300k完整 checkpoint 开始，恰好128次真实 full `Agent.train`。它们分别检验 `Breset → Bwhole` 和 `Awhole → Amin` 的输出图变化；**不要求 Bwhole 与 Amin 的参数、carry 或损失相等**，二者的 rep 梯度本来不同。

两个新终态若分别与真实加载的旧 B、当前 A 完整终态逐叶 dtype/shape/bytes 相同，就能排除“这两个观察图变化改变了本次 native 评价所用128步模型终态”的解释。该条件针对冻结对象的终点，是两条路径的充分证据；它不证明所有可能模型、输入、GPU、训练长度的等价性，也不证明每个中间状态与历史完全相同。它不能把 r1 收益直接提升为任务信息压缩或选择性保护结论。

### 不把缺失的历史证据补成事实

现有 `run_branch`/`run_a` 收据保存逐轮 train root、counter、metrics JSON、原 typed `metrics_digest`，以及初末完整 state digest；**没有保存每轮完整参数、returned carry、replay outs、实际内部 NJ 子keys或 sampled latent**。

| 比较项 | 两路径对象可以实际验收的内容 | 不能声明的内容 |
| --- | --- | --- |
| 初态与终态 | 真实加载 anchor，并逐叶比较完整 params（含 optimizer/moments/norm/targets）、完整 counters 和 typed state digest | 只有 actor 权重相等，或只有 JSON 中声称的 hash 相等 |
| B 的128轮指标 | Bwhole 实际 typed metrics 只删除确切五个 `dl_allrep/*` 后，其完整 key 集与旧 B 相同，使用原 `tree_digest` 与旧 B 该轮 `metrics_digest` 精确比较 | 丢弃其它不同键、把新指标填入旧 hash 或放宽数值容差 |
| A 的128轮共同指标 | Amin 实际共同指标按原 `jsonify` 转换后，与旧 A 去除确切五个 whole 指标的 JSON 树精确比较，包含浮点 signed zero；明确 `reference_common_typed_digest_available=False` | 原 A 共同指标的 dtype/shape/bytes 已被独立恢复或逐字节证明 |
| carry 与 replay outs | 新路径每轮保存真实返回值的 typed digest/leaf manifest，并核有限性、空间、计数和 raw prevact 合同；128次实际 `init_train` 记录 | 它们已与不存在的旧逐轮 carry/outs 字节证据比较 |
| 随机流 | 与原 B/A 收据逐项比较实际 `prepare_batch` uint32 root、seed counter、Agent counters；original 随机源码及调用拓扑不变，包装不新增 `nj.seed` | 历史所有内部子key或随机样本逐字节比较已经完成 |

原 A 的 JSON 指标比较是其真实保存表示上的精确比较，不冒称 typed byteexact。新路径可以保存更完整的 typed 证据，但不能回填或覆盖原收据。不得在 optimizer 编译图内新增 key、latent、carry 或特征输出来制造这些证据；应读取原本已经返回的 carry/outs，或在 JIT 边界之外读取完整 state。额外诊断本身的调用数、host transfer 和耗时须记录。

可选的证据增强是恢复旧 A **全部**metrics的类型与shape，再要求原 `tree_digest` 与该轮既有完整 `metrics_digest` 完全相同；只有这个旧完整hash认证通过后，才可投影共同键并声明共同指标typed比较。恢复使用明确的原源码contract/实际typed schema，保留signed zero、整数范围和精确表示；不能凭JSON数值相同或未认证的dtype猜测宣布typed相等。该增强不改变两路径终点对象；未做或无法认证时仍用上表的JSON边界，不能声称增强已通过。

### 何时才需要四条新轨迹

如果采用者另外要求“128轮每一步的完整 params/optimizer/norm/carry/common outs均直接跨观察图逐字节等价”，历史缺失不能由终态相等补足。此时才需要再复演 `Breset` 与 `Awhole`，形成 `Breset ↔ Bwhole`、`Amin ↔ Awhole` 两对新轨迹；两条 legacy 变体的终态仍必须精确锚定原 B/A。四条总计512次更新，属于另一个更强对象，须单独采用和冻结预算，不能自动加到本两路径候选。

两路径是本次终点归因目标的最小选择。一次 GPU 单步仅能发现起点的反例，不能替代本128步对象；默认重演四条也不是终点目标的必要条件。终态相同不能排除中间差异后来消失，而这种差异未改变被冻结的评价模型终态。

## 全部不变的科学输入与执行语义

- 原 baseline train seed0、300k、size50m、`dmc_quadruped_walk`、64 RGB/proprioFalse/camera2/repeat1；原完整 config 仅使用既有 offline `enable_policy=False` 构造，不新增 public policy、report、evaluation 或环境调用。
- 原完整 params/optimizer/moments/norm/slowvalue/counters 标准 `Agent.load`，不 regex load、不清 momentum；标准 load 将 `batches`恢复为 checkpoint `updates`，保持既有语义。
- 原 cal IDs `[9,14,15,18,20,24,30,31]`、八真实完整1001帧、原 `contexts.pkl`、原 `offline_batch`、`DL-reset-learning-r1` layout/stepID、batch16/loss64/context1及原 batch digest不变。八cal各重复两次，另一cal真实terminal作context，reset0..63作loss；仍是有限离线reset富集对象。
- 每轮 `init_train(16)`，原初始化root `_seeds(0)`；`prepare_batch`实际 uint32 root与原 B/A逐条一致，不按新路径名派生训练key。原数据/contexts/checkpoint前后 hash不变。
- free nats1、dyn/rep系数1/.1、reward_grad/repval_grad=True、ac_grads=False、其余 loss/optimizer/normalizer/slowvalue不变；M2/rep_probe/DtLatch overlays off。
- 官方 `opt/step/value` int32 scalar由74493到74621；Agent updates/batches各+128，其余保存 counter不变。每次真实 `opt/updates`与完整 state核对，不只累计循环次数。
- 训练调用方式沿用既有对象：`Agent.train`后调用 `take_train_result`取得该次完整metrics与outs。原对象没有回写实际 Replay，本候选也不回写；不能偷偷改成普通replay loop。
- 包装在 Agent/JIT 构造之前安装，原 `RSSM.loss`只调用一次。Amin在训练图内部完全没有五个 whole 指标输出，不能在 host端删除键之后就声称图已精简。Bwhole不得先调用已有停止 rep 的包装再尝试“恢复”梯度。

## 最小实现接口与复用边界

候选实现以新 `scripts/dl_rep_observation_guard.py` 和 `tests/test_dl_rep_observation_guard.py` 为工程准备文件，不修改原19生产文件、`dl_reset_all_rep.py`或r1 driver。最终实现签名以采用时冻结源码为准；最小职责应为：

```python
observation_branch(mode, rssm_class=None)  # mode严格为Bwhole或Amin，构造前上下文
run_guard(agent, checkpoint, calibration, source_hashes,
          frozen_context_entries, mode, *, reference_b, reference_a,
          check=lambda: None)
```

`observation_branch`复用原loss、同一压力公式，明确三reset＋五whole的输出集合及全rep SG开关；不创建 NJ 变量、参数或随机调用。`run_guard`复用 `offline_batch`、`tree_digest`、`optimizer_step`、实际计数/key守卫，但不能直接调用有旧branch身份与压力守卫的 `run_a` 来伪装 Bwhole/Amin。两个reference为已认证的原B/A学习收据；helper没有CLI、文件认证、设备设置或科学source读取。

新的外围driver负责原字节认证、资源/父预算/owned退出、真实 checkpoint anchor加载及完整逐叶比较；array-only helper不能自行宣布真实来源或GPU验收。采用时新增每轮typed state/params、carry、outs、实际root、metrics及common projection收据，持久化 leaf manifest/digest与两份完整终态即可，不要求把256个完整中间checkpoint全部写盘。观察须在原 `_train`外进行，不能改变其 aux结构来记录新数据。

源码可复用 `dl_rep_native.authenticate_old` 的旧B/19pins/source/完整退出认证思路、`dl_reset_joint.load_cal` 的hash-before-decode读取、`dl_resources`的fresh资源/实际CUDA绑定，以及owned deadline/迟写封口。现r1 parent的phase、7200秒和固定native评价不能直接当本对象driver使用；需新独立入口、版本、目录、退出收据和源码pins。

## 采用前必须冻结的认证证据

1. **旧 B anchor**：原 `/data/Policy_Discrepancy/runs/dl-reset-joint-20261012-c030e26/primary-attempt02/` 的 `B-checkpoint.pkl`、`B-complete.json`、`contexts.pkl`、原父/CLI/freeze/独审链及19原文件。既有 checkpoint/complete/context SHA依次为 `c878652b0fca1313ed9d1536eec838e2972e3dea611e4b309f94b32a63272670`、`b2596ff268873725e54a4008d760961e55a67c00c5318477b5ccc98ff280b8fb`、`c8a6a964455e2a579c4ed76f1de19cc38931955279b36b45445310414451e1ab`。B最终fullstate digest为 `9e670873990510f97174f07b04a407ed290ad762057a33e8884818d5e2fd013d`；仍须核真实原字节并实际加载比较。
2. **当前 A anchor**：`DL_REP_NATIVE_RESULTS.md` 指向原producer实际SHA `fc4219f36449783f8751a9890a125cd218a2807f` 和 `/data/Policy_Discrepancy/runs/dl-rep-native-20261012-fc4219f/primary-attempt01/`。采用前从认证的 `A-complete.json`、`A-checkpoint.pkl`、`old-authenticated.json`、freeze及退出链冻结路径、原字节SHA、typed初末state/params digest和128条收据；本文未读取这些科学载荷、未填入未知A终态hash。若r1仍未完整封口，只能注明已认证学习phase的范围，不能写整个r1已验收。
3. **原输入**：checkpoint SHA `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9`、config SHA `11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445`、batch digest `3bfdef05a0dde48f04f37a67f51da86a236420e41f7cf9476a8ae9bcf67deeec`，初始fullstate digest `216fe9be592190709fe4f22ad23ab89808baa9eb49b618046855861a11276c78`；全部32source只hash，科学数组仅解八cal，未开启旧24或任何native episode载荷。
4. **环境和编译**：与原B/A相同的实际Python/JAX/jaxlib/Ninjax/NumPy/ml_dtypes/CUDA/XLA/设备拓扑/dtype/编译相关环境；记录全部依赖、flags、设备UUID和新源码pins。环境不一致是复演混杂，不能失败后换flags直到获得exact。静态源码及CPU fixture是前置，不是GPU通过证据。
5. **新运行完整链**：独立attempt、父预算、live parent认证、两worker终止、externalCLI exit0、无late marker、真实GPU start/end与峰值采样、所有原输入末核；成功结果在deadline内完成写盘并封口。CPU auditor只认证文件和比较证据，不能写成独立GPU optimizer重演。

历史依赖或编译flags未被原收据保存时，标记为历史未知；现在测得的版本不能补作历史证明。可记录新终态是否重现anchor，但“除观察图外条件相同”的更强因果声明仍有环境认证缺口，不将未知默认成一致。采用前须由root明确可认证的环境范围及保留限制。

## 精确比较、失败记录与停止

所有state数组均比较 container/key集合、dtype身份（包括真实BF16）、shape和原bytes；signed zero、NaN/Inf、漏leaf、额外leaf、漏optimizer状态、counter冲突均不能忽略。新终态从独立文件重新load后再次使用标准 `agent.save()` 与anchor真实state逐叶比较；不比较pickle文件本身的序列化顺序来代替数值树比较，也不只核SHA字符串。

比较失败输出首个update、字段/leaf路径、dtype/shape、两侧原hash、首差flat index、差异数、signedzero信息、有限的abs/relative差诊断。原不相等的payload/digest、完整失败退出和差异记录保留在新attempt；浮点误差大小只用于诊断，精确判据没有容差。不得round、cast后忽略差别、删除不相等的输出、换key/batch/版本或增加更新数求等价。

上述leaf细节按实际可得证据报告：两份真实终态可定位原字节首差；legacy指标若只有总typed digest而无旧leaf schema，需报告该轮整体hash不等及可定位的JSON差别，将无法识别的旧dtype/shape/byte位置写为不可得，不编造首差。JSON完全相同也不能掩盖typed hash不等。

为定位endpoint差异，工程有效时两条固定128路径均保留完整结果；某一路出现数值差不触发追加轨迹。无效输入、非有限值、真实optimizer/计数不完整、资源冲突或超时则失败停止，不能发布equivalence或拼接未完整路径。

| 结果 | 允许的结论 | 后续边界 |
| --- | --- | --- |
| 两条终态、其可得共同指标及全部工程守卫精确通过 | 在本冻结输入/环境/128更新终点上，去掉或增加五个观察指标没有改变r1评价模型终态；观察图这个终点替代解释被收窄 | 不扩大为逐轮全部旧carry/子key相同、普通replay、选择性保护或方法有效 |
| 任一state bit/common可得表示不等 | 精确观察等价性未建立，保存差异；r1暂不能严格宣称actual A−旧B仅由rep SG造成 | 当前r1结果和裁决不改，不将所有差异归为图融合；另审定位或未来对称/分离图方案 |
| 工程或证据不完整 | 无法裁决本guard | 不把checkpoint终态相等的部分信息当完整验收，不追加预算补齐 |

原r1状态和native回报只引用其既有报告；本guard没有新的性能检验或统计显著性门。压力 `active=rep>1`只表示严格超额，不等于所有非零rep梯度；不改变原active定义或使用该值作为本等价门。

## 资源与预算建议：均未批准

**建议**一张独占A100顺序执行两个新GPU worker，无需双卡并行；原GPU4的分配和r1预算不自动授权本对象。采用时依照AGENTS重新确认用户指定服务器/物理GPU、fresh占用和实际CUDA UUID，不挤占未知任务或自行换卡。本对象不创建env、不physics、不render；记录0新EGL/渲染调用，不能拿旧EGL证明冒称本次实际使用渲染。

**建议硬预算**：整个新GPU父入口1800秒、最多一张卡/0.5GPU小时，包含认证、imports、加载/编译、256次full train、逐轮host读取/hash、保存/reload/比较及完整退出；另0GPU CPU证据审计180秒。两者均为候选单attempt上限，尚无吞吐充分保证，不是总科研预算、预计小时或资源已预约。

建议来源：已记录当前A128学习phase为143.612114秒；旧0/B/R完整主入口505.44秒、峰6629MiB。新对象不含native96,000步和render，但新增逐轮fullstate/carry/outs host读取与hash，两次独立编译和checkpoint reload仍有未知开销，不能用上述时间保证1800秒内完成。6.47GiB历史采样峰不是新瞬时峰或最低显存要求；实际内存/传输、GPU峰、全过程时间和失败成本须记录。超时停止，不自动延长到r1的7200秒。

建议产物根为 `/data/Policy_Discrepancy/runs/dl-rep-observation-<date>-<sha>/primary-attemptNN/`，独审sibling `independent-auditNN/`，本机ignored镜像 `analysis/outputs/dl-rep-observation/`；只将协议、代码、测试及小型结果报告入Git，不提交checkpoint、RGB、cal arrays或全量state/outs。

## 待root审定与下一步

仍待审定：采用两路径终点对象还是另立四路径全轨迹对象、当前A认证anchor完整pins、真实编译环境一致性、新父/独审预算、实际资源分配及helper/driver确定版本。首选两路径endpoint guard；A共同metrics的legacy JSON边界和历史carry/outs/子key缺失必须保留。若要求超出已有证据的更强结论，应改变候选规格而非偷换验收口径。

可立即进行候选helper/CPU构造、hash/错配/精确比较/owned父退出测试及只读交审；真实输入读取和GPU执行须在采用、版本封口和资源核查后按所冻结范围开展。没有本guard通过前，未来普通replay对象应明确额外观察图归因缺口；本guard通过也不自动采用普通replay或第二信号方案。继续当前chat，不自动创建chat或发送跨chat消息。
