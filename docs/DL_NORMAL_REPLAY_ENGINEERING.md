# DL 普通私有 replay 工程验收

2026-10-15，版本DL-normal-private-replay-probe-r1。Agent在既有完整预验证授权内采用的必要工程前置；
不采用[下一机制草案](DL_NORMAL_MECHANISM_DRAFT.md)的训练组数、更新数或科学判据。

**当前结果**：18:23:39 sv2真实1M CPU私有导入及合成回写工程验收完成，actual/tested2ae18a5，
外286.732秒、峰RAM31.85GiB、0GPU；没有神经网络学习或方法效果结论。下方保留实施过程及两次失败。

## 固定对象与验收范围

输入为[已核验的真实N1M持久化数据](DL_REPLAY_INVENTORY.md)，固定原run、1280 NPZ、16链、
1,001,008行/1008 reset/1M动作。inventory原始SHA为
`3339b71ca033d5fc065130751fe163aca05a684b2f14f570ab2018219a3a706d`。
本次只在私有内存建立官方Replay，原数据、原训练checkout和在线状态不写入。

`scripts/dl_normal_replay.py`先认证inventory和每个NPZ原始字节，再完整CRC/schema/stepid/onehot/链核验；
保留原UUID、successor及历史cache，独立检查官方999,984个合法65步起点。
使用官方annotation与Consec(length64,consec1,prefix1)，实际输入16×65；
完整65步输入及64步输出stepid认证后，调用官方Replay.update并逐位置验证重叠时后写覆盖。
缺失/错误回写不能沿官方静默KeyError路径跳过；异常部分应用后该私有臂不可继续。

`scripts/dl_normal_replay_probe.py`生产CLI不开放fixture、计数、inventory pin或预算覆盖。
固定工程seed20261035，只抽一个16序列计划；显式合成deter+1及stoch onehot滚动测试私有cache，
再次同计划读取时独立核全65步位置及所有raw字段字节，再以原值回写清空pending。
这些不是神经网络输出，不是模型训练或压缩效果；一个工程计划不定义科研抽样/更新预算。
结束前重新核全部1280文件原始SHA、原run元数据和真实Git blobs，输出只在原run外fresh目录。

## CPU预算与证据要求

生产入口固定Linux、900秒全入口alarm，CUDA_VISIBLE_DEVICES空/JAX_PLATFORMS=cpu/MUJOCO_GL=disable；
自身RLIMIT_AS最多96GiB并记录实际上限和峰RSS。私有数组预计约30.6GiB，不能用压缩NPZ大小估RAM。
不请求GPU backend、模型、optimizer或环境；包间接导入JAX不等于执行GPU，不声称完全没有JAX import。
XML+pmon只读核自身PID的前后GPU进程快照，查询缺失需明示，不能假写双查询通过。
接受真实完整验收还须外部exit0、900秒内及owned进程组清理证据，单独JSON不能替代终态。

## 当前验收状态

本机Windows三组16项工程检查15通过、1项Linux真实alarm/owned进程组跳过，unittest 2.121秒。
三组为tests.test_dl_normal_replay、tests.test_dl_normal_replay_probe、tests.test_dl_replay_inventory。
fixture覆盖跨三chunk、重叠回写、未知ID/坏cache、输入污染、漏写及独立回读拒绝篡改；
它不构成真实1M导入、Linux监督或学习验收。

下一在本机提交确定源码后，只同步sv2新隔离clean checkout，先Linux16项，再执行真实CPU全量probe。
截至本段写入尚未执行真实1M私有导入，0新增GPU作业；R5仍仅sv3物理4原f845入口，执行checkout不移动。
历史cache混合年龄、sampler/queue/env不可恢复的限制即使工程通过也保持。

### Linux attempt01保留失败（2026-10-15 18:16）

确定源码`b8f14a4d9471df2fcf823b7d5f6b540224241b52`已push并仅部署sv2新隔离checkout。
实际Linux16项中15通过；真实alarm用例已取得exit124、owned进程均退出，
但fixture先exists再读取/proc时碰到进程刚被reap，抛ProcessLookupError，故整体验收失败、真实1M probe未启动。
证据在`/data/Policy_Discrepancy/runs/dl-normal-replay-probe-20261015-b8f14a4/attempt01/`，
linux-tests-terminal.json记录exit1、外2.718362秒、无残留，失败日志不覆盖。
本机仅修fixture为直接读取stat；若条目已消失仍满足原“已退出或Z/X”判据，读得到时仍严格Z/X。
不改生产代码、时间/内存/计数或科研语义；新确定提交、新attempt重新完整Linux验收后再真实CPU导入。

### Linux修复后及入口attempt02/03

修复来源`2ae18a54722ab4e96f1775d64a0884f998f573da`已push；sv2 clean Linux16项全过，
未移动原训练或R5执行checkout。外部launcher生成第二次尝试时误将输入路径的attempt01也替换成attempt02，
生产入口在inventory认证前拒绝不存在的路径，exit1/外0.215115秒、无残留、未读真实NPZ；
`/data/Policy_Discrepancy/runs/dl-normal-replay-probe-20261015-2ae18a5/attempt02/`完整保留。
此为启动参数错误，不归因数据或方法，不改源码/协议来绕过拒绝。

正确原输入路径已逐字核对，attempt03使用同一确定提交的新clean checkout：
`/data/Policy_Discrepancy/dl-normal-replay-code-20261015-2ae18a5-attempt03`。
再次Linux16项全过（unittest 2.452秒、外2.718126秒/exit0/owned无残留），
18:18:52.931858北京时间开始真实CPU probe PID1278439，whole900秒/96GiB保持。
输出在`/data/Policy_Discrepancy/runs/dl-normal-replay-probe-20261015-2ae18a5/attempt03/`；
该时点等待其权威终态；后续实际结果见下节，不能把此前fixture通过称为真实1M验收。

## 真实CPU验收完成（2026-10-15 18:23:39）

actual/tested均为`2ae18a54722ab4e96f1775d64a0884f998f573da`，上述sv2 attempt03完整exit0，
probe外286.732150秒/900秒；含clone、checkout及16项测试的整个控制入口290.344214秒。
实际私有导入271.240985秒，峰RSS34,195,369,984字节（约31.85GiB），
RLIMIT_AS实际soft/hard均103,079,215,104字节（96GiB），owned进程无残留。

真实1280 chunks/16链/1,001,008行/1008 reset/1M动作完整导入，官方999,984合法65步起点登记通过；
一个固定16序列工程计划经官方annotation/Consec得到16×65，合成1024行cache回写后完整65位置独立回读通过，
所有raw字段逐字节不变、pending已清空。本真实计划1024行均唯一，重叠后写覆盖证据仍只来自有限fixture，
不冒称此真实计划也碰到了重叠。原1280文件/11,421,580,553字节的末遍原始SHA及run元数据前后均稳定。

probe.json为41,290字节，SHA256
`b3b4680feb9d54ed7bc69eae89a0e52b9eb15b9554eda92a99d3c04f3acea4f7`。
外层probe-terminal.json/terminal.json独立记录exit0、时限和owned清理；本机小镜像在ignored
`analysis/outputs/dl-normal-replay-2ae18a5/attempt03/`。
第二Agent只读核对该原始JSON/日志SHA、全部230项Git blobs、固定库存及元数据，并独立按filename顺序和seed
重建16序列计划及999984起点计数，逐项一致；真实计划含一处跨chunk、无重叠。
此复核没有再次扫描原11.42GB、解码CRC、重演Replay或读取原始XML/pmon，不称第二次独立全量执行。
自身PID1278439前后XML+pmon均未发现GPU进程；包间接import JAX为true，未请求backend/模型/环境，
0 optimizer更新、无学习或GPU任务。进程快照不构成独立driver/backend执行证明。

通过的是真实历史普通数据的完整私有导入与一个工程计划的合成回写，
不是正常神经网络多步学习、信息压缩/保护、第二信号或性能验收；历史cache/online不可恢复限制不变。
后续仍在当前chat监督原R5，下一机制的科研预算和判据另行冻结，不以本工程通过代替科研依据。

## 实际tiny多步反馈前置（2026-10-15）

新增`tests/test_dl_normal_learning.py`，本机真实公共Agent连续三次train/take_train_result/strict_update，
完整八类原loss、默认free-nats1/imagH15/optimizer/norm/slow target均保留；三次opt updates1/2/3，
保存真实非零moments后完整save/load并继续，无清零或regex。全部保存state/carry/outs/metrics有限，
optimizer状态逐次变化、最终NN参数变化，原NPZ SHA稳定；只核原总grad_norm有限且正，不声称逐叶独立gradient验收。

关键是consec_train1产生consec0：真实_apply_replay_context用每批第0行cached prefix覆盖传入carry。
fixture明确用start2→3→4，后批第0context恰为前步actual network outs回写过的位置，
逐字节核两次反馈与三次真实context/prevact选择；独立counterfactual传入不同carry仍由consec0选cache。
重叠两行以actual后行网络输出为最终writer，跨真实fixture chunk核读回，不以只连续调用train充当反馈证明。

本项是生成12行/3chunks的tiny CPU F32、B2/T3/context1、image64/action12。
显式隔离patch小schema/计数，保留原图/flags核验；实际原始输入4行，不是生产65步。
兼容receipt字段legal65_starts=9仅是该小fixture的4步窗口数，N1M及source Git认证明确False。
不构成真实N1M/BF16/16×65多步学习、gate作用或方法效果。原完整1M CPU验收actual/tested仍2ae18a5，
本新fixture不替代或改写它。与四项[新读出数学检查](DL_NORMAL_READOUT.md)及旧三组联跑，
本机21项20pass/1 Linux alarm skip，11.358秒；确定提交Linux验收待执行。

确定源码`e237ed8bef91899258a6d8076d0f29f9e358ef8b`仅同步sv2新隔离clean checkout，
上述21项Linux全过（unittest17.987秒/外18.288242秒/完整入口19.267469秒，exit0/0GPU/owned无残留）。
233项源码及三norm前后真实blob一致，独立Agent复核存储Git/日志/收据通过，未二次执行远端测试。
证据`/data/Policy_Discrepancy/runs/dl-normal-cpu-20261015-e237ed8/attempt01/`，
服务器canonical/旧实际2ae完整replay与f845 R5源码checkout不移动；不因新tests通过升级旧1M学习范围。
