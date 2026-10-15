# DL 普通私有 replay 工程验收

2026-10-15，版本DL-normal-private-replay-probe-r1。Agent在既有完整预验证授权内采用的必要工程前置；
不采用[下一机制草案](DL_NORMAL_MECHANISM_DRAFT.md)的训练组数、更新数或科学判据。

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
