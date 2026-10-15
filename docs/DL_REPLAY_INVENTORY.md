# DL 普通持久化 replay 内容取证

版本 **DL-replay-inventory-r1**，2026-10-15。Agent在完整预验证授权内采用的必要CPU前置，
不是新的训练/算法协议，不改变正式M2或原R4失败。

## 目的与已知前提

普通active压缩/保护的下一机制对象需要真实普通数据。只读文件名盘点已发现1280个NPZ、
16条无环链、总11,421,580,553字节；此前未检查内容/CRC/逐文件SHA，不能直接沿官方loader静默跳坏chunk。
原训练来源SHA `13ec72a214b0c03b886b2c7fac52c8f089966dec`。
精确run：`/data/Policy_Discrepancy/runs/dl-pilot-r3-20261013-13ec72a/dmc_quadruped_walk/N-seed0-attempt01`；
输入仅该run下的`replay/`及原`protocol_manifest.json`、`final_state.json`。

## 固定核验范围

采用`scripts/dl_replay_inventory.py`独立解析，而不调用`Replay.load`：

- 先后两遍完整目录/文件SHA稳定库存；固定原N1M的1280文件/16链/1M真实动作及原记录reset数。
- 逐NPZ真实ZIP CRC、唯一精确成员、NPY header/长度/dtype/shape及有限值，`allow_pickle=False`。
- image uint8[64,64,3]、reward float32、first/last/terminal bool、action float32[12]、stepid uint8[20]、
  dyn/deter float32[4096]、dyn/stoch float32[32,32]及真实onehot/flags/末行action语义。
- filename UUID/successor/长度、每行UUID16字节＋大端4字节索引、跨chunk边界与时间顺序、无断链/合流/环。
  每条链尾指向未落盘的非零successor是源码flush后创建空chunk的合法情况，明确列出，不能伪造零闭包。
- 同源Git、实际配置/context1/batch16×64/consec1、计数及metadata前后SHA；不解checkpoint pickle。

一处失败就拒绝整个本取证，不放宽dtype/flags、忽略CRC、跳chunk或把坏文件排除后宣布通过。
新输出在原run外fresh目录，禁止写原run/replay，禁止把NPZ/检查点提交Git。

上述UUID/缓存前提已核实际`elements==3.22.2`及冻结13ec72a源码：UUID大端/base62顺序与
`UUID(256)`实际向量一致（uuid.py SHA256 `e8829a81c80058e4f130a5821acd9db62de5f72fbb02ad4b10262ccfa3006d6a`）；
reset后仍posterior onehot采样，enc/dec entry为空，finite不入replay，BF16缓存经outer wrapper转float32。
这是源码预期，不代替真实NPZ内容核验。

## CPU执行与预算

本机提交/有限测试后，仅把确定版本同步至sv2新隔离clean审计checkout，原训练源码不移动。
使用已有`/data/Policy_Discrepancy/envs/dreamer/bin/python`，CUDA_VISIBLE_DEVICES为空、JAX_PLATFORMS=cpu、
MUJOCO_GL=disable；不导入或执行model/env/GPU。
完整外部入口固定**900秒/0GPU**，含Git/metadata、双遍SHA、CRC/解码、所有schema/链及输出；
超时只终止本次owned进程组并保留外部失败日志，不重命名为通过。
产物目录`/data/Policy_Discrepancy/runs/dl-replay-inventory-<date>-<auditsha>/attempt01/`。
保存确定执行SHA、完整命令/环境、时长/exit、inventory JSON及SHA；本机只镜像小报告。

## 即使通过仍不能声称

源码`Replay.save`只首次落盘chunk，此后内存`replay.update`的cached dyn不重写已存NPZ；
因此它是**跨保存时点的历史cache**，不是1M终点在线buffer cache的逐字节快照。
sampler RNG、online queue、runner环境状态没有保存；不能无缝续训或恢复原online流。
consec由stream抽样后注入，不应要求NPZ内存在；缓存context和采样后前缀规则必须由未来机制协议单独固定。
内容/来源通过只是持久化数据可用于另立离线受控对象的前置，
不构成标签有效、普通压缩损伤、第二信号有效、保护收益或正式准入。

## 实际结果

实际执行/tested SHA为`7ac29906e9d8953b8b87eb6ba134d3ddcdac9f46`，sv2新隔离clean checkout：
`/data/Policy_Discrepancy/dl-replay-audit-code-20261015-7ac2990`。
本机5项CPU fixture通过；同提交clean Linux5项全过（unittest .195秒、外入口.364786秒）。

真实输出`/data/Policy_Discrepancy/runs/dl-replay-inventory-20261015-7ac2990/attempt01/`，
2026-10-15 16:51:48.905610北京时间终止，exit0、外入口247.14559346297756秒/900秒、0GPU、owned group已清理。
完整1280 chunks/16链、11,421,580,553字节，1,001,008行含1008 reset，恰好1,000,000动作。
全CRC/schema/stepid/onehot/flags及两遍全部文件SHA通过，无跳坏文件；16链尾均为唯一未落盘非零successor。
inventory.json为1,209,489字节，SHA256：
`3339b71ca033d5fc065130751fe163aca05a684b2f14f570ab2018219a3a706d`。
独立只读核验库存JSON/terminal、SHA/计数/16链及所有1280文件名无环无合流；
未独立第二次解码所有CRC，不能把元数据复核称作第二遍独立内容重演。
本机仅镜像上述小报告到ignored `analysis/outputs/dl-replay-inventory-20261015-7ac2990/`。

已验收的是普通持久化数据可用性；历史混合年龄cache与不可恢复online/RNG限制仍成立，
尚未开始正常active replay压缩/保护干预，也没有由此产生第二信号或方法有效结论。
