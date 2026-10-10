# DL 验证日志与证据索引

版本 **DL-engineering-r4 / DL-protocol-r1**，2026-10-10。整体目标**未完成**：真实r1两阶段主诊断完成，当前幅度判据未满足且关键标签覆盖不足；真实size50m attempt03也因logging optimizer状态差异未通过。必要工程修复及独立H100机会诊断进行中。进程exit0、文件完整或工程fixture通过均不等于方法有效。

本日志区分历史证据和当前状态。运行状态依据本轮根 Agent 已核查的收据；最近一次资源快照为 **2026-10-10 17:39 北京时间**，不代表持续空闲。科研定义、冻结分割与筛选标准仍以 [DL_PREVALIDATION.md](DL_PREVALIDATION.md) 和 [DL_PREVALIDATION_REVIEW.md](DL_PREVALIDATION_REVIEW.md) 为准。

## 版本和同步边界

项目根 `D:\program\PD\dmr3`；本机隔离工作树 `C:\Users\97370\.codex\worktrees\1f9a\dmr3`，分支 `codex/dl-prevalidation`。DL-r0 归档提交 `2b9641d`；合并基准完整 SHA `ae8372e42a815824d2e4ec3a527bfb985a81972b`，含 main `595e982f1a3939775879728009e2105ee516e86d`。

| 确定版本 | 范围与实际验证 |
| --- | --- |
| `9e6da39e59ad347f53db93af649dc8e8b63425ad` | DL-code-r1 方法实现、诊断和 runner；clean 提交上完成本机完整 tiny Agent、回归与 CPU 物理复原；真实 pilot/采集使用此方法代码 |
| `1280bea7bf409e9cc6fa975704bcb370b8880a0a` | DL-engineering-r2，仅改资源空闲判定及测试；完整 XML/pmon/UUID 作业库存判占用，显存作为真实 telemetry 保留；clean 提交上资源/runner 26项通过 |
| `3721103a081e6e667ca2329a0b221cf8d0aeb5ef` | DL-engineering-r3，仅改 size50m 梯度验收 helper 及其测试；对 probe 实际访问的参数逐叶检查，未读取状态明确列为不依赖；clean 提交上2项 helper 测试通过 |
| `55a527c620fb14339ee00f76f045a5dfddc81068` | DL-engineering-r4，将CUDA prior/v动态传入独立局部梯度检查，避免闭包常量触发transfer guard；没有改变stop-gradient、均值、free-nats或容差；本机helper2项通过，真实GPU7 attempt03五worker执行完成但paired整体验收失败 |

最近真实size50m工程执行SHA为 `55a527c620fb14339ee00f76f045a5dfddc81068`。9e之后至55的三个提交未改 `dreamerv3/` 方法代码，也未改科学配置、筛选阈值或精确比较条件。后续logging分离和机会helper尚需另记确定提交及实测；不能将文档提交或当前HEAD写成所有旧运行的执行版本。

sv3 已部署独立worktree `9e6da39`、`1280bea`、`3721103`、`55a527c`，位置 `/data/Policy_Discrepancy/dl-worktrees/<版本>/`。原 `/data/Policy_Discrepancy/repo` 保持干净d9be557，不修改原作业checkout；SV2仅只读取得checkpoint，没有代码同步或GPU运行。DL分支**未合入main**；初次push连接reset失败，随后成功推送至GitHub `origin/codex/dl-prevalidation` 的3721103。当次成功不冒充后续55或新修改已推送；最终远端SHA须实际再核。不覆盖canonical main既有本地修改。

## 环境与本地证据

隔离环境为 `%LOCALAPPDATA%/Codex/dl-validation-env`，Windows/Python3.12.14、JAX/jaxlib0.4.33、NumPy1.26.4、elements3.22.2、ninjax3.6.3、optax0.2.5、chex0.1.90、portal3.8.1、dm-control1.0.48、MuJoCo3.15.0；关键版本对齐服务器 lock，`pip check`通过。Windows Python3.12 与实际 sv3 Linux Python3.11.16 仍有差异，本机验收不能替代服务器验收。未升级服务器环境；进程 DLL 路径补充只存在本机外部隔离环境。

确定提交的本机证据根（Git ignored）：`analysis/outputs/dl-prevalidation/validation-9e6da39/`。目录名保留最初验证版本，其中后续 r2/r3 日志实际 SHA 见表，不把目录名当执行版本。

| 证据与实际版本 | 已观察到的结果 | 限制 |
| --- | --- | --- |
| `full-agent.log`，clean `9e6da39` | 完整 tiny Agent 测试1项通过，77.71秒；验证 off/alpha0/rho0/logging/strengthκ0 旁路的参数、梯度、loss、latent 与优化更新精确一致，非法有效 reward 明确失败 | tiny 模型，非真实 size50m、非方法有效性 |
| `regression.log`，clean `9e6da39` | 92 passed、1 skipped、13 subtests passed，20.20秒；覆盖探针、TwoHot、mask/terminal、梯度、置换、诊断和 runner 等工程路径 | 保留 skipped 状态；跨 shard fixture 不是多 GPU 方法需求或真实任务验收 |
| `physics.json`，clean `9e6da39` | 3 seeds×4位置×7候选，共84项 H10重复检查，完整积分状态/任务/RNG/counter一致，最大误差0；float32 H1标签一致 | 实际 CPU 物理分支，禁止渲染；不验 segmentation、冻结 size50m 信号或 gate/return收益 |
| `resource-r2-regression.log`，clean `1280bea` | 资源/runner 26项通过，3.15秒；包含真实 `dl_train.run` 控制流 fixture，以及1MiB无作业边界、非空作业库存拒绝 | 资源查询使用 mock；flow 使用真实 serial Driver/ProtocolState/EpisodeReturn/Checkpoint保存，但 agent/replay为double；Windows fixture关闭 checkpoint清理，未验 Linux清理或真实训练 |
| `helper-r3.log`，clean `3721103` | helper 2项通过，5.50秒；小型生产 Agent 上实际读参数的 detached probe 梯度严格为0、状态不变 | 仅验实际访问参数；不将未读参数声称为直接梯度实测；真实 size50m r3待验 |
| `helper-r4.log`，确定 `55a527c` | helper2项通过，5.43秒；代码已提交，其他dirty变化为文档 | 未完整执行真实worker局部梯度路径；由GPU attempt03补验，不能据本项宣布整体验收 |

初步 dirty 工作树证据保留在 `analysis/outputs/dl-prevalidation/diagnostic-cpu-r1/`、`runner-flow-20261010T073136Z/`、`runner-resource-20261010T074442Z/`。前者含 `tests-frozen.log`、`physics-final.json`、`cpu-validation-frozen.json`；旧文件哈希、失败和未提交标识不覆盖。当前验收优先引用上表的确定提交证据。

### DL 已发现的工程失败和修复

- 静态 gate=1 仍执行乘法会产生后续 optimizer 末位差；改为真正的 alpha0/rho0/strengthκ0 旁路，没有放宽精确比较。
- JAX ordered callback 在 Dreamer transfer guard 下创建 host token；改用 unordered invalid-signal callback，实际非法训练 fixture 仍明确失败。
- 物理 float64 reward 与采集 float32 标签误作同精度比较；分开物理重复误差与 float32 编码一致性。
- 按诊断 batch4 的 clean carry推进会使历史随机流受选点影响；每帧统一 batch1 advance，诊断 carry丢弃并实测 invariance。
- P 索引移动可能只是互换相等权重；新增实际权重变化和释放量检查，未改变门槛求通过。
- Windows regression 发现旧 `freeze_c.py` 的 JSON LF/CRLF 转换导致旁边 checksum 不等。在 main `595e982` 的独立源码归档复现同一失败；r1 改为写入实际被 hash 的 UTF-8 bytes。未改变 c 公式、统计、容差或 Linux 字节；确定提交回归已完成，不能再记为待重验。
- **DL-engineering-r2 资源误报**：17:07北京时间现场 GPU4/6/7各1MiB，但完整 XML/pmon 无 PID；GPU5有实际 collect PID1293785、2585MiB。原 `memory.used==0` 条件误阻健康卡。对齐既有正式监督器的作业库存语义，仍拒绝任何计算/图形/未知 PID、库存不可用和 UUID/绑定异常；照实记录1MiB，不写成0，不放宽科学或数值判据。预检不是原子资源锁，启动和初始化仍须核查实际映射。
- **DL-engineering-r3 size50m helper 失败**：GPU7 的 `size50m-cp1000000-attempt01` 在 detached probe 梯度检查中失败，Ninjax 目标集合含该 objective 未读取的参数，尚未执行 optimizer update。改为先追踪实际访问，再以完整叶名检查所有被读共享参数，记录未访问状态，并禁止 probe 创建/修改状态。保留 attempt01 原日志；r3 用新 attempt02重跑，不删除失败或放宽梯度为0的判据宣布通过。

## 冻结 checkpoint 来源

可审计清单：[dl_checkpoint_sources.json](dl_checkpoint_sources.json)。这是只读 sv2/lyg0360 的 source provenance 复核，不是新的 GPU 推理或方法证据。源为 `dmc_quadruped_walk`、baseline logging、同一训练 seed0、clean vision64、size50m，原训练 SHA `48d71f1541721208322442f5a1d5977124f3e888`。固定选择300000与1000000训练动作阶段，相隔700000动作，没有按 return搜索更好 checkpoint。

| 实际训练动作阶段 | checkpoint SHA256 | 成功 optimizer更新 |
| --- | --- | --- |
| 300000 | `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9` | 74493 |
| 1000000 | `c9c893081485c8ad22468611d046bc5bf9fbe463d55996fda15cf85b2a0323fe` | 249493 |

清单核对各513050095 bytes、checkpoint hash、done/latest、evaluation metadata、更新counter、源 manifest/config，终点参数与 final_state一致。源原始进程 exit code不可获得，不能伪造exit0；完整产物与传输核查另记。`pickle.counters.actions` 是训练 policy 调用计数，不是训练环境动作预算；阶段依据 `evaluation_snapshot.json.actual_action_step`。

**两个阶段不是两个独立训练 replicate**；独立采集 episode 仅支持各冻结模型内的诊断区间，不能当跨训练 seed 泛化。正式源审计也不等于四组 M2 全部正式验收。清单只提交小型路径、哈希和元数据，参数文件不入 Git。

## GPU 和真实方法验收

实际运行根：`/data/Policy_Discrepancy/runs/dl-prevalidation-20261010-9e6da39/`，以下简称 `R`。运行目录名保留 r1，不替代每条 manifest/config 的实际 SHA。每条新 attempt保存 launcher/hash、开始结束时间、exit-code、真实资源与 CUDA/EGL probe、GPU采样和原始日志；失败不覆盖。

| 工作/实际物理 GPU | 状态与证据位置 | 验收范围 |
| --- | --- | --- |
| 300k checkpoint pilot采集 / GPU4 | exit0，`R/pilot-cp300000-attempt01/`，实际方法/执行 `9e6da39` | 2完整 episode，仅1 calibration/1 blind，非32-episode Stage-C |
| 上述 pilot score / GPU6 | exit0，`R/pilot-cp300000-score-attempt01/`，实际执行 `9e6da39` | 每episode4位置、H10，共8位置；工程/数值诊断，不能做完整筛选裁决 |
| 300k checkpoint完整采集 / GPU4 | exit0，`R/cp300000-collection-attempt01/` | 32完整episode、8 calibration/24 blind；仍待完整独立视觉与结果审计 |
| 1m checkpoint完整采集 / GPU5 | exit0，`R/cp1000000-collection-attempt01/` | 同样32完整episode、8/24分割；仍待完整独立视觉与结果审计 |
| 1m checkpoint完整score / GPU5 | exit0，`R/cp1000000-score-attempt01/`，实际执行 `1280bea`、方法代码 `9e6da39` | 每episode32位置、H10；参数冻结，完整分析及独立数组审计完成，科研筛选未通过 |
| 300k checkpoint完整score / GPU4 | exit0，`R/cp300000-score-attempt01/`，实际执行 `1280bea` | 对应32-episode完整采集/32位置，独立分析与审计完成；不以1m或pilot替代 |
| 1m checkpoint真实size50m paired / GPU7，attempt01 | 失败，`R/size50m-cp1000000-attempt01/`，实际执行 `1280bea` | helper工程失败、无optimizer update；不是方法失败或size50m通过 |
| 同一真实size50m paired / GPU7，attempt02 | 失败exit1，实际 `3721103` | probe读取参数梯度严格零已执行；局部梯度编译捕获CUDA数组违反transfer guard，无optimizer update，不放松guard |
| 同一真实size50m paired / GPU7，attempt03 | 失败exit1，实际 `55a527c`；五worker各自complete | off/alpha0/rho0状态、logging独立梯度/loss/latent、N非rep梯度等核查完成；各自单步可精确重放。但logging两个optimizer二阶状态hash不等，paired整体不通过 |

完整采集的具体执行 SHA 以各自 manifest为准；`9e6da39` 和 `1280bea` 的采集/score方法实现一致，资源 r2不重命名为科学修订。尚无 DL 18条短训练启动；size50m单步 replayed update验收即使后来通过，也不等于短训练或正式性能实验。

### r1真实裁决、视觉及工程失败

完整科学裁决与精确汇总见 [DL_R1_RESULTS.md](DL_R1_RESULTS.md)、[dl_r1_scientific_summary.json](dl_r1_scientific_summary.json)。两stage原analysis与全部64个score文件保持不变；当前代码只纠正“充分活跃但释放不足”误记inconclusive的分类，不改阈值。保护幅度未满足、label/cue不足、P对照不足分别记录，不能由修复代码宣布科学通过。

attempt03唯一logging差异为 `opt/state/2/1/rew/head/logits/bias` 和 `opt/state/2/1/val/head/logits/bias`，实际模型参数及独立总梯度完全相同。日志附加图影响优化器融合舍入是**待检验的工程解释**，尚不能作已证因果；不允许强制round状态或放宽严格hash。必要修复将独立编译只读logging诊断与训练核心分离，须证明参数/抽样/随机流不变，再以新attempt复验。

校准视觉证据在ignored `operational-evidence-r1/`：两stage各固定episode4/6/10的frame50/500/900、NPZ与来源hash、两张montage及逐像素JSON。root已实际查看18帧：背景仅替换天空，geometry/ground/边缘/影子保持；protected RGB精确相等。灰body保留姿态轮廓，实际移除外观纹理而非全部运动学线索。此有限QA不能担保全数据语义或排除所有背景尾部误保护。

同目录 `run-resources.json`（小型镜像 [dl_run_resources_r1.json](dl_run_resources_r1.json)）保存实际start/end/exit-code及资源证据hash、1秒GPU采样：300k/1m的32episode采集walltime分别341/335秒、score198/178秒，各采集峰值2585MiB、score2511MiB；attempt03峰值6665MiB、09:54:19–10:06:22 UTC，exit1。峰值为1秒采样观察值，不是连续上界。完整GPU数据、config/环境与UUID映射留服务器原目录。

### 独立长后果机会分支

协议见 [DL_OPPORTUNITY.md](DL_OPPORTUNITY.md)，`DL-opportunity-r1`；新seed20261012、每stage32episode/16预定位置、H100累计主后果，原H1信号/gate不变。只确认一次动作持续机会，不称闭环policy价值或新方法通过，任何结果均不启动短训练。原r1已公开结果不用作新未见数据。

precommit真实CPU H100审计84候选×独立短prefix，完整记录动作100步float32逐step对照采集精确、完整integration/task/counter重复最大误差0；这支持物理机制。20项诊断/opportunity CPU测试与6subtests通过（含真实tiny冻结scorer），实际来源55a527c加prefix/analyzer补丁；确定提交后须另存实测版本。新GPU采集/评分及机会结论尚待实际执行。

### DL pilot 已核查的小型证据

本机镜像为 `analysis/outputs/dl-prevalidation/bootstrap/`：`pilot-collection-complete.json`、`pilot-independent-integrity.json`、`pilot-independent-numeric-summary.json`、`pilot-mask-check.json`、`pilot-mask-montage.png` 和 `pilot-episode000.npz`。

- pilot两 episode各1001帧，collection参数不变；score/collection checkpoint与参数一致，plan/complete及两个score artifact哈希核对通过。prior logits、prior action/std和reward支持跨当前观测变体精确相同。
- 独立 NumPy float64复算对保存的float32信号最大差：`e`约8.09e-7，TwoHot entropy约5.36e-8；有限字段检查通过，物理重复最大误差0。记录的是实际误差，不新增或放宽验收容差。
- episode000 的 frame50/500/900背景替换中，被保护RGB逐像素相同；可变背景面积分别15.625%、12.5%、10.9375%。这是三帧pilot像素证据，不自动证明全数据语义保留，完整32-episode视觉审计仍待完成。
- pilot出现背景新增保护和mode扰动影响的具体示例：episode0/position1的背景A新增保护约0.03534；小KL mode扰动下 clean `e`最大变化约0.04011、保护量最大变化约0.00587。它们提示完整诊断应保留背景与mode反例；8位置不足以裁决整体筛选门槛，也不据此调参后继续称盲测。

### DL 真实设备绑定与资源历史

sv3 初期多次 SSH 在认证前关闭，失败证据 `bootstrap/sv3-ssh-20261010-145416.txt` 保留。15:30北京时间恢复；15:31:14快照实际 hostname=lyg0326、GPU4–7均A100-SXM4-80GB/81920MiB、0MiB/无计算图形进程。原 checkout干净、Python3.11.16、pip check通过、/data约1.1TiB可用；这是当时环境快照。

15:36:53–56北京时间在 sv3 GPU4 / UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23` 完成既有 `scripts/probe_formal_devices.py`。helper从 `ae8372e42a815824d2e4ec3a527bfb985a81972b` 导出，SHA256 `3954d6051287141c7bf27b6ce68da8a813f1b798357c216f44a1680bbaf54194`；PID1248755 的CUDA UUID/pmon C+G均只在GPU4，exit0，训练动作/Agent更新为0，probe随后结束。证据：服务器 `/data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/gpu4-device-probe.json`，本机 `bootstrap/gpu4-device-probe.json`。这只是首次绑定探针，不替代后续运行证据。

后续每条实际任务都另做该任务版本的 CUDA/EGL probe，绑定单个 `CUDA_VISIBLE_DEVICES` UUID、`CUDA_DEVICE_ORDER=PCI_BUS_ID`、物理 `MUJOCO_EGL_DEVICE_ID`、`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`；进程内CUDA0。不能沿用bootstrap旧空闲时间，或把GPU4绑定证据外推GPU5–7。r2 fresh resource记录以完整XML+pmon+UUID证明当时无可见作业，实际1MiB保留。17:39北京时间 sv3 GPU4–7均0MiB，是运行结束后的瞬时快照，不构成下一启动核查；GPU0/2未知其他作业未触碰。

XLA报告driver CUDA12.8比PTX编译器12.9.86旧，关闭并行编译；不改驱动/环境，实际耗时包含该影响。最低一张独占A100可串行推进，四张用于独立任务并行，无需多卡模型。六组×3seed×100k=180万动作，旧吞吐外推约20.4 GPU小时纯训练；诊断、编译、评价及新探针开销另计，尚非本轮实测预算或完成时刻承诺。

## 待完成清单

1. 完成必要logging分离修复及确定提交测试，在GPU7新attempt严格重验size50m B16/T64/context1、完整optimizer/模型状态、抽样/随机流、probe零梯度、局部缩放与单步重放，保留所有失败。
2. 新H100机会分支完成确定SHA、CPU完整参考/前缀复验，再在fresh GPU4/5完成各32新episode采集和16位置评分；审计float64真实序列、源hash和独立机会交集，不借旧r1数据。
3. 按H100唯一主窗逐stage裁决；不足停止该长窗对象，必要改进须另立有因果对象和独立数据的版本。核心gate、对照和压缩判据未通过前不能进入短训练。
4. 若DL-B/C及对照有效性满足，再在确定SHA、真实校准文件和完整冻结配置上启动18条100k短训练；否则保存失败/证据不足，论证必要修订并换版重验。尚无短训练或正式性能结论。
5. 逐项审计原五个目标，形成证据支持的正式候选、适用条件和剩余局限；缺真实gate/替代解释证据时不写“可正式实验准备就绪”。最后核对main/GitHub同步，报告最终文档SHA与各实际测试/运行SHA之间差异。

继续本次已授权任务，无需新建聊天或重新申请已有算力权限；不自动创建或发送其他 chat。模型来源核查、进程完成、工程通过和科研结论分别记录。
