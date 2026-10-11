# DL 正确生产动作来源：结果与资源

版本 `DL-source-r1` / 实现 `DL-source-engineering-r2`，2026-10-11。

**已验证事实**：新的真实 production 来源完成，文件/接口独立审计及八条 calibration 的独立 CPU 物理复放通过。它修复了旧 diagnostic 绕过 Clip/Normalize 的来源问题，未检验第二信号、gate 或训练性能。旧结果按 [DL_ACTION_CONTRACT.md](DL_ACTION_CONTRACT.md) 限制在 `legacy_raw_suite` 对象，原数值和停止裁决保留。

## 1. 实际版本、来源与运行

实际采集、服务器 clean source 五项测试的完整 SHA 为 **`f2d24da1831da2ffca22d313e7f534c96fc0d3eb`**。该提交已 push 到 `origin/codex/dl-prevalidation` 并核远端；未合入 main。只部署 sv3 隔离 worktree `/data/Policy_Discrepancy/dl-worktrees/f2d24da`；原 `/data/Policy_Discrepancy/repo` 仍干净 `d9be5574878d2c3f117cc6667be844f9a527eb84`，未向 sv1/sv2 同步本实现。

| 项目 | 实际记录 |
| --- | --- |
| source 选择 | Agent 在用户完整前验证授权内冻结；不写成用户逐项批准或方法结论 |
| 模型/任务 | 原 baseline seed0、300000训练动作 checkpoint；size50m、`dmc_quadruped_walk`、vision64、proprioFalse、camera2、repeat1、sampled eval policy |
| checkpoint SHA256 | `2ab693c8ded4d8f2186bd489aa8b77498448ba2f2c331090159da4e5f08056b9` |
| 参数 digest | `14859c4e129b51aa488c10eab09f16d56fd110c76239b2c9072f415829d111c2` |
| 配置文件 SHA256 | `11428512fdc9b9e04a5384e419c5abcef93e5429db0a3327d05ccb0d4bd20445` |
| collection seed/计划 | 20261020；32完整episode、每条1001帧/1000实际动作；8 calibration＋24 blind，无替换 |
| calibration episode | `[9,14,15,18,20,24,30,31]`，按事前算法决定 |
| 位置 | 每episode预定16个H100 eligible位置；不按收益或D/K筛选 |
| 动作契约 | `production_clip_unify_normalize_v1`；实际 `DMC + main.wrap_env`，raw float32留模型carry，实际suite applied float64另存 |
| 资源 | sv3 / lyg0326 / 物理GPU4；UUID `GPU-9c070d5a-ac68-4c26-9522-a790ad874b23`；进程内CUDA0，EGL物理4，实际CUDA/EGL同卡probe及worker收据通过 |
| attempt02 | `/data/Policy_Discrepancy/runs/dl-production-source-20261011-f2d24da/cp300000-attempt02/`；worker PID1752617，已终止 |
| 启止（北京时间） | launcher 13:00:42–13:08:22；有界source supervisor 440.142416秒，exit0，未超时，worker terminal |
| 显存 | 1秒采样426点、观察峰值2585MiB；不是连续显存上界 |
| 硬上限 | 单卡3600秒，覆盖加载/JIT/采集/全部重演/写盘；这是上限，不是预计耗时 |
| 训练状态 | optimizer updates0；加载参数digest保持；updates/batches74493、训练policy calls18769保持；没有gate拟合或修改 |

source的每episode eval seed按固定计划设置、eval calls推进1000，最后eval seed3261730323；与加载checkpoint的eval seed3404107006/eval calls1001不同是合法收集行为，不能宣称全部counter不变。checkpoint的训练policy calls18769不等于300000环境训练动作。

环境为 Python3.11.16、JAX/jaxlib0.4.33、numpy1.26.4、dm-control1.0.48、MuJoCo3.15.0、elements3.22.2、ninjax3.6.3。完整配置和环境收据在 `collection/collection_plan.json`；CUDA_VISIBLE_DEVICES/UUID、MUJOCO_EGL_DEVICE_ID、实际pmon和graphics映射在本attempt的 resource/probe 和 complete 收据。旧CUDA12.8/PTX12.9.86并行编译警告保留，未改驱动。

## 2. 验收范围与独立性

collector对全部32条执行工程核对：总32032帧/32000实际policy动作；512个位置的100步 recorded-applied物理复放共51200步，逐transition float32 reward bytes、完整float64 integration/task/RNG/counters/reset与FromDM done一致，最大state error0。512位置在冻结source RGB、实际pre-policy BF16 carry及原eval key/counter上，实际公共policy的raw sample和after-carry hash逐字节相同。末帧raw/applied零占位明确 `action_executed=False`，不映射、不执行。

这些是collector重演证据；以下两个脚本独立检查其原产物，未把collector自报等同独立model重演。

| 独立检查 | 实际范围与结果 |
| --- | --- |
| NumPy/stdlib＋ml_dtypes artifact审计 | ml_dtypes仅注册BF16 dtype；全32文件原字节hash及Git文件/probe/配置/预算/监督收据绑定；只打开8 calibration NPZ，核raw/applied有序dtype映射、terminal mask、完整snapshot、actual BF16 carry/prevraw/逐叶hash/eval来源；passed，无模型/环境/GPU调用 |
| 官方CPU physics审计 | 独立写restore/map；全32字节hash，只读8 calibration；声明envseed的首次reset完整state/task/RNG与snapshot0一致；128 H100窗口/12800物理步，直接执行记录applied，F32 reward bytes/flags/task/RNG/counters/expected done exact，初始/恢复/逐步integration最大误差均0；15.014750秒，无JAX/render/GPU |

CPU物理审计没有执行实际FromDM wrapper，而是按官方timestep核saved done；实际wrapper执行由collector/真实CPU wrapper测试证明。独立审计没有独立运行政策或重渲染；冻结RGB政策重演不证明renderer或完整闭环逐位复现。盲测24条仅核原字节hash及工程收据，未打开其数组用于科学分析、未计算D/K/C/U或选择方法。

实际source/CPU结果不证明观测控制价值、第二信号增量、目标子集辨别、活跃压缩效果、S/P/W充分性或训练性能，**方法目标仍未完成，不启动短训练**。

## 3. 失败、修复和测试版本

attempt01保持原目录 `/data/Policy_Discrepancy/runs/dl-production-source-20261011-bb78e8c/cp300000-attempt01/`，执行 `bb78e8c625c141d862293d26fc32d5ca64a598cb`，12:53:35–12:54:46北京时间exit1，worker50.438587秒。首episode的host BF16 carry进入公共policy `_stack`时触发implicit transfer guard；未写episode NPZ/complete，无来源结果。不得把失败改称局部成功或续跑。

r2只在该重演入口按原 `local_sharding(agent.policy_mirrored)`显式device_put原dtype/结构/数值carry；未放宽guard、容差、seed、窗口或科学判据。attempt02从头采用同一已冻结计划，不替换episode。

- clean `bb78e8c`：sv3 action-contract3项8.971秒/source4项4.154秒及pip check通过；action八固定case实际wrapper有序字节映射与物理核对、实际tiny模型raw/carry一致。
- clean `f2d24da`：sv3 source5项10.805秒；新增实际公共tiny Agent/BF16/strict disallow下host carry重演通过。本机同5项6.722秒。原action helper/tests未改，沿用bb78确定版本结果。
- 独立physics自检：真实CPU1600步、最大误差0；reward/applied/integration/NaN/done五类篡改均拒绝。fixture不代替本节真实来源验收。

这份结果及候选记录是后续文档变化，不改变上述实际采集/测试SHA；最终交接文档提交与f2之间的差异、push/sync状态须按实际Git另记。

## 4. 收据与复核命令

attempt02完整数据保留服务器 `collection/episode_*.npz`；checkpoint/replay/大数组不提交Git。小收据本机镜像 `analysis/outputs/dl-action-contract/f2d24da/attempt02/`；失败镜像为同根 `bb78e8c/attempt01/`，CPU收据为各SHA的 `cpu/`。

| 文件/脚本 | SHA256 |
| --- | --- |
| `collection_plan.json` | `633efe28d551d5793ac8f0ed5f0e36e937e7f1af3c8edefc1d9f5fa278146fcc` |
| `collection_complete.json` | `d155f577d19d2bbc051ec56b95f15e26afc774c130c3c9728a7fe599f295f3cf` |
| `supervisor_complete.json` | `a8d9dbc0cd90a5d66361b2082e8ae3a51322d2d450ce9dd13ddbe763209ead75` |
| artifact审计器 | `ec7207ca797bad5ac81569ec3c19209778810301b2b511aa985c826fbcd2f6d5` |
| `independent-source-audit-r1.json` | `8f93788ca212118139fab135036fd47ccbad79f04a34139b5dfa7d2dd26f7516` |
| physics审计器 | `459446a3a7df704d843e952c6fbe94a962bc76a510d470519877751a82a0df65` |
| `independent-physics-source-audit-v1.json` | `8ed3528fae18dc7a3043c767da87f873642a40a1a028ce86247f0521c3264722` |
| 外部launcher | `0fa8c9fbac0b812337b7d4130b2b58d037b2ec9230c7c3b7a1708b85f5711bd9` |

launcher与两个独立审计器在服务器 `/data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/`；审计器本机源在ignored `analysis/outputs/dl-action-contract/`。完整启动命令由launcher/hash保留；source实际入口：

```bash
/data/Policy_Discrepancy/envs/dreamer/bin/python scripts/dl_production_source.py \
  --checkpoint /data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/sources/cp300000-agent.pkl \
  --config /data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/sources/config.yaml \
  --output /data/Policy_Discrepancy/runs/dl-production-source-20261011-f2d24da/cp300000-attempt02/collection \
  --resource-record /data/Policy_Discrepancy/runs/dl-production-source-20261011-f2d24da/cp300000-attempt02/resource.json
```

复核使用同一实际环境，仅CPU；输出须新路径，不覆盖本次收据：

```bash
MUJOCO_GL=disable CUDA_VISIBLE_DEVICES= /data/Policy_Discrepancy/envs/dreamer/bin/python \
  <independent-source-audit-r1.py> --dataset <attempt02>/collection \
  --expected-source-sha f2d24da1831da2ffca22d313e7f534c96fc0d3eb \
  --repository /data/Policy_Discrepancy/repo --output <new-artifact-audit.json>
MUJOCO_GL=disable CUDA_VISIBLE_DEVICES= /data/Policy_Discrepancy/envs/dreamer/bin/python \
  <independent-physics-source-audit-v1.py> --dataset <attempt02>/collection \
  --expected-source-sha f2d24da1831da2ffca22d313e7f534c96fc0d3eb \
  --output <new-physics-audit.json>
```

## 5. 资源结论与未决事项

**实测资源**：本次正确生产来源约440.14秒/2585MiB，即source supervisor约0.12226 GPU小时；完整launcher含fresh/probe约460秒。此前真实size50m完整五组单步工程检查668秒/6675MiB，见 [DL_VALIDATION_LOG.md](DL_VALIDATION_LOG.md)。这些是已测单项，不能相加后宣称剩余完整验证预算。GPU4在13:09:36北京时间核查已释放；GPU4–7均0MiB、compute/pmon无作业；GPU0/2未知作业未干扰。是时间快照，不是持续空闲保证。

**资源建议**：最低1张独占A100串行；2张可并行确实独立的诊断/工程环节，4张分配足够，无需占满或多卡模型。仅此A100环境已验；更小显存卡可行性未测。CPU文档/统计/物理审计不用GPU。单卡3600秒是每个已冻结source/诊断的硬上限，不是所有前验证耗时；旧20.4 GPU小时是旧条件训练外推，不能用于修订方法当前总预算。

**尚未冻结候选**：[DL_FEEDBACK_DIAGNOSTIC.md](DL_FEEDBACK_DIAGNOSTIC.md) 论证一次当前RGB省略、未来正常各自闭环。其32状态×4paired RNG/H100、奖励dtype、key派生、terminal规则、统计及3600秒预算尚须根Agent最终审查/冻结与工程验收，不由source通过自动执行。平均机会阳性不替代selective目标和第二信号增量；非正/不确定也不能否定所有后验保护。

首选继续本聊天的完整授权任务：先论证并冻结一个确实解决反馈补偿/控制价值缺口的有限对象，再验证其实际policy/RNG/physics语义；同时明确它与独立第二信号辨别之间的证据关系。当前可立即做候选审查，GPU运行依赖确定协议与代码验收；短训练仍依赖目标子集、增量和有效对照证据，不沿用旧门强行启动。总体目标保持未完成。
