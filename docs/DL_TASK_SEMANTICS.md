# DL 固定状态的任务语义与机制边界

版本 `DL-task-semantics-r1`，2026-10-11。本文是完整前验证授权内的Agent设计与证据记录，不是用户逐项确认的新方法，也不改正式M2协议。**固定32初态的实际CPU查询与自检已通过，零环境动作/模型/GPU；它只核任务语义，不证明关键信息或方法效果。** 科学来源和此前负结果见 [DL_FEEDBACK_RESULTS.md](DL_FEEDBACK_RESULTS.md)。完整方法目标仍未完成。

## 1. 两条需要分开检验的路径

**即时行为路径**：共同历史之外的当前信息改变现有actor的动作，并改善有限真实控制后果；减少对这项信息的压缩可能保留这种能力。现有actor利用信息是这条具体路径的前置，仍不充分证明训练保护有效。

**后续学习路径（工作假设）**：当前后验保存可学习的任务信息，即使现有actor还未使用，减少有害压缩也可能改善后续actor学习。当前frozen q相对p没有收益，不是这条路径失败的必要证明。不能把一次观测省略替代训练中的表示变化。

两条路径均未获得完整支持。第二条至少还需要：外部控制关键性；q在历史之外保有可学习信息；原free-nats活跃的正常训练确实损伤该信息；保护能保留它；后续actor学习确有收益。只做posterior-head KL下降、固定readout变差，可能是人工删信息或坐标漂移，不能替代正常训练的因果证据。readout应允许等量重新拟合，并与功能变化、OOD和正常其他loss共同检查。

**理论局限**：若actor尚未依赖有用latent，Dt可以为零而raw K非零；Dt必要门会漏掉这类信息。raw K又不能区分可学任务信息与无关新奇。是否保留Dt为必要门、引入阶段/可靠性机制，当前均未决定；不得直接把K越大解释为越该保护。

## 2. 本次CPU对象和必要性

先核实已有图像中的姿态推测及任务覆盖，避免因目测足部朝上就宣布跌倒、关键状态或actor失效。连续官方任务量只能描述状态，不能证明动作排序、当前RGB的信息增量或actor利用。

| 项目 | 本次固定范围 |
| --- | --- |
| 输入 | production `DL-feedback-r1` / engineering-r2 attempt02；实际运行SHA `0d49c589b525ba0c26f9f85bf52c23573dc19507` |
| 位置 | episode `[9,14,15,18,20,24,30,31]` × positions `[1,300,600,900]`，全部32初态 |
| 初态 | 每位置固定pair0/q；核全部八分支的完整初态一致 |
| 访问 | 先核authority、固定来源和全部8个calibration artifact hash，再读取可信记录；不打开24blind数组 |
| 官方任务 | `quadruped.walk`，原suite/任务RNG、step/reset及完整MuJoCo integration snapshot |
| 操作 | `mj_setState`、`forward`和查询；不执行环境transition、控制、模型、渲染或优化器 |
| 查询 | torso upright、局部三维速度、desired speed、官方upright/move reward分量、乘积与task reward、物理time、step count和remaining clock |
| 完整性 | 包版本及官方task模块hash；restore与保存snapshot按原1e−8匹配（实测0），查询前后完整integration/task/RNG/clock/reset严格字节不变；非有限或输入不符失败 |
| 资源 | sv3 CPU，GPU使用0；120秒POSIX硬限加monotonic核查，只终止自身进程 |
| 输出 | launcher要求新的 `task-semantics-qa01/` 目录；查询器拒绝已有output文件，允许同一目录保存selftest；保留32连续量、来源/hash和限制 |

这是看到原负结果之后的探索性语义核查，不宣称独立科学重复或预注册验证。对象按原固定坐标完整查询，不按D/K/e/U或图像外观筛点，不设critical/fallen/should-protect阈值。privileged physics量只供离线诊断，不能悄悄加入vision-only gate输入。

实现复用既有独立CPU物理审计器的authority/restore/statecompare，并绑定其脚本hash。因此本查询是派生实现，不是第二后端或新增独立物理审计。仅恢复状态的自检不替代后续真实动作对照。

已冻结查询器 [dl_task_semantics.py](../analysis/dl_task_semantics.py)，SHA256 `ddf89d47521965387da680e5b0cd2bea455e2e5ec42f50022ac13812bc93ea64`；依赖 [dl_feedback_physics_audit_v1.py](../analysis/dl_feedback_physics_audit_v1.py) SHA256 `54e5b5a24fc9971ef8f454979a1ee9a9bc47c536f52bcac42b4c50efd96ef1ab`。两份源码与此前ignored原件字节一致，纳入Git以使后续查询可复现；后者只是归档既有审计器，不新增独立审计结论。新代码不改变Dreamer或gate。

本机官方quad synthetic active snapshot的九项selftest通过0.438秒，包括query写入单ULP和消费task RNG的反例；不执行reset/step/render，不是自然状态验收。实际入口另用外部 `timeout 120` 覆盖import和写盘，保留内部POSIX/monotonic检查。

## 3. 后续决定需要什么证据

下一项是有限动作与接口信息诊断，交叉审查后的冻结规格见 [DL_ACTION_INFORMATION.md](DL_ACTION_INFORMATION.md)。当前尚未实现或运行，不因逻辑可能存在正子集而消耗blind。

需要分别回答：

1. 同一有限动作集合的真实排序或选择相关代价是否随可辨识任务状态改变？若所有控制奖励同比缩放、同一动作始终近优，仍不证明需要辨别状态；低奖励、姿态差异或控制敏感性本身不足以定义关键线索。
2. 合法64×64 RGB及正常历史能否获得该区别？当前RGB相对已有历史是否有新增信息？单帧看见本体不证明速度或隐含状态可辨。
3. q latent是否保存该信息、现有actor是否利用？信息可学习性与当前actor能力分开裁决，不把一次q/p样本收益当精确标签。
4. 若actor已利用，是否有active rep造成能力损伤的实际学习证据？若actor未利用但信息可学，是否值得另立学习介导对象？两者都不能直接沿用旧失败gate或短训练入口。

保持原有限平均对象停止：不追加原对象的样本/RNG/H/checkpoint，不用新对象覆盖其负裁决。新信号、gate、压缩学习配置、强度/对应关系/reward-only对照均仍需独立冻结与验收。

## 4. 实际执行与结果

实际执行和Linux自检SHA均为 `57270c7b92b8fd9da2bce5d1d6d9f377229564e2`；两份tracked查询源码与原ignored脚本字节相同。仅部署sv3 clean独立工作区 `/data/Policy_Discrepancy/dl-worktrees/57270c7`。canonical repo保持clean `d9be5574878d2c3f117cc6667be844f9a527eb84`，旧0d49c58执行工作区未改动，sv1/sv2未同步。

2026-10-11 19:20:16–19:20:29北京时间，launcher exit0。Linux selftest九项通过0.435633秒，完整查询32状态通过12.221363秒；外部timeout120及内部POSIX/monotonic启用，未超时。全部256分支初态绑定、32官方reward分量及F32初obs编码通过；restore/query最大integration误差均0，查询前后完整integration/task/RNG/clock/reset字节不变。GPU/环境transition/reset/control/render/model调用均0。root另核输出字节hash、完整episode/position顺序、clock、所有32前后hash及退出/clean状态；这是收据审查，不是第二次官方查询执行。

本次结果目录：

```text
/data/Policy_Discrepancy/runs/dl-feedback-20261011-0d49c58/cp300000-attempt02/task-semantics-qa01/
  task-semantics-v1.json / selftest.json
  query.log / selftest.log / input-sha256.txt / code-sha256.txt
  git-sha.txt / git-status-after.txt / python-version.txt
  start.txt / end.txt / exit-code.txt / output-sha256.txt
```

小收据完整镜像在ignored `analysis/outputs/dl-feedback/0d49c58/cp300000-attempt02/task-semantics-qa01/`。结果JSON SHA256 `61a6edbad985f033601bc4f6242e13c07fbb1b0d307840c874f8fe543459217a`；Linux selftest `52c792ba5003ffe473329a97f70d0d78a14b6ad1ac37f5df3284c94da72daf51`；launcher `f09bdd6ce432248cb01234174d012e9d00e5a351315dcb443c84f1cda8c762ef`。launcher独立比对既有plan/result/complete的公布hash，再调用查询器，不只依赖metadata中写着旧SHA。增量Git bundle SHA256 `9b05341899ac36fa2a49c3117ef919cb998c0a994ca25bf2decc41f5bf582dae` 已核传输/verify；无需GitHub可完成确定版本的sv3同步。

全32点连续量范围：torso upright `[-0.999956434,0.999024578]`，local vx `[-0.192920138,1.598224940]`，task reward `[0.000014827,0.999512289]`。固定顺序 `[1,300,600,900]` 的部分主要量如下；完整三维速度、分量、snapshot/hash以JSON为准。

| episode | 四位置torso upright（显示六位） | 四位置local vx（显示六位） |
| --- | --- | --- |
| 9 | .685144 / .994375 / .993971 / .942463 | −.048034 / .125946 / .701067 / .218448 |
| 14 | .676010 / .994603 / .933644 / .985232 | −.032405 / .620582 / .551375 / .909640 |
| 15 | −.651412 / −.991529 / −.999956 / −.755085 | −.001795 / .224438 / .180652 / .737531 |
| 18 | .927596 / .975004 / .997398 / .978901 | .081596 / .598911 / .490167 / .772117 |
| 20 | .070724 / −.971125 / −.892356 / −.882815 | −.118559 / .102062 / .066771 / .141407 |
| 24 | .593561 / .969891 / .999025 / .966037 | −.081466 / .825526 / 1.598225 / .771280 |
| 30 | −.621252 / −.988113 / −.999506 / −.983853 | −.008518 / .188399 / −.022481 / −.192920 |
| 31 | .121756 / −.940907 / −.837295 / −.998686 | .090422 / −.073530 / .083600 / .077221 |

**有限事实**：覆盖同时包含接近正立、局部前向速度达到walk目标的状态，以及torso z轴接近朝向世界负z的状态；后者与此前图像的反转外观一致。不能由这些量判断足部接触、可恢复性、正确动作或当前信息增量。也不能把整个checkpoint概括成“actor不会任务”，或把低reward状态直接当保护目标。

**执行命令**（W为上述57270c7工作区，F为上述原feedback目录，R为新task-semantics-qa01，P为 `/data/Policy_Discrepancy/envs/dreamer/bin/python`）：

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu MUJOCO_GL=disable timeout 120 "$P" "$W/analysis/dl_task_semantics.py" --physics-auditor "$W/analysis/dl_feedback_physics_audit_v1.py" --self-test --output "$R/selftest.json"
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu MUJOCO_GL=disable timeout 120 "$P" "$W/analysis/dl_task_semantics.py" --physics-auditor "$W/analysis/dl_feedback_physics_audit_v1.py" --dataset "$F" --expected-feedback-sha 0d49c589b525ba0c26f9f85bf52c23573dc19507 --output "$R/task-semantics-v1.json"
```

本次没有资源占用查询，未申请GPU；上一张19:06:18北京时间快照的GPU4–7均空闲不作为持续保证。科学目标、信息readout、有限动作诊断和学习机制仍未验收。新动作/读出规格已交叉审查冻结，尚未实现或运行。

为保持上述源码SHA256，新增 `.gitattributes` 仅对两份新归档Python文件指定 `text eol=lf`。实际Windows `core.autocrlf=true/core.eol=crlf` 下用checkout-index导出到新ignored目录，两个hash均与公布值一致；不全仓库renormalize，不改执行代码或科学语义。
