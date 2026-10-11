# DL 固定状态的任务语义与机制边界

版本 `DL-task-semantics-r1`，2026-10-11。本文是完整前验证授权内的Agent设计与证据记录，不是用户逐项确认的新方法，也不改正式M2协议。当前先冻结只读CPU查询对象，实际结果待下文补记。科学来源和此前负结果见 [DL_FEEDBACK_RESULTS.md](DL_FEEDBACK_RESULTS.md)。完整方法目标仍未完成。

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
| 完整性 | 包版本及官方task模块hash；restore/查询前后完整integration、task/RNG、clock/reset不变；非有限或输入不符失败 |
| 资源 | sv3 CPU，GPU使用0；120秒POSIX硬限加monotonic核查，只终止自身进程 |
| 输出 | 新的 `task-semantics-qa01/` 目录；保留32连续量、来源/hash和限制；已有目录拒绝覆盖 |

这是看到原负结果之后的探索性语义核查，不宣称独立科学重复或预注册验证。对象按原固定坐标完整查询，不按D/K/e/U或图像外观筛点，不设critical/fallen/should-protect阈值。privileged physics量只供离线诊断，不能悄悄加入vision-only gate输入。

实现复用既有独立CPU物理审计器的authority/restore/statecompare，并绑定其脚本hash。因此本查询是派生实现，不是第二后端或新增独立物理审计。仅恢复状态的自检不替代后续真实动作对照。

已冻结查询器 [dl_task_semantics.py](../analysis/dl_task_semantics.py)，SHA256 `ddf89d47521965387da680e5b0cd2bea455e2e5ec42f50022ac13812bc93ea64`；依赖 [dl_feedback_physics_audit_v1.py](../analysis/dl_feedback_physics_audit_v1.py) SHA256 `54e5b5a24fc9971ef8f454979a1ee9a9bc47c536f52bcac42b4c50efd96ef1ab`。两份源码与此前ignored原件字节一致，纳入Git以使后续查询可复现；后者只是归档既有审计器，不新增独立审计结论。新代码不改变Dreamer或gate。

本机官方quad synthetic active snapshot的九项selftest通过0.438秒，包括query写入单ULP和消费task RNG的反例；不执行reset/step/render，不是自然状态验收。实际入口另用外部 `timeout 120` 覆盖import和写盘，保留内部POSIX/monotonic检查。

## 3. 后续决定需要什么证据

下一项首选对象是独立的有限动作关键性诊断，先冻结动作集合、后续控制定义、真实成本、配对规则、预算和停止规则，再执行。当前尚未冻结，不因逻辑可能存在正子集而消耗blind。

需要分别回答：

1. 同一有限动作集合的真实排序或代价是否随可辨识任务状态改变？低奖励、姿态差异或控制敏感性本身都不足以定义关键线索。
2. 合法64×64 RGB及正常历史能否获得该区别？当前RGB相对已有历史是否有新增信息？单帧看见本体不证明速度或隐含状态可辨。
3. q latent是否保存该信息、现有actor是否利用？信息可学习性与当前actor能力分开裁决，不把一次q/p样本收益当精确标签。
4. 若actor已利用，是否有active rep造成能力损伤的实际学习证据？若actor未利用但信息可学，是否值得另立学习介导对象？两者都不能直接沿用旧失败gate或短训练入口。

保持原有限平均对象停止：不追加原对象的样本/RNG/H/checkpoint，不用新对象覆盖其负裁决。新信号、gate、压缩学习配置、强度/对应关系/reward-only对照均仍需独立冻结与验收。

## 4. 实际执行与结果

待实际CPU查询及完整性审查后填写。方法有效、当前信息增量、actor能力和压缩学习作用当前均未由本查询建立。
