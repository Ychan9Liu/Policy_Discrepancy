# DL 正常 reset 有限校准结果

2026-10-12，科学版本 `DL-natural-reset-calibration-r1`。实际执行、clean Linux 工程和独立审计完整 Git SHA **`aa9fb7f777a28e8a82d57c31f9e749b5912af316`**；规范固定在 [DL_NATURAL_RESET.md](DL_NATURAL_RESET.md)。状态：该有限 CPU 对象及独审验收完成，按规范停止。整体方法尚未完成，原 B 的 MC-inconclusive、旧对象停止裁决及正式 M2 协议保持。

## 本轮得到的有限事实

正确 production 来源的八个正常 reset0 中，**episode9 与18的闭 δ 近优控制集合不相交**；全部28对只有这一对满足固定存在性条件。其最小共同最坏 regret 为 `.2404058425460036`，比固定 δ `.2` 高 `.0404058425460036`。该余量是描述，未用作新的筛选阈值。

这说明：在原七首控制、幅度 `.25`、共同19步 normalized-zero 尾、H20 和该两个初态上，没有一个首控制同时达到各自 `.2` 近优。它为当前观测可能有用提供外部物理对象，**不证明 RGB 可辨、同 h/prior、q 可学习、actor 利用、Dt/K/第二信号有效、正常 rep 损伤或保护带来性能提升**。不能把一对状态当作总体覆盖、功效或完整方法验证。

## 实际执行和工程验收

只在 **sv3/lyg0326 Linux CPU，0 GPU** 执行，CUDA 不可见、MUJOCO_GL=disable、BLAS/OMP 单线程；无 JAX、checkpoint/model、renderer、actor、optimizer 或训练调用，没有本轮 GPU 占用核查，不引用旧快照为当前空闲保证。

- 主对象：01:51:08–01:51:15 北京时间，父 PID2001945、worker2001954；父完整入口 **6.6502385209314525秒**，外部 CLI **6.722968969028443秒**，exit0、未超时、worker terminal、owned-group cleanup、完整哈希链通过。180秒为硬限，非预计耗时。
- 八次官方原 seed fresh reset 与原完整 integration 都字节相同，实际最大误差 **0**，task/RNG/clock/reset/done 一致。八条原0→20参考共160步：原 raw→applied、F32 reward/flags exact；完整状态实际最大误差 **0**，沿原 `1e-8` 守卫。源只存 F32 reward，不声称 source F64 oracle。
- 新主1120步及 stored-applied 复放1120步完整完成，合计主对象 **2400实际 transition**；56条完整记录，无 earlyterminal/time-limit/reset/padding。新 F64 reward、flags、完整 integration/task/RNG/clock/reset/done 明确 byte-exact。
- 最终 Windows 43项：40pass/3POSIXskip，3.081秒；确定提交 Linux **43项全过、无skip，4.547秒**，外部测试4.770秒，pip check通过。fixture覆盖不同误差契约、闭δ及反例、白名单/可信payload/源退出链、owned超时/晚写和generic callback范围，不代替真实任务或方法效果。

实际源码在 `/data/Policy_Discrepancy/dl-worktrees/aa9fb7f` 干净 detached checkout；server canonical仍clean `d9be5574878d2c3f117cc6667be844f9a527eb84`，没有部署到sv1/sv2。本机canonical main及其原dirty两docs未改；本轮结果文档与GitHub状态以最终收据为准，不冒充执行SHA。

## 独立 CPU 审计

01:51:41–01:51:45，北京时间；父 PID2002143、worker2002144，父完整入口 **4.58881506556645秒**，外部 **4.671188522130251秒**，180秒有限对象/exit0/无timeout/full chain verified，**0 GPU**。

独立 math.fsum 复算全部 H20成本、regret、闭集合、28对 minimax、零侧与分类，连续绝对容差在真实输入前固定 `1e-10`，集合/整数/分类与余量零侧 exact。主/独审完整 analysis 数值最大差 **1.7763568394002505e-15**，由已镜像的两份小 JSON 联核；没有用容差修改 δ 成员。

另外完整执行 **1120 stored-applied 官方 CPU transition**，新 F64/F32 reward、discount/flags及完整状态全部 byte-exact。原 source退出链、source三元数据、全部32原字节、八cal完整初态与新记录及源码前后绑定通过。独审科学数组只读原八cal和新八episode，不开24blind、RGB或carry。

独审复用同一官方模拟器及已pin的恢复/快照原语，不称独立模拟器；八fresh reset和160源参考认证复用primary收据，**不称独审重新执行了它们**。两本轮真实对象共3520控制transition，fresh reset八次；数值/物理一致是工程证据，未验证模型信息或压缩机制。

## 全部八初态及28对

控制编号0–6依次为 zero、all+、all−、lift-LR+、lift-LR−、extend-diag+、extend-diag−。下表 F64 累计 reward 越大越好，展示六位小数；裁决使用原完整F64，不根据展示舍入或tie容差。normalized zero 的 lift applied 为 `.05`，不称零力矩。

| episode | zero | all+ | all− | lift-LR+ | lift-LR− | extend-diag+ | extend-diag− | range | 闭 Aδ 编号 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 9 | 10.916913 | 11.157818 | 10.701513 | 10.802798 | 11.029109 | 11.024856 | 10.805331 | 0.456305 | 1,4,5 |
| 14 | 9.059116 | 9.214385 | 9.056056 | 9.106260 | 9.022902 | 8.983941 | 9.133103 | 0.230444 | 0,1,2,3,4,6 |
| 15 | 2.030488 | 2.047026 | 1.978452 | 1.963775 | 2.086284 | 2.024738 | 2.010506 | 0.122509 | 0,1,2,3,4,5,6 |
| 18 | 5.837847 | 5.535375 | 6.144429 | 5.883790 | 5.773360 | 5.904023 | 5.756743 | 0.609054 | 2 |
| 20 | 0.309793 | 0.335317 | 0.298583 | 0.321751 | 0.299153 | 0.292470 | 0.343691 | 0.051221 | 0,1,2,3,4,5,6 |
| 24 | 4.825301 | 4.910359 | 4.852102 | 4.817639 | 4.842762 | 4.889720 | 4.759318 | 0.151040 | 0,1,2,3,4,5,6 |
| 30 | 2.283281 | 2.273302 | 2.248063 | 2.228779 | 2.343608 | 2.278011 | 2.302101 | 0.114828 | 0,1,2,3,4,5,6 |
| 31 | 3.077719 | 3.053432 | 3.104060 | 3.079164 | 3.077335 | 3.172844 | 2.983788 | 0.189056 | 0,1,2,3,4,5,6 |

仅9、14、18的range≥δ，其他五个初态七控制都在δ近优集合中，不能把“winner编号不同”称关键。完整原始成本、regret、集合和全部对存于 result.json，镜像及hash见下文。

| episode A | episode B | minimax regret − δ | 闭集合不交 |
|---|---|---:|---|
| 9 | 14 | −0.200000 | 否 |
| 9 | 15 | −0.160742 | 否 |
| 9 | 18 | 0.040406 | 是 |
| 9 | 20 | −0.191626 | 否 |
| 9 | 24 | −0.200000 | 否 |
| 9 | 30 | −0.129694 | 否 |
| 9 | 31 | −0.080588 | 否 |
| 14 | 15 | −0.160742 | 否 |
| 14 | 18 | −0.041671 | 否 |
| 14 | 20 | −0.191626 | 否 |
| 14 | 24 | −0.200000 | 否 |
| 14 | 30 | −0.129694 | 否 |
| 14 | 31 | −0.091875 | 否 |
| 15 | 18 | −0.092168 | 否 |
| 15 | 20 | −0.160742 | 否 |
| 15 | 24 | −0.160742 | 否 |
| 15 | 30 | −0.200000 | 否 |
| 15 | 31 | −0.138454 | 否 |
| 18 | 20 | −0.154892 | 否 |
| 18 | 24 | −0.141744 | 否 |
| 18 | 30 | −0.104455 | 否 |
| 18 | 31 | −0.131216 | 否 |
| 20 | 24 | −0.191626 | 否 |
| 20 | 30 | −0.158493 | 否 |
| 20 | 31 | −0.148779 | 否 |
| 24 | 30 | −0.134404 | 否 |
| 24 | 31 | −0.179361 | 否 |
| 30 | 31 | −0.134404 | 否 |

## 产物、资源及下一问题

服务器根目录 `/data/Policy_Discrepancy/runs/dl-natural-reset-20261012-aa9fb7f/`，主 `calibration-attempt01/`、独审 `independent-audit01/`、完整启动命令/外部终点/Linux测试/部署收据 `engineering/`。本机ignored镜像 `analysis/outputs/dl-natural-reset/dl-natural-reset-20261012-aa9fb7f/`，manifest列28文件930204字节全部hash复核；原source/checkpoint/replay未复制或入Git。一次本机打印统计表遇GBK不能输出unicode减号，验证已完成但helper exit1；ASCII重跑helper exit0，未重新执行科学对象或放宽规则。

关键hash：primary result `9a11da123f9743f3815f6a36bf3da26641900217708990830ef15c092d49c8be`；complete `678451d9a2b94807016e1be9bd84469d74941a5ba1d68b3df9740e4007b4f774`；audit `c76a6473617ecf10824efc0bcd8eebb4042075354cfcf74e4e0d65546d3ae7f0`。其余预算/计划/源码/源pins/所有八NPZ/收据由 mirror-manifest及完整artifact链提供。

**资源回答**：当前物理校准/统计/独审0GPU，已有规模checkpoint诊断1独占A100可串行，2卡仅用于相互独立步骤并行，不必占满用户分配的sv3物理GPU4–7。上一B约46秒/.013GPUh/采样峰2.45GiB只代表其对象；本轮CPU耗时与旧单步工程峰值均不能外推新训练显存或完整剩余GPU小时。正常压缩/短训练的组数、步数、配置尚未冻结，没有可信的整体GPU小时承诺；下一每个GPU对象再记录硬上限和实际用量。

**首选继续当前完整授权chat，先冻结下一完整模型/读出规范**：实际同h/prior及共同p样本、当前RGB合法来源、q相对共同历史的独立真实控制增量、共享F/候选第二信号及同容量D/K对照、whole-episode训练/留出/一次启用blind与停止规则。物理阳性不能直接授权读blind或启动gate；所有8cal联报，不能只挑9/18制作效果例子。normal rep还缺合法reset训练context、完整optimizer下的正常B/停止新增rep梯度R、重建history/重新拟合读出以排除prior赶上/坐标漂移，以及选择性保护和后续学习性能。上述建议尚未采用/执行，整个方法未宣告可正式实验。
