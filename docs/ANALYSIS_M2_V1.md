# 05 · M2 v1 单训练 seed0 分析

日期：2026-10-12，Asia/Shanghai。状态：**16条全部完成协议产物验收并纳入；05复算、门控表和图形完成。** 这是正式任务的单seed探索性结果，不是跨seed方法有效性或训练稳定性结论。本轮不启动训练、不调参、不改冻结协议或训练源码。

## 结论与证据层级

- **已验证产物事实**：sv2的最终16条审计于2026-10-10 22:59:10完成；05于2026-10-12只读复核。每条100万训练动作、249493成功更新、21点各10完整episode；总计336点、3360episode、3991888成功更新。正式c均有效，工程数据/工程c、失败后处理和重复记录未纳入。
- **描述性比较**：quadruped_walk的Dt在主、次指标均高于三个对照；quadruped_run高于baseline/shuffle，低于constant；reacher_hard主指标未超过baseline/shuffle，次指标高于baseline/constant但低于shuffle；hopper_hop的Dt全程回报很低，两个指标均低于所有对照。总体AUC略高而末段均值更低，没有一致的四任务M2支持。
- **门控事实及解释边界**：Dt在raw rep KL>1位置实际减弱，quadruped两任务更明显。不能把“不胜constant”泛化为所有gate无效，也不能把walk单seed正向结果写成细粒度任务信息因果保护。全程作用强度和数据/参数轨迹没有被constant或global shuffle精确匹配。
- **助手建议，未批准协议**：02优先设计原四组的多训练seed复验，检验walk优势与hopper退化能否重复；再依据机制目标决定条件shuffle。具体seed、任务覆盖、预算、判据、配置、资源仍由用户/02/03确定。DtLatch仍是未批准候选，本结果不授权实施。

## 事实来源与版本核对

必读来源已核对：AGENTS、CHATS、STATUS、RESEARCH（含未提交DtLatch候选）、METHOD_SPEC、EXPERIMENTS C/C.1、CODE_PROTOCOL_V1、FORMAL_CONSTANT_SHUFFLE_PD1、FORMAL_BASELINE_DT_PD1、analysis/README与启动HANDOFF。启动文档仍写“待终点”，以本次读取的服务器原始终点记录补齐，不自动回退本机HEAD。

服务器终点审计：`/data/Policy_Discrepancy/repo/analysis/outputs/formal-four-groups-pd1-20261010-final-attempt01/`，含RUN_INDEX、completion、return-audit六表、audit与tables。completion的verified_run_count=16、full_four_groups_complete=true；表格SHA256=`41fa0030b8ec27e11930c083f0cbf78a0bcdc0a258d3a6c8071f6ba42da2dd87`。

- **T，逐条实际训练SHA**：`48d71f1541721208322442f5a1d5977124f3e888`。16条各自manifest、index、final_state完全一致；这是核验后的结果，非沿用启动交接假设。sv2仓库当前仍为该SHA且干净。
- **A，逐条最终外部审计SHA**：`d239c3fb7e5b77ddf205fe994c46c2e5a6acce6a`。已将Git对象中scripts/audit_formal_pd1.py的SHA256与服务器实际部署脚本和completion源hash逐字节核对：`7a990107ccb2c390a1cc78456e6042f621682bc37500775c56695b235d5938c0`。04统计器本身从服务器训练checkout执行，不能把其tool_commit误当外部审计版本。
- baseline/Dt较早attempt02审计源`a8b62436827721c49a77af55663dbb5225e1ed39`，旧completion源hash=`c33f0179756a8c49a8e9fa9a0c577984472e9e991896280a2c1e35300abac65f`，与Git对象一致。最终16条均再次由A审计。
- **05实际分析及测试SHA**：`ad325482a46b0c9dd3c3709cccca6a63bea9ba5d`。在该确定提交执行正式analysis-final与19项单元测试，均通过。后续交付提交仅增加本报告/STATUS/EXPERIMENTS结果索引；完整最终SHA记在本轮HANDOFF/DELIVERY及最终回复，避免文档自引用。无训练源码变化。
- 本机初始HEAD=`595e982f1a3939775879728009e2105ee516e86d`；已有RESEARCH/STATUS未提交候选归档保留。本轮STATUS只提交05新增段及日期，不混入原候选改动。本轮只本机提交，未推送或同步源码到服务器；sv1/sv3未访问/同步。

## 16条实际运行与完成状态

| 任务 | 组 | run ID（各自 attempt01） | sv2物理GPU | 训练/最终审计版本 | 状态 |
| --- | --- | --- | ---: | --- | --- |
| hopper_hop | baseline | [formal-m2-v1-pd1-seed0-20261009:dmc_hopper_hop:logging:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-seed0-20261009-dmc_hopper_hop-logging-01) | 0 | T / A | 验收完成，纳入 |
| quadruped_run | baseline | [formal-m2-v1-pd1-seed0-20261009:dmc_quadruped_run:logging:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-seed0-20261009-dmc_quadruped_run-logging-01) | 1 | T / A | 验收完成，纳入 |
| quadruped_walk | baseline | [formal-m2-v1-pd1-seed0-20261009:dmc_quadruped_walk:logging:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-seed0-20261009-dmc_quadruped_walk-logging-01) | 2 | T / A | 验收完成，纳入 |
| reacher_hard | baseline | [formal-m2-v1-pd1-seed0-20261009:dmc_reacher_hard:logging:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-seed0-20261009-dmc_reacher_hard-logging-01) | 3 | T / A | 验收完成，纳入 |
| hopper_hop | dt | [formal-m2-v1-pd1-dt-seed0-20261009:dmc_hopper_hop:dt:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-dt-seed0-20261009-dmc_hopper_hop-dt-01) | 4 | T / A | 验收完成，纳入 |
| quadruped_run | dt | [formal-m2-v1-pd1-dt-seed0-20261009:dmc_quadruped_run:dt:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-dt-seed0-20261009-dmc_quadruped_run-dt-01) | 5 | T / A | 验收完成，纳入 |
| quadruped_walk | dt | [formal-m2-v1-pd1-dt-seed0-20261009:dmc_quadruped_walk:dt:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-dt-seed0-20261009-dmc_quadruped_walk-dt-01) | 6 | T / A | 验收完成，纳入 |
| reacher_hard | dt | [formal-m2-v1-pd1-dt-seed0-20261009:dmc_reacher_hard:dt:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-dt-seed0-20261009-dmc_reacher_hard-dt-01) | 7 | T / A | 验收完成，纳入 |
| hopper_hop | constant | [formal-m2-v1-pd1-constant-seed0-20261010:dmc_hopper_hop:constant:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-constant-seed0-20261010-dmc_hopper_hop-constant-01) | 0 | T / A | 验收完成，纳入 |
| quadruped_run | constant | [formal-m2-v1-pd1-constant-seed0-20261010:dmc_quadruped_run:constant:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-constant-seed0-20261010-dmc_quadruped_run-constant-01) | 1 | T / A | 验收完成，纳入 |
| quadruped_walk | constant | [formal-m2-v1-pd1-constant-seed0-20261010:dmc_quadruped_walk:constant:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-constant-seed0-20261010-dmc_quadruped_walk-constant-01) | 2 | T / A | 验收完成，纳入 |
| reacher_hard | constant | [formal-m2-v1-pd1-constant-seed0-20261010:dmc_reacher_hard:constant:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-constant-seed0-20261010-dmc_reacher_hard-constant-01) | 3 | T / A | 验收完成，纳入 |
| hopper_hop | shuffle | [formal-m2-v1-pd1-shuffle-seed0-20261010:dmc_hopper_hop:shuffle:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-shuffle-seed0-20261010-dmc_hopper_hop-shuffle-01) | 4 | T / A | 验收完成，纳入 |
| quadruped_run | shuffle | [formal-m2-v1-pd1-shuffle-seed0-20261010:dmc_quadruped_run:shuffle:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-shuffle-seed0-20261010-dmc_quadruped_run-shuffle-01) | 5 | T / A | 验收完成，纳入 |
| quadruped_walk | shuffle | [formal-m2-v1-pd1-shuffle-seed0-20261010:dmc_quadruped_walk:shuffle:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-shuffle-seed0-20261010-dmc_quadruped_walk-shuffle-01) | 6 | T / A | 验收完成，纳入 |
| reacher_hard | shuffle | [formal-m2-v1-pd1-shuffle-seed0-20261010:dmc_reacher_hard:shuffle:attempt01](https://wandb.ai/ychan9liu-tsinghua-university/PD_1/runs/formal-m2-v1-pd1-shuffle-seed0-20261010-dmc_reacher_hard-shuffle-01) | 7 | T / A | 验收完成，纳入 |

各行完整训练/审计SHA、run/attempt ID、原始与本机目录、W&B URL、完整manifest/config、GPU UUID/CUDA/EGL映射、输入哈希、原退出码、原状态和最终审计状态，逐条记录于`analysis-final/run_manifest.json`与CSV。RUN_INDEX原state仍是pending字段；05依据已完成的独立completion写accepted，不改原索引。全部attempt01从头训练，没有择优保留训练尝试。

四个服务器原始根目录为：

| 组 | 原始根目录（下接task/attempt） |
| --- | --- |
| baseline | `/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-seed0-20261009/<task>/logging-attempt01` |
| Dt | `/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-dt-seed0-20261009/<task>/dt-attempt01` |
| constant | `/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-constant-seed0-20261010/<task>/constant-attempt01` |
| global shuffle | `/data/Policy_Discrepancy/runs/formal-m2-v1-pd1-shuffle-seed0-20261010/<task>/shuffle-attempt01` |

共同配置：clean纯视觉64×64 RGB、size50m、seed0、repeat1、batch16×64、replay_context1、16训练env/1串行评价env、train_ratio256、CUDA/bfloat16、A100-SXM4-80GB；free_nats1、dyn/rep scale1/0.1、ac_grads=false、reward_grad/repval_grad=true。实际baseline v=1；Dt alpha20；constant取对应正式c；shuffle同次自身完整batch/time源权重全局置换、独立key。prealloc=False、JSONL+W&B project PD_1、无scope。逐任务完整配置排除mode/c/frozen_c_file/logdir后四组精确一致；环境版本与各流派生声明、所有评价seed映射一致，未从baseline checkpoint分叉。

环境：sv2/lyg0360，Python3.11.16、JAX/JAXlib0.4.33、NumPy1.26.4、dm-control1.0.48、MuJoCo3.15.0、W&B0.30.0；原dreamer依赖锁hash=`cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22`。完整环境在逐条manifest，设备均单logical CUDA0，CUDA_VISIBLE_DEVICES为对应物理卡UUID，MUJOCO_EGL_DEVICE_ID为对应EGL物理枚举，复用04逐PID实际映射证据。

## 完整性与修复影响

05从每条原始update_receipts流核对ID连续0..249492、动作开始stamp单调且与每点评价update ID对应；无重复、invalid0。训练动作字段1000000包含预填充，第一更新开始2032动作；reset另记，不计动作预算；每条评价动作210000，等于210个长度1000的非reset完整episode之和。计步语义复核对应训练源码，不声称有逐转移训练动作轨迹可重新计数。

评价网格严格0/50000/…/1000000；每点唯一schema v2行、10个预定episode ID、显式complete、return/length/seed与固定snapshot ID/update一致；无重写或冲突。05只读核对336份snapshot metadata、latest/done及服务器agent.pkl存在，全部步号与更新号匹配，复用04完成审计。未再次加载大参数payload或独立哈希全部参数；这限制了参数内容的独立验证层级，不影响冻结统计复算完整性。

已记录修复均在训练外部：baseline监督按实际作业判断占用并接管旧PID；W&B数字镜像末位舍入导致旧收尾失败，后改为规范化精确JSON字符串+SHA256传输。科学源值/容差、训练/梯度/RNG/冻结配置不变；d239相对a8仅补完整配置精确字段。前三baseline原始exit code不可回收，保留null，不伪造exit0。旧失败和旧表仍留在服务器，attempt02为后处理尝试而非训练重跑。训练源与已验收工程修复的语义/渲染边界见CODE_SIZE50M_REPAIR。

2026-10-12 12:32:22只读W&B交叉检查16条均finished，exact_values_json/hash均与04权威payload一致；没有向W&B写入本轮分析。Finished作为交叉检查，不替代上述产物验收。本轮无需训练重跑，没有阻断16条比较的缺失正式评价点或正式c。

## 正式c复算

| 任务 | 正式c | S | N | 成功匹配更新数 |
| --- | ---: | ---: | ---: | ---: |
| hopper_hop | 0.8169140208772256 | 24831065.62310791 | 30396180 | 50000 |
| quadruped_run | 0.9287514026396865 | 47192358.03814697 | 50812691 | 50000 |
| quadruped_walk | 0.9492462655872804 | 47199733.726623535 | 49723381 | 50000 |
| reacher_hard | 0.9187557996879948 | 26834834.727371216 | 29207799 | 50000 |

各任务只取自身正式logging baseline更新开始动作`[100000,300000)`的raw rep KL>1位置；逐成功更新累加原float32收据到宿主float64 S、整数N。S/N、全部50000个匹配update ID、原receipt hash、artifact/sidecar hash、baseline配置/训练SHA与final/matching_result精确一致，且冻结时间早于constant启动。完整c来源/冻结时间/hash在frozen_c.json/CSV。其他组三个matching收据字段按训练源码为0，不表示门控不活跃；其诊断取实际train/dt日志。

c只匹配baseline该窗口的**活跃位置平均候选系数**。它不匹配全程平均权重、惩罚值、梯度方向/范数或optimizer更新。诊断50k区间按快照更新边界汇总，区间包含末边界更新，而c按半开窗口归属；不能用阶段均值替代精确S/N。

## 原始return与冻结指标复算

每episode来源为`environment_interface.reward`，定义为`undiscounted_sum_non_reset_rewards_repeat_already_summed`。源码EpisodeReturn在非reset转移累加原始reward，is_last完成才输出；DMC接口已处理repeat，本次repeat1。schema v2 episode记录与兼容scores/returns_raw别名一致，JSONL protocol/eval_return与评价文件的mean精确一致。**未保存/镜像逐transition reward轨迹，不能声称已从每步奖励独立复算episode累计值**；本次独立复算从原始逐episode return开始，奖励来源由显式schema及训练源审查支持。

每点先`fsum(10 returns)/10`；主指标`sum(Δ动作步×相邻点均值/2)/1000000`；次指标严格80/85/90/95/100万五点算术平均。四任务原始return等权，未归一化、不平滑、无缺点填补。统计器require-complete退出0，另用21等间距点端点半权重的直接算式和尾部五点直接算式交叉验证；逐任务/总体/差值三表与04完全一致。展示四舍五入两位，CSV/JSON保留完整精度。

主指标（return AUC / 100万训练动作）：

| 任务 | baseline | Dt | constant | global shuffle |
| --- | ---: | ---: | ---: | ---: |
| hopper_hop | 253.83 | 0.86 | 241.00 | 223.73 |
| quadruped_run | 467.40 | 600.16 | 616.14 | 488.37 |
| quadruped_walk | 638.16 | 796.25 | 573.05 | 639.84 |
| reacher_hard | 734.07 | 730.24 | 676.48 | 732.40 |
| 四任务等权 | 523.37 | 531.88 | 526.67 | 521.09 |

次指标（80–100万五点评价均值）：

| 任务 | baseline | Dt | constant | global shuffle |
| --- | ---: | ---: | ---: | ---: |
| hopper_hop | 316.75 | 1.01 | 314.92 | 300.43 |
| quadruped_run | 810.95 | 846.08 | 891.41 | 760.31 |
| quadruped_walk | 937.79 | 944.89 | 911.83 | 924.06 |
| reacher_hard | 926.82 | 957.66 | 929.06 | 966.52 |
| 四任务等权 | 748.08 | 687.41 | 761.81 | 737.83 |

三个Dt差值（B=baseline，C=constant，S=global shuffle）：

| 任务 | AUC Dt−B | AUC Dt−C | AUC Dt−S | 末段 Dt−B | 末段 Dt−C | 末段 Dt−S |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hopper_hop | -252.97 | -240.14 | -222.87 | -315.74 | -313.91 | -299.41 |
| quadruped_run | +132.75 | -15.99 | +111.78 | +35.13 | -45.33 | +85.77 |
| quadruped_walk | +158.09 | +223.21 | +156.41 | +7.09 | +33.05 | +20.83 |
| reacher_hard | -3.83 | +53.76 | -2.16 | +30.84 | +28.60 | -8.86 |
| 四任务等权 | +8.51 | +5.21 | +10.79 | -60.67 | -74.40 | -50.42 |

![seed0未平滑四任务学习曲线](../analysis/outputs/m2-v1-seed0-20261012/analysis-final/learning_curves.png)

![seed0主次指标的三个差值](../analysis/outputs/m2-v1-seed0-20261012/analysis-final/dt_differences.png)

walk的优势主要体现在更早获得高return，末段优势小于AUC差值；run的constant末段更高，Dt对baseline的优势不能排除整体减弱KL；reacher不同指标排序不一致；hopper Dt所有非初始点均很低，保留原始结果而非删点/重跑。四任务平均的正AUC来自任务间正负抵消，末段负差同样必须报告。

## 门控是否实际作用

诊断来自20个50k动作区间内的成功更新scalar mean，区间更新数从快照update ID得到（首区间11993、后续各12500，共249493）。按更新数加权汇总；不是把20行简单平均。每组所有区间active_weight_available=1、invalid_D_frac=0。源w和实际v分开读取；baseline实际v均值1、超额减少0，constant实际均值等于冻结c的float32前向值。

全程诊断：D均值及D=0比例覆盖有效loss位置；活跃比例为raw rep KL>1；活跃权重是**逐更新活跃区均值的更新加权均值**，不冒充全程逐活跃位置加权统计。超额KL减少比例为`Σupdate mean((1-v)*(R-1)+) / Σupdate mean((R-1)+)`，不是free-nats常数减少比例，更不是梯度向量/范数减少。

| 任务 | 组 | D均值 | D=0比例 | free-nats活跃比例 | 实际活跃权重均值 | 超额KL减少比例 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hopper_hop | baseline | 0.07617 | 0.966% | 57.04% | 1.0000 | 0.00% |
| hopper_hop | dt | 0.00279 | 1.013% | 55.42% | 0.9516 | 5.81% |
| hopper_hop | constant | 0.05106 | 0.920% | 68.59% | 0.8169 | 18.31% |
| hopper_hop | shuffle | 0.03980 | 0.890% | 65.62% | 0.8485 | 15.26% |
| quadruped_run | baseline | 0.01195 | 0.014% | 97.68% | 1.0000 | 0.00% |
| quadruped_run | dt | 0.01333 | 0.012% | 98.58% | 0.8656 | 16.87% |
| quadruped_run | constant | 0.01719 | 0.016% | 96.56% | 0.9288 | 7.12% |
| quadruped_run | shuffle | 0.01389 | 0.015% | 97.46% | 0.8755 | 12.37% |
| quadruped_walk | baseline | 0.01262 | 0.018% | 96.85% | 1.0000 | 0.00% |
| quadruped_walk | dt | 0.01933 | 0.016% | 98.61% | 0.8480 | 20.75% |
| quadruped_walk | constant | 0.01205 | 0.018% | 97.56% | 0.9492 | 5.08% |
| quadruped_walk | shuffle | 0.01123 | 0.019% | 97.45% | 0.8990 | 10.35% |
| reacher_hard | baseline | 0.02696 | 1.087% | 37.12% | 1.0000 | 0.00% |
| reacher_hard | dt | 0.02497 | 1.113% | 37.93% | 0.9134 | 10.40% |
| reacher_hard | constant | 0.02382 | 1.151% | 42.68% | 0.9188 | 8.12% |
| reacher_hard | shuffle | 0.02393 | 1.283% | 39.30% | 0.9348 | 6.96% |

![seed0按50k区间的门控诊断](../analysis/outputs/m2-v1-seed0-20261012/analysis-final/gate_diagnostics.png)

Dt组分阶段：

| 任务 / Dt组 | 早期0–10万 D均值 | 早期活跃比例 | 早期活跃权重 | 后期80–100万 D均值 | 后期活跃比例 | 后期活跃权重 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hopper_hop | 0.00077 | 63.15% | 0.9837 | 0.00439 | 55.77% | 0.9318 |
| quadruped_run | 0.00309 | 99.71% | 0.9535 | 0.01372 | 97.59% | 0.8538 |
| quadruped_walk | 0.00210 | 99.45% | 0.9694 | 0.02200 | 97.86% | 0.8406 |
| reacher_hard | 0.01672 | 85.12% | 0.9187 | 0.02809 | 20.04% | 0.9119 |

完整D的p10/p50/p90、源w/实际v均值、mode差异、actor均值/方差差异、源/实际权重与raw KL协方差及近零活跃比例在gate_intervals.csv；阶段汇总在gate_summary.json/CSV。w与实际v的p10/p50/p90保留在inputs各运行的原始metrics.jsonl。**日志p10/p50/p90是每批分位数的均值，不能恢复全阶段合并分位数、直方图或尾部事件频率。** D=0使用原精确零计数/1024重建区间比例，非近零阈值；近零指标沿用源码D<=1e-6，条件比例从联合比例/活跃比例获得。

四任务Dt全程活跃系数平均减少约4.84%/13.44%/15.20%/8.66%，超额KL减少约5.81%/16.87%/20.75%/10.40%。这证明训练目标在活跃位置有非零作用；没有预设“充分”阈值，不新增判据。quadruped作用较明显，不能以全无活跃梯度解释其比较结果；hopper早期权重接近1，作用较弱，低return与低D之间因果方向无法识别。reacher后期活跃位置仅约20.04%，限制了gate覆盖，即使少数位置有减弱，也不能据此保证足够检验任务信息保护。

shuffle四任务索引换位比例均约99.9024%；这不等于权重数值不同位置的比例。其源w/raw KL协方差全程约−0.07901/−0.08883/−0.08350/−0.00998，置换后约0.000030/0.000043/−0.000001/0.000014，符合整体关联被破坏的诊断。尤其hopper源活跃均值0.8163、实际0.8485；不能拿源活跃值替代shuffle实际作用。Dt与shuffle使用各自训练轨迹，本组batch权重集合保留不代表两组权重分布相同。

walk的Dt超额减少20.75%，constant5.08%、shuffle10.35%；run对应16.87%/7.12%/12.37%。因此walk同时优于两个对照，提供**当前对照范围内的对应关系价值候选证据**，仍不能排除更强/不同压缩强度、raw KL大小关联及训练轨迹变化等解释。平均系数匹配不是损失/梯度作用完全匹配，更强M2需另定条件shuffle及有效换位标准。

## 证据缺口与不作的结论

单训练seed0不能估计seed间变异、显著性、训练稳定性或统计等效；10个评价episode不是10个训练seed。未设置最小增益/等效范围，不按本结果追选阈值。clean运行不能支持抗干扰结论；没有任务信息/nuisance配对干预、延迟线索干预、逐位置联合分布、实际rep梯度向量或optimizer反事实。日志可描述mode差异及actor响应，不能据此标记高D信息对任务有用。DtLatch未实现，不用本数据验证其奖励代理假设。

跨组评价seed/环境配置声明相同，但0步两quadruped任务的原return并非逐位相等，全部长度相同。观察值如下，不放宽比较容差、不替换任何点：

| 0步评价与本任务baseline比较 | mean return差 | 最大单episode绝对差 |
| --- | ---: | ---: |
| quadruped_run / dt | -0.084084 | 0.397068 |
| quadruped_walk / dt | +0.126589 | 1.140896 |
| quadruped_run / constant | -0.041374 | 0.759802 |
| quadruped_walk / constant | +0.055085 | 0.586532 |
| quadruped_run / shuffle | -0.063376 | 0.739947 |
| quadruped_walk / shuffle | -0.471558 | 4.958387 |

CODE_SIZE50M_REPAIR已在独立工程证据中确认同物理状态的EGL图像可能出现少量像素差异；本16条没有逐帧轨迹，不能将当前差异具体归因于渲染，也未重新读取0步完整参数payload证明初始化逐位一致。冻结v1未要求真实训练/评价跨进程逐位相同；本差异不构成预算/评价完整性失败，16条均按正式协议纳入，同时降低严格配对因果解释强度。无现存日志可补出的逐reward/逐像素数据不得编造。

## 可复算交付与资源快照

本机独立目录：`D:\program\PD\dmr3\analysis\outputs\m2-v1-seed0-20261012`。输入1161文件全部原路径/SHA256一致，336快照；inputs.tar.gz SHA256=`74276ca2f35af2b0a08f01189748d5f594da8e3dddd3f0e624c9ffacf8253220`。原始文件映射/哈希在inputs/COLLECTION.json，mirror_validation.json保存整包校验。大checkpoint/replay留sv2原目录，未提交到Git。

有效交付仅analysis-final：运行清单、完整RUN_INDEX、正式c表、return-audit六表/JSON、门控区间/阶段表、初始评价差异、三张PNG/SVG、validation.json和output_hashes.json。最终复算tables.json hash=`3d02f5cca66644877e55e45a79fd160949ff9b8d72c45920bf09e1235d75e9ef`；与04六张表的全部解析JSON内容完全相同，字节hash不同仅来自本机CRLF与服务器LF换行（归一换行后字节相同），不是数值或审计字段变化。早期analysis-attempt01/02/03/05为分析代码调试失败，04/06为成功开发核查，均保留，不作为正式结果版本；没有训练重跑/产物覆盖。运行命令、测试日志和尝试说明在本目录。

本机用bundled Python3.12，绘图依赖安装到本输出plot_deps（Matplotlib3.10.8、NumPy2.4.2），不修改服务器dreamer/base-py311。复算命令（PowerShell，输出须新目录）：

```powershell
cd D:\program\PD\dmr3
$py = 'C:/Users/97370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$env:PYTHONPATH = 'D:/program/PD/dmr3/analysis/outputs/m2-v1-seed0-20261012/plot_deps'
& $py -m analysis.analyze_m2_v1 --inputs analysis/outputs/m2-v1-seed0-20261012/inputs --output analysis/outputs/m2-v1-seed0-20261012/NEW_RECALC
& $py -m unittest tests.test_return_metrics tests.test_m2_analysis -v
```

复采另用`-m analysis.collect_m2_v1 --output analysis/outputs/NEW_COLLECTION`；服务器来源目录固定，不覆盖旧输入。分析本身仅CPU，19项测试和真实16条require-complete均在分析SHA通过，不把fixture当真实任务验收或方法有效性。首要结果表/科学解释已写入本报告并提交，生成outputs保持项目Git忽略规则。

资源只读快照2026-10-12 12:43:45北京时间：原16条PID均已退出，原M2作业已释放sv2物理GPU0–7，没有这些16条仍运行；八卡目前有PID3757333–3757340、目录`/home/lyc/denoising_sv2_formal_20261012`的其他作业，非本次16条，项目归属未确认，记未知作业，不干扰，不称八卡空闲。证据CURRENT_RESOURCE.json。本轮无GPU作业启动，后续资源状态必须重新检查，不沿用此快照。

## 下一步与可复制交接

首选继续现有02｜方法与机制，由用户手动复制本报告prompt，不创建chat或发送消息。02可立即审议原四组多训练seed复验及机制证据需求；执行依赖用户确认的新协议、03完整配置和服务器/GPU分配。建议先检验walk正向与hopper负向可重复性，避免同时引入新gate造成归因混杂；任务覆盖/seed数量/预算/判据仍未決，不替用户选值。完成标准：写清确认/否决理由、科学主张、复验比较集合、未决配置及03/04验收要求，不直接启动。

如需提高初始化/像素归因证据，可由用户交04并协调03只读补齐；这不是当前16条验收阻断项，也不授权重跑：

```text
项目根D:\program\PD\dmr3；训练48d71f1541721208322442f5a1d5977124f3e888，最终审计d239c3fb7e5b77ddf205fe994c46c2e5a6acce6a，05分析ad325482a46b0c9dd3c3709cccca6a63bea9ba5d。
必读AGENTS、ANALYSIS_M2_V1（初始评价差异/证据缺口）、CODE_SIZE50M_REPAIR、EXPERIMENTS与本机outputs/m2-v1-seed0-20261012。
仅只读CPU审查原16条eval_snapshots/0000000可信checkpoint的初始参数hash及seed/config来源，特别是两quadruped；给出与baseline逐位一致/不一致及证据，必要时交03解释。不启动训练、不改源码/容差/alpha/c、不择优重跑；不存在的逐reward/像素轨迹注明无法补齐，不重新生成冒充原始日志。结果放新outputs目录，更新必要报告，按AGENTS交付完整SHA、证据、限制与prompt。
```

给02的prompt（最终交付SHA见HANDOFF/最终回复）：

```text
继续02｜方法与机制，项目根D:\program\PD\dmr3。先核实际HEAD和未提交修改，不回退。
交接版本：本轮HANDOFF/最终回复中的完整最终SHA；实际分析/测试ad325482a46b0c9dd3c3709cccca6a63bea9ba5d；训练48d71f1541721208322442f5a1d5977124f3e888；16条最终审计d239c3fb7e5b77ddf205fe994c46c2e5a6acce6a。
必读AGENTS.md、docs/CHATS.md、docs/STATUS.md、docs/ANALYSIS_M2_V1.md、docs/EXPERIMENTS.md C/C.1、docs/METHOD_SPEC.md、docs/RESEARCH.md、docs/CODE_SIZE50M_REPAIR.md、analysis/README.md；复算证据analysis/outputs/m2-v1-seed0-20261012/analysis-final及inputs/COLLECTION.json。
已确认：原四任务clean视觉size50m seed0 repeat1 alpha20、free_nats1、dyn/rep1/0.1、既定梯度开关、baseline窗口[100000,300000)S/N冻结c、四组从头100万动作、21点×10完整episode、原始return AUC/预算与固定五点末段/四任务等权不改；归一化暂缓。16条有效并纳入，正式c有效；无训练重跑。DtLatch仍未批准。
结果：walk两个指标均胜三个对照；run胜baseline/shuffle但不胜constant；reacher排序混合；hopper Dt很低。总体AUC Dt−B/C/S=+8.51/+5.21/+10.79，末段=−60.67/−74.40/−50.42。gate有活跃作用，但全程强度/梯度未匹配。单seed、跨进程初始评价小差异及未保存逐reward/像素/梯度限制须保留。
本任务只审议下一科学协议。首选制定原四组多训练seed复验以核查walk优势及hopper退化可重复性；具体seed/任务覆盖/预算/判据待用户确认，不由05结果自动批准。更强M2的条件shuffle分层/边界/退化/有效换位也未定；不要同时自动引入DtLatch或调alpha/c。
允许只读证据、理论解释、在用户授权内维护方法/实验文档；禁止启动训练、自行分配GPU、改训练源码或冻结v1、归一化/追选阈值/把episode当训练重复。
完成标准：明确确认/否决理由、主张与比较集合、已定/未决字段、03实施与04验收要求；输出docs/METHOD_SPEC.md、docs/EXPERIMENTS.md及STATUS中明确的新版本或未决清单，保留v1历史。完成按AGENTS给完整最终/实际测试SHA、差异/同步范围、证据限制、下一步及可复制prompt。不自动建chat或发消息。
```
