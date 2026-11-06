# seed1 全任务、seed0/1 等权均值及 DtL 三任务分析

2026-11-06，05 Analysis。按用户本次明确答复纳入 acrobot_swingup，共20任务；此前19任务草稿仅保留为分析过程证据。Dt 与图例 PPD-KL 是同一组；DtL 是独立方法，不把其结果混入两seed的 Dt 均值。

**seed1 没有复现 seed0 的总体 AUC 优势：20任务等权 ΔAUC 为 −8.91，末段差为 +24.07。两seed均值为 +25.11 / +37.26，但只有9/20任务的平均AUC为正，11/20任务在两个seed间发生AUC方向反转。DtL只在 finger_spin 的这个 seed0 上同时超过两个对照；两个quadruped任务均低于原Dt。**

## 1. 来源、完成状态和纳入范围

11月6日10:38（sv1/sv2）及10:40（sv3）北京时间只读检查正式索引、终点文件和逐条审计。完整20任务的seed1来源已经有38条新训练终点及两条复用hopper；本报告纳入全部40条，并补读acrobot的seed0 baseline/Dt；11:08–11:09再次采集核对45条完整原产物；新增来源及逐条采集时间见input20和acrobot-seed0-inventory.json。迁移前的未启动/阻塞占位不计作额外运行，不择优挑选失败尝试。

| 分析部分 | 实验条数 | 证据与状态 |
| --- | --- | --- |
| seed1，20任务×baseline/Dt | 40 | 38条新运行索引为 run_audited_pending_group_and_WB；复用hopper seed1两条为 accepted。全部本次读取并复核完整原始评价/终点/收据。 |
| seed0，同20任务×baseline/Dt | 40 | 复用既有正式分析的原始输入镜像：原四任务8条＋新增15任务30条；本次补读acrobot两条原始产物。重算全部评价；原四任务本机完整收据再次逐行复核，30条沿用此前完整收据审计。不是重新下载全部seed0大产物。 |
| DtL，三任务seed0 | 3 | sv2物理GPU5/6/7分别为quadruped_run/walk/finger_spin；均accepted。supervisor-final于北京时间11月6日05:22:29记录passed，三条各249493条DtL更新收据。 |
| 合计 | 83 | 1743个固定评价点、17430个完整评价episode；没有插值、缺点重加权或工程数据。 |

seed1的逐条产物可用于完整20任务比较，但**尚不能将其称为03已经完成整批终点合并和W&B精确收尾**。本次只读分析不改03原索引，也不写W&B summary。seed0运行清单保留历史索引原state；部分旧state仍写pending，最终验收依据是原独立审计报告，不把历史字段误当新的运行状态。

新读取45条（40 seed1＋3 DtL＋acrobot seed0两条）均核对：1000000训练动作、249493更新和batch key、210000评价动作、21点各10个完整1000动作episode；原始episode均值与JSONL一致；固定snapshot的步号/update ID/标识/SHA/latest/done及服务器参数文件存在；普通更新收据全量连续、非法计数0，正式baseline c的[100000,300000)窗口S/N、冻结值与完整收据hash一致。DtL额外748479条收据全部与普通更新对齐，20区间均值与JSONL相符，21点评价隔离记录齐全。对旧seed0本机小镜像不要求缺省未下载的大参数payload，复用此前服务器存在性核查。

episode return来自 evaluations.jsonl 的 return_raw，声明为非reset奖励的未折扣求和，action repeat已在环境内累加。本次从10个episode的原始return复算均值与指标，**没有逐评价转移reward流可独立重新求和，也没有重新反序列化所有大checkpoint验证参数内容**；固定参数/奖励累计实现复用03源码和原验收证据。训练replay或既有GIF不替代固定快照评价。

W&B只作独立展示交叉核对，结果保存在输出目录 wandb20_crosscheck.json；精确指标取原始评价数据。具体本次线上读取结果在第8节注明。

本轮不启动训练、不调参、不执行agent/环境或GPU探针，不清理服务器、不更换训练checkout。上表GPU是这些训练的实际历史分配，**本次没有核查当前卡是否空闲**。

## 2. 固定统计口径

每点先对10个完整episode求平均原始return，记作 R_i，动作网格为0、50000、…、1000000。

- 主指标：Σ[(R_i+R_(i+1))/2×50000]/1000000，即21点梯形 AUC / 1M。
- 次指标：80万、85万、90万、95万、100万这五点评价均值的算术平均，以下称“末段”。
- 两seed均值：先在同任务同组内对seed0、seed1指标各给1/2权重；总体再对20任务等权平均原始return。保留逐任务、逐seed差值；不做归一化、不引入效果阈值或等效范围。
- 所有图用未平滑评价值，折线连接实际网格点；两seed图淡线是单个seed，粗线是均值，不是置信区间。10个评价episode不是10个训练seed。

| 范围 | baseline AUC | Dt AUC | ΔAUC | baseline末段 | Dt末段 | Δ末段 | AUC/末段正差任务数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| seed0 | 628.05 | 687.17 | +59.12 | 754.38 | 804.83 | +50.45 | 10/20；15/20 |
| seed1 | 692.86 | 683.95 | -8.91 | 811.23 | 835.30 | +24.07 | 9/20；10/20 |
| mean_seed0_seed1 | 660.45 | 685.56 | +25.11 | 782.81 | 820.07 | +37.26 | 9/20；11/20 |

以上“正差任务数”仅是数值符号计数；+0.10等很小差值也计入，不能解释为统计显著、有效率或已设定的优胜阈值。

## 3. seed1：逐任务结果

| 任务 | baseline AUC | Dt AUC | ΔAUC | baseline末段 | Dt末段 | Δ末段 |
| --- | --- | --- | --- | --- | --- | --- |
| acrobot_swingup | 213.65 | 195.99 | -17.66 | 338.75 | 275.52 | -63.24 |
| cartpole_balance | 973.63 | 974.95 | +1.32 | 994.30 | 995.71 | +1.41 |
| cartpole_balance_sparse | 936.12 | 968.32 | +32.19 | 999.76 | 1000.00 | +0.24 |
| cartpole_swingup | 770.48 | 761.18 | -9.29 | 850.14 | 852.10 | +1.96 |
| cartpole_swingup_sparse | 0.00 | 151.64 | +151.64 | 0.00 | 555.74 | +555.74 |
| cheetah_run | 678.63 | 649.94 | -28.69 | 874.91 | 856.62 | -18.29 |
| cup_catch | 870.75 | 899.47 | +28.71 | 976.26 | 973.10 | -3.16 |
| finger_spin | 950.54 | 950.06 | -0.48 | 984.60 | 984.26 | -0.34 |
| finger_turn_easy | 631.89 | 662.75 | +30.86 | 814.94 | 865.50 | +50.56 |
| finger_turn_hard | 639.34 | 710.68 | +71.34 | 955.40 | 833.28 | -122.12 |
| hopper_hop | 265.17 | 323.30 | +58.13 | 316.26 | 457.71 | +141.45 |
| hopper_stand | 834.84 | 723.24 | -111.60 | 875.32 | 861.14 | -14.18 |
| pendulum_swingup | 723.95 | 738.02 | +14.07 | 798.18 | 826.54 | +28.36 |
| quadruped_run | 644.20 | 505.99 | -138.21 | 894.35 | 881.53 | -12.82 |
| quadruped_walk | 707.50 | 554.54 | -152.95 | 946.00 | 924.68 | -21.32 |
| reacher_easy | 801.26 | 800.05 | -1.21 | 943.70 | 939.06 | -4.64 |
| reacher_hard | 690.63 | 683.23 | -7.40 | 904.94 | 925.14 | +20.20 |
| walker_run | 672.76 | 571.04 | -101.72 | 811.42 | 752.58 | -58.84 |
| walker_stand | 934.17 | 948.03 | +13.86 | 984.65 | 984.75 | +0.10 |
| walker_walk | 917.68 | 906.60 | -11.09 | 960.70 | 961.12 | +0.42 |

![seed1学习曲线](../analysis/outputs/seed1-seeds01-dtl-20261106/result20-final/seed1_learning_curves.png)

seed1的关键现象：

- **acrobot_swingup：seed1的Dt低于baseline。** AUC213.65→195.99，差−17.66；末段338.75→275.52，差−63.24。seed0的差为+33.92/+32.03；两seed均值AUC+8.13但末段−15.61，同样发生方向反转。

- **quadruped_run / quadruped_walk主要是早中期学习延迟。** ΔAUC为−138.21/−152.95；末段差缩至−12.82/−21.32。以30万点为例，run的baseline/Dt约493.33/97.05，walk约832.01/108.44。不能只看百万步接近便抹掉样本效率损失。
- **walker_run既学得慢，末段也落后。** AUC−101.72，末段−58.84，区别于仅早期延迟的情形。
- **cartpole_swingup_sparse的正差主要来自baseline这一seed全程未取得评价回报。** 原始210个评价episode return均为0；Dt后期才抬升，85万至100万达到约662—674。+151.64 AUC / +555.74末段是这个训练重复的真实差值，但不能只凭它宣称降低失败概率。
- **finger_turn_hard体现“前期更快、后期更差”。** AUC+71.34但末段−122.12，不能只挑一个指标解释为全面改善。
- **pendulum_swingup没有重现seed0的大幅优势。** 这次两组都在早期学会，AUC只差+14.07；finger_spin同样两组都达到高回报，AUC−0.48、末段−0.34。

## 4. seed0/1 等权均值：正的总体掩盖任务与seed差异

| 任务 | baseline AUC | Dt AUC | ΔAUC | baseline末段 | Dt末段 | Δ末段 |
| --- | --- | --- | --- | --- | --- | --- |
| acrobot_swingup | 195.98 | 204.11 | +8.13 | 325.74 | 310.14 | -15.61 |
| cartpole_balance | 968.16 | 963.60 | -4.57 | 993.94 | 993.44 | -0.50 |
| cartpole_balance_sparse | 953.13 | 941.45 | -11.69 | 987.56 | 999.56 | +12.00 |
| cartpole_swingup | 717.99 | 767.84 | +49.85 | 851.78 | 855.74 | +3.96 |
| cartpole_swingup_sparse | 211.91 | 327.16 | +115.25 | 369.04 | 658.73 | +289.69 |
| cheetah_run | 691.16 | 668.16 | -23.00 | 853.06 | 843.97 | -9.08 |
| cup_catch | 903.14 | 923.44 | +20.30 | 973.17 | 973.97 | +0.80 |
| finger_spin | 711.05 | 709.22 | -1.82 | 740.71 | 775.53 | +34.82 |
| finger_turn_easy | 692.95 | 664.83 | -28.11 | 852.65 | 826.55 | -26.10 |
| finger_turn_hard | 614.20 | 671.90 | +57.71 | 879.02 | 824.89 | -54.13 |
| hopper_hop | 259.50 | 162.08 | -97.42 | 316.51 | 229.36 | -87.15 |
| hopper_stand | 808.47 | 759.17 | -49.31 | 889.85 | 887.42 | -2.43 |
| pendulum_swingup | 378.24 | 714.74 | +336.50 | 449.83 | 807.17 | +357.34 |
| quadruped_run | 555.80 | 553.07 | -2.73 | 852.65 | 863.80 | +11.15 |
| quadruped_walk | 672.83 | 675.40 | +2.57 | 941.90 | 934.78 | -7.11 |
| reacher_easy | 829.42 | 822.55 | -6.87 | 938.73 | 955.30 | +16.57 |
| reacher_hard | 712.35 | 706.74 | -5.61 | 915.88 | 941.40 | +25.52 |
| walker_run | 487.26 | 631.68 | +144.42 | 580.51 | 778.70 | +198.20 |
| walker_stand | 940.66 | 945.38 | +4.73 | 980.25 | 980.27 | +0.01 |
| walker_walk | 904.88 | 898.71 | -6.17 | 963.37 | 960.65 | -2.71 |

![两seed等权学习曲线](../analysis/outputs/seed1-seeds01-dtl-20261106/result20-final/two_seed_mean_learning_curves.png)

![逐seed和均值差值](../analysis/outputs/seed1-seeds01-dtl-20261106/result20-final/seed_differences.png)

平均AUC为正的只有9/20任务，但pendulum_swingup、walker_run和稀疏cartpole的较大正差使总体仍为+25.11。这是冻结原始return等权口径下的结果，不等同于大多数任务受益。不能以改权重或移除不利任务替代该总体。

两个seed中AUC方向都为正的四任务是cartpole_swingup_sparse、cup_catch、finger_turn_hard、pendulum_swingup；方向都为负的五任务是cheetah_run、finger_spin、reacher_easy、reacher_hard、walker_walk；其余11任务发生反转。这里的“同向”只描述两个观测，不声称跨seed稳定性。

quadruped_run的ΔAUC从seed0的+132.75变成seed1的−138.21，均值−2.73；quadruped_walk从+158.09变为−152.95，均值+2.57。因此，原seed0上两个quadruped的优势没有在seed1复现。walker_run从+390.56变成−101.72，均值仍+144.42，也不能将这个均值当成每次训练都会获得的收益。hopper_hop从−252.97转为+58.13，说明原失败式结果并非所有seed都会重现，但两seed均值仍明显为负。

pendulum均值+336.50 AUC主要由seed0 baseline的低回报拉开。已有hopper/pendulum的seed2/3数据不混入本次指定seed0/1均值；完整可靠性讨论还应另读其既有报告，不把“两个seed”表述成全部可用重复。

## 5. 原Dt是否实际作用：不能统一归因于alpha过小

以下为seed1全训练成功更新的诊断；各区间按更新数加权。活跃位置权重是日志中“每次更新活跃位置均值”的更新平均，不冒充把所有活跃位置重新合并后的全局均值。超额KL削弱比例是 Σ(1−w)max(KL−1,0) / Σmax(KL−1,0)，只反映已实现损失量，不等于梯度范数下降比例。

| 任务（seed1 Dt） | 平均D | D=0比例 | free-nats活跃比例 | 活跃位置权重均值 | 超额KL削弱比例 |
| --- | --- | --- | --- | --- | --- |
| acrobot_swingup | 0.02280 | 0.70% | 37.80% | 0.9369 | 6.40% |
| cartpole_balance | 0.01178 | 0.39% | 15.90% | 0.8603 | 15.85% |
| cartpole_balance_sparse | 0.03396 | 0.59% | 18.25% | 0.8916 | 9.40% |
| cartpole_swingup | 0.01429 | 0.97% | 23.66% | 0.8491 | 13.42% |
| cartpole_swingup_sparse | 0.03483 | 0.98% | 32.30% | 0.9604 | 5.72% |
| cheetah_run | 0.00614 | 0.56% | 45.69% | 0.9269 | 9.45% |
| cup_catch | 0.05893 | 2.52% | 38.13% | 0.8692 | 16.49% |
| finger_spin | 0.29078 | 2.24% | 49.07% | 0.6609 | 37.58% |
| finger_turn_easy | 0.04394 | 0.51% | 46.23% | 0.8715 | 15.84% |
| finger_turn_hard | 0.08315 | 1.26% | 51.60% | 0.8287 | 17.61% |
| hopper_hop | 0.06434 | 0.37% | 71.00% | 0.7900 | 29.60% |
| hopper_stand | 0.02551 | 0.30% | 56.81% | 0.8636 | 20.45% |
| pendulum_swingup | 0.08932 | 3.40% | 13.46% | 0.8867 | 10.08% |
| quadruped_run | 0.01041 | 0.03% | 95.63% | 0.8982 | 13.79% |
| quadruped_walk | 0.01111 | 0.02% | 99.32% | 0.8884 | 13.61% |
| reacher_easy | 0.03496 | 1.03% | 32.69% | 0.8927 | 12.86% |
| reacher_hard | 0.03171 | 1.23% | 40.33% | 0.9202 | 9.20% |
| walker_run | 0.09286 | 0.01% | 99.36% | 0.6662 | 39.02% |
| walker_stand | 0.03467 | 0.06% | 90.72% | 0.8360 | 24.95% |
| walker_walk | 0.05886 | 0.01% | 99.08% | 0.7606 | 30.03% |

finger_spin的活跃位置平均权重约0.661，超额KL削弱37.58%，两组回报仍接近；walker_run约0.666和39.02%，回报却落后。这两个例子都不支持“门控没打开所以没优势”的解释。quadruped_run/walk活跃比例约95.63%/99.32%，超额KL削弱约13.79%/13.61%，也不是free-nats把门控全部屏蔽。

现有数据更支持区分三件事：任务是否已接近高回报平台、学习过程是否走向不同轨迹、削弱rep约束是否有利于该任务。只有baseline/Dt这两组，不能区分整体减弱rep KL与位置对应的贡献。原四任务的constant/shuffle对照只支持其既定范围，不能自动推广到新增15任务或新seed。

## 6. DtL三任务：同seed0的三组比较

三条DtL均为seed0，下面只与对应seed0 baseline/Dt比较，绝不与两seed平均对照混算。正式配置为clean视觉、size50m、repeat1、alpha20、rho0.5、H1 reward-support、normal_only=False、kl_tolerance=1e-5、kappa=-1、bins=[-1]；原rep_probe关闭。没有新constant/shuffle或强度匹配臂。

| 任务（seed0） | baseline AUC / 末段 | Dt AUC / 末段 | DtL AUC / 末段 | DtL−baseline AUC / 末段 | DtL−Dt AUC / 末段 |
| --- | --- | --- | --- | --- | --- |
| quadruped_run | 467.40 / 810.95 | 600.16 / 846.08 | 582.02 / 804.89 | +114.62 / -6.06 | -18.13 / -41.19 |
| quadruped_walk | 638.16 / 937.79 | 796.25 / 944.89 | 446.58 / 908.75 | -191.57 / -29.05 | -349.67 / -36.14 |
| finger_spin | 471.56 / 496.82 | 468.39 / 566.80 | 696.02 / 986.28 | +224.46 / +489.46 | +227.63 / +419.48 |

![DtL三任务学习曲线](../analysis/outputs/seed1-seeds01-dtl-20261106/result20-final/dtl_learning_curves.png)

- **finger_spin：本次DtL改善最明确。** AUC696.02，分别比baseline/Dt高224.46/227.63；末段986.28，比两者高489.46/419.48。曲线在55万点已进入高回报区，并在最后五点评价保持约985—988，不只是挑出百万步单点。但seed1原baseline/Dt也都达到约984，说明“高回报”并不是DtL独有的可达行为；不能将跨seed观测当新的配对实验。
- **quadruped_run：前中期相对baseline有收益，整体仍未超过原Dt。** AUC比baseline高114.62、比Dt低18.13；末段反而低于两组6.06/41.19。
- **quadruped_walk：学习明显延迟。** AUC比baseline/Dt低191.57/349.67；晚期已达到较高回报，末段仍低29.05/36.14。当前记录不是“始终学不会”，也不能因后期追上部分回报忽略AUC差。

DtL实际门控为 `v=stop_gradient[1−rho×e×(1−1/(1+alpha×D))]`，其中无有效邻接的位置v=1，`e=max(exp(−KL(y||Pq))−exp(−KL(y||Pp)),0)`。rho0.5意味着权重下界0.5，不意味着每个位置实际削弱一半。

| 任务（seed0） | DtL free-nats活跃比例 | DtL有效邻接的平均e | DtL实际平均权重 | DtL超额KL削弱比例 | Dt超额KL削弱比例 | DtL最小实际权重 |
| --- | --- | --- | --- | --- | --- | --- |
| quadruped_run | 97.76% | 0.001848 | 0.999765 | 0.3058% | 16.87% | 0.512444 |
| quadruped_walk | 95.63% | 0.003763 | 0.999692 | 0.3003% | 20.75% | 0.509238 |
| finger_spin | 40.81% | 0.001902 | 0.999815 | 0.0662% | 45.90% | 0.500419 |

![DtL门控强度](../analysis/outputs/seed1-seeds01-dtl-20261106/result20-final/dtl_gate_diagnostics.png)

**本次DtL总体接近baseline的权重1，作用强度远小于原Dt。** 有效邻接比例约98.34%，两个quadruped的free-nats活跃率也高；弱作用并非主要因为无有效邻接或没有活跃KL，而是记录到的reward-support信号e均值很小。偶尔存在接近0.5的局部权重，因此不能说DtL完全未触发。按现有v<1−1e-6诊断阈值，活跃位置中满足该减弱阈值的比例分别约41.31%、37.27%、8.27%；这个阈值统计不能替代减弱幅度，许多变化非常小。

原Dt在这三个seed0运行的超额KL削弱为16.87%、20.75%、45.90%，DtL仅0.306%、0.300%、0.066%。各组沿不同轨迹采样，分母也不同；这些比例描述各自实际训练，不是同一batch上的反事实梯度对照。不能凭finger_spin收益证明DtL更会挑位置，也不能凭quadruped退步直接否定保护机制。

现有DtL日志没有单列活跃位置平均v，本报告保持该字段为空；D_valid_mean只在有效邻接上平均，而原Dt的D_mean口径不同，不能直接拿两者作同分布比较。没有记录足够的任务相关因果标签或同batch强度匹配梯度，无法从这批结果确定位置选择价值。

## 7. 版本与配置可比性

| 范围 | 纳入条数 | 实际训练SHA |
| --- | --- | --- |
| baseline/Dt seed0 | 10 | 18b85c4d85ff456e6822626f17cb14cc0f44d2b8 |
| baseline/Dt seed0 | 4 | 6e468ad4a8252f21401eab6a9761626b33659f0e |
| baseline/Dt seed0 | 7 | c7689d50f4caed0fef89912c65280b8d2558b126 |
| baseline/Dt seed0 | 7 | dbd28b6703d51d61e5d3485dbfea6c413075ea85 |
| baseline/Dt seed0 | 4 | d8997f64a301bfdcfe08758faeae8957a7395766 |
| baseline/Dt seed0 | 8 | 48d71f1541721208322442f5a1d5977124f3e888 |
| baseline/Dt seed1 | 24 | bef29480df12d07659c640b22218804dd1d69e63 |
| DtL seed0 | 3 | f1f33bc9a74eb112f091a29f09307f81ba72513a |
| baseline/Dt seed1 | 14 | 77699380851701f7a03a9d13985c5cc0de5d0702 |
| baseline/Dt seed1 | 2 | 743089f40662b031a5cf2b6bf988decb52ba0b28 |

逐条配置、运行/尝试ID、服务器/物理卡与UUID、原始目录、W&B链接、来源hash在 result20-final/run_manifest.json/csv；新索引原件在sv1/2/3-inventory.json，旧输入hash在input20/seed0.json。没有回退HEAD或假设所有运行同版本。

83条manifest的依赖锁及包版本相同：Python3.11.16、JAX/jaxlib0.4.33、dm-control1.0.48、MuJoCo3.15.0、NumPy1.26.4；设备均为A100-SXM4-80GB。baseline/Dt共同配置比较仅有seed、入口scope、组mode及logdir差异；DtL另增加方法参数和背景适配器的clean缺省字段，未启用静态/动态背景。科学共同项仍为64×64视觉、无proprio、16环境、batch16×64/context1、train_ratio256、free_nats1、dyn/rep=1/0.1、ac_grads=False、reward/repval梯度开、prealloc=False、JSONL+PD_1。

source_comparison.json逐历史Git blob比较：agent.py、rssm.py、core/driver.py、dmc.py、jax/agent.py、依赖锁在10个训练SHA间一致。rep_probe差异是已有mode诊断修复；protocol_v1差异是任务scope校验/manifest标签，见原工程报告，不据入口名变化推断科学口径变化。DtL由独立dtl_agent/包装器启用新方法，不能用原agent.py相同声称整个DtL训练与baseline逐位相同。

迁移后控制器与实际训练提交可以不同。下面是此前归档bound-plan中的声明版本（输入路径/hash见controller_provenance.json），不是从训练SHA猜测审计版本；多数逐条run_audit本身没有独立audit_execution_sha，本次不补造该字段。原四任务seed0的最终外部审计为d239c3fb7e5b77ddf205fe994c46c2e5a6acce6a，来源见原ANALYSIS_M2_V1；DtL supervisor-final明确记录f1f33bc9a74eb112f091a29f09307f81ba72513a。

| 正式来源根 | 声明控制器SHA |
| --- | --- |
| formal-m2-vision20-hopper-stand-dt-20261105-sv2 | a81b797348e03a3c1767afa50b523472995ceb85 |
| formal-m2-vision20-seed1-reallocated-20261104-sv1 | d6b9604ddf860d451d74d3cf6513fb66973a9094 |
| formal-m2-vision20-seed1-20261103-sv2 | 77699380851701f7a03a9d13985c5cc0de5d0702 |
| formal-m2-vision20-seed1-20261103-sv1 | bef29480df12d07659c640b22218804dd1d69e63 |
| formal-m2-vision20-seed1-remaining-20261105-sv2 | 666ce9031760f37e6f3fbd604f526621e9cffa43 |
| formal-m2-vision20-walker-dt-20261105-sv2 | 2f7e6358fa45f125ac630aabb8b5d9f3ef3bfa2a |

保留的工程限制：DtL Linux CPU上active-vs-baseline初始weight=1的严格梯度/优化器等价断言曾失败，最大差2.980232238769531e-8；原DtL也有同一失败。原实现/本次集成同模式617数组逐位一致，旁路及真实CUDA短程已通过；没有改容差，也不把保留失败改写为通过。详见[DtL运行与验收记录](DTL_THREE_SEED0.md)。旧mode诊断修复的loss/grad/RNG回归依据见[诊断修复记录](CODE_MODE_DIAGNOSTICS_REPAIR.md)。同seed同配置依然不保证跨进程/服务器的初始输入及完整训练轨迹相同，不能把seed差异全部归于方法或这个CPU边界。

## 8. 交付、复算与下一步

最终20任务采集、表图分析及四项手算/范围测试SHA：2e54499d96d8e5ab47235301469e6fd5de274b40。
初期19任务分析为ba59fc981f2621da176744346e32020b66c925b3；清单元数据修复为9a57256717359c8b30c5fc89cfe7918575ab9e23。用户明确纳入acrobot后，在新输入/输出目录重新采集45条原始来源并重算83条。旧19任务数据及DtL三任务的逐运行指标保持相同；20任务总体按新范围重新等权计算。最终文档交付SHA见输出目录DELIVERY.json，避免文档自引用。

本次没有训练源码修改或服务器代码同步。期间主分支并入另一窗口的DCS动态背景准备提交，涉及8个其他文件；本次四项分析测试不代表验收那些改动。

最终结果目录：`analysis/outputs/seed1-seeds01-dtl-20261106/result20-final/`，输入为`input20/`。早期`result/`（46个W&B URL不完整）和`result-final/`均是19任务范围，不作为本次最终全任务结果；REPORT19_DRAFT.md保留被用户范围更正前的报告。最终83条URL完整。输入与表图/报告hash、命令、线上交叉核对及声明控制器来源同放本次输出根。

| 文件 | 内容 |
| --- | --- |
| seed_metrics.csv/json | seed0和seed1各任务baseline/Dt、AUC/末段及差值 |
| two_seed_means.csv/json、overall.csv/json | 两seed等权均值及20任务等权汇总 |
| dtl_comparison.csv/json | seed0三任务三组及DtL对两对照差值 |
| learning_points.csv/json | 1743个未平滑评价均值 |
| gate_intervals.csv/json、gate_summary.csv/json | 20个动作区间诊断及按成功更新加权汇总 |
| run_manifest.csv/json、config_differences.json | 逐运行配置、实际SHA、路径、来源hash和差异 |
| verification.json、input/*.json | 严格完整性结果及逐episode原始输入 |

运行于项目根，使用已有本机Matplotlib环境：

```powershell
$python = 'C:/Users/97370/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$env:PYTHONPATH = 'D:/program/PD/dmr3/analysis/outputs/m2-v1-seed0-20261012/plot_deps'
& $python -m unittest tests.test_seed01_dtl_analysis -v
& $python -m analysis.compare_seed01_dtl analyze --include-acrobot --input analysis/outputs/seed1-seeds01-dtl-20261106/input20 --output analysis/outputs/NEW_SEED01_DTL_RESULT
```

要重新采集只读原产物，使用本次固定库存 `--inventory analysis/outputs/seed1-seeds01-dtl-20261106` 和全新目录：`-m analysis.compare_seed01_dtl collect --include-acrobot --inventory ... --output analysis/outputs/NEW_SEED01_DTL_INPUT`。输入必须包含完整两个seed配对；缺评价点/episode、重复配对或缺seed会拒绝，不插值。4项手算/范围测试覆盖AUC/末段、缺点评价、等权seed及19/20任务严格范围逻辑；真实83条原输入完整检查通过。五张图已检查标题、seed范围、图例和曲线，无平滑或episode伪重复置信带。

北京时间2026-11-06 11:10:46只读W&B核对：40条seed1＋3条DtL＋acrobot seed0两条均为finished；45条共945点评价网格与原始记录一致，读取的return均值最大绝对差为0；终点动作/更新/评价动作/episode数也一致。这确认线上展示，不替代原产物验收或seed1整批精确summary收尾。没有写W&B。早期19任务核对及一次稀疏网格查询分页重试的记录保留，见COMMANDS.md。

当前最值得交02决定的是：**先区分训练重复差异与门控强度，再讨论放大参数。** 原Dt已经在部分无收益任务上明显削弱rep KL；直接统一加alpha没有本批证据支持。DtL则需要先审视H1 reward-support导致的实际弱作用，并明确未来要比较的是位置选择还是整体正则强度。优先围绕finger_spin收益的可复现性、quadruped两seed反转与DtL强度差制定有限验证；具体seed、参数/强度匹配控制和预算由02与用户冻结，本次不追加实验。

03尚需完成seed1全20任务唯一来源合并、原始return/c审计集合及W&B精确终点回读，记录控制器/审计执行版本并更新运行文档。现有20任务两seed原始评价已齐，不因后处理未收尾而虚构缺点评价，也不把该缺口隐去。

两个训练seed只能给探索性证据，三条DtL仍各只有一个训练seed。不能声明跨seed统计显著、训练稳定性改善、等效或机制已被证实。评价episode数增加不会补足训练重复数。
