# DL-NM1：cal-only执行规范及后续学习草案

2026-10-15。**Agent在完整预验证授权内采用C，版本DL-NM1-cal-r1；后续R/T部分仍为DL-NM1-trial-r0建议。**
它将[独立审查提案](DL_NM1_PROTOCOL_REVIEW.md)与[机制草案](DL_NORMAL_MECHANISM_DRAFT.md)收敛为一次有限对象；
此次root内部仅采用cal-only的下列对象/资格/预算，不称用户逐项确认，不因此采用4096学习、6小时或性能门。
后续建议值在本稿中写全，末节区分已采用与仍待决定。
未查看R5中间D/e/Y、cal系数或科研结果；不按R5正负、NM1终点或所需阳性修改下面的对象。
R5原科学规范及工程r2各自不变，旧失败不改；原M2不变。新mask/只读取批/scratch工程并行实现中，
normal_only有限CPU fixture或源码存在不是真实N1M cal验收；确定源码CPU验收与实际资源条件仍须完成。

## 对象、已知事实与假设

本对象问：从同一个完整N1M初态，在固定普通持久化replay上，有限减少**非reset位置**的rep压力，
是否形成真实参数更新差异，并在共同真实历史上保留固定读出族可访问的任务状态量？
这种效应是否超出预定义标量释放、层内置换和reward-only对照，并与完整Agent的native行为相容？

已验证前置是：原N1M完整checkpoint身份；1280 NPZ/16链/1M动作的内容库存；严格私有cache导入/合成回写；
tiny CPU三步真实训练反馈；固定读出数学程序。它们不等于真实N1M普通学习、逐叶梯度或方法有效验收。
历史replay包含不同年龄的cache，不能恢复原sampler、在线queue、环境或N1M终点全部cache。

工作假设：B的普通完整更新会降低某个任务量的有限可访问性，N_nr可部分避免，且有效S/P/W不能完全解释。
尚无证据的推测包括：原r3慢学来自信息损伤、tiny释放足以产生收益、H1识别了应该保护的信息。
本对象不是早期r3原因重演，不把B从N训练所得初态分叉解释为历史B组。

## 固定输入与分阶段对象

checkpoint/config使用原`/data/Policy_Discrepancy/runs/dl-r4-utility-inputs-20261015-0655/`下
`agent.pkl`、`config.yaml`；原始SHA分别为：

- `c6a7d92e24669a515bb44693faf73e442d9868f9fc814eec0e22e069aa594d24`
- `877f8470cb4f7f45d143e3099fbd3f55a828991afe33779b7b04c4264f6c8377`

固定`dmc_quadruped_walk`、size50m/BF16、RGB64/camera2/proprioFalse/action-repeat1/root seed0，
原loss scales/free-nats1/reward_grad/repval_grad/ac_grads、optimizer/moments、norm及slow targets不变。
训练模式不得改成`training=False`以避开norm写入；不清momentum、不冻结actor/prior、不删其它loss。
完整post-load state须与原输入逐叶核同，policy副本和pending状态另行封口，不能只核weights。
C建议仅关闭outer `jax.enable_policy`同步包装，不创建policy执行副本；仍保留完整pol/value训练与所有state叶，
不得把此选项当冻结actor。此时policy-copy检查明确记不适用；R/特征/native另启正常policy worker并真正sync。

replay精确来源为原run
`/data/Policy_Discrepancy/runs/dl-pilot-r3-20261013-13ec72a/dmc_quadruped_walk/N-seed0-attempt01/replay/`，
源Git `13ec72a214b0c03b886b2c7fac52c8f089966dec`，库存raw SHA
`3339b71ca033d5fc065130751fe163aca05a684b2f14f570ab2018219a3a706d`。
固定1280文件、16链、1,001,008行、1008 reset、1M动作和999,984个合法65行起点；不静默skip。
每次新入口认证原字节与末遍稳定性，输出一律放原run外fresh目录，原NPZ不写入。

| 阶段 | 固定对象或后续建议 | 推进条件与边界 |
| --- | --- | --- |
| C：已采用cal-only | 128个独立抽样批次；完整初态只读反事实；不累计学习/不回写cache | 可独立完成或失败；资格与系数先封口、CPU核验后才讨论T |
| R：参考来源 | 完整独审R5的全部处理前历史，或原已采用的条件normal-reference | 仅按完整性选择，不能按科研正负；C不等待这一来源 |
| T1：有限链 | 0/B/N_nr/S_nr，后三者各4096成功完整更新；初态与终态读出/native | C_N与C_S有效、来源可用且本阶段另行采用；不搜索中间终点 |
| T2：选择性 | P_nr/W_nr各4096更新，从共同0重新开始 | C_P/C_W事前有效，T1损伤/保留链及N相对有效S的增量成立、无明确native损害；不重新挑B/N |

0为不更新的共同完整初态，最多五个学习臂。T2是事前声明的条件分支，未执行必须明记原因；
未执行或无力的P/W不能被当作已排除的解释。C结束不自动启动R、T1或T2。
不要求先证跨任务/训练seed正式胜出，也不把R5科研阳性作为普通压缩存在性前提。

## 六对象的精确定义与新mask

在消费context后的16×64 loss位置上，令原H1邻接mask为m，当前`obs.is_first`为r，
额外干预mask为`A=m & ~r`；`K=KL(q || stopgrad(p))`为原rep raw KL，`E=max(K-1,0)`，
`f=1-1/(1+20D)`，`e=max(exp(-d_q)-exp(-d_p),0)`，沿原H1真实下一reward/实际action。
所有权重stop-gradient，alpha20、候选rho=.5、原KL有效性tolerance=1e-5；其余位置权重1。

| 对象 | 实际rep权重 | 所控制的有限解释 |
| --- | --- | --- |
| 0 | 无更新 | 前后anchor |
| B | 1，原完整生产loss/`dt_latch.mode=off` | 新离线对象完整loss基准 |
| N_nr | `1-A*.5*e*f` | 原候选在普通位置的总效应 |
| S_nr | `1-A*kappa*f` | 预定义标量释放量与D形状 |
| P_nr | 在A/活跃性/D/K层内置换N_nr完整权重 | 层内权重与对应位置的关系 |
| W_nr | `1-A*rho_W*e` | 去掉D后该reward-support来源是否足够 |

现有`dt_latch.transition_mask`不排除当前reset；N_nr是新显式子集语义，不是原N已如此。
并行工程新增`dt_latch.normal_only=False`兼容默认，C各活动反事实显式True；后续活动学习臂True仍是建议。
False保留原行为/指标。
开启时原`probe.m/v/f_D`、H1标签清理/有效性检查保留，另算`m_intervention=A`和`v_intervention`，不能覆写probe冒充原信号。
原`dl/valid_*`仍指raw H1；`candidate_v_mean`、保护比例与`calibration_num/den`使用有效干预A语义，
新增`dl/intervention_valid_*`明确子集，不宣称开启后所有旧统计仍是原m总体。固定rho=.5 N目标另立`nm1/*`收据。
P的分组mask也用A。普通判断只读裁context后的真实is_first；不删除reset loss、不变mean分母或free-nats。
原rep损失仍为`max(K,1)`；释放代理E忽略常数floor，不能将其误称全部loss、梯度或Adam位移匹配。

## 计划、consec、随机流与计数

cal/训练均按官方合法起点顺序均匀有放回生成16-start批次，不按D/e/K/后果筛窗。
采用C的root=20261101、cal tag=`0x4E4D4341`；train tag=`0x4E4D5452`仍为后续建议。精确plan seed定义为
`int(np.random.SeedSequence([20261101,tag]).generate_state(1,dtype=np.uint64)[0])`。
按固定NumPy1.26.4的PCG64、`cache.make_plan(seed, updates=128或4096)`生成，保存全部原始(UUID,index)、
完整65行stepid及抽样次序并raw hash封口。两计划可重叠；独立随机流不等于独立数据副本，须报重复/源链覆盖。

**配置是`consec_train=1`，不是配置consec=0。** 官方`Consec(length=64,consec=1,prefix=1)`每draw只有一片，
从index0取片，实际批字段`data['consec']`全为int32的0；实际输入16×65，`replay_context=1`。
因此`_apply_replay_context`每批选择第0行cached dyn carry，prevact为原action`[:,:-1]`，
loss obs/stepid与输出cache为`[:,1:]`的64行。annotation强制的第0行is_first随后被裁去。
真实返回carry继续传入，但在这一consec0路径由新批cache覆盖；不是连续consec1的carry路径，也不造zero context。

共同post-load实际计数须记录。`Agent.load`将n_batches恢复为checkpoint updates而非任意保存batches；
本N1M预期n_batches=n_updates=opt.step=249493，actions62563/eval_actions1001/eval_seed1289196729，实际不符就拒绝。
cal第j批用真实`_seeds(249493+j)`的prepare语义生成key，但不调用会递增共同计数的prepare_batch。
128批每次都从同一个初态计算，故cal各批**opt.step仍为249493**；不是128次累计更新。
cal key与后续训练前128个counter的key相同，输入计划不同，不宣称两者RNG完全独立。
T每臂第j次调用prepare_batch消费counter249493+j，j=0..4095，结束updates/batches/opt.step为253589，actions不增加。

P直接沿已有源码：`fold_in(fold_in(PRNGKey(config.seed=0),0x444C),uint32(opt.step.read()))`，
**不使用plan seed、local j或新的随意P seed**，也不消耗Ninjax训练随机流。T的pre-step=249493+j；
C每次复原初态，P pre-step固定249493。分层变化使同key仍可对应不同置换，不能称128个独立P key。
保留真实原始key、P perm、全局step与配置收据；不将arm label混入共同训练root。

T各臂fresh私有cache初始字节相同，随后严格执行
`prepare_batch → train(dict(prepared)) → delayed outs/metrics为空 → take_train_result → strict_update`。
核全部65输入及64输出stepid，不只首行；预验后官方顺序写入，重叠按后batch-row最后写入者。
不用Prefetch、不模拟原online queue；cache/norm/模型的后续分化是干预后果，不强迫相同。

## C：独立cal-only的实现契约

C只需已认证checkpoint/replay，不需要R5统计、source标签、native或任何完整机制阳性。
原`PrivateReplay.batch`会设置_pending并要求strict_update，**不能直接循环读取128批而跳过回写**。
并行工程新增的`read_batch`及`dl_nm1_counterfactual`须先有限验收：同官方_getseq/assemble/annotation/Consec路径，返回相同16×65，
不产生pending、不改cache、sampler/metrics/counters或原始文件；禁止手清_pending、合成/原值回写伪装只读。
对真实计划逐批核源stepid/原始字段和cache不变，首末完整cache digest及原文件hash稳定。

反事实必须保留`training=True`与全部原loss、真实optimizer/norm/slowval更新；所谓只读是**不提交结果**。
优先采用原`agent._train`生产核心：dona与allo的每一叶都从共同初态新建独立device copy，
允许原核心只donate这些scratch buffer；不能只复制Python dict或把活的agent.params/policy副本传给donation。
不调用公共Agent.train，不回赋返回params，不创建pending sync/outs，不推进Python计数或cache。
每调用后原完整state、policy copy、counters/carry及cache封口均不变；scratch opt.step可到249494后丢弃。
核心只接收批次副本、不接收host cache对象；完整cache hash在首末核，不要求每个scratch调用重扫30GiB。
`nj.pure(model.train)`是可论证的辅助实现，但不是未经验证的原生产编译图替代；本稿首选前者。

每个cal批次以同初态、同实际context、同key执行B、B重复、rho0旁路和N_nr真实核心；
比较真实新参数/optimizer/norm/slow-target输出，保留全部128批，不选择更有利的批。
B/B与rho0/B的参数、carry、replay输出及共有训练指标须字节一致，忽略只有活动诊断才存在的额外字段。
失败归为工程/数值测量不合格，不放宽容差；原输入保持不变不能由“调用了pure函数”自行证明。

另用独立完整loss/`nj.grad`参考JIT取得B/N_nr梯度与N/S/W/P reference-forward收据；
复制并丢弃norm scratch，禁止optimizer/slow-target写入或创建叶，保留完整loss而不裁成单rep loss。
它与生产核心分开编译，可能改变BF16融合/舍入/抽样；**不能把reference梯度叫生产optimizer内部逐叶bit-gradient**。
梯度差与真实生产参数位移分报，核两者的前向/样本差异；不将差异自动归因为压缩信息损伤。
现有logging collector只为logging构建、允许norm scratch、没有optimizer更新，不能直接充当上述全部接口。

保存可独立重算encoder/posterior差值统计的参考梯度及B/N实际新参数原值或无损差分、源叶清单/shape/dtype/hash；
posterior路径由真实访问叶与RSSM obs/obslogit源码映射封口，不能把整个dyn混称posterior。
逐批联报模块范数/方向、实际不同参数数、ULP尺度、生产前向差异及完整scratch-state digest。
其它模块至少保存完整叶hash/有限性/位移统计；无原张量的部分不声称CPU独立重算其optimizer运算。

## C：系数、对照效力与停止分类

在128批共同初态N_nr reference-forward上，保存实际F32 K/e/f/A及`E_f32=max(K_f32-1,0)`；
解析系数将这些实际值提升F64后逐项乘积并用`math.fsum`汇总：

`kappa=.5*sum(A*E*e*f)/sum(A*E*f)`；`rho_W=.5*sum(A*E*e*f)/sum(A*E*e)`。

分母为零、非有限或系数不在[0,.5]，对应对照未校准；不填默认值。冻结解析值、实际F32系数、
实际F32权重及其实现的释放`math.fsum((1-float(v_i))*float(E_i))`；不复用GPU F32归约充作F64汇总。
生产原F32 release指标另报，不用只有解析公式相等代替实际精度匹配。
所有门在C开始前封口，不根据C结果换量化/阈值、加cal批、放大rho或换e归一化版本。

| C资格 | 已采用固定条件 | 失败含义 |
| --- | --- | --- |
| C_N覆盖 | 至少128个不同原始位置、至少8源链，A且K>1且实际F32 v_N<1 | 普通干预覆盖不足，非稀有子群普遍无效 |
| C_N作用 | 至少16/128批在同一预先封口的encoder或posterior路径同时有非零reference-gradient差和真实核心参数位移差；所有比较有限 | 候选在该初态/精度下作用不足，或参考/生产差异待定位；不扩U补救 |
| C_S、C_W | 共同N reference-forward的F32释放>0，按该forward算出的S/W实际F32权重释放相对N偏差≤1% | 该标量对照未获数值校准，不能用它排除解释 |
| C_P | 原保护活跃位置中实际权重差>1e-6的比例≥50%；P释放相对同state N目标偏差≤10% | 置换无力或剂量不匹配，不可宣称对应位置无关 |

P在cal的A且K>1位置上取D的1/3、2/3线性quantile与K的median，重复边界去重后冻结。
按已有A/活跃性/D/K全局B*T分层、side='right'置换完整v；不临时粗化分层、改key或移除singleton求通过。
C_P的保护分母沿现有源码`A & (K>1) & (v_N<1-1e-6)`，空分母未获资格；
N覆盖的`v_N<1`与P的1e-6效力阈值不同，均须报告。联报移动率、真实权重变化率、singleton及每层计数。
这里校准的是共同参考forward上的F32标量，不声称S/W/P不同生产编译图的全部参数位移已经匹配。

资格算式固定如下：C_N逐批计算`(enc_grad_diff_nonzero & enc_step_diff_nonzero) OR
(post_grad_diff_nonzero & post_step_diff_nonzero)`后计通过批数≥16；不是必须同一模块单独通过16批。
这里比较数值差，不以+0/−0或hash变化代替非零作用。完整张量必须有限；零范数时角度明确不可用，不伪填0或另判失败。
P cal分层quantile、C/T释放与效力都pool全部位置访问次数，不按UUID去重，也不平均batch比例；
P变化率=`sum(protected & (abs(v_P-v_N)>1e-6))/sum(protected)`，
剂量偏差=`abs(sum(release_actual)-sum(release_N))/sum(release_N)`。
只有C_N覆盖的128位置/8链使用原始stepid去重；重复曝光不能增加这一覆盖数。

C失败可形成一个完整的有限停止结果，不需等4096或native。C_N失败停止本候选的后续学习；
C_S失败不自动降级成新B/N协议；C_N/C_S通过才具备T1的必要资格，仍不等于T1已采用。
C_P/C_W失败可保留T1的有限问题，但T2/完整选择性主张不可激活。不得在同一cal追试新候选直至通过。
这些128/8/16、1%/10%/50%是本次C采用的有限覆盖/精度选择，不是功效保证或普适常数。

## T：4096学习与对照资格

建议每学习臂U=4096完整更新，16×64 loss位置共4,194,304次曝光，约为buffer行数4.19倍；
重采/重叠/边缘不均使它不是每行恰好4遍或419万独立样本。只取初态0与终态作主比较，不在中间挑最好checkpoint。
T1先B/N_nr/S_nr，三臂顺序按PCG64 seed20261103对固定列表一次置换并封口；
T2对固定[P_nr,W_nr]独立使用seed20261105一次置换。0只评价一次；顺序不根据终点变更。

S/P/W逐更新记录实际释放与其**自己当前state**上的固定rho=.5 N_nr目标，训练累计相对偏差≤10%
才作为本对象的标量释放近似控制；P还须整个训练保护权重变化率≥50%。N目标零则比值不可用。
W配置rho=rho_W时原candidate_excess_release也会用rho_W，不能误用为固定.5 N目标；须新独立nm1收据。
同state目标指同一批次forward的K/e/D上代入固定N公式，不冒称另一个N生产JIT反事实或完整gradient匹配。
跨臂实际累计释放同时报告，不在训练中重校准kappa/rho_W，也不强制cache/状态回到相同轨迹。
匹配释放不匹配全部梯度/Adam位移；无效S/P/W只允许有限B/N结论，不能把无显著差叫等效。
缺失更新、非有限、cache丢写或硬时限失败保留全证据，不拼接/续跑成完整对象，不自动延U或加seed。

## R与读出：固定来源、独立物理量和终点判据

沿已采用[正常参考来源](DL_NORMAL_REFERENCE.md)与[读出定义](DL_NORMAL_READOUT.md)：
完整独审R5可用时使用全部512个处理前current状态及各自真实prefix，family0..7 fit/8..31 held。
只在R5全对象完整性/独审不足时用独立normal-reference的32×14点、t64..896，
env24576+f/eval0x4E4D3100+f，fit112/held336；不复用R5 partial或300k来源冒充N1M，不因科研负号切来源。
R5每unit独立prefix不能拼成一条episode；新normal-reference每family一条连续1000步历史。

标签固定为当前官方Physics的torso_velocity()[0]主、torso_upright()辅；CPU逐全部source transition重放并验证只读query。
标签不是关键观测/应保护/最优动作真值。N1M完整load及source冻结先于feature使用；不以e/head/actor效用作独立标签。
每个模型真正reset后沿共同真实RGB与source实际raw连续重推q历史，各模型自己的h供当前q/p共享；
使用source原公开EVAL root/counter，保存实际子key对接，不喂模型新预测raw、不用旧latent/cache；p query不进入q历史。
只读特征提取不得改训练state/计数/cache；源码需明确current wrapper的新namespace与原public-policy oracle对应，不能冒用R4白名单。

固定h4096与官方unimix概率32×32，API为`softmax(model.dyn._dist(logits).output.dist.logits,-1)`；
h+q_prob、h+p_prob各自refit，h-only描述；各组只用fit中心化/group RMS，ridge .1、mean MSE、无罚截距。
使用已实现纯NumPy读出，不搜lambda/维数/非线性probe。kernel trace≤2、斜率effective df≤20，另加截距1。
概率比actor一次sample更丰富，p是h的非线性函数；不将q-p差叫条件MI或actor利用。

主量为velocity_x的`d_damage=R_B(hq)-R_0(hq)`及`d_rescue=R_B(hq)-R_N(hq)`。
建议两者都满足paired family-bootstrap双侧95%下界>0且点差≥.01×fit标签方差，才称有限损伤/保留链建立。
所有模型用0源fit标签方差作共同尺度，零方差则该归一化主量不可用，不填0/0；报告常数预测与绝对物理单位MSE。
先family内等权再24 held family等权，10000次paired bootstrap、seed20261040、linear分位、所有差共用resample；不refit。
这是固定fit/模型/训练流条件下的pointwise区间，不传播fit或训练seed不确定性，不宣称选择后精确覆盖。

T1损伤/保留链或`R_S(hq)-R_N(hq)`的95%下界>0未建立，不追加预算或为完整选择性链执行T2。
这利用完整强主张本来必需的S比较提前停止，不把跨零解释为S/N等效。若T2激活且有效，完整选择性支持还需
`R_S(hq)-R_N(hq)`、`R_P(hq)-R_N(hq)`、`R_W(hq)-R_N(hq)`三个95%下界均>0；不挑最好对照。
upright、hp、h、q相对p优势为辅助，不能救活失败主量；不同风险模式须如实区分prior追上、促进学习与保护相容。

## Native与结论边界

每个实际完成对象一次32新episode，env32768+j/真实EVAL seed0x4E4D4500+j、counter0..999，j0..31；
保持原生posterior+sampled actor+各自真实反馈、1000步上限，fresh环境，不作context替换或fixed-openloop。
从该对象完整state加载/sync policy后封口，评价后训练state/counters/cache不变，eval计数恢复并核真实policy副本。
共同seed计划不保证RGB逐位相同；native测操作性行为，不翻案旧pixel-identity失败。早终止不制造后续transition。

主指标actual F64 episode return/1000；按32 episode计划配对bootstrap10000次，seed20261104，linear双侧95%。
N−B下界>0且点差≥.01时，才加称该有限对象有native改善；95%上界<−.02是明确损害，停止当前性能候选/T2。
没有native正区间不否定信息链，也不证明安全或非劣；区间宽必须标性能未知。
32评价episode不是32训练seed；单初态/单任务/一次离线更新不建立正式跨任务性能。

最强反例必须写入最终结论：N只有loss数字变化而无真实位移；B没有绝对hq损伤；S/P/W同样足够；
旧head变差但refit恢复；只有p改善而q未降；读出改善而native不改善或明确受损。
即使所有环节相容，也不识别唯一中介、全部信息量、任意读出族或nuisance/OOD鲁棒性。
不因失败自动改为归一化e/非法rho；那是新候选、新规范，不能在本对象续搜阳性。

## 工程预算、CPU独审与产物

以下明确区分**C已采用硬限/资源**与**后续未采用建议**，均不是ETA：

- C：固定sv3物理GPU5、UUID `GPU-79704918-e096-92b4-252c-e3244497a8b3`，一张独占A100串行；
  在当前R5权威终态及完整独审或失败封存后，fresh核空闲、身份与资源记录才执行，不并行抢占另一卡、不停止陌生作业。
  整个入口固定3600秒（最多1占卡小时），含导入/JIT、全部128反事实、张量证据/封口/清理；独立CPU全入口900秒。
  C没有环境/渲染，不启动EGL；真实compute UUID与owned进程仍需前后核验。CPU验收本身不授权提前GPU执行。
- T1+条件T2+全部对应特征/native：尚未采用的一个总入口21600秒（6GPUh）建议，不是每臂6小时；CPU全对象独审1800秒建议。
  只有采用后才生效，不用这6小时给C未实现工作兜底，不在条件分支重新起算预算。
- R若需新source，沿原已采用1800秒单卡+900秒CPU；这不是改R5或改其物理资源分配。
  T具体资源仍待明确；C的物理5安排不自动授权R/T换卡或并行。

因此推荐计划上限C+T为7占卡小时，条件新source再+.5；单独设备probe另记。
其中只有C的1小时在本次采用，T的6小时未采用；这个混合上限不是完整研究剩余总GPU小时，不保证完成或功效。
每臂私有数组实测约31.85GiB主存，串行释放；cal梯度/参数、T全部回写证据的磁盘与峰RAM在实施前估算并记录，
可无损压缩，不删必要证据求省时；不能拿NPZ11.42GB当RAM。
源码本机实现/有限测试/提交后仅同步承担任务的新clean checkout，启动前fresh资源检查；不在服务器改源码。

C CPU审计：先全部raw hashes、终态/owned退出/完整Git blobs与配置/输入库存，再可信decode；
核128计划/全部65跨度、context/mask/P key/perm、F64系数及F32权重、覆盖/匹配门、保留张量的差值算术和原state/cache不变收据。
CPU不重新执行NN/反传/Adam；生产核心确曾运行与fullstate只读属于认证实现/完整收据证据，不冒称独立NN重算。

T CPU审计：认证所有臂4096连续更新/root/counters、完整更新/缓存回写图与最后写入者、输入末遍SHA、初终state与评价前后封口；
按全部已存cache输出独立顺序重放回写并核对逐批输入cache digest，不恢复sampler；所有窗口/更新不可抽样替代。
核reference来源完整物理重放/标签身份（已有审计证据可按完整原图hash复用）、每模型feature/target对齐和fixed-fit统计；
native全部真实transition按官方CPU physics独立重放、raw→applied/F64/F32/flags/full snapshots/终态精确核验。
不得把hash链、stored RGB、NN feature或梯度收据说成CPU重执行模型/render；这些边界单列。
若1800秒不足，当前对象工程失败，不用只审部分或扩大容差发布联合结论。

建议fresh产物分为`nm1-cal-<git>/attempt01/`、`nm1-trial-<git>/attempt01/`及各run外独审输出；
保存manifest/actual config/budget/process/resources、plan、原始日志、cal完整与资格分项封口、每臂state/update/cache/feature/native图及终态。
C完整失败与T未启动也是明确交付；生成结束待独审与验收完成须分开。只在真实证据闭合后给相应有限科研判读。

## 已采用范围与后续尚需root决定

| 决定 | 当前状态 |
| --- | --- |
| 本次范围 | 已采用C DL-NM1-cal-r1；不自动采用R/T学习或性能门 |
| 新语义 | C已采用normal_only显式True、默认False兼容；原probe m/v保留、另记A |
| C固定量 | 已采用128批/上述PCG64 namespace与真实prepare keys；128/8/16门；每批B/B/rho0/N核心 |
| C数值与P | 已采用S/W1%、P50%与10%、active D-tertiles/K-median；原seed0/global optstep P key |
| C反事实证据 | 已采用原_train独立scratch位移+独立reference梯度、无pending/回写；不要求虚假的bit-gradient等价 |
| C资源与硬限 | 已采用sv3物理5/R5后fresh核查、3600秒GPU+900秒CPU；实际执行Git/配置/证据体积待工程封口 |
| 后续科学预算 | U4096/T1-T2触发/21600秒总T/1800秒CPU仍待采用；不能据终点更改 |
| 主辅判据 | 固定readout已另采用；本稿.01 fit方差门、32 native/种子与+.01、−.02界仍待采用 |
| 来源与读出接口 | 完整性选择沿既有条件规范；实现/实际来源/新feature wrapper和完整独审尚未完成，不由C自动启动 |

本次采用仅本节C；尚未实际运行。执行时必须将阶段/数值/资源/版本绑定真实Git blobs与完整收据；未采用部分保持建议。
改变科学量须另版并先封口，不回填旧结果；必要工程修复不能改变上述科学语义。
下一首选是当前chat完成C的窄实现与有限CPU验收，再用确定提交和实际分配资源执行C；
完成标准是128批完整、共同初态/cache只读、生产位移/参考梯度分报、系数/资格有效或明确失败及CPU审计闭合。
这一步无需先证明正式效果，也不得以交接代替当前已授权工程工作。
