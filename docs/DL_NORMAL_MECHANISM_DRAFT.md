# DL-NM1 普通回放机制对象草案

2026-10-15，版本DL-normal-mechanism-r0。**跨Agent只读审查后的建议，尚未采用训练协议或启动新训练。**
用户完整预验证授权允许Agent论证必要干预；不把这些选择写成用户逐项确认。
本稿形成时R5仍只运行cal，没有读取中间D/e/Y或为求阳性调整其设计；R5规范、7200秒和旧失败不变。

## 拟检验的有限链条

同一个完整N1M初态，普通非reset/free-nats活跃位置的rep干预，是否改变实际梯度/optimizer位移，
并在共同真实历史上保留可访问的任务状态量、改善整个更新Agent的native行为？
这三环相容时支持有限保护机制；不识别唯一信息中介，也不要求先完成正式跨任务/训练seed效果。
固定旧head变差、单看KL/q−p变化、非零gate或标量loss释放，均不能单独证明信息被压缩损坏。

## 最小组与完整目标的区别

首选分阶段0/B/N/S：0为共同初态只读anchor，B为原完整loss，N为原候选gate，S为弱D形状的强度对照。
这是建议，不是已冻结组数、更新数或GPU预算。B从N训练所得N1M分叉，不是历史B/D重演，不能归因早期r3差异。

| 对照 | 回答的问题 | 无力时不能声称 |
| --- | --- | --- |
| B | 候选rep干预的总效应 | 保护改善该对象 |
| S | 是否只是该标量压缩释放总量改变 | 排除一般减压解释 |
| P | 活跃、D/K层内的对应位置是否重要 | 选择位置有效/条件对应性 |
| W | Dt相对reward-only是否有必要 | 双信号或Dt不可少 |

局部B/N/S能先判断链条是否存在；完成完整“双信号选择性”主张仍需有效P/W，不能把首轮局部通过当整个goal完成。
若S同样好，结论止于一般减压；若W足够，需据证据重新论证方法，而不能坚持Dt必要性。

## 正常学习接口与数据真实性

普通持久化replay已全内容核验，见[DL_REPLAY_INVENTORY.md](DL_REPLAY_INVENTORY.md)。
保存cache跨更新年龄混合，sampler/queue/env不可恢复；此处只能另立受控离线普通数据对象，不冒称恢复原online流。
同初态必须包含完整params、optimizer/moments、norm、slow target、counters及同一post-load封口；
不默认清momentum或冻结prior/actor，不删除rec/dyn/reward/con/imag/repval等原分支。

两臂固定同raw stepid/obs/action抽样清单和训练root，初始cache相同，此后各臂私有cache按官方流程回写。
cache/carry/norm的后续差异是学习中介，不能强迫多步所有tensor相同并仍称正常反馈。
真实流程保留官方Replay annotation、`Consec(length64,consec1,prefix1)`和replay_context1，
批数据实际为16×65；不能跳过annotation、裁掉context或改成全zero carry。

**源码已核的项目实际时序**（dl_pilot_train.py与protocol_v1.py）：
prepare_batch→train(dict(batch))→要求delayed outs/metrics为空→take_train_result取得当前更新→私有replay.update。
通用Agent.train默认挂起一call是源码事实，但不能据此把本项目已用过的同步回写换成延迟回写。
新离线schedule可明确去掉Prefetch，不能称复刻原在线队列。所有out stepid跨度要核对，防止官方update的KeyError被静默跳过。

正常严格私有cache适配已作为必要工程实施（scripts/dl_normal_replay.py及有限fixture），
不定义科研更新数/折、不加载真实学习模型或授权训练。每臂数组约30.6GiB以上，建议串行，不能按NPZ11.42GB估RAM。

## 独立任务读出和真实行为

建议复用完整通过来源/退出/physics审计后的R5全部pre-treatment source，省去另采GPU数据；
此后新对象仍称探索性，不能重新把已开数据称blind。R5主signal阳性不是不同压缩存在性问题的逻辑必要前提，
但数据完整性前置不能跳过，R5未完整/独审时不得把prefix当已认证新数据。

最小独立标签建议为官方Physics的torso_velocity()[0]（局部前向速度，主）和torso_upright()（辅），
与真实quadruped walk奖励构成有源码依据；已有analysis/dl_task_semantics.py的只读查询可复用。
不重用e的shadow nextreward、H1 head或R5 actor-effect作为信息真值；这些物理量也不是“正确动作/应保护”真值。
仅每unit current t共512点；若改用prefix帧，须事前定义权重，不能让长prefix或帧数充当独立重复。

每个0/B/N/S模型从真正reset沿相同RGB和实际raw完整prefix重新推断，不能用旧cached latent/carry，
也不能喂其自己新预测的raw。当前q/p共享该模型重新算出的h。分别重拟合同容量/同预算读出，
只用fitting family标准化；分报h、h+q、h+p绝对held risk与q−p，family为统计单位。
此结论只涉及固定读出族可访问的两个任务量，不是信息论总量/最优控制真值。
独立新native episode评价完整更新Agent；跨模型RGB微差属于操作性行为差异/随机性，
不能沿用旧失败对象要求初帧逐位相等的估计对象，也不能把旧native失败改为通过。

判读：B相对0绝对q读出变差、N相对B保留，且行为方向相容，才是该有限对象的保护相容证据。
B未损坏而N学得更好，应称促进学习；只有q−p变小而p改善/q未降，可能prior追上；
旧head失效而refit恢复是坐标/利用问题；readout好而行为不改善不证明性能收益，反之也不能归因该信息机制。

## 匹配与尚未冻结的关键项

令a=m*max(K−tau,0)、f=f_D，独立普通cal上S可解析取
`kappa=rho*sum(a*f*e)/sum(a*f)`；W可取`rho_W=rho*sum(a*f*e)/sum(a*e)`。
合法范围均不超过rho，零分母是未校准；不把N放大去追强D。只在cal冻结，不据held收益调值。
匹配该标量不等于匹配全部gradient/Adam更新；真实释放漂移与模块gradient/位移必须联报。
P须验active protected权重真正变化、singleton和释放偏移；无力对照不能拿来排除替代解释。

仍待执行前一次冻结：更新预算U/完整入口时限、唯一读出算法与lambda、native episode/seed/order，
S/W独立cal清单及释放漂移资格、P分层/有效性判据、合法随机流和完整验收链。
旧128更新和历史11.5GPUh既不证明功效，也不决定新对象ETA；不得看结果再延长U、换标签或搜索lambda求支持。
本稿没有补默认值。首选继续当前chat监督R5到终态并独审，完成严格replay工程，再据证据采用一次有限机制协议。
