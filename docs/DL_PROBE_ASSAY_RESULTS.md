# DL 固定合成探针：检测范围结果

2026-10-11。科学版 `DL-probe-assay-r1`／工程r1，冻结规范见 [DL_PROBE_ASSAY.md](DL_PROBE_ASSAY.md)。实际执行与 clean Linux 测试 SHA 为 **`8b7f7a9daf80bd199b4be5391edad6cb128f2931`**。本对象只检查指定测量工具的有限检测范围，不是 Dreamer 方法或真实信息验收。

**执行及独立数值审计已验收，停止本固定assay。** 原真实 H20/H100 裁决保持；最终始终 `stop_after_fixed_synthetic_assay`，不进入 gate／训练，不读取 blind。

## 实际执行与资源

| 项目 | 已验证执行记录 |
| --- | --- |
| 服务器／设备 | sv3／lyg0326，Linux CPU；GPU 0 张，本assay未启动GPU作业，未核查当前占用 |
| 版本／依赖 | clean SHA 8b7f7a9daf80bd199b4be5391edad6cb128f2931，Python3.11.16／NumPy1.26.4；五个线程环境固定1 |
| 数组／随机流 | 8人工组×4state×4sample；h F64 4096维、z F64 精确onehot 32×32；seed20261022，两个固定PCG64 child，基础流一次生成／四case复用 |
| 数值测量 | 原stats字节pin不改；λ1／train-only组RMS／112训练row／16留出row／8fold；合成七控制utility／δ.2／原2000人工组bootstrap |
| 完整执行 | 22:10:32–22:10:33北京时间（预算及终点文件时间）；父CLI terminal总耗时1.2324890201秒／120秒，exit0／无超时 |
| 进程／终点 | worker1920334 terminal，仅自建group清理；四case及输入／结果／plan／summary hash链通过父验收，`complete_verified=true` |
| 工程验收 | 本机21项中17通过／4POSIX跳过（.449秒）；确定commit sv3 Linux21项全部通过（2.598秒），含失败／超时后代清理、无关sentinel保留、验收及末次成功写盘超时拒绝 |
| 边界 | 0真实dataset／blind／Dreamer或actor／模型／physics／render／optimizer；只保存构造的NPZ，不接受真实数据入口 |

硬预算成功检查覆盖最后收据写盘。终端中的 `terminal_elapsed_seconds` 包含此写盘；文件收据的 `elapsed_seconds` 在该写入前采样，两者不能混称。超时后的清理／失败证据尽力保存，不算预算内科学完成。

服务端输出：`/data/Policy_Discrepancy/runs/dl-probe-assay-20261011-8b7f7a9/attempt01/`；源码：`/data/Policy_Discrepancy/dl-worktrees/8b7f7a9/`。本机 ignored 镜像：`analysis/outputs/dl-probe-assay/8b7f7a9daf80bd199b4be5391edad6cb128f2931/attempt01/`。全部14个镜像文件SHA与server逐一相同，完整清单另存同级 `mirror-manifest.json`。

实际命令（父入口不提供预算／seed／算法搜索参数）：

```text
cd /data/Policy_Discrepancy/dl-worktrees/8b7f7a9
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= /data/Policy_Discrepancy/envs/dreamer/bin/python scripts/dl_probe_assay_supervisor.py --output /data/Policy_Discrepancy/runs/dl-probe-assay-20261011-8b7f7a9/attempt01 --expected-git 8b7f7a9daf80bd199b4be5391edad6cb128f2931
```

CLI终点与Linux测试日志分别保存于本机 `analysis/outputs/dl-probe-assay/sync/attempt01-cli-8b7f7a9.log`、`linux-preflight-8b7f7a9.log`。原server canonical仓库干净d9be5574878d2c3f117cc6667be844f9a527eb84保持，未改本机main／sv1／sv2；GitHub是否同步以最后Git收据为准，不以bundle替代远端验证。

## 四个固定结果

每组constant=.75；有声明的feature-only decoder每sample均选正确动作，reward=1。全部四case外部合成选择依据成立：32state／8人工组／168跨组不相交δ集合对；168是组合计数，统计单位仍为8人工组。下表区间只是固定人工组描述，不是真实任务置信结论。

| case | q reward | p reward | q−constant（名义95%） | q−p（名义95%） | 指定控制与范围 |
| --- | ---: | ---: | --- | --- | --- |
| sparse | .75 | .75 | 0 [0,0] | 0 [0,0] | q含factor0/class1的精确线性decoder，原程序全部选a2，指定有限检测漏检 |
| replicated | 1 | .75 | .25 [.25,.25] | .25 [.25,.25] | q完整选择恢复；预定强cue对照符合预期 |
| history | 1 | 1 | .25 [.25,.25] | 0 [0,0] | 共享h可读，两接口输入／输出精确相同，双方恢复选择 |
| identity | .75 | .75 | 0 [0,0] | 0 [0,0] | q/p输入／输出精确相同；没有已声明q特有decoder，不判漏检 |

history原数值分类未通过q−p门是预期共享信息结果，不是异常。两branch没有decoder的case/interface不能因此称无信息；有限nuisance可偶然相关。未按结果改λ／seed／维数／utility／sample／metric／预算，未只报告阳性对象。

**有限解释**：在明确已知线性解存在且有.25选择收益优势的指定稀疏对象中，该固定学习程序没有恢复有用选择。这足以限制“固定probe阴性⇒接口没有决策信息”的一般推断。replicated／history对照表明程序能检测其他规定的信息结构；它们不是稀疏失败只由λ造成的因果排除。复制cue改变了组能量几何和信号结构，不能用于归因真实失败或保证真实任务功效。

**不能据此推出**：旧真实cal8的q确有信息、prior没有信息、RGB相对完整历史有新信息、正常rep KL删除信息、D+K／DtLatch gate有效或性能改善。原r1实际接口零增量、H100负平均及停止门仍保持。

## 独立审计与证据定位

独立审计 `DL-probe-assay-independent-audit-r1` 已在同一sv3 CPU完成，父监督terminal总1.1428153403秒／120秒、exit0／worker1923993 terminal／only-ownedgroup清理；audit正文写盘前耗时1.066235756秒。未导入原worker／stats，用全部特征值的独立kernel-eigh重算64fit，固定abs容差1e−10、无relative tolerance或cutoff；16200连续标量比较的最大差3.0975222387e−14。所有输入／mean recipe字节、choice／reward／regret／bootstrap及边界exact相同，计出的counterexample仅sparse，全部原始输入hash前后不变。

审计只读取原14个合成产物及实际执行8b的六个Git blob。原direct-solve weights／dual只存digest metadata，独立eigh不声称重获其bytehash；独立检查相应预测、objective、weight norm与算术残差。历史OS清理／末次发布时间只核原收据，不声称事后独立复现过去进程；Linux21项process fixtures及实际父终点是另一工程证据。无真实模型／physics或method效果验收。

独立脚本为ignored `analysis/outputs/dl-probe-assay/independent_probe_assay_audit.py`，SHA256 `2d8e9c56918d6959ad56bd3ab19da11ed68afb4c18d205e1685833de50eb931f`，并非8b提交的跟踪源码。父launcher同级 `audit01-launch.py`，SHA256 `45eeec438e4bd6bc0ab8fa65297f6c6f2dcb8b635bc2815c26a4a6b528ce9129`；两者镜像到sv3 bootstrap并核hash。审计产物在原attempt下 `independent-audit01/`，本机同结构镜像；CLI终点在 `analysis/outputs/dl-probe-assay/sync/audit01-cli-8b7f7a9.log`。最终成功须audit passed true及完整64fit，父进程验收audit正文hash，四份文件与server逐一同。

```text
cd /data/Policy_Discrepancy/dl-worktrees/8b7f7a9
env CUDA_VISIBLE_DEVICES= OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python /data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/audit01-launch.py
```


关键源码／产物 SHA256：

| 文件 | SHA256 |
| --- | --- |
| analysis/dl_probe_assay.py | 03e71d7e3c88dfb04489eeb847e8469237bda2da13731d472541c929a079ca9d |
| scripts/dl_probe_assay_supervisor.py | 17d43f8b7bad98413563b60a6b157f0f8f5ce269ef80dc30144392825be9aa0d |
| 原 scripts/dl_action_information_stats.py | c1d4cfa85707616ffaf0ca35581cf2d90b1ce6b9d48dc1f56bded2a3e0613b35 |
| plan.json | 6f86f0310e3af709178549180b10ab25afdd8b22829b0dbb356e9b332dc1edad |
| summary.json | a613f571a63505261cb81f6b4c5f75e3367b018f2d5a6cac86f67a2f3c2645d1 |
| assay_complete.json | 2f7502edf1248bd4e9121440692c37666e2c6e2aa0c6c37baac3f4e0ea15cb84 |
| supervisor_complete.json | fe4277a999c962b3b109c35652e987d03336698b903889c5eefd56d3d4653c41 |
| independent-audit01/audit.json | 3b46a84add338d7c4b72ebe9b1ab3cfa462d70b5f28a751197642a46a4c5db13 |
| independent-audit01/supervisor_complete.json | 2c5f2e06633d9882eabc52aa0d2065935c10ec50e6b917d21798d66b71e8bed9 |

完整随机流initial／final state、基础h／codes／onehot及所有输入字节hash在plan／各case proof；各fold预测、选择和代数metadata在result。Git bundle SHA c00a039fbcd962ae3dc98fd34c1d3844ee57d7c9b21577523d20a7f6765f9237，仅同步sv3隔离工作区。

## 下一问题与算力口径

审计已通过，停止此assay。首选继续当前总体授权聊天，另立低自由度、整episode inner／outer隔离的真实可学信息探针候选；先冻结容量、共同q/p选参、选择目标、唯一主量和停止规则。不能将已见cal8探索称新确认、用24blind选参、默改旧r1或由此启动torque／gate／训练。

**审查过但未采用，已撤回首选建议**：一个两规则尺度对照曾比较原mean λ1与固定sum penalty γ1（每fit λ=1/Ntrain）。这是两种惩罚标度的敏感性探索；γ1没有自然最优性，亦无稀疏／非线性信息覆盖保证，不能凭本toy结果采用。另有必要反例：把每row精确复制k份时，mean λ1预测不变，而sum γ1等效λ降k倍、可改变预测；四sample不是独立episode，行数标度不能当自然样本量。因此Agent撤回其“自然样本标度”的理由与首选推荐；它至多是对复制敏感的尺度对照，不能作为信息充分检测器。任何另立新探针的隔离建议是8outer整episode LOEO，每outer剩7ep再7-way inner LOEO；inner训练6ep=96row、outer训练7ep=112row，每fit重新train-only处理，q/p共同选规则。尚未采用的inner选择建议按pooled q/p选择utility，再以MSE及强正则作预定tie，不强制剔除constant-like结果；outer主量仍真选择utility，MSE另列。精确tie／共同权重、solver与λ1等价验收、阴性停止／探索性报告及有限预算仍待冻结；不在本assay追加任何fit。解析去收缩会放大小特征值／nuisance、插值高维少样本，不能无留出地绕过这些问题。首选下一规范须保持episode／state等权、MC内平均及精确复制不变性，使用固定mean λ或另有独立依据的事前范围；目前范围仍未确定。

当前规范／合成／统计工作GPU需求0。已有规模的单项模型诊断最低1张独占A100串行，2张可并行独立工作；已分配sv3 GPU4–7足够，未发现本阶段须四卡同时使用的环节。已有约2.5／6.5GiB峰值属于不同单项采样，不能保证未来训练显存或完整预算。总剩余GPU小时未定，没有本轮当前GPU占用核查或释放GPU的新事实。

完整“独立第二信号→normal active rep损伤与保护→排替代解释→后续学习／性能”的方法目标仍未完成；工程成功和本合成反例均不代替它。
