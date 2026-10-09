# 04 · 原始 return 独立复核与真实 size50m CUDA 补验

日期：2026-10-09。依据：EXPERIMENTS C/C.1、CODE_SIZE50M_REPAIR、CODE_PROTOCOL_V1 和用户本轮有限工程授权。没有正式百万步训练、正式 c 或 M2 效果结论。旧报告及原始产物完整保留。

**后续正式准备索引（未启动）**：完整配置、16条新计划索引及sv2 GPU0–3的CUDA UUID/EGL实际映射已准备，见 `docs/M2_V1_FORMAL_PREPARATION.md`。本报告的真实训练实测SHA与边界不变；后续只有无训练设备探针和CPU配置解析，不能写成新增训练验收或正式启动授权。

## 版本与执行范围

接手本机 HEAD 为 `27261bf6df875939195879ab23e1dc161507fe5c`，工作区干净，未回退。它相对统计实测 `1f5b5a385346da01e1108c90114fed721c8bbaf2` 只有三个文档变化；相对完整 CPU agent 实测 `c4de170928e9314559be793634209320d97fe0b4`，运行路径不变，后续增强离线审计/测试及文档。

- 独立旧统计复算、四任务新增 CUDA 主运行、完成重开、中断拒绝、冻结输入和 39 项 CPU 回归的实际 SHA 均为 `27261bf6df875939195879ab23e1dc161507fe5c`。
- 本轮发现 EGL 设备未固定后，在本机只修改 `scripts/run_size50m_integration.py`：为子进程设置并记录 `MUJOCO_EGL_DEVICE_ID`。提交 `51ca42803e5c211d1be26cc164b9012f3388839d`，没有改 agent、runner、统计器、科学参数或测试；独立渲染绑定补验实际执行此提交。
- 最终报告提交和同步状态在完成回复给出完整 SHA；文档提交不能写成重新完成了训练。所有服务器源码均从本机 Git bundle 快进同步，没有服务器直接修改项目源码。

新服务器根目录 `R=/data/Policy_Discrepancy/runs/engineering-return-cuda-20261009-27261bf/`。sv1/sv3 的 R 仅保存只读核查及环境记录；新增 CUDA 仅在 sv2。每条运行保存完整 config、manifest、命令、seed、依赖、GPU、环境变量、资源采样、退出状态和原始 stdout；运行目录名不替代实际 manifest SHA。

## 独立复核旧统计与16条原始运行

本机新输出 `analysis/outputs/return-v1-04-independent-27261bf/`。另一个新目录 `return-v1-04-require-complete-27261bf/` 实测 `--require-complete` **生成审计后退出2**，stdout/命令/退出码保存在 `return-v1-04-independent-verification-27261bf.json`。

1. 核对 03 的 `return-v1-validation-summary-1f5b5a3.json` 中六项产物哈希全部正确；统计审计引用的102个唯一输入文件（96个源小文件、3个证据包、3份快照审计）和 run index 哈希全匹配。
2. 用当前确定提交重新执行生产 CLI，六张 CSV 和 `tables.json` 与 `return-v1-engineering-audit-1f5b5a3/` **逐字节相同**。独立以 `math.fsum(returns)/episode_count` 核查48点评价均值，沿用事前 `atol=1e-9,rtol=1e-12`；数量、动作步、seed、ID精确核对。
3. 三台实时只读16条原目录，共96个 manifest/evaluations/config/execution/audit/final 文件的哈希与本机镜像全同。重新读取48个可信实际 checkpoint，参数摘要和更新counter与原04快照审计全同，终点参数与final一致。新实时结果在 `return-v1-04-live-verification-27261bf.json` 及 `return-v1-04-27261bf-tools/sv*-old-live-audit.json`。
4. 原始运行版本仍为 `1393548fb6e46d28f92a834204f7b09495a3eab1`；四模式完整性/工程 c 来源与先前验收保持一致，未重跑、覆盖或合并旧运行。它们均为4098动作、间隔2049、每点2episode、窗口 `[2200,3600)`。
5. 旧日志的 `scores` 映射为原始 return 有源码依据：该提交 evaluate 累加接口reward、Driver逐条完成episode；FromDM 的 reset reward为0、DMC ActionRepeat已累计reward，通用包装无奖励变换。**这属于经核对源码和运行长度/seed/快照支持的来源推断**；旧日志没有逐transition reward/terminal原始轨迹，不能声称重新直接求和验证了每一个reward。

旧表保留96条episode、48个工程评价点；`source_complete=true`只表示各自2episode工程配置满足。正式点 `protocol_complete=false`，AUC/tail、任务/总体指标和三项Dt差值均为null（CSV空），不是0。没有除4098、缩区间、换末段点、补插或重新加权。原始return协议已冻结，归一化score暂缓，不再列为02未决阻塞。

## 新增真实 CUDA 主验收

四个目录为 `sv2:R/<task>/report-return/`，模式 logging-only，全部从头初始化。固定 clean、image=True/proprio=False、size50m、seed0、repeat1、alpha20、free_nats1、dyn/rep1/0.1、ac_grads=False、reward_grad/repval_grad=True；CUDA/bfloat16、batch16×64、replay_context1、16训练环境/1评价环境、train_ratio256、replay5e6/chunk1024/online/uniform均保留。

工程区别：budget4098、间隔2049、评价 `[0,2049,4098]`、每点2完整episode、匹配 `[2200,3600)`、engineering_fixture=True、report_every_actions=2049、prealloc=False、logger=jsonl。正式协议为100万/5万/21点×10episode、窗口 `[100000,300000)`、engineering_fixture=False、默认report关闭；未自行改变它。prealloc/logger是本轮工程执行值，正式执行前仍须统一记录。

| task | JAX物理GPU | 主进程退出 | 动作/成功更新 | 报告/评价审计 | 重开 |
| --- | --- | --- | --- | --- | --- |
| dmc_hopper_hop | sv2 GPU0 | 0 | 4098/517 | 2/3 | 0、关键产物不变 |
| dmc_quadruped_run | sv2 GPU0 | 0 | 4098/517 | 2/3 | 0、关键产物不变 |
| dmc_quadruped_walk | sv2 GPU1 | 0 | 4098/517 | 2/3 | 0、关键产物不变 |
| dmc_reacher_hard | sv2 GPU1 | 0 | 4098/517 | 2/3 | 0、关键产物不变 |

每条均有16初始reset、259训练policy调用、517训练batch key；预填充计入4098，6000评价动作排除。真实训练环境各仅约256动作，仍不覆盖训练episode末尾/reset循环和长期replay周转。

`R/new-resource-and-quality-summary.json` 保存独立审计。24条新schema v2完整episode、12点评价通过：scores=returns_raw、mean=mean_return_raw、原始reward来源/累计定义、完整episode ID/长度1000、派生环境seed、独立动作随机流、目标与实际动作步、唯一snapshot_id及更新ID一致。12个实际checkpoint读取其参数和counter；metadata、latest、完成标记与完整tag核对。报告和评价的8/12行指纹只在runner逐项相等后写入；独立审计又将其参数摘要/counter与实际快照交叉核对，并比较同边界报告/评价的训练fingerprint。训练参数、未来训练key的计数来源、消费batch、replay sampler RNG/metrics保持不变。

新表在 sv2 `repo/analysis/outputs/return-v1-04-cuda-audit-27261bf/`。生产CLI `--require-complete`生成审计后退出2；保留24条episode/12个源完整工程点，正式AUC/tail仍空，不当作正式结果。所有原始命令与配置可由execution/manifest复现；旧16条四模式不与新增logging合并择优。

四个新logging完成后，又从本轮各自隔离的收据冻结工程c，350个窗口成功更新全部纳入；新c依次为0.9999513867034315、0.9999360508578164、0.9999402865341731、0.999928712985215。生成工具实际51ca428、来源实际27261bf。校验和通过，四次正式freeze均明确拒绝工程来源，没有正式c。证据 `R/engineering-freeze-audit.json`；本轮未用这些c新跑constant，也不替换旧主矩阵的工程c。

## 冻结输入、恢复及去重边界

- `R/frozen-input/{first,second}/`：同初始参数、冻结视觉batch和训练seed0的真实size50m CUDA计算，两进程四次更新的损失、参数及匹配S/N完全相同，容差0；两份result文件SHA-256均 `0db8a213eda038b4940a44f131eb8787be38210be10e5f6ba10bea9c89034998`。合成输入rng12345仅为fixture数据生成，未新增训练seed。完整config和命令分别保存；这不是真实DMC轨迹或方法效果。
- `R/dmc_hopper_hop/interrupted/`：2049动作/5更新后工程开关故意中断退出1；相同提交/命令重开退出1，明确拒绝不能完整恢复环境/replay的训练续跑。manifest、两类审计、收据、评价、latest及实际checkpoint全部文件哈希前后相同，无final_state。不是无缝恢复。
- 在新真实收据上验证同ID同值幂等、冲突拒绝；独立小日志注入验证重复持久化ID拒绝。零ledger与五条真实收据的偏离明确拒绝。最后一项是**使用真实收据的小ledger fixture**，不是重启原旧零动作故障目录；原目录保持只读，03已有原零动作checkpoint证据继续保留其范围。正常replay重采样不按transition内容去重。
- 本机15项纯统计测试通过；sv2 CPU两逻辑host设备的 return/protocol/probe/overlay 共39项回归通过，日志在 `R/independent-cpu-regression.log`。没有新增/放宽容差。真实DMC图像非逐位确定性的03定位不撤销，不能用独立真实轨迹参数完全相等替代上述冻结输入/状态检查。

## 环境、资源与EGL设备安排

sv1初查HEAD27261bf、sv2/sv3为d9be557，均干净。sv1八张卡已有作业，sv3 GPU0已有作业，未触碰；sv2八张卡初查空闲，约2TiB `/data`与992GiB可用RAM。只将实际承担CUDA的sv2快进27261bf及后续执行提交，sv1/sv3只读审计，不同步新版本。三台pip check通过、freeze哈希均 `cd50c7936690f9adbc5c35f99b3df2f7a977cef3222582f988979b11f65b6f22`，Python3.11.16/JAX0.4.33/dm-control1.0.48/MuJoCo3.15.0；未升级依赖或删除base-py311。

| task | 指定计算卡采样峰值MiB | 预热动作/s | 每点2episode耗时s | 保存量GB |
| --- | --- | --- | --- | --- |
| hopper_hop | 9539 | 24.42 | 12.38–14.86 | 2.132 |
| quadruped_run | 8088 | 24.40 | 12.75–15.16 | 2.144 |
| quadruped_walk | 6650 | 24.24 | 12.69–15.05 | 2.144 |
| reacher_hard | 6649 | 24.64 | 11.32–13.79 | 2.129 |

吞吐取第二训练区间，排除前64更新与评价；包含环境/replay/收据和本轮报告开销。评价计时包含snapshot IO和串行episode，未含后续checkpoint保存完成。原短预算线性估算仍只能提供约12小时/条量级，不能证明百万步、预分配或scope日志的需求。

**上述四运行只绑定CUDA计算卡，没有显式固定EGL渲染卡**，不能将指定卡采样峰值读成整条运行独占总显存。期间GPU1训练时GPU0有额外图形分配；安装的dm_control/MuJoCo EGL源码确认未设 `MUJOCO_EGL_DEVICE_ID` 时自行遍历设备。即使JAX隐藏其他卡，EGL也需要单独约束。这修正了旧报告调度建议中“只选空闲CUDA卡即可”的不足，旧峰值/外推保留历史和这一限制。

51ca428的独立绑定补验在 `R/render-pinned/dmc_hopper_hop/`，budget/间隔2048、窗口 `[2032,2048)`、每点2episode、report2048，其余固定设置不变。子进程退出0，2048训练动作/5成功更新、1报告/2评价指纹、4完整episode，两个实际快照counter/参数与metadata/final匹配；计算卡采样峰值8088MiB。`CUDA_VISIBLE_DEVICES=1`及`MUJOCO_EGL_DEVICE_ID=1`明确记录。

全部GPU/pmon在 `R/render-pinned/all-gpu-resources.jsonl`，SHA-256 `fe5eb94d99bf131b9951d2a5f59bff7d4ecb64c8efc4dc44843911ebdb1a0bac`。按执行记录主PID2725242核查：计算与C+G上下文只在GPU1；观察到16个G进程也只在GPU1，和配置的16个DMC worker一致。**已验证的是pmon实际落卡；这16个G进程的worker归属是结合数量/时序/配置的来源推断，未采集完整父子进程树。** sv2 GPU1 UUID为 `GPU-1e27fd5e-4231-926a-754f-e7d4c37d41ab`。

首版全卡断言被GPU0出现的非本轮execution PID2726553（类型C）触发，包装器退出1，主运行仍退出0；此进程事后已退出，未确认命令或项目，未操作它。保留首版断言及原始全卡数据，独立PID/图形归属审计写 `R/render-device-audit.json`，不放宽数值容差、不把其他卡计算进程计作本轮渲染。全卡GPU0峰值765MiB含未归属计算，排除于本轮显存估算；其他卡枚举期间观察到1MiB小分配，不称为全部设备始终零占用。仅本次sv2/index1落卡映射获证实，其他服务器/卡启动前仍须现场核对。

未来每次运行须同时检查计算与渲染卡、记录两个环境变量和实际UUID；不能仅因JAX卡空闲就承诺不使用其他卡的图形资源。新代码解决子进程显式绑定，不替代资源独占安排或完整进程树审计。

包装器质量记录：quadruped_walk主进程正常退出后，两次同进程包装器的立即重开被1MiB占用严格拒绝；包装器退出后归零，独立执行进程随后成功重开，未放宽占用规则。现象出现在验收包装进程存活期间，未证明只是驱动退出延迟。首版资源审计假定task/logging目录及冻结文件，不能用于新report-return路径；保留失败尝试后改用独立只读路径审计，没有移动或伪造输入。这些包装器失败不改写为训练失败，也不隐藏。

## 交付、判断与下一步

小证据包：sv2 `/data/Policy_Discrepancy/runs/engineering-return-cuda-20261009-27261bf-evidence.tar.gz`，198301字节/207条成员，本机下载核对SHA-256为 `c4f1193c3ac462ee73c9b984bc2a5be214df9a9266508cfa59b1cc8b0078f447`。包含配置、日志、资源、审计、全部外部验收脚本版本和新统计表，排除checkpoint参数/replay/tmp/bundle。完成标记、latest与小snapshot metadata保留。二进制checkpoint/replay仍在sv2，未提交大产物。

本机镜像 `analysis/outputs/return-v1-04-cuda-source-51ca428/`；证据包、旧实时审计、新摘要及辅助脚本在 `analysis/outputs/return-v1-04-27261bf-tools/`。汇总与文件哈希索引为 `analysis/outputs/return-v1-04-verification-summary-51ca428.json`；表格和详细审计均保留在忽略的outputs。外部脚本未写入服务器仓库，不算服务器改源码；它们由本机编写上传到R，哈希及源码随小证据包保存。

该汇总SHA-256为 `8d1b3b119dd6f1819a159c6c2372c6a221df56a705f5180cb63b763305e4c406`，含60个工程评价点均值的独立核对、正式指标全空检查、各输入/产物及外部脚本哈希、恢复/渲染/冻结证据与包装器失败记录。新表的本机镜像离线复算可使用 `analysis/outputs/return-v1-04-cuda-local-index-51ca428.json`，输出另选新目录；完成回复另给最终提交下的复算结果，不覆盖服务器原表。

同步bundle：d9be557→27261bf 的执行包SHA-256 `f2c3f760225dcc9508d2aa74cf063dbcaf3b4b3e1d623f1714f48dcb416d5730`；27261bf→51ca428 的绑定包 `1ac72b38f7da4ee6aaf87626335d0f895f9c99b1079b38e88770a18e4dee4769`。本机/sv2哈希相同、bundle verify及快进通过；最终文档同步另在完成回复记录。sv2 postflight没有运行中的GPU进程、/data约2TiB可用；它是当时快照，后续新作业必须再次检查。本轮自己的所有进程已结束。

本轮满足所列**有限预算工程补验**，没有观察到新的agent/runner/统计故障。原始return已经确认；真实渲染非逐位确定性、旧reward缺少逐transition轨迹、长期训练与正式设备/logger/prealloc资源没有长程实测，均保留边界。工程通过不代表M2有效；正式指标和正式c仍不存在。

首选下一步继续现有04，准备可审阅的正式完整执行配置、计算/EGL资源映射和16条运行索引规划；不再让02重选原始return，不自动创建chat或发送消息。正式启动须由用户另行授权；授权后先四任务正式logging-only baseline，完整完成后逐任务冻结正式c，再从头执行Dt/constant/shuffle。任何工程c不得复用。

可复制交接prompt：

```text
你是现有04 | 运行与结果，项目根目录D:\program\PD\dmr3。
从完成回复的最终完整SHA接手；有限CUDA主验收27261bf6df875939195879ab23e1dc161507fe5c，
EGL绑定补验51ca42803e5c211d1be26cc164b9012f3388839d。HEAD变化先核差异，不回退。
必读AGENTS、CHATS、STATUS、EXPERIMENTS C/C.1、RESEARCH、CODE_PROTOCOL_V1、
CODE_SIZE50M_REPAIR、SERVER_INTEGRATION_SIZE50M、SERVER_RETURN_CUDA_AUDIT和analysis/README。
M2四任务/clean仅视觉/size50m、seed0、alpha20、free_nats1、dyn/rep1/0.1、
ac_grads=false、reward_grad/repval_grad=true及四组均冻结。
原始return口径已确认，score归一化暂缓且不阻塞。正式预算100万含预填充，
排除reset/评价；21点每点10完整episode，固定快照、采样动作、独立流；
正式logging baseline窗口[100000,300000)成功更新活跃位置S/N逐任务冻结c，
四组从头训练，工程c不可复用，无活跃/非法值停止，不改alpha或增加seed。
本轮有限CUDA报告/schema/指纹/快照/完成重开、冻结输入与中断拒绝证据已记录；
真实渲染非逐位确定性保留，不要求独立新轨迹参数相等。
下一任务只准备正式完整配置、计算与EGL实际UUID/映射、空闲资源及新目录索引，
给用户可审阅材料；没有新授权前不得启动正式训练。
服务器repo=/data/Policy_Discrepancy/repo，环境=/data/Policy_Discrepancy/envs/dreamer，
不删除base-py311，不触碰已有作业，不在服务器直接修改源码。
代码修改在本机提交再只同步实际执行服务器；每次运行前检查GPU占用并绑定/记录
CUDA_VISIBLE_DEVICES与MUJOCO_EGL_DEVICE_ID，确认EGL枚举实际落卡。
旧证据R=/data/Policy_Discrepancy/runs/engineering-return-cuda-20261009-27261bf。
新表/审计用analysis/outputs新目录，大产物留runs，正式指标缺失仍为空。
完成后更新文档，给完整最终/实际测试SHA、差异/同步范围、证据、限制、下一步prompt。
正式运行只有用户另行授权后按baseline→逐任务正式c→其余三组推进；
不自动创建chat或向其他chat发送消息。
```
