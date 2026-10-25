# 剩余八任务 · sv1/sv2 baseline→Dt

2026-10-24，03/04合并窗口。用户明确“sv1 GPU2、7，sv2 GPU0–5，跑剩下的任务”。本批承接原DMC16清单的后八个环境，**8任务×logging baseline/Dt，共16条**，每卡同一任务先baseline，逐条审计通过后从头启动Dt。前八任务的旧缺项另列，不用它们替代本批任务或改变本批资源分配。

## 2026-10-25 14:10北京时间只读状态（最新）

**8条baseline均训练结束且已有逐条审计通过；3条Dt运行、5条Dt因指定卡外来占用被阻止，尚未启动。** 每条baseline为1000000动作、249493更新、21×10完整episode、退出0；索引状态为 `run_audited_pending_group_and_WB`，尚非全批accepted。此处交叉检查已有audit/评价日志/最后收据及进程，没有在本轮重新执行完整终点审计或从短程fixture推断真实完成。

| 服务器/物理卡 | task | baseline c（本任务正式来源） | Dt状态 | 最近更新开始动作 | 已完成评价点 |
| --- | --- | ---: | --- | ---: | ---: |
| sv1/2 | finger_turn_easy | 0.836720146108 | 运行 | 310828 | 7（至30万） |
| sv1/7 | finger_turn_hard | 0.808270270433 | 运行 | 373088 | 8（至35万） |
| sv2/0 | hopper_stand | 0.814043895789 | 占用阻塞，未启动 | — | 0 |
| sv2/1 | pendulum_swingup | 0.983906275907 | 占用阻塞，未启动 | — | 0 |
| sv2/2 | reacher_easy | 0.914984271616 | 占用阻塞，未启动 | — | 0 |
| sv2/3 | walker_run | 0.489778632710 | 占用阻塞，未启动 | — | 0 |
| sv2/4 | walker_stand | 0.800742077389 | 运行 | 394300 | 8（至35万） |
| sv2/5 | walker_walk | 0.631685994358 | 占用阻塞，未启动 | — | 0 |

c为既有逐条audit记录的S/N冻结结果（显示值经四舍五入，完整值/来源hash在snapshot.json），不输入Dt，不新增constant/shuffle。本轮没有重算全部S/N，不能把读取audit结果描述为新的一次独立c全量复算。三条Dt进程/监督器均存活，最近收据invalid_count=0，已完成评价点均10完整episode；只读W&B于14:11:53—55确认三条running、完整配置一致。W&B当时最后已发布_step分别为easy25万、hard30万、stand30万，早于收据进度；保持原显式step缓存/上传节奏，不用W&B _step代替实际动作计数。

sv2五条Dt于09:50:35—09:52:51北京时间交接时被 `Assigned GPU has foreign/unverified jobs` 门禁拒绝，无PID、无manifest/评价/训练动作，非训练失败或恢复续跑。14:10复核物理GPU0/1/2/3/5分别有root用户的JaxGCRL目录计算进程1622975/1628146/1618095/1641959/1626072，对应PGID1338124/1338125/1338126/1338127/1338129，不在本批进程祖先/session中；项目路径明确不同，其余项目归属不推测。原本批进程已释放这五卡，**卡当前不空闲**；没有停止外来作业、共享或换卡。三条Dt仍占用sv1/2、sv1/7、sv2/4，各17个本项目C/G PID。blocked是终止状态，不是等待空闲后自动重试的queued；补开须先落实资源、刷新占用并使用不覆盖原证据的明确启动记录，不重启整个已有campaign。原分配未变时可在确认释放后按既有授权继续；换卡需用户明确。

两台运行checkout仍为sv1 `6e468ad4a8252f21401eab6a9761626b33659f0e`、sv2 `dbd28b6703d51d61e5d3485dbfea6c413075ea85`，工作区干净；三份监督器仍在。可用空间sv1 405,347,250,176字节、sv2 918,613,098,496字节。此轮仅只读服务器/线上W&B与本机文档记录，没有训练/GPU作业启动或停止、算法/配置/测试修改、服务器源码或Git对象同步；状态文档仅本机提交。此前补测SHA `5c6dd4fae0b35ac9d769dd00bc7d2845dfd226eb` 和所有工程/失败/短时冲突证据继续保留。

本轮证据与可复制继续prompt：`analysis/outputs/dmc8-remaining-status-20261025-1410/` 的snapshot.json、wandb-readback.json、DELIVERY.json、HANDOFF-current03.txt。八任务整体仍不完整，不能报告总体AUC/tail或方法有效；前八旧六条Dt也不由本次核查补跑。首选继续现有03监测三条运行，优先协调sv2五张原卡释放或由用户重新分配，以补开未启动五条；不自动建chat/发消息。以下10月24日段落为启动与故障历史。

## 当前分配与声明

| 服务器/实际host | 物理GPU | 运行任务 | 卡内顺序 |
| --- | ---: | --- | --- |
| sv1 / lyg2153 | 2 | dmc_finger_turn_easy | baseline→审计→Dt |
| sv1 / lyg2153 | 7 | dmc_finger_turn_hard | baseline→审计→Dt |
| sv2 / lyg0360 | 0 | dmc_hopper_stand | baseline→审计→Dt |
| sv2 / lyg0360 | 1 | dmc_pendulum_swingup | baseline→审计→Dt |
| sv2 / lyg0360 | 2 | dmc_reacher_easy | baseline→审计→Dt |
| sv2 / lyg0360 | 3 | dmc_walker_run | baseline→审计→Dt |
| sv2 / lyg0360 | 4 | dmc_walker_stand | baseline→审计→Dt |
| sv2 / lyg0360 | 5 | dmc_walker_walk | baseline→审计→Dt |

冻结配置不变：M2，clean纯视觉64×64 RGB、size50m、训练seed0、alpha20、free_nats1、dyn/rep=1/.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true；batch16×64、replay_context1、16训练env、train_ratio256、repeat1、CUDA/bfloat16、prealloc=False。每条100万动作含预填充、不含reset/评价；0..100万每5万评价，共21点，每点恰好10完整episode，固定快照、采样动作、独立评价流。报告在正式配置关闭；训练/report/评价/shuffle的原RNG实现和派生不变。

每任务自身完整logging baseline按成功update开始动作 `[100000,300000)` 对有效loss位置raw rep KL>1累计S/N，候选w=1/(1+20D)、实际训练权重1；正式c供审计，不传给Dt。本批无constant/shuffle，不使用工程c、默认c或调alpha；N=0/非法值停止报告。保持原始return、实际x梯形积分/固定百万分母、80/85/90/95/100万指定末段五点和缺项不重加权。仅seed0两组描述，不支持强M2选择性归因或跨seed稳定性。

正式根分别为 `/data/Policy_Discrepancy/runs/formal-m2-dmc16-remaining8-serial-20261024-{sv1,sv2}/`；工程根为 `/data/Policy_Discrepancy/runs/engineering-dmc8-remaining-20261024-{sv1,sv2}/`。运行入口仍为m2_dmc16/protocol_dmc16，受限控制器 `scripts/run_dmc8_remaining.py`，结果profile为 `dmc8-remaining`。W&B继续在线 `ychan9liu-tsinghua-university/PD_1`，唯一run ID/resume=never，JSONL同时保留。

**sv1分片调整**：GPU2一度被外来作业占用，完整sv1计划未启动。两个任务各用独立合法card计划，正式/工程根分别追加 `-sv1-gpu7/`（hard）及 `-sv1-gpu2/`（easy），每份仅本任务baseline/Dt。GPU2后来空闲，在原用户授权、原卡/任务/配置下继续，不挪到GPU7、不改变科学配置。原sv1工程失败目录保留。原 `...-sv1/plan/sv1-plan.json` 只是未绑定静态计划，不能当作已运行4条。

22:34:14北京时间只读核查：指定8卡全部0MiB、无C/G PID；sv1其他0/1/3/4/5/6卡有Ray未知作业，sv2 GPU7有未知Python作业，均不使用/不干扰。22:38再次核查资源与依赖，sv1可用590.5GB、sv2可用1,077.4GB；后续启动仍逐卡刷新，旧快照不是预约。两台env `/data/Policy_Discrepancy/envs/dreamer` 的Python3.11.16、JAX/jaxlib0.4.33、numpy1.26.4、dm-control1.0.48、mujoco3.15.0、elements3.22.2、ninjax3.6.3、wandb0.30.0一致，pip check均通过，本次未安装/修改依赖。UUID及全部版本见工程根resources/runtime-environment JSON。

## 工程实现与边界

共享first8控制器增加带原默认值的工程参数（host/GPU集合、validator、容量检查、finisher、进程session），旧入口/范围保持；新控制器固定两台server、GPU和任务，不做自由调度。agent/RSSM/rep_probe/JAX agent/协议runner/DMC源码、目标/梯度/reduction/RNG均未修改。每台只运行自己的2或6任务，JSON索引明确host shard；本机生成的16条配置、8任务随机流及两台计划固定同一Git提交。

新训练进程各自拥有独立POSIX session/process group。baseline退出审计后若仍有GPU进程，只在可核实其属于该baseline进程组时有界等待释放（90秒）；遇到未知/外来作业立即阻止后继，不等待抢卡、不共享、不替换。启动前再次require_free；记录实际占用/进程组证据。这处理可能的退出清理竞态，不认定旧六条阻塞已被证明由清理竞态造成。旧记录没有阻塞时完整PID所有权证据。

每台完成只标本机shard完成，并生成逐任务指标/W&B读回。`dmc8-remaining` 仍要求完整八任务，单台缺另一台的总体指标为null，不因sv1只有2任务或sv2只有6任务而重新归一化。两台完整审计产物汇齐后在独立CPU输出中执行require-complete，才能标整个八任务完成；不自动与前八/原四任务合并。

监督器拒绝已有campaign根，失败/重跑使用独立尝试，不覆盖、不将环境/replay不完整恢复称为无缝。容量门槛延续每条60GB工程估算，sv1四条至少240GB、sv2十二条至少720GB；不改replay/检查点配置。此门槛不是运行期空间保证，需持续检查。

## 前批状态核查（与本批分开）

22:40:34北京时间只读核对旧 `formal-m2-dmc16-first8-serial-20261022/supervisor-final.json`：**complete=false**；八条baseline和cheetah_run、finger_spin的Dt已结束并有逐条audit，另六条Dt因当时GPU占用被阻止，目录未创建。不能写“原16条已完成”，也不由本次新8任务补齐旧统计。此处读取已有审计字段，不冒充本轮重新完整验收；源hash及逐条final/c/audit索引保存在本次sv1工程根 `previous-first8-state.json`。本轮不重开或挪卡运行旧六条，后续缺项需用户明确资源安排。

## 验收进度与下一步

实际代码和本轮测试提交 **`dbd28b6703d51d61e5d3485dbfea6c413075ea85`**。本地49项CPU检查18.195秒通过；首轮Windows的POSIX getpgid mock未创建属性导致单项测试报错，修正仅测试mock，失败日志保留。两台同提交各93项CPU回归无跳过通过（sv1 52.833秒、sv2 48.967秒），覆盖原phase1/2回退/梯度、shuffle/随机流、协议计步/去重/恢复/评价、return统计、原范围和本批固定分配/串行屏障/外来占用拒绝。完整命令在两台工程根check-commands.json。

两台显式CPU/float32完整agent各no-report、with-report、one-eval三条，12动作/9更新/5点评价，退出0；报告/评价变化后的参数、收据和训练随机计数精确一致，共同参数hash `41a913d9d50f02df34e387ba514354e51ca924aeefed36a643aebcd2fabb6be4` 与上一版18b85c4实际补验一致。这是dummy/debug agent fixture，不是真实任务或方法有效证据。代码/目标/随机流源文件不变另有本机哈希记录。

两台生产W&B logger线上写入/读回均通过，finished、seed0、sentinel=1、0动作0更新。指定8卡CUDA计算与EGL渲染PID同卡探针通过：sv1物理2/7→进程内CUDA0→实际EGL2/7；sv2物理0..5→各进程CUDA0→实际EGL0..5，UUID及完整枚举/compute/pmon在gpu*-binding.json。其他卡不参与。

两台通过Git bundle verify/fetch/fast-forward同步上述确定提交，服务器更新前无活动项目训练、工作区均干净，无服务器源码手工修改。sv1从18b85c4同步，bundle SHA256 `78a7405dc7f3165b60422fb816fa40ea007c4cffdbb3f8d442b511ff26cad4c9`；sv2从48d71f1541721208322442f5a1d5977124f3e888同步，bundle SHA256 `663b5caf09e9c42e344f29a53822cca108f3f59601ca0c7e0ad948ace4908e4d`。仅sv1/sv2，无GitHub推送或sv3同步；RESEARCH/STATUS/proposal/TeX现有草稿保留。

八任务×两模式真实size50m CUDA/bfloat16短程在独立工程目录运行。工程用4098动作、每2049动作评价1完整episode、窗口 `[2200,3600)`、报告间隔2049，均不进入正式配置。实际CUDA/正式启动证据后续补记；未把工程运行写成正式结论。

### 资源冲突、版本和当前正式运行

22:48:46北京时间，sv1 GPU2出现外来PID2167282/2167285（其进程组2167224，非本项目baseline组2160981），已阻止启动Dt工程测试；GPU7空闲，但旧批量工程入口在GPU2处退出，GPU7的Dt当时尚未执行。本次没有共享、终止外来作业或自行替换资源。按AGENTS资源规则报告并向用户询问处理，在冲突持续时不启动。**23:04:28及23:05:52重新核查外来作业已退出、GPU2无C/G PID**，据用户原GPU2授权继续同一任务；不是把询问未答复当作批准，也没有变更资源分配。消除冲突的原始快照在旧sv1工程根gpu2-conflict-cleared.json。该次明确捕获的外来占用只能解释本次边界，不能据此确定两天前旧六条Dt的阻塞根因。

为让已授权GPU7独立推进，增加显式 `split-card`：仅能从未绑定计划选用户已指定的同一卡、同一任务，分片保留完整八任务cohort定义，独立目录/run ID、科学配置只差logdir；不允许替换任务或未分配卡、不覆盖已有文件。新增代码/本机50项测试（18.226秒）与sv1实际测试版本 **`6e468ad4a8252f21401eab6a9761626b33659f0e`**。相对dbd28b6仅 `scripts/run_dmc8_remaining.py` 和 `tests/test_dmc8_remaining.py` 两文件，增加资源分片，不改任何训练算法/协议runner/随机流。sv1同步前无活动项目训练，bundle hash `c234e62ca89dec615801302279946046092ce9cdcb4caefeb6c03aeed36ee0f6`；94项服务器回归52.978秒和3条显式float32 agent通过、原参数hash不变、GPU7实际CUDA/EGL通过，真实任务短程/线上logger按同版本补验。不将两台不同实际运行SHA隐藏为同一次代码实测；两版本的训练算法源文件逐字节相同。

sv2继续保持已经验收的dbd28b6完整SHA运行，不在活动训练期间切换checkout。12条真实size50m/CUDA/bfloat16工程验收全部通过，每条4098动作/517更新/3×1完整episode/2报告，合计6204更新、24次报告与36点评价指纹隔离；所有baseline匹配活跃位置非零，logging实际权重精确1，Dt实际平均权重0.9992706418—0.9999337781，实际作用非平凡，但不是收益证据。

**sv2于22:53:47北京时间正式启动**，CPU监督器PID1251084，GPU0–5依任务表各一条baseline，六条Dt排队。22:53:44启动前六卡无C/G PID、/data可用1,051,796,803,584字节；未使用GPU6/7。22:58:23启动审计通过：六条0点各10完整episode、410—506连续成功更新且invalid_count=0，102个本项目C/G PID均同卡且归属对应agent session，W&B全部running、完整配置一致、独立startup exact JSON/hash线上读回通过。snapshot与manifest版本/预算/梯度配置核对一致，JSONL保留，详见sv2工程根formal-startup-audit.json。startup摘要仅为当时状态，非正式AUC/实时进度。

GPU7在6e468ad4完整SHA完成本任务2条真实size50m CUDA/bfloat16短程，1034更新、4报告、6点评价指纹隔离通过；logging实际权重1、活跃位置非零，Dt平均实际权重约0.99972774/0.99993648，非平凡但不是收益证据。**23:04:31启动GPU7正式baseline**，CPU监督器2181514、Dt排队。启动前GPU7无C/G PID、/data可用581,828,259,840字节。GPU2按同6e468ad4版本在独立新目录重新工程验收；同机同SHA、未变依赖的CPU94项/3条float32 agent/0动作线上logger证据复用并逐文件记录来源/hash，GPU2设备探针与实际任务短程重新执行，不复用旧DBd或他卡的CUDA证据。没有因重复CPU验收而增加正式seed/预算。

GPU7 **23:12:13启动审计通过**：0点10完整episode、1897连续成功更新、invalid_count=0，17个C/G PID在GPU7及对应agent进程组；完整实际配置/快照与计划一致，W&B running、startup exact JSON/hash读回通过，最近更新开始动作9616。读取W&B的辅助审计脚本首轮把remote.path列表传入只接受字符串的API，读回报错，修正为显式entity/project/run-id字符串后通过；失败辅助记录保留，不涉及训练代码、状态或随机流，不放宽检查。sv1期间两次SSH认证前断开，随后使用持续连接；不是运行中训练失败，未改SSH入口/凭据。

GPU2独立新目录在6e468ad4版本完成本任务2条真实size50m CUDA/bfloat16验收，1034更新、4报告、6点评价隔离通过；logging活跃位置非零/实际权重1，Dt平均实际权重0.99970653/0.99994268。**23:14:36启动GPU2正式baseline**，CPU监督器2199050、Dt排队。23:14:35启动前GPU2无C/G PID、/data可用576,750,313,472字节；没有换卡或共享。至此本批8条baseline均已启动、8条Dt等待各自前驱审计，16条真实CUDA短程矩阵全部通过（SV2十二条+sv1两个单卡各两条，共8272更新、32报告、48点评价）；原sv1失败尝试中两个额外logging工程短跑保留但不混入此16条矩阵或正式结果。

**GPU2正式初次尝试失败与重开**：该进程在JAX“Compiling train and report”期间SIGSEGV（exit=-11），尚未创建manifest/0点评价/update收据或初始检查点，Dt被正确阻止；不将前段“已启动”写成持续成功。日志/actual index/supervisor-final完整保留在原 `...-sv1-gpu2/`，工程根formal-failed-attempt01.json、native-crash-diagnostics.json记录详情。kernel dmesg读取无权限（rc1），不能据此确认驱动、硬件或库级根因；已有driver/PTX版本警告也不单独构成SIGSEGV原因。

23:18:53在原GPU2、原6e468ad4版本/配置/seed/预算从头重开，正式根 `.../formal-m2-dmc16-remaining8-serial-20261024-retry02-sv1-gpu2/`、工程根 `.../engineering-dmc8-remaining-20261024-sv1-gpu2-retry02/`，CPU监督器2203655；该次仍在backend_compile收到SIGSEGV，零训练动作/无评价/无收据，故不是成功续跑。仅增加诊断环境变量 `PYTHONFAULTHANDLER=1` 打印堆栈，没有调整算法、JAX/XLA精度/参数或关闭检查。两次正式失败和全部诊断目录均保留，不覆写、不读取其检查点，不按return选尝试。

### 原生编译故障定位与retry03

- 反事实检查：失败正式配置只编译、不创建replay/logger、不训练或评价，缩短TMPDIR仍同样SIGSEGV（`jit-diag-gpu2-20261024/`）。此前成功的精确工程配置在同一新启动上下文也崩溃（`jit-control-engineering-gpu2-20261024/`），因此不是正式预算/评价点使训练目标改变，也没有据此调整它们。
- 找到启动上下文差异：sv1交互式登录shell把 `/usr/local/cuda-12.4/bin` 加入PATH、设置 `LD_LIBRARY_PATH=:/usr/local/cuda-12.4/lib64` 和 `CUDA_HOME=/usr/local/cuda-12.4`；原成功工程/仍运行GPU7的非交互启动无这些变量。库版本元数据相同不等于动态加载环境相同。进程白名单环境及stack limit比较在retry02工程根launch-context-comparison.json，未输出凭据。
- 保持失败正式模型配置及原长TMPDIR，在独立进程恢复已验证GPU7的干净非交互父环境，再覆盖GPU2/本任务字段，**23:32:39完整训练/报告编译通过，0训练动作/0更新**（`jit-clean-context-gpu2-20261024/`）。参数digest `83547769a7a093579ddd01896d8df1c41707ee439d9b9f9127a1988371c2cf00`。这是实际CUDA/size50m编译定位证据，不是新增正式训练或收益。结果验证了启动上下文恢复可解决当前编译故障；PATH/LD_LIBRARY_PATH/CUDA_HOME未逐个拆分，不把某一个具体动态库的内部缺陷写成已证明根因。
- 正式retry03使用 `/data/Policy_Discrepancy/runs/formal-m2-dmc16-remaining8-serial-20261024-retry03-sv1-gpu2/`，工程根对应 `engineering-dmc8-remaining-20261024-sv1-gpu2-retry03/`。仍为6e468ad4完整SHA、相同科学配置/seed/预算，新增runtime-environment-policy.json固定成功父环境的PATH、清除系统CUDA/LD_PRELOAD/PYTHONPATH注入；不改或安装服务器依赖，不复制GPU7模型/随机状态。Python fault handler仅作诊断，训练目标/随机流不变。
- 23:35:42首次retry03监督启动被新的外来GPU2进程（PID2222422—2222425）在require_free处拒绝，**训练根未创建**。已报告资源冲突，没有共享/停止对方/换卡；这不是再次编译失败。23:45:58复核外来进程退出、原GPU2无C/G PID后，按原分配用同bound plan启动监督器 **2232259**，新日志 `formal-supervisor-start02.stdout`、命令记录 `formal-supervisor-command-start02.json`；此前被拒绝的supervisor日志/记录保留。实际GPU2运行版本及状态以此记录和最终启动审计为准。

补充环境回归测试提交 **`5c6dd4fae0b35ac9d769dd00bc7d2845dfd226eb`**：2项本机及两台CPU检查通过，复现交互环境污染与显式正确继承上下文、验证目标GPU/W&B身份覆盖和CPU/MPS flags隔离、原环境不被修改。仅新增测试；通过Git bundle同步对象，在当前各自实测实现版本上执行已提交测试blob（hash见launch-context-tests.json），没有切换活动checkout或声称以5c6dd4f运行正式agent。具体动态库根因仍是工程限制，不影响冻结v1或构成hopper gate因果证据。

### 启动验收与资源冲突复核（23:59快照）

GPU2 retry03/start02于 **23:51:57北京时间** 通过实际启动审计：代码6e468ad4完整SHA，0点评价恰好10完整episode/快照动作0-update0，1229连续成功更新、invalid_count=0，17个C/G PID在原GPU2及正确agent session，CUDA UUID/EGL2/进程内CUDA0一致；线上W&B running、完整配置及startup exact JSON/hash读回通过。当前agent PID2232292、监督器2232259，根为 `formal-m2-dmc16-remaining8-serial-20261024-retry03-sv1-gpu2/`。白名单记录确认LD_LIBRARY_PATH/CUDA_HOME等注入已清除。没有读取旧失败检查点或复用GPU7模型/RNG。

因此八条baseline均已完成真实启动验收，八条Dt尚未启动。23:52:55（sv2）与23:54:24（sv1）的只读快照记录：

| 服务器/卡 | task | 连续成功更新 | 最近更新开始动作 | 已完成评价动作点 | 当时状态 |
| --- | --- | ---: | ---: | --- | --- |
| sv1/2 | finger_turn_easy | 2222 | 10916 | 0 | baseline运行/Dt排队 |
| sv1/7 | finger_turn_hard | 18061 | 74272 | 0,50000 | baseline运行，但有外来资源冲突/Dt排队 |
| sv2/0 | hopper_stand | 21601 | 88432 | 0,50000 | baseline运行/Dt排队 |
| sv2/1 | pendulum_swingup | 21661 | 88672 | 0,50000 | baseline运行/Dt排队 |
| sv2/2 | reacher_easy | 21622 | 88516 | 0,50000 | baseline运行/Dt排队 |
| sv2/3 | walker_run | 21612 | 88476 | 0,50000 | baseline运行/Dt排队 |
| sv2/4 | walker_stand | 21572 | 88316 | 0,50000 | baseline运行/Dt排队 |
| sv2/5 | walker_walk | 21516 | 88092 | 0,50000 | baseline运行/Dt排队 |

每个已完成评价点均10完整episode，收据ID从0连续、invalid_count=0。八卡仍有136个本项目C/G PID，没有本批训练卡释放；sv1可用571,747,737,600字节、sv2可用1,039,445,635,072字节。计数是有时间戳的快照，最近更新开始动作不是精确当前总动作。

**运行后外来占用，未通过持续独占检查**：23:52末次独占断言发现sv1 GPU7新PID2235998；23:53:28核查用户cgy的PID2235998/2236001、PGID2235891，启动时间约23:52，明确不在本项目agent2181554的进程祖先或session中。23:54快照仍有这两个外来计算进程。此前GPU7启动前和23:12启动审计均无外来进程；本项目没有主动共享/停对方/换卡，也未因冲突擅自丢弃仍在运行的baseline。已按AGENTS资源规则报告并请用户决定协调释放、停止本项目或重新分配；处理未确认前不据该卡启动新的工作，后继Dt仍须require_free。保留 `delivery-resource-ownership-investigation.json` 及 `delivery-resource-and-running-snapshot.json` 的原始pmon/归属证据；该快照的exclusive检查为false，**不是通过**。冲突可能影响性能/后续可执行性，当前无loss/梯度非法值；不能仅凭收据有限就保证独占或裁决最终结果有效性。其余七卡当时未发现外来C/G PID。用户处理与下一次资源复核后再更新事实，不把旧启动审计当实时资源保证。

**23:58:59 sv1 / 23:59:02 sv2复核，冲突解除**：GPU7上述两个外来进程已退出，无需改变用户分配；八卡当时只发现136个本批C/G PID，八条baseline仍running、八条Dt queued，尚无训练卡释放。sv1 hard19866更新/最近开始动作81492、easy4027/18136；sv2六条23925—24070更新/最近开始动作97728—98308，全部收据ID连续invalid0，已完成评价点均10完整episode。可用空间sv1 567,467,507,712字节、sv2 1,039,443,705,856字节。当前资源询问因外来作业退出而解除，不是把未答复当资源或共享批准。原运行未中断、不重开、不追加GPU工作；后继仍逐卡刷新require_free。短时外来并发的性能/逐位数值影响未单独量化，留作最终有效性审查限制，不把此后快照通过改写为全程独占。两台 `delivery-final-resource-snapshot.json` / `delivery-final-actual-RUN_INDEX.json` 是最新有时间戳的证据，旧冲突和失败记录不覆盖。

本机小证据包：sv1 480文件、SHA256 `115fb046a8618bb749105760a517491db99eddab5569a8dc8ba8491a4853ed8a`；sv2 307文件、SHA256 `07a97579ad8a40f3935877b3a2ad658f55a9cd661043f4bb40dc204b6d370dfc`。镜像只含小配置/日志/诊断/工程收据与启动证据，不含检查点或replay，完整最终审计仍需源服务器数据。最终交付在5c6dd4f之后只更新文档；相对sv1运行版另有2项新增测试，相对sv2运行版还包括显式单卡分片控制器。只向sv1/sv2同步已验证Git对象，不移动活动checkout，无GitHub/sv3同步。完整最终SHA/对象同步及副本hash见本机DELIVERY.json。

### 复现、输出和未完成项

本机证据目录 `analysis/outputs/dmc8-remaining-start-20261024/`（Git忽略）包含原/单卡bundle、CPU日志、资源/冲突/同卡映射、版本差异、16条最终工程矩阵及启动审计的小镜像；没有提交检查点/回放或密钥。服务器每个实际工程根有bound-plan.json、formal-supervisor-command.json、formal-startup-audit.json、integration/acceptance.json和diagnostic-summary.json；源命令/配置/环境/哈希均可复核。GPU2同机CPU/logger证据的复用来源另有 `reused-same-SHA-environment-evidence.json`。

```sh
# 同提交CPU回归完整13模块列表见工程根check-commands.json。
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=2 OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -m unittest -v tests.test_dmc8_remaining tests.test_dmc8_supervisor tests.test_dmc16_preparation tests.test_return_metrics
# 已有实际运行不要再执行run或覆盖其目录。全部结束后只读重审计用新输出：
CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 /data/Policy_Discrepancy/envs/dreamer/bin/python -m scripts.run_dmc8_remaining audit-host --plan ACTUAL_ENGINEERING_ROOT/bound-plan.json --output NEW_HOST_AUDIT_DIR
# 收集全部八任务真实快照/收据/评价证据后，全体索引的独立审计：
python -m analysis.return_metrics --profile dmc8-remaining --index GLOBAL_ACTUAL_INDEX.json --output NEW_RETURN_AUDIT --require-complete
```

正式百万动作、21×10全网格、正式c、全部Dt和两机八任务最终统计尚未完成，不能将排队的Dt称为已运行或验收完成。后续交付SHA晚于实际测试版本的差异、同步范围与各卡最新状态在DELIVERY及最终回复说明。

首选继续当前03窗口监测同批16条，保留GPU7短时资源冲突限制，逐卡确认无占用后完成baseline→Dt；每条百万动作/21×10/正式c/收据/快照/原始return与W&B审计完成再供05分析。当前可继续原已授权队列，前八旧六条Dt另需用户分配资源，不能挪用本批卡补跑。现存科研边界仍交02与用户决定，不扩大任务/seed/组、不自动建chat或发消息。可复制的完整继续prompt在 `analysis/outputs/dmc8-remaining-start-20261024/HANDOFF-current03.txt`，由同一窗口继续，无需新建chat。
