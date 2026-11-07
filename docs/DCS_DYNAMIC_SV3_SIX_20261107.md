# Vision20 dynamic：sv3六对串行与sv2调度接管

用户在原40条dynamic批次运行中指定“sv3 gpu1、3~7，挂六条排六条，同任务串行”。从原后备队列按VISION20顺序迁出接下来的六对，共12个从未启动的正式槽位；不新增任务、组、训练seed或总训练次数。原40条拆为sv2保留28条、sv3执行12条，每卡baseline审计通过后再执行同任务原Dt。

## 资源与固定配置

| sv3物理GPU | 任务 | 顺序 | baseline PID / 启动时间 |
| --- | --- | --- | --- |
| 1 | dmc_cartpole_swingup | baseline→Dt | 4081688 / 21:13:27 |
| 3 | dmc_cartpole_swingup_sparse | baseline→Dt | 4081697 / 21:13:27 |
| 4 | dmc_cheetah_run | baseline→Dt | 4081720 / 21:13:31 |
| 5 | dmc_finger_spin | baseline→Dt | 4081735 / 21:13:31 |
| 6 | dmc_finger_turn_easy | baseline→Dt | 4082098 / 21:13:35 |
| 7 | dmc_finger_turn_hard | baseline→Dt | 4082121 / 21:13:35 |

20:44:59（北京时间，以下时刻均UTC+8）首次核查sv3六卡均0MiB、无C/G进程；GPU0/2存在未归属占用，不使用或干扰。/data可用447,155,154,944字节。20:52再次复核六卡空闲；20:57:49完整agent工程启动前仍空闲，可用445,470,953,472字节。GPU均NVIDIA A100-SXM4-80GB、driver570.158.01。每条实际启动仍逐卡重新检查，旧快照不替代实时独占核查。

科学设置全部继承 [原dynamic启动报告](DCS_DYNAMIC_LAUNCH_20261107.md)：seed0、size50m、64×64 RGB仅视觉、百万实际训练动作（含预填充、排除reset/评价）、repeat1、alpha20、free-nats1、dyn/rep=1/0.1、ac_grads=false、reward_grad/repval_grad/repval_loss=true、DtLatch关闭。16训练环境、batch16×64、context1、train_ratio256、CUDA/bfloat16、prealloc=false。baseline为logging-only实际权重1，Dt为原PPD-KL；不运行constant/shuffle。

评价固定快照、采样动作、独立随机流，0至100万每5万一次共21点，每点恰好10完整episode。baseline匹配窗口[100000,300000)，按成功更新开始动作步归属、有效活跃位置累计S/N；非法值或无活跃位置停止报告，不用工程c或默认c替代。

DCS dynamic仍使用DAVIS train60视频训练/独立val30评价、video_alpha1、num_videos=-1、background seed=-1、cache_size2、14任务上游地面规则及另6任务0.3。声明SHA256仍为 `c51f0760c481057571719c932c6d84e400c9122760b94e0e4f959ead816691a7`。素材从sv2复制同一归档到sv3同一绝对路径，完整归档SHA256 `e3d0b5b77c3d031b000a19e0e25e3e2cac65d183755601bc2cf066df1a2aa492`，6208帧解压及逐帧审计通过，训练/评价视频不重叠；不重新下载另一版素材。

实际训练继续固定 `5245096ee52de5ba1ff6bcafb96d19298b74dcad`，两服务器训练checkout同为 `/data/Policy_Discrepancy/checkouts/dcs-dynamic-5245096ee52d`。只因服务器迁移改变logdir、主机/物理设备、输出相关环境路径；正式科学config逐字段比较只允许logdir不同，原logical run_id、W&B ID/group、root seed及原seed派生保持，原训练/评价/报告/shuffle RNG实现和算法文件未改。

## 调度接管与防重复语义

新增 `scripts/run_dcs_sv3_six.py` 与 `scripts/dcs_process_adoption.py`，代码/服务器CPU实测提交为 **`0b222ea905ad0bcfe45ff6315ba7655435d25de4`**。旧sv2两个控制器没有在线迁出接口，直接在原输出路径占位会使旧控制器抛异常并清理其仍运行的子进程，因此不采用该方式，也不修改旧控制器源码或运行内存。

接管流程只针对已核对UID、完整命令、cwd、PID和进程开始标识的两个本项目监督器。先仅对监督器PID发SIGSTOP，不向训练PID或其进程组发信号；八条既有训练继续运行。随后重新核对40槽索引、12个来源都是无PID/无开始时间/无目录的backlog，确认旧监督器直接子进程恰为已登记的八条训练。独占创建新的sv2控制器根和观察器锁，原八条训练通过原PID、父PID、/proc开始标识接管观察，不重启、不加载检查点、不改变工作目录或环境。

新监督器ready后才发布不可静默重用的source-claim；sv3正式入口要求完整claim、plan hash、同声明12条CUDA验收及精确代码版本。ready前失败会只停止新的观察器并恢复旧监督器；发布后旧监督器保持暂停，不能手动恢复旧队列。新sv2索引只含保留28槽，排除迁出的12槽；sv3索引只含12槽，logical run_id不变，合并恰为40个唯一ID。

原八条训练结束时，暂停的父监督器暂时保留子进程退出状态；新监督器读取可核对PID开始标识的Linux等待状态并执行原终点审计，记录成功/非零退出。某旧监督器的全部已登记训练子进程退出、退出证据落盘后，才终止该已暂停监督器，训练进程不会因此被终止。普通观察器异常不向接管的训练发信号；已存在的5GB紧急容量保护另有受验证的原训练进程组停止路径。PID重用、父进程变化、旧监督器被恢复、重复claim或输出路径已存在均拒绝，不承诺观察器/环境/replay无缝恢复。

源sv2八卡授权及科学配置保持。迁出后，各源lane跳过且仅跳过所转移的任务，继续原顺序剩余28槽。新的sv2/sv3索引是后续运行状态来源；原启动记录和原两个控制器索引保留为历史，不应与新索引相加计算56/68条，也不能恢复旧控制器来补它们显示的backlog。

## 工程验收与路径

本机新增控制/分配测试通过，Linux专属信号/进程组测试在Windows明确跳过，未写成本机已验证。最终0b222ea在sv2、sv3分别运行27项CPU测试，162.966秒和161.038秒，均通过、无skip。包括原DCS协议与GPU2接续回归，以及迁出12/保留28分区、仅logdir变化、固定run/W&B身份、已启动/缺失/重复来源拒绝、后继依赖、完整接管seal、PID重用/错误进程组拒绝。

真实Linux子进程fixture验证：暂停父监督器时子进程仍增加进度；30个固定seed随机值逐项保持；非零退出7经父进程保留的等待状态准确读出；有活跃训练子进程时不得终止旧监督器。该fixture不替代真实训练连续性核查，也不声称GPU轨迹逐位一致。数值均值核对沿用rtol/atol=1e-12，离散配置、身份、hash和随机序列要求精确一致；未放宽原算法数值容差。

```text
JAX_PLATFORMS=cpu MUJOCO_GL=disable CUDA_VISIBLE_DEVICES= XLA_FLAGS=--xla_force_host_platform_device_count=2 python -m unittest tests.test_dcs_sv3_six tests.test_dcs_dynamic_gpu2 tests.test_dcs_protocol -v
```

20:54:10六卡零动作CUDA/EGL绑定全部通过，实际执行固定训练5245096；计算与图形均位于所分配物理GPU。20:58:05开始12条真实size50m CUDA工程运行，每卡baseline→Dt串行，每条4098动作/517更新，0/2049/4098三点评价各1完整episode，匹配窗口[2200,3600)，两次CUDA报告只读和三次评价隔离检查。所有缩小预算值、c和seed用途均为工程验收，不能替代正式百万步结论。

21:08:44十二条全部accepted：合计6204次成功更新、36个完整评价episode、24次报告只读检查、36次评价隔离检查；两组十二条实际计算/渲染进程归属均已单独留档。末次诊断中六baseline的actual_weight_mean全部为1、active_effective_frac为0；六Dt的actual_weight_mean为0.9998960–0.9999443，active_effective_frac为0.7399349–0.9978638，十二条rep_active_frac均为1。这只验证gate确实作用于活跃位置，不代表收益；工程c约0.9998916–0.9999503，明确不用于正式实验。精确数值见P/integration-diagnostics-summary.json，原始报告/评价fingerprint及诊断日志保留。

## 实际接管与正式启动

21:11:47再次检查sv3 GPU1、3–7均1MiB且无C/G进程，GPU0/2仍为未分配的其他作业，可用419,179,900,928字节，满足六卡360GB门槛。21:12:07开始执行固定0b222ea的takeover并成功；新sv2监督器PID265311，旧监督器213688/235467仅管理进程暂停，八条训练PID未变。来源claim SHA256为 `5a8c0bbb80a8d4da9e24be26befef6cbdef6190e8ff6fee7fe5191f2edb0aff2`，plan SHA256为 `fc3923aebe481d0b49b2e7a9fa8310292018bef182ee5fef2d4f2a31cc2d780c`；同字节claim复制至sv3后才允许正式启动。

21:13:11创建sv3正式监督器PID4081511，六个baseline依次启动，同任务Dt由监督器在baseline终点审计通过后自动串行启动。两台监督器均以CPU协调环境运行，训练子进程仍使用确定的5245096 checkout及各卡UUID/EGL绑定；未将协调进程分配至GPU0/2。

21:13:28真实连续性核查通过：sv2八条训练PID、/proc开始标识、父PID、cwd和命令保持，接管前成功收据的完整字节前缀SHA256保持，且分别新增462、622、1794、562、510、598、597、544次更新；转出12个来源输出目录仍不存在。这里只证明实际接管未重启训练、未改写已发生记录且更新继续，不声称同输入跨服务器逐位重跑一致。证据为P/source-continuity-before.json与after.json。旧两个索引不能再作为当前调度状态，也不可SIGCONT恢复旧监督器。

21:19:25–21:19:42六条正式启动审计全部通过：每条0动作/0更新的同一固定快照恰好10个完整1000步episode，共60个；原始return算术均值以rtol/atol=1e-12核对，快照标识和配置/seed/声明hash精确一致。读取时已各有73–195次连续成功更新、无非法匹配值。六条实际主进程与计算/渲染子进程均属于所分配物理GPU，环境未继承LD_LIBRARY_PATH/LD_PRELOAD/CUDA_HOME/CUDA_PATH/PYTHONPATH/XLA_FLAGS。

W&B六条online、同原group和逻辑ID、完整config精确核对；仅写入`startup/*`命名空间的精确JSON/hash并读回，不覆盖训练指标。网址见P/formal-startup-audit.json；项目为 [PD_1](https://wandb.ai/ychan9liu-tsinghua-university/PD_1)，原group为`dcs-dynamic-vision20-seed0-20261106-5245096`。正式百万步AUC/末五点完整性仍为false，不报告工程或零点评价为正式结果。

21:19:18（sv2）/21:19:53（sv3）收尾快照：sv2为8运行/8就绪/12backlog，sv3为6运行/6就绪；合计40唯一ID、20任务各baseline/Dt两组，即14运行/14就绪/12backlog，0失败/阻塞、0终点完成。sv2原八条成功更新23775–26640，最近更新开始动作97128–108588；sv3六条190–305更新，最近更新开始动作2788–3248。两端/data分别385,092,608,000与412,151,267,328字节。这些是带时刻的运行快照，不是停止点；本批占用sv2八卡和sv3所分配六卡，未释放训练卡。

准备/证据根P（两服务器）：`/data/Policy_Discrepancy/runs/engineering-dcs-sv3-six-20261107/`。

- sv3：`asset-sync.json`、`assets-audit/`、`bindings/`、`integration/`、`live-devices/`、`tests-server.log`、`sync-tests.json`。
- 双端：`closeout.json`、`closeout-RUN_INDEX.json`、`audit-operators/`及`evidence-sv2.tar.gz`或`evidence-sv3.tar.gz`；sv2另有`source-continuity-before/after.json`、`takeover-execution.json`；sv3另有`formal-launch-receipt.json`、`formal-startup-audit.json`与`formal-frozen-evidence/`。
- sv2接管根S：`/data/Policy_Discrepancy/runs/dcs-dynamic-vision20-seed0-20261107-sv2-remaining28/`。
- sv3正式根T：`/data/Policy_Discrepancy/runs/dcs-dynamic-vision20-seed0-20261107-sv3-six/`。
- 本机镜像与交付收据：`analysis/outputs/dcs-dynamic-sv3-six-20261107/`，不提交模型、replay或大产物。
- 监督器checkout（两服务器）：`/data/Policy_Discrepancy/checkouts/dcs-sv3-six-0b222ea905ad`；Python均 `/data/Policy_Discrepancy/envs/dreamer`，依赖与原准备一致，未安装/删除环境或base-py311。

两个canonical仓库HEAD保持 `d8997f64a301bfdcfe08758faeae8957a7395766` 且工作区干净。固定代码经Git bundle校验、fetch和独立checkout同步，不在服务器改源码；控制器bundle SHA256 `056ada665d368fd4ce2cd02f3166da1d7d69a59ec69f5475bfd9ce5c7a404b2a`。本机开始HEAD为 `5c13aa9f2614f05645476cf7a343ead2fa80225b`，相对上一交付f7e56a4只有sv3清理文档/STATUS，已先检查并保留。现有AGENTS/CHATS/RESEARCH/STATUS及论文草稿不混入本次提交。最终完整SHA及同步收据以DELIVERY.json为准。

本机已逐字节校验两份证据归档及292个文件：sv2为24件，归档SHA256 `170fc69522aca27e7da4635ff8a126d9808dbc0d0a90df367cced08e3c88c502`；sv3为268件，归档SHA256 `b6658a3bdf20fb4788612f1ec0f56238252136d0c99f0171e7f05d235c4f592f`。清单及复算在本机`mirrors/`、`mirror-verification.json`、`combined-closeout.json`；归档不含checkpoint、replay、DAVIS帧或凭据。本次最终记录相对0b222ea只改报告/STATUS/EXPERIMENTS文档，不影响已验收或在跑代码。同步限GitHub及实际执行的sv2/sv3 Git对象/ref和独立checkout；未访问或同步sv1。

实际运行完整命令、CPU协调器环境、设备映射和每次资源快照写入P的启动收据。正式目标入口为 `python -m scripts.run_dcs_sv3_six target-run`；源接管入口为 `python -m scripts.run_dcs_sv3_six takeover`，均要求确定checkout和未使用的独占根，不可当作重复运行命令。

## 限制及后续

原GPU辅助额外渲染的低位像素非确定性仍保留，未改容差、生产渲染、alpha或梯度；该辅助接口不在本批训练/评价/report路径使用。工程/启动验收不等于方法有效，也不等于40条终点完成。

sv3正式六卡初始要求360GB，后继不足60GB等待；两端低于5GB仅停止本批相应受控训练，保留失败证据，不擅自删数据。原sv2保留28条的后续累计产物仍可能需要继续清理或扩容。

首选继续当前合并03窗口，按S的28槽和T的12槽监测、逐条完成百万动作及21×10评价/匹配/快照/收据/W&B终点审计，再将完整40条产物交05分析；不新建chat、不自动发送消息，不以旧控制器历史索引代替新状态。

可在当前03窗口继续的prompt（不自动派发）：

```text
项目D:\program\PD\dmr3。继续合并03已授权的Vision20 dynamic seed0共40条，不新增训练。
先核查HEAD和未提交修改，保留全部现有工作；交付完整SHA与同步记录读
analysis/outputs/dcs-dynamic-sv3-six-20261107/DELIVERY.json。
必读AGENTS.md、docs/CHATS.md、docs/STATUS.md、docs/EXPERIMENTS.md、
docs/DCS_DYNAMIC_SV3_SIX_20261107.md、docs/DCS_DYNAMIC_LAUNCH_20261107.md。
实际训练固定5245096ee52de5ba1ff6bcafb96d19298b74dcad；双端监督器及CPU实测固定
0b222ea905ad0bcfe45ff6315ba7655435d25de4；后续文档提交不等于更换在跑代码。
授权sv2 GPU0–7保留28槽，sv3 GPU1/3/4/5/6/7执行报告所列六对12槽，
每卡baseline终点审计通过后才启动同任务Dt，原总40槽和run ID保持。
科学配置不变：size50m、视觉、dynamic DAVIS train60/val30、alpha20、seed0、
百万动作/repeat1、free-nats1、dyn/rep1/.1、原梯度开关；21点各10完整episode，
baseline匹配[100000,300000)有效活跃loss位置S/N；不跑constant/shuffle或DtLatch。
声明与素材hash、完整配置及设备映射以报告/plan为准，不用工程c。
当前权威索引是/data/Policy_Discrepancy/runs/下
dcs-dynamic-vision20-seed0-20261107-sv2-remaining28/actual-RUN_INDEX.json和
dcs-dynamic-vision20-seed0-20261107-sv3-six/actual-RUN_INDEX.json，合并恰40唯一ID。
旧sv2两个监督器只管理进程被冻结，训练八条已接管；不得SIGCONT旧监督器，
不得把旧R/C历史backlog重新启动。新监督器会在对应原训练全部退出后处理旧父进程。
允许在既定资源和配置内监测、审计及普通工程故障修复；修复须本机测试提交再同步，
保留旧失败证据、独立新尝试，不直接改服务器源码、不自动重试或声称无缝恢复。
后继不足60GB等待、低于5GB保护停止本批受控作业；不能擅删数据或换卡/共享卡。
先读实时索引、PID、GPU/容量和收据，报告带检查时间的状态及有意义变化。
完成标准为40条分别百万动作、249493成功更新、21×10完整评价及冻结快照、
baseline匹配有效、实际配置/设备/原始return/W&B可追溯，完成后交05分析。
产物留对应runs独立目录，小证据放新的analysis/outputs目录，更新报告/STATUS，
最终说明完整SHA、实测/实际运行版本、差异和同步范围；工程通过不代表方法有效。
新的方法或科学复验交02与用户确认；继续现有窗口，不自动建chat或发消息。
```
