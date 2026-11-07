# Vision20 dynamic：2026-11-07正式启动

本轮继续用户已授权的20任务×baseline/原Dt，共40条，seed0、sv2物理GPU0–7。用户清理后要求“已清理，按原计划启动”；19:54:40（北京时间，服务器时钟）创建控制器，先启动七条。GPU2起初有外部评估作业，用户确认等待原卡释放；释放后补接，quadruped_walk baseline于20:05:58启动。**20:11:56启动验收通过，8条运行、8条就绪、24条backlog、0失败/阻塞、0终点完成**，原40条授权和调度范围保持。

本记录是正式启动与启动核查，不是百万动作终点验收或方法有效性结论。11月6日的准备、工程验收、GPU辅助渲染限制继续见 [DCS_DYNAMIC_VISION20.md](DCS_DYNAMIC_VISION20.md)。本次没有修改算法、训练代码、科学配置或依赖。

## 配置与调度

- size50m、64×64 RGB仅视觉、seed0、每条100万实际训练动作（含预填充，排除reset/评价）、repeat=1。16训练环境，batch16×64、context1、train_ratio256、CUDA/bfloat16、prealloc=false。
- baseline为logging-only，实际rep权重1；Dt为原PPD-KL，alpha20、`agent.rep_probe.mode=dt`、DtLatch关闭。本轮不运行constant/shuffle，也不把工程c用于正式实验。
- free-nats=1，dyn/rep=1/0.1，ac_grads=false，reward_grad/repval_grad/repval_loss=true。匹配按[100000,300000)成功更新的开始动作步归属，记录有效活跃loss位置候选权重和S与计数N；正常replay重复采样保留，无效匹配停止报告。
- 评价0至100万每5万一次，共21点，每点10完整episode；固定参数快照、采样动作、独立评价随机流。原始return不折扣、不除长度；未完成全网格不能报告正式AUC或末五指定点指标。
- DCS dynamic：DAVIS train60视频训练、独立val30视频评价，video_alpha=1、num_videos=-1、background seed=-1、cache_size=2；14任务沿上游地面规则，另6任务地面alpha=0.3。完整任务映射和素材hash绑定声明 `c51f0760c481057571719c932c6d84e400c9122760b94e0e4f959ead816691a7`。不与clean结果混合。
- 每卡串行，前一条完成并通过审计才启动下一条；首轮baseline→同任务Dt，之后按既定20任务顺序补入。失败、资源冲突或审计失败阻塞该lane，不自动跳过、不复用旧目录、不自动重试。

| sv2物理GPU | 首条baseline | 后继Dt | 本轮开始状态 |
| --- | --- | --- | --- |
| 0 | hopper_hop | hopper_hop | 首条运行、Dt就绪 |
| 1 | quadruped_run | quadruped_run | 首条运行、Dt就绪 |
| 2 | quadruped_walk | quadruped_walk | 等外部作业退出后，20:05:58启动首条、Dt就绪 |
| 3 | reacher_hard | reacher_hard | 首条运行、Dt就绪 |
| 4 | acrobot_swingup | acrobot_swingup | 首条运行、Dt就绪 |
| 5 | cup_catch | cup_catch | 首条运行、Dt就绪 |
| 6 | cartpole_balance | cartpole_balance | 首条运行、Dt就绪 |
| 7 | cartpole_balance_sparse | cartpole_balance_sparse | 首条运行、Dt就绪 |

GPU2另有cheetah_run、walker_stand的baseline/Dt四条依赖，合计六条最初阻塞。19:58:30再查，PID207042已退出，GPU2无C/G进程。原控制器把资源冲突设为终止性lane状态，不会因卡释放而自动恢复；因此新增独立接续入口 `scripts/run_dcs_dynamic_gpu2.py`，按用户等待原卡的指令补接，不迁卡、不重启其他七条。

接续入口只接受原40条plan中GPU2六个精确未启动槽位：第一条必须是原资源阻塞，后五条必须是原依赖阻塞；拒绝PID、开始时间、配置/命令/路径变化。以独占创建的新控制器根作claim，再要求原训练输出路径均不存在；保留原run ID、W&B ID、logdir、命令、config、seed与背景映射，不产生额外训练重复。原控制器索引保持原样，原控制器对这些终止性阻塞槽位没有启动路径。

接续控制器根C为 `R-gpu2-continuation-20261107`（即在正式根完整路径后追加该后缀）。其 `actual-RUN_INDEX.json`管理六条；`combined-RUN_INDEX.json`按原run_id替换六个槽位并保留其余34条，始终40个唯一run ID。后续全批状态及05读取须合并两个实际索引，不能只看原R索引误报六条仍阻塞，也不能将两个索引相加为46条。C控制器运行期间刷新合并快照；C结束后快照可能早于原七lane的后续变化，必须核对checked_utc并从两个新鲜索引重建。

原R控制器保留六条终止性阻塞，因此它自己的最终passed可能为false，即使其余34条均accepted；这不单独构成补接六条的失败证据。全批完成条件须逐条核对合并后的40条accepted、原R其余34条及C六条的终点审计和原始收据，不能把任一控制器局部passed等同于40条验收。

每条开始前再查GPU2独占、60GB容量、训练checkout精确SHA/干净状态；前一条成功且终点审计通过才前进。错误、资源再次占用或审计失败阻塞后继，低于5GB仅停止自己的GPU2子进程；重复调用和已有输出拒绝，不声称可以无缝恢复环境/replay。

20:05:33启动前再次核GPU2为2MiB、无C/G进程，UUID与原绑定证据相同；20:05:40接续控制器PID235467启动，20:05:58训练PID237001启动，/data余量406,776,619,008字节。控制器checkout为 `/data/Policy_Discrepancy/checkouts/dcs-gpu2-e04b1e8f517e`，训练cwd仍是5245096 checkout；`P/launch-20261107/gpu2-launch-receipt.json`及C下claim/startup/execution记录两层来源。接续原样使用第一次从未创建的attempt01路径和W&B ID，属于补启未运行槽位，不是恢复或覆盖已有训练。

## 启动资源与容量

19:51首次复核GPU0/1/3–7空闲；GPU2 PID207042为root所属、路径 `/data/gbr_pong_three_goal_bbf_noln_winpoint_v54_formal_20261107_r2/` 的评估进程，使用另一个Python环境，与本项目训练无关。未读取凭据、停止或干扰该作业。19:54:40启动前快照仍只有该GPU2进程，其他七卡0MiB且无C/G进程；控制器对每条再独立执行实时占用检查。

启动前/data可用414,379,216,896字节。此前480GB是每卡60GB的保守首批规划门槛；现有空间高于按既有size50m产物估计的首八条约200GB。根据本轮清理后明确启动的指令，由03作容量执行判断，使用已有 `--accept-capacity-deficit`，仅豁免首批480GB门槛，不修改代码或科学配置；收据保留实际余量及判断依据，不把用户指令改写成其亲自确认了具体字节缺口。

后继仍要求/data至少60GB，否则排队等待；运行中低于5GB会停止该控制器自己的子进程并保留证据。40条累计约1TB只是规划估计，当前空间不保证整批完成，需要后续继续清理或扩容。此次没有删除任何数据，既有Denoising清理记录与本次启动分开。

## 版本、命令与证据

实际正式训练、控制器及此前CPU/CUDA验收均为完整SHA **`5245096ee52de5ba1ff6bcafb96d19298b74dcad`**。运行前复核独立checkout干净、声明hash及同SHA同声明的八条完整agent acceptance匹配；素材实际hash在每条agent创建前重新核查。

- checkout：`/data/Policy_Discrepancy/checkouts/dcs-dynamic-5245096ee52d`。
- 环境：`/data/Policy_Discrepancy/envs/dreamer`，沿用11月6日依赖快照；本次未安装或升级软件。
- 准备根P：`/data/Policy_Discrepancy/runs/engineering-dcs-dynamic-preparation-20261106`。
- 正式根R：`/data/Policy_Discrepancy/runs/dcs-dynamic-vision20-seed0-20261106-5245096`。名称采用11月6日准备标识，实际正式启动日期为11月7日。
- 控制器PID213688；原始命令、资源、容量判断及子进程启动记录在 `P/launch-20261107/{request,launch-receipt}.json`、`supervisor.log`、`R/startup.json`、`R/actual-RUN_INDEX.json`、各条 `execution-initial.json`。不是可无条件重复执行的命令。

```text
cd /data/Policy_Discrepancy/checkouts/dcs-dynamic-5245096ee52d
/data/Policy_Discrepancy/envs/dreamer/bin/python -u -m scripts.run_dcs_dynamic run --plan /data/Policy_Discrepancy/runs/engineering-dcs-dynamic-preparation-20261106/formal-plan-all40/plan.json --evidence /data/Policy_Discrepancy/runs/engineering-dcs-dynamic-preparation-20261106/device-binding-5245096 --accept-capacity-deficit
```

GPU2补启的实际命令（只能由独占claim执行一次）：

```text
cd /data/Policy_Discrepancy/checkouts/dcs-gpu2-e04b1e8f517e
/data/Policy_Discrepancy/envs/dreamer/bin/python -u -m scripts.run_dcs_dynamic_gpu2
```

W&B沿用 `ychan9liu-tsinghua-university/PD_1`，独立group `dcs-dynamic-vision20-seed0-20261106-5245096`，online、resume=never、每条新run ID。19:57:11–19:57:17只读远端已确认七条均running、完整config逐字段相同、group正确，原始 `wandb-initial.json` 留档。20:01:24–20:01:45进一步核对七条的零点评价、快照、原始return均值、成功更新连续ID/有限匹配值及实际C/G进程父链；各10完整1000动作episode，更新数329–417，每张卡只含对应训练及其子进程，CUDA可见UUID与EGL落同卡。

七条启动审计 `P/launch-20261107/formal-startup-audit.json` 为passed=true；读取时的完整收据前缀和稳定配置/快照文件以hash冻结到 `frozen-evidence/`，不将继续写入的日志当静态全文件。W&B线上config逐字段相同，启动核查的精确JSON及SHA256写入独立 `startup/*` summary字段并读回一致，不覆盖训练指标。原logger的step0记录可暂存在partial history，直到后续step推进才提交；此启动summary不等于21点评价已全部上传。

20:11:33–20:11:42 GPU2同样通过真实size50m CUDA启动核查：零点10个完整1000动作episode、冻结快照一致、149条连续有效更新、实际CUDA/EGL同卡独占、W&B完整config与精确启动摘要读回一致，证据 `P/launch-20261107/gpu2-startup-audit/formal-startup-audit.json`。同目录冻结该条证据前缀；这是一条真实正式运行的启动验收，不是CPU/debug替代CUDA。

20:11:56最终合并实时核查 `final-live-snapshot.json` 为passed=true：八卡均对应本批训练及其子进程，40个run ID唯一，80个零点评价episode，原七条PID不变、两个控制器仍存活且无supervisor-error。前七条观察到4169–4309更新、最近更新开始动作步18704–19264；GPU2为254更新、3044动作步。/data可用405,703,831,552字节。完整八条分项、物理GPU/PID、同卡进程及冻结40槽视图在同目录 `combined-live-index.json`，均带快照时间，不当作后续实时状态。

本次开始本机HEAD为 `d245a915bf7438731e88f87c00a4afae5211eafb`，相对上次交付83bb51c仅增加Denoising清理文档与STATUS记录。新增接续控制器和测试提交为 `e04b1e8f517e946fed0e554b064f02c21299165e`；训练仍固定5245096。工作期间main出现并行离线分析提交 `1ae54b5` 及其文档提交 `d889bc3`，均先检查保留，接续入口通过merge合入，未reset或丢弃草稿。

相对实际训练5245096，本机增加辅助渲染诊断、独立离线分析及测试、接续控制器和文档；生产训练/算法调用路径未改。最终SHA、相对两个实际执行SHA的差异和同步范围在本机 `analysis/outputs/dcs-dynamic-vision20-20261107-launch/DELIVERY.json`。原有AGENTS/CHATS/RESEARCH/STATUS草稿、其他分析报告及论文文件保留，不混入本次提交。

e04b1e8通过bundle同步sv2，SHA256为 `c99fd3106e1daa36cd7b2d85bdc037362cd42f03626059598cf64bd1d1010354`；服务器Git验证前置83bb51c、fetch确定ref并创建干净独立checkout，`gpu2-sync.json`和`gpu2-tests-server.json`保存收据。canonical repo工作区仍干净、HEAD保持d8997f64a301bfdcfe08758faeae8957a7395766，未在服务器修改源码。GitHub远端查询超时，最终推送及交付bundle以DELIVERY收据为准；本次未连接或同步sv1/sv3。

## 当前验收与后续

完整agent工程依据继续是11月6日同5245096的本机/服务器81项CPU回归、八卡绑定、四代表任务×两组共八条4098动作size50m CUDA运行。此次使用原已验收代码，未重复耗GPU跑相同fixture；正式启动核查单独保留，不能据此宣称40条终点已验收。

接续入口新增8项CPU测试，连同原DCS协议10项共18项，本机20.497秒通过，sv2实际提交e04b1e8在CPU上125.604秒全部通过。覆盖六槽原命令/config/RNG字段精确保持、已启动/篡改来源拒绝、原控制器不能推进阻塞槽、重复claim/已有路径拒绝、审计依赖、40条合并去重、训练checkout漂移拒绝，以及完整控制器fixture中baseline审计通过→Dt失败→四后继阻塞、仅清理自身进程。首次误用缺少PIL/portal的旧本机venv产生导入错误，换回既有DCS环境；Windows路径比较问题在提交前修正，失败日志保留，未影响服务器训练。离散配置/状态要求精确相同，没有放宽数值容差。命令：

```text
JAX_PLATFORMS=cpu MUJOCO_GL=disable XLA_FLAGS=--xla_force_host_platform_device_count=2 python -m unittest tests.test_dcs_dynamic_gpu2 tests.test_dcs_protocol -v
```

GPU辅助额外渲染的少量低位RGB差异仍未定位，严格失败未改标通过；该接口未用于本批生产训练/评价/report，不承诺CUDA轨迹逐位可复现。报告只读、评价随机流隔离的既有完整agent工程证据保持，后续新科学复验仍由02与用户确认。

首选继续当前合并03窗口监测已启动八lane、容量及自动后继，逐条终点审计与W&B收尾；完成标准为合并后的40条各自百万动作、21×10完整评价、匹配/收据/快照一致、无静默覆盖或重复启动，再交05独立分析。八卡仍有本项目运行，没有释放；现有容量可能在后续阻止新一条启动，需要用户后续清理或扩容，未经授权不删除产物。当前没有新增chat或自动发消息。
