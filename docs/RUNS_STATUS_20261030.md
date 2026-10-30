# 03合并窗口 · 全批任务完成情况（2026-10-30）

**后续执行索引**：本报告保留13:40核查原貌。用户随后分配sv2七卡（排除GPU6）、sv3 GPU4—7补十一条Dt；14:25时sv3四条启动验收通过、sv2七条因GPU0/2/4新外来作业尚未启动。当前状态及后续证据以`docs/MISSING_DT_RUNS_20261030.md`和STATUS最新段为准，不将下表11条“从未启动”继续当作实时状态。

**核查时间：北京时间13:40资源/运行索引，13:43—13:45线上W&B与已有completion。** 本轮只读sv1/sv2/sv3及线上W&B，未启动/停止训练、创建环境、调用agent或修改科学配置。下列数量按已授权任务/seed/组槽位计，工程短跑、GPU2零动作编译失败尝试不计入完成数。已获授权的运行没有因本报告而取消；当前缺项并非仍在后台排队。

## 总览与验收层次

| 范围 | 计划训练条数 | 已训练结束且有逐条审计 | 从未启动 | 整体状态 |
| --- | ---: | ---: | ---: | --- |
| 新增前八环境，seed0 baseline/Dt，sv1 | 16 | 10（8 baseline+2 Dt） | 6 Dt | 不完整，已有10条等待所属批次收尾 |
| 新增后八环境，seed0 baseline/Dt，sv1/sv2 | 16 | 11（8 baseline+3 Dt） | 5 Dt | 不完整；其中sv1两对4条已完整验收，sv2七条待所属批次收尾 |
| hopper_hop补seed1/2/3，baseline/Dt，sv3 | 6 | 6 | 0 | 六条完整验收/统计/线上精确摘要通过 |
| **本轮新增合计** | **38** | **27** | **11** | 无本批训练运行中；未整体齐全 |

新增16环境单独看是 **32条中完成21条，剩11条Dt从未启动**，不是32条全部完成。16条baseline均完成，Dt仅cheetah_run、finger_spin、finger_turn_easy、finger_turn_hard、walker_stand五条完成。

另列历史原四任务四组seed0的16条：既有全体completion为true/verified_run_count=16，05已分析；本轮只读复核该completion和原表hash仍一致，不重新独立审计全部历史大产物。若把历史原16条与本轮新增38条仅作运行数量盘点，为54个授权槽位中43个已训练结束、11个未启动；**这不是合并科研指标或43条全部完成当前批次收尾**。原四任务/新增16任务/hopper补seed保持各自范围，不自动拼成20任务或多seed总体。

27条本轮已结束运行均退出0，尾收据update ID249492（共249493更新）/开始动作1000000，21点严格0..100万每5万，点内各10完整episode；索引中均有run_audit。线上27条均finished、最后_step=1000000，full config与实际计划一致。**其中10条已完成最终指标和精确W&B摘要验收**（hopper六条+sv1 finger_turn两对四条）；另外17条状态仍是run_audited_pending_group_and_WB（前八10条+sv2后八7条），原始曲线上传结束不等于所属批次最终汇总完成。

本轮读取已有run_audit并交叉核对尾收据/评价/完成标记/表hash及线上传输，没有重新执行全体原始收据、checkpoint或S/N的独立重算；既有工程/正式验收边界保留。训练结束、逐条审计、单卡shard完成、全批完成四者分别记录。有限工程通过和完成数量均不构成方法有效或强M2选择性因果证据。

## 新增16环境逐任务状态

下表baseline均为百万动作/21×10/已有逐条审计；“Dt完成”也满足同预算/网格。

| 批次 | task（省略dmc_） | baseline | Dt |
| --- | --- | --- | --- |
| 前八 | acrobot_swingup | 完成 | 未启动，旧GPU0占用门禁阻止 |
| 前八 | cup_catch | 完成 | 未启动，旧GPU1占用门禁阻止 |
| 前八 | cartpole_balance | 完成 | 未启动，旧GPU2占用门禁阻止 |
| 前八 | cartpole_balance_sparse | 完成 | 未启动，旧GPU3占用门禁阻止 |
| 前八 | cartpole_swingup | 完成 | 未启动，旧GPU4占用门禁阻止 |
| 前八 | cartpole_swingup_sparse | 完成 | 未启动，旧GPU5占用门禁阻止 |
| 前八 | cheetah_run | 完成 | 完成；全前八批仍不完整 |
| 前八 | finger_spin | 完成 | 完成；全前八批仍不完整 |
| 后八 | finger_turn_easy | 单卡shard验收完成 | 单卡shard验收完成 |
| 后八 | finger_turn_hard | 单卡shard验收完成 | 单卡shard验收完成 |
| 后八 | hopper_stand | 完成 | 未启动，sv2 GPU0门禁阻止 |
| 后八 | pendulum_swingup | 完成 | 未启动，sv2 GPU1门禁阻止 |
| 后八 | reacher_easy | 完成 | 未启动，sv2 GPU2门禁阻止 |
| 后八 | walker_run | 完成 | 未启动，sv2 GPU3门禁阻止 |
| 后八 | walker_stand | 完成 | 完成；sv2 shard仍不完整 |
| 后八 | walker_walk | 完成 | 未启动，sv2 GPU5门禁阻止 |

以上11条blocked目录均不存在，无manifest、评价或训练动作，不是训练中断后等待恢复。blocked为终止状态，不会自动空闲重试；前八supervisor-final complete=false、后八sv2 complete=false。三条在10月25日14时仍训练的Dt现均已结束，不能再报告其“31/37/39万运行中”。GPU2此前两次零动作SIGSEGV和retry03最终成功的证据不覆盖；GPU7曾有短时外来并发，性能/逐位影响未量化，仍列为最终有效性限制。

## 已完成的终点与线上精确摘要

已有completion及对应tables.json的SHA256逐项一致；10条正式精确JSON/hash与线上summary读回一致：

| 范围 | completion北京时间 | 条数 | tables.json SHA256 |
| --- | --- | ---: | --- |
| hopper seeds1/2/3 | 10月22日02:30:43 | 6 | 5ce0f26287085711949c7aa71be0f9bc9d86a6477a2c2ab23bbf61fc3090970e |
| finger_turn_hard 单卡 | 10月25日21:06:12 | 2 | 355e8800952e6267361c903ff576e150778d9610c26391887289f838c9e62a78 |
| finger_turn_easy 单卡 | 10月25日21:46:11 | 2 | f972d67075f54b42fc336e0872b577f0874506e50ffbfaf1099eed57eb0c8428 |

两个finger单卡completion各自host_complete=true、cohort_complete=false，当前两张小表的八任务总体仍为null，没有因缺其他任务重加权。hopper六条的独立profile完整，可交05按既有协议分析；不与seed0自动合并，不用episode当独立训练seed。本轮不选择效应解释或新增科研复验。

历史原四任务表hash仍为 `41fa0030b8ec27e11930c083f0cbf78a0bcdc0a258d3a6c8071f6ba42da2dd87`。旧四组completion字段为return_tables_sha256，新shard字段为tables_sha256；本次核查按各自schema比对，未改来源数字或容差。

## 当前资源与补跑条件

13:40各卡C/G进程及13:43—13:45项目命令行扫描均没有本轮训练/监督进程；本批已释放原分配的全部卡，**释放不等于卡当前空闲**：

| 服务器 | 当前非本批C/G作业所在卡 | 其他卡的本次快照 | /data可用空间 |
| --- | --- | --- | ---: |
| sv1 | 0、1（cgy Ray作业，项目未识别） | 2—7无C/G PID | 52,328,312,832字节，约52.3GB |
| sv2 | 0（cmy python/java）、6（root python3.10，项目未识别） | 1—5、7无C/G PID | 821,247,361,024字节，约821.2GB |
| sv3 | 0、2（root python3.10，项目未识别） | 1、3—7无C/G PID | 725,263,654,912字节，约725.3GB |

sv1低于原单条60GB容量下限，不能直接在该机按旧计划补开；需空间整理/迁移方案，不能擅自删checkpoint/replay释放空间。sv2原GPU0当前仍有外来作业，原1/2/3/5已无C/G作业，但不是对下一次启动的预约。前八旧卡分配与后八最新映射不能互相覆盖，任何迁移/换卡/增卡仍由用户指定。本轮仅状态核查，不在此过程中补跑或重启整个campaign；补跑应在明确有效的资源范围及空间条件内保留旧blocked索引，生成新的受限启动记录/尝试，不能篡改为已自动续跑。

## 实际版本、本轮变更与证据

| 范围 | 实际训练及既有工程测试SHA |
| --- | --- |
| 新增前八 | 18b85c4d85ff456e6822626f17cb14cc0f44d2b8 |
| 后八sv1 | 6e468ad4a8252f21401eab6a9761626b33659f0e |
| 后八sv2 | dbd28b6703d51d61e5d3485dbfea6c413075ea85 |
| hopper六条 | 743089f40662b031a5cf2b6bf988decb52ba0b28 |

sv1/sv2/sv3当前checkout分别仍6e468ad4/dbd28b6/743089f4，均干净；前八已在18b85c4结束，不能把当前sv1 HEAD当该批实际训练版。额外启动上下文2项测试来源 `5c6dd4fae0b35ac9d769dd00bc7d2845dfd226eb`。本轮没有新数值测试或完整agent/GPU验收，也不需要为只读状态重跑训练测试。

本轮初始本机HEAD `256b5a22ae656b9f301937998ce889aa5ad3f3d1`，相对上次状态40419cb新增的是hopper GIF分析脚本和文档，非训练路径。本轮仅新增状态报告/索引和日期，实际代码/科学协议/试验不变。六个关键训练源码与前八及后八运行版本相同；对hopper旧版则protocol_v1.py存在随后新增的按protocol选择任务集合校验，不能声称四个旧运行SHA源码全相同；hopper仍运行其固定743089f4。本轮检查记录具体Git差异，没有把当前HEAD冒充已在全部服务器实测。全部原AGENTS/CHATS/RESEARCH/STATUS/proposal/TeX草稿保留，STATUS仅本节及日期入此次提交。

本机证据 `analysis/outputs/all-runs-status-20261030-1340/`：snapshot.json（38槽位、尾收据、网格、现有audit/目录/版本及资源）、online-and-completion.json（27条线上状态/配置、10条精确摘要、completion/表hash及历史16条标记）、status-validation.json、runtime-version-check.json、DELIVERY.json及HANDOFF-current03.txt。源index/final/completion/table路径与hash逐项保存；没有复制大检查点/replay或提交密钥。只本机提交，未推送GitHub、同步服务器Git或直接改服务器源码；最终完整SHA见DELIVERY/最终回复。

## 下一步

首选继续当前03，落实11条缺失Dt的资源/空间方案，按原冻结设置补齐，再完成32条完整指标/线上精确摘要和全体审计。可立即进行小产物复核、只读空间盘点和补跑CPU准备；实际启动依赖有效分配、空闲检查及容量，不因观察到空卡自行迁移。科学配置和指标已冻结，无需重新等待02选择alpha/c/seed；新的方法因果复验另交02与用户。已齐全的hopper六条可由用户交现有05分析，依赖既有完整审计而非新增运行；不自动建chat/发消息。完整可复制prompt在本轮HANDOFF-current03.txt，最终交付不以交接替代缺失运行。
