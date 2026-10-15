# R5 独审入口工程修复

2026-10-15，Agent 在准备完整独审时发现并修复两项工程缺口；这不是用户逐项确认的新科研选择。
运行中的 R5 producer 仍为 `1584ef31c322fc3371ea9ce220c06fe7e654b23a`，原执行 checkout、
512 单元计划、科学规范、统计与物理复放代码均不修改。新 auditor 须另行提交，在 clean 隔离
checkout 完成 Linux 验收后使用；本文件不声称真实 R5 已完整生成或已通过独审。

## 外部启动权威

旧独审只核 run 内成功 terminal 和原 PID 不活跃，不能证明外层 `Popen.wait()` 的真实退出。
正式 CLI 现在必须同时提供四个参数：

```text
--launcher-terminal /绝对路径/terminal-attempt01.json
--launcher-sha <该文件原始 SHA256>
--launcher-launch /绝对路径/launch-attempt01.json
--launcher-launch-sha <该文件原始 SHA256>
```

旧 r2 实际 terminal 只有 `git_sha,engineering_version,parent_returncode,parent_terminal,
elapsed_seconds,external_failure,terminal_at_utc,supervisor_terminal,
supervisor_terminal_sha256,main_log_sha256`；没有 argv/run/parent identity，也没有 launch JSON pin。
因此必须外部独立固定 launch 和 terminal 两份原始文件，不能仅凭同目录文件名把 launch 当作已认证。
不向旧产物补写字段，不重写旧启动记录。

在任何可信 pickle 解码前核验：真实 parent exit0、无外层失败、外层实际耗时不超过原 12600 秒；
嵌套 supervisor 与 run 内原文件及 SHA 一致；launch 的 Git/run/cwd/完整 argv、原输入路径、
物理 GPU5/CUDA/EGL 环境、原 parent PID/start_ticks/PGID、12600/12610 预算一致；
原启动器源码与主日志 SHA、实际命令指向的 producer 源码与原 source pin 一致。
12610 仅是抓失败的外层 watchdog，不放宽 12600 成功门。

正式 CLI 强制执行外部权威验证，完整图认证入口再验证一次。已有直接 Python fixture API 保留，
不得拿直接 fixture 调用冒充正式 CLI。独审结果继续保留原 `run_git`、实际 `audit_git`、run 路径
与完整 512/32 认证状态；执行控制器须另存完整 argv、两个外部 raw pins、真实独审退出与结果 SHA，
供 predecessor retirement 绑定。新 auditor Git 不能替换原 producer Git。

## 进程身份与预算

元数据 helper 使用独立 session；创建时记录 PID/start_ticks/PGID。发组信号前核原 leader 身份，
并扫描同 PGID 的活成员必须仍属于原 SID、start_ticks 不早于原 leader。空组不发信号，
leader 已 reap 但原 session 尚有活 descendant 时继续真实清理；确认无活成员后才删除 owned 登记。
PID 复用或组身份不明不授权 kill。原 parent 和每个 family worker 的原 session 也须无活成员；
不清理或干扰其他作业。

独审仍是全入口 1800 秒、0 GPU，包含 import/Git helpers/全文件 hash/可信解码/官方 CPU
物理复放/统计/输出与 owned 清理。超时或证据不足保留失败，不改门、不延时补审部分对象，
不自动重启 producer 或重新采样。原 RGB/NN/策略子 key 仍仅认证存储证据，不独立重演。

## 验收范围

本机针对性测试含实际旧 launch/terminal schema、坏 raw pin/退出码/预算/argv/原身份/嵌套
supervisor/源码反例、正式 CLI 缺外部权威拒绝、PID 复用不发 kill，以及原单元官方 CPU 物理复放。
Linux 另须执行真实 reaped-leader/活 descendant 清理用例。fixture 不是完整 512 个真实 N1M 单元验收。

上述 7c 身份修复与 producer 提交逐函数 AST 对照：`audit`、`validate_record`、`validate_pretreatment`、`replay`、
`official_transition`、随机计划/label/key、策略与 bundle 数值检查不变；只改元数据清理、退出认证、
resource 身份检查入口和 CLI。`analysis/dl_live_statistics.py`、共享物理实现和所有科研阈值不改。

## 确定提交的 Linux 验收

实际测试提交 **7c4affc49d4535b83804cb443471141645fbe658**，sv3 新 clean 隔离目录
`/data/Policy_Discrepancy/dl-r5-audit-code-20261015-7c4affc`；原 1584 producer checkout 不移动。
七个 R5/R4 相关模块共 55 项全部通过、无 skip，unittest 24.445 秒、测试进程
24.750257851 秒、含 clone/checkout/source binding 的完整入口 25.775878075 秒/300 秒。
实际 SSH exit0、26.375 秒，0 GPU、owned 空；Linux reaped-leader/活 descendant 用例实际执行。
246 份实际 Git blob 与 working bytes 一致，前后 source manifest 相同，含本工程规范。

证据目录 `/data/Policy_Discrepancy/runs/dl-r5-audit-cpu-20261015-7c4affc/attempt01/`，
本机 ignored 镜像 `analysis/outputs/dl-r5-audit-cpu-7c4affc/attempt01/`。
`linux-tests.log` SHA256 `42267ef1da716fb0b3a50e6574c9e97f1985aaaae249bedad728be40f36be57d`；
`source-binding.json` SHA256 `b83a21d302f7f816f2669522e14a8bbeb8623ed62edd2678a4ecd5bb802e66db`；
`terminal.json` SHA256 `3fb74e9b6e7f6d4b78e0ea388d07908127649e320a9b728646346ff39e050927`。
本机 `ssh-terminal.json` SHA256 `92022bf84134cb25b101c083ddc6f31b2c0d4f6b0ad047afcfb526ba7f038678`。
不读取真实 R5 中间科学值，不构成完整 512 单元/N1M 独审或方法有效结论。

独立复核重算全部246个实际Git blobs、bundle/controller/helper pins与日志相符。
精确边界：原测试前核working bytes，测试后的manifest再次来自Git blobs；
21:31:26另做一次0GPU postcheck，逐246份working bytes核同、Git clean、owned空，0.205502872秒/120秒。
该记录是后来补核，不追称为原测试时的工作区末遍检查。外部
`/data/Policy_Discrepancy/runs/dl-r5-audit-cpu-20261015-7c4affc/postcheck.json`
SHA256 `468317e6f72d7082322a63a32e20d5088250c22c72b3043e2cb09f3e42898470`。

另已准备 CPU 独审外层控制器（ignored，独立静态复核、仅语法编译，尚未真实执行），
实际 `Popen` 前到退出及 owned 清理固定 1800 秒；脚本自身安装/预核不冒称已包含在此计时。
终态版本 `DL-r5-audit-launcher-r1`，绑定原 producer launch/terminal 双 raw pins、实际 auditor Git、
完整 argv、结果原始 pin、真实 returncode/elapsed、原 PID/start_ticks/PGID/SID 及组清空状态。
退出非零、超时、身份或组未释放都进入 `external_failure`，不凭结果 JSON 单独宣称成功。
后续 C 的 predecessor retirement 同时绑定完整 audit report 和此退出权威；不使用其中科研统计选分支。
准备后的控制器已仅复制至
`/data/Policy_Discrepancy/runs/dl-r5-audit-inputs-20261015-7c4affc/launch.py`，
raw SHA256 `72d98c21161c4372b21981b20a59a1d040914e30ee40892cae7c128a49ec565f`；
以上为首次独审前的准备记录；此后已实际执行并失败，见下节，不构成成功独审。

## 首次真实全对象独审暴露的 dtype 契约修复

原 producer `1584ef31c322fc3371ea9ce220c06fe7e654b23a` 已完整生成 512 单元/32 family，
外层真实 exit0，主入口约 9288.792 秒。首次使用 7c 独审约 8.845 秒后 exit1：元数据闭包认证
已过，在第一条 record 的 `validate_bundle` 因错误地将 mode 与 logits/sample 全部要求 BF16 而拒绝。
该失败没有完整物理复放或科研统计裁决，不改写为 producer 失败或方法阴性。

原始产生路径决定的精确契约为：

| 字段 | 原 dtype / shape | 源码依据 |
| --- | --- | --- |
| h | BF16 / [4096] | 原 recurrent core 与 RSSM compute dtype |
| qlogits、plogits | BF16 / [32,32] | 原 `_logit` 与 `_prior` |
| qsample、psample | BF16 / [32,32] | `_observe`、`R4Policy.derive` 显式 `nn.cast` |
| qmode、pmode | F32 / [32,32] | `OneHot._onehot_with_grad` 显式 F32；`representative_signal` 仅 stop-gradient，不作 `nn.cast` |

7c 的 N1M 几何 fake 将 mode 也写成 BF16，掩盖了这个缺口；此前真实 tiny policy 测试没有把
其实际混合 dtype bundle 交给 N1M 几何全审。不能用此前 55 项通过声称本问题已经验收。
新修复仅按上述原路径核验每类 dtype，不接受任意 float、不在 audit 中 cast 原数据，不改科学算式、
统计、seed、物理或判败门。原 run 与失败 attempt 均保留；需新 auditor Git、fresh Linux 验收后，
在新的独审目录重新审同一个完整对象，不重采数据或延长原主入口。

root 在完整原 7c metadata/四外部 pins 认证后，先核将被解码的同一 trajectory bytes SHA，再只输出
上述 dtype/shape。首 record 为 `family_00/unit_00/trajectory.pkl`，8,987,356 bytes，SHA256
`3549ae306a11366cf53d2feb16f01b5315a09523363738fd8ceee19e142426de`。
成功 diagnostic 元数据为 server `dl-r5-audit-dtype-20261015-7c4affc/attempt02/metadata.json`，
SHA256 `de0a4d50ebc092b9fca5544fa2c1bd00c43dccb724216a4b93b3e8acfc412c86`，
8.757242301 秒/120 秒、exit0、0GPU，不输出 D/e/Y 或其它科研值。
attempt01 因 root 将脚本命名为 `inspect.py` 遮蔽 stdlib 而 import 失败，未认证/解码；保留原失败，
attempt02 改为 `metadata_probe.py`。这一 metadata 核对不是完整物理或统计独审。

本机修复后 `tests.test_dl_live_audit` 共 14 项，13 通过、1 项 Linux `/proc` 测试在 Windows 跳过，
4.505 秒。新增真实 CPU 官方 `representative_signal`/categorical 路径证明 BF16 logits 产生 F32 mode、
显式 cast 的 sample 保持 BF16；N1M 几何 fake 改为原精确混合 dtype，并在重封原 hash 后分别拒绝
q/p 的 BF16 mode、F32 logits 与 F32 sample。仅 shape/counts 相同不足以通过。
新版固定提交和 clean Linux 补验待 root 完成，不冒称这些 fixtures 是真实 N1M 全对象验收。

## dtype修复与必要接口的确定提交补验

actual/tested **277c18001d353d9101a3622959351d3994699adb**，sv3 fresh clean
`/data/Policy_Discrepancy/dl-r5-audit-code-20261015-277c180`。十模块77项全过、无skip，
unittest38.080秒、进程38.517741790秒、完整入口39.615237942秒/300秒；真实SSHexit0/40.218秒、0GPU、owned空。
252份实际源码及三份额外规范working bytes前后均同Git blobs，最终clean。
真实Linux descendant与symlink拒绝、真实Categorical混合dtype、新末帧/RandomState/官方query反例均实际执行。
第二Agent逐Git blob、原日志、bundle/controller/helper及外部SSH终态独立复算一致，未重跑测试。

证据 `/data/Policy_Discrepancy/runs/dl-r5-repair-cpu-20261015-277c180/attempt01/`，
本机ignored `analysis/outputs/dl-r5-repair-cpu-277c180/attempt01/`。
`linux-tests.log` SHA256 `73283a4a8210aff4fe54e468e9f8c2c162c15b537352cc10afd520d95b443f9c`；
`source-binding.json` SHA256 `92b2593d0eb3c2025dc123680e1ef49db50e117a49ed4c42a8d21305de25918f`；
`terminal.json` SHA256 `c16adb7b26f234ec2106fd8f3135710c888c18672cfeaf8d832ed1b37c7fb7d9`；
本机SSH终态 SHA256 `c1067e1c9ebc600ffdb2c4b8f0c477603b85ed5305b583a3e7f69766601b232d`。
有限CPU通过不代替真实512全审。

修复版实际完整CPU独审已完成，输出新目录
`/data/Policy_Discrepancy/runs/dl-r5-independent-audit-20261015-277c180/attempt01/`，
真实parent459454/start409578209/PGID=SID=459454；外层继续使用原已核SHA72d98控制器、1800秒整体预算。
22:21:07实际外层exit0，完整655.220153340秒/1800秒，external_failure为null、原进程结束且owned组清理完成；
本机SSH观测实际exit0/656.375秒，原handle98628已结束。完整512unit/32family认证，
247872个source步、51200个outcome步、512个shadow步的物理和奖励byte-exact。
这不独立重执行NN、renderer、public policy或subkey sampling，不证明信息独立性或方法有效。

成功`audit.json`947544字节，SHA256 `144c4490e69283ab14ad1a96fd8cbd9c47945875797a91ac6d420dade9e216cd`；
外层`terminal.json` SHA256 `d6164a94bdb7b9a95579454d539643d705f808d798d8f44043dad3d0101c8614`；
外层`launch.json` SHA256 `0704040141c34e05e67b23469f0d3890075dd6eebb5d8aa66214d7a51942d144`；
`audit.log` SHA256 `7e2a697b9c3f8c20ed7d80b5f6b31d86fad5ece00d86ea1dfc2ab63843115711`。
本机完整小镜像`analysis/outputs/dl-r5-independent-audit-277c180/attempt01/`。
原7c失败目录不覆盖，原1584数据不重采；成功prior physics audit Git是277c完整SHA。
冻结统计主检验未支持且sham出现implementation concern，详[实际结果](DL_R5_R2_RESULTS.md)；
完整性通过与科学阳性分别记录，不能互相代替。
