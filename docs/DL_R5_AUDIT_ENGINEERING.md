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

与 producer 提交逐函数 AST 对照：`audit`、`validate_record`、`validate_pretreatment`、`replay`、
`official_transition`、随机计划/label/key、策略与 bundle 数值检查不变；只改元数据清理、退出认证、
resource 身份检查入口和 CLI。`analysis/dl_live_statistics.py`、共享物理实现和所有科研阈值不改。
