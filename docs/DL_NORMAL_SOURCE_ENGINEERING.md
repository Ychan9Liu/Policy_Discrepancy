# R5 正常参考 source index 工程

2026-10-15，`DL-normal-source-index-r1`。Agent 在完整预验证内采用的必要 CPU 工程，
不是 T、真实来源执行预算或机制结论。当前仅 generated fixture；不读取真实 R5 科学值。

实现 `scripts/dl_normal_source_index.py`，只认证并发布固定来源索引。没有模型、环境、
物理重演、标签查询、来源采样或科研统计裁决；不修改 R4/R5、正常特征或读出算法。

## 完整权威先于解码

正式 `authenticate_and_index(args)` 与 CLI 都强制提供：

```text
--run <原 R5 run>
--expected-git <当前 index 的完整执行提交>
--prior-physics-audit-git <已完整成功的原独审完整提交>
--launcher-terminal <原 producer 外层 terminal>
--launcher-sha <该原文件 raw SHA256>
--launcher-launch <原 producer 外层 launch>
--launcher-launch-sha <该原文件 raw SHA256>
--audit-report <原完整独审 audit.json>
--audit-report-sha <其 raw SHA256>
--audit-terminal <独审外层 terminal.json>
--audit-terminal-sha <其 raw SHA256>
--audit-launch <独审外层 launch.json>
--audit-launch-sha <其 raw SHA256>
--output <run 外 fresh index.json>
```

原 producer 四参数不能从 sibling 名称推权威或省略；新增完整独审三文件各有外部显式 pin。
原 run Git 固定 `1584ef31c322fc3371ea9ce220c06fe7e654b23a`，原 physics-audit Git 必须由
调用方显式给出已完整成功的完整 SHA，且与独审 report、外层 launch/terminal及实际 argv相同。
`7c4affc49d4535b83804cb443471141645fbe658` 首次真实独审在 BF16 dtype检查失败，
没有完成 physics/statistics，不能作为通过来源；成功独审版本仍待后续真实验收。
当前 metadata 重新认证使用当前 index 的 clean 确定 SHA，不冒称该 SHA 重演过原 physics。
当前 auditor 与 physics/primitive/statistics 五份依赖源码逐 raw Git blob 核同调用方指定的
成功独审 Git，并核实际 imported 路径。原1584数据不因独审修复重采。

采用实际 `DL-r5-audit-launcher-r1` schema：独审真实 returncode0/无 external failure、
完整退出≤1800秒、原 PID/start_ticks/PGID/SID 及 owned 组已结束、报告 full raw pin、
实际完整 argv/CPU 环境、原 producer 两份外部 pin、log/controller raw pins。controller
SHA 固定 `72d98c21161c4372b21981b20a59a1d040914e30ee40892cae7c128a49ec565f`。

复用新 R5 `external_authority` 与 `authenticate`，显式安装原四 pin 的 CLI authority；
不走旧直接 fixture API 的无 authority 分支。完整独审 report 必须认证全部32×16及
全部官方 physics/rewards，库存与当前完整 run 一致。完整 run 图及外部权威稳定后才调用
`trusted_pickle`，该函数对实际同一 bytes 再验 raw SHA 后解码 `trajectory.pkl`。
处理结束再次扫全 run、外部完整权威、依赖原字节和 clean/current Git；不接受 partial。

metadata `authenticate` 仍核原 calibration/统计输入封口图及解析关系；本入口不调用
`statistics.analyze`、不重演 physics、不使用科学正负筛选任何 source。独审报告的统计
字段不是来源取舍条件。外层真实退出由其已认证收据支持，索引不独立证明远程 SSH 历史。

## 发布的边界与对齐

严格按 family0..31、replicate0..15输出512条，fit为前8 family/128条，held为后24/384条。
每unit仍是一条独立 prefix，不能拼成同family的一条 episode；不能按长 prefix 增加独立重复。
相同 counts 不足以对齐：核 global/local ID、唯一 source ID、trajectory raw pin/path、原随机
t/plan/env/eval seed、连续公共 EVAL root/counter及 fit/held 顺序。

索引只发布原文件引用、半开范围与 typed hashes：

- `observations[0:t+1]`、`snapshots[0:t+1]`；
- `raw[0:t]`、`applied[0:t]`及这t次 transition 的 reward/discount/flags；
- `keys[0:t+1]`，含当前 t 决策的事前公共 root；
- 每frame的原 obs/snapshot hash、counter/root及 previous raw 索引/hash；
- 当前标签位置严格对应 `snapshot[t] / observation[t] / reward_f64[t-1]`。

快照的 `snapshot_sha256` 明确采用
`snapshot_hash_format=DL-r5-snapshot-json-array-task-pickle5-sha256-r1`：
先对 `str(dtype)+str(shape)+integration.tobytes()` 作SHA256；再将该hash、clock、
reset/wrapper bool及 `SHA256(pickle.dumps(trusted_task_dict, protocol=5))` 组成JSON列表，
按 `separators=(',', ':')` 编码后作SHA256，精确沿原R5 snapshot hash协议。
这里任务字典包含官方 `np.random.RandomState`，保留其完整RNG状态；不把任意object放入
共享张量 `tree_digest`。本函数只处理调用方已可信解码的快照，单独不认证来源。
query helper的独立readonly快照hash若采用其它格式，调用层必须分别核同一个saved snapshot，
不得直接将不同格式的hash等同。

当前 t 是非reset、非last、非terminal的物理帧，但其 outgoing **source raw 不可用**；
`raw[t]` 已属于随机干预，不能拿来当正常历史。索引记录 `source_raw_available=False`、
`source_raw_index=None`，不补零、不借 native qraw、不发布 postintervention 新轨迹。
后续特征在当前 t 应只读取后结束，不再推进 source prevraw。索引没有旧 latent/carry；
新模型必须 reset 后沿共同原 RGB/raw 重推自己的 posterior 历史。

`project_prefix` 是纯投影，单独不认证来源；row中的 `projection_authenticates_source=False`
不能被改称通过真实来源。只有完整正式入口的外层权威图才认证这些行属于该 run。
typed prefix hash只含处理前数据；source ID还绑定全 trajectory/raw run库存以保持完整来源身份，
不会根据其 opaque hash或D/e/Y数值选行。

## 有限验收与限制

generated fixture 验证：同 counts 的 ID/顺序/trajectory/fit-held错位；partial与整体图变化
在可信 pickle前拒绝；坏/缺外部 pin在载入 auditor前拒绝；原真实外层 schema中的坏退出/
预算/argv/producer pin拒绝；原 reward的i→i+1、EVAL key、clock/reset偏移拒绝；
当前t截断和postintervention数据不进入prefix hash。没有运行真实来源、NN或官方物理。
本机验证命令如下，10项中9通过、1因Windows无symlink权限跳过（1.275秒，exit0）；
`..`、已有输出拒绝已在该测试内执行，alias拒绝须由clean Linux补验。未设置额外DLL路径：

```powershell
& 'C:/Users/97370/AppData/Local/Codex/dl-validation-env/Scripts/python.exe' -m unittest tests.test_dl_normal_source_index -v
```

测试还核显式成功独审 Git 及其 raw dependency blob，拒绝缺失或缩写 Git。
快照fixture包含真实 `np.random.RandomState`及缓存Gaussian状态：专用hash匹配原R5实现，
哈希查询不改变RNG，改变RNG状态会改变快照及prefix hash，pickle5 roundtrip保留结果。
输出路径先核父链无symlink、解析父目录与`..`、再核run外fresh，不能经alias写回producer图。

真实入口限定原 sv3 Linux 主机，使原 PID/session 结束检查不被其它主机的空 `/proc` 替代。
CLI 已固定全入口900秒/纯CPU的工程上限，含 import/hash/解码/输出；当前不据此启动真实
source 执行，实际执行预算尚待root采用。实际运行还需独立外层真实 exit0、完整耗时及输出
raw pin，不能单靠 index JSON 自证入口退出。失败保留，不扩时、补源、拼接或按科学符号换来源。
来源认证不是机制、读出或性能阳性，也不自动启动 T。

**确定提交Linux补证**：actual/tested `277c18001d353d9101a3622959351d3994699adb`，
十模块77项全过/无skip、完整39.615237942秒/300秒，真实SSHexit0/0GPU。
本模块十项均实际执行，含Windows因权限未执行的symlink拒绝；真实RandomState快照协议和路径别名反例通过。
252源码及本规范等三norm的working bytes前后同真实Git blobs，最终clean，第二Agent独立复算一致。
证据见[共同工程记录](DL_R5_AUDIT_ENGINEERING.md)。修复版277c已在22:21:07完整独审成功，
可以作为显式prior physics audit Git；其主科学检验未支持不能用于选择更有利的source。
当前没有执行真实source index，fixture不是该真实来源验收；实际900秒CPU执行预算尚未启用。
