# 正常来源当前任务量标签：有限 CPU 工程

**最新真实验收**：actual/tested `47db6265311bc04dbdcc79439842eb8b6e707097`已完成固定512/32，
247872真实官方prefix步、512reset、全部当前任务量query readonly；整体controller797.825秒/900秒，
root实际SSHexit0/798.329秒。输出rawpin及内外层退出边界见[实际结果](DL_NORMAL_LABEL_RESULTS.md)。
下文未执行描述是实现/本机验收时的历史范围，不能覆盖或冒充这次真实完整证据。

2026-10-15，`DL-normal-labels-r1`。这是完整预验证内的必要工程实现与本机 fixture
验收记录；**尚未激活真实来源执行预算**，不采用 T、GPU 特征执行或任何机制结论。
本轮没有读取真实 R5、SSH、GPU、学习、提交或运行真实512条标签。

**组合Linux测试隔离修复**：actual `01fbef2331e96776ef22a1e1b5a4313bbe586f4d` 首次24模块
完整174项组合在272.589秒结束，外层SSHexit1/276.828秒；1个error、2原optional skips。
唯一error为mock512成功case继承其它模块已导入的JAX，被生产入口正确的纯CPU进程要求拒绝。
不能把该失败报告为完整Linux通过，也不是实际512来源或模型失败；原日志保留
`analysis/outputs/dl-nm1-repair-cpu-01fbef2/attempt01/`。
仅将此mock coordinator测试放进独立CPU子进程，与正式CLI隔离一致，不删sys.modules/JAX，
不改变任何生产纯CPU限制、源码或判据。本机新5项2.200秒/exit0；预先import jax的父进程中
同5项2.059秒/exit0。下一以新确定Git重做24模块整体600秒Linux验收，真实标签尚未执行。

**后续采用边界（root，2026-10-15）**：Agent在完整预验证授权内采用一次整体900秒CPU真实入口，
前提是本实现进入确定Git且clean Linux有限验收通过、原prior277c成功完整独审及全部authority pins可认证。
本页下方“未激活”描述实现/本机验收时状态；新的采用不把测试升级为真实标签验收。
900包含source/index与所有512条prefix/末遍认证/落盘，不另叠source预算；失败保留、不拼partial、
不按R5科学符号换来源；实际版本/运行/外层退出/完整标签仍待root记录。不采用GPU特征、T或科学阳性。

实现为 `scripts/dl_normal_labels.py`，测试为 `tests/test_dl_normal_labels.py`。
只读当前查询复用 `analysis/dl_normal_task_query.py`；完整来源认证复用
`scripts/dl_normal_source_index.py`。这些旧接口及原 physics/R5/feature/index 本轮未改。

## 固定范围与前置

入口固定32 family×16 unit=512条；前8 family/128条为 fit，后24/384条为 held。
保持原有序 source ID、trajectory raw pin、plan、env seed、时间 t、fit/held 和 prefix hash，
不按 D/e/Y、标签值或科学正负选取来源。相同 counts 不能代替逐 ID 对齐。

整体硬上限是 **900秒/0GPU**，从入口首个计时点开始，含 import、首次完整 source index
认证及哈希、全部512次 fresh suite/reset、全部 prefix 物理步、末尾重新认证和写盘。
source index 的内部上限受这个共同 deadline 限制，不能将其900秒另加在标签预算之外。
实际完整吞吐未测；超时或任一来源失败时不能发布部分成功、续拼或自行加预算。
脚本/fixture 可用不等于已获真实执行授权，确定提交、clean Linux 验收及实际预算由 root 后续处理。

`run(args)` 首先调用正式 `source_index.authenticate_and_index(args)`；CLI 强制原 run、
当前完整 `--expected-git`、已完整成功的 `--prior-physics-audit-git`，以及下列五个文件各自的
显式 path/raw SHA256。原 producer 四个参数不得缺失或从旁边文件名推定权威：

```text
--launcher-terminal / --launcher-sha
--launcher-launch / --launcher-launch-sha
--audit-report / --audit-report-sha
--audit-terminal / --audit-terminal-sha
--audit-launch / --audit-launch-sha
--output <原run外、parent已存在的fresh labels.json>
```

来源固定原 R5 production Git `1584ef31c322fc3371ea9ce220c06fe7e654b23a`。
必须认证真实 producer 退出、成功 prior 全512物理独审及其外层完整退出、原进程/组退役、
完整原 run 库存、当前 clean Git 及实际 imported 源码。失败/partial prior 不能作为来源。
当前 auditor/physics/primitive/statistics 五份复用依赖必须逐字节同已成功 prior Git；
labels/index/query 三份当前模块另由当前真实 Git blob及实际 imported路径绑定。

## 单条 continuous replay 与当前读数

完整来源已认证后，对实际将解码的同一 `trajectory.pkl` bytes 再核 raw SHA/bytes，才解可信
项目 pickle。`label_record(record,row,inventory_sha,a)` 单独不是来源认证入口，fixture也不能
借它宣称真实 run 已验收。

每条先重新 `project_prefix`，要求与原 index row逐字段相同，然后：

1. 从 `plan.env_seed` 创建官方 quadruped/walk 并真正 reset；核完整 static MJB及 snapshot0。
2. 独立核 `raw[0:t]` 到 `applied[0:t]` 的原有序 dtype/clip/normalize 映射，按原 applied 连续走 t 步。
3. 每步核原 F64 reward、原 F32编码、F64 discount、first/last/terminal及完整 integration/task/RNG/
   clock/reset snapshot，要求字节相同，不能以旧 `1e-8` 比较代替最终 exact 检查。
4. 在当前 t 调 `query_current`，以 `reward_f64[t-1]` 和 `observation[t].reward` 核当前官方奖励。
   不把随机干预的 outgoing `raw[t]` 用作正常历史或执行它，不使用 t+1/shadow 的奖励或状态。

标签 `velocity_x` 是 **torso-local velocimeter x**，不是 world x；`upright` 为当前官方 torso
upright。两者是 privileged 当前任务量，**不是 actor-critical、观测关键信息或最优策略真值**。
它们不能单独证明 actor 利用、独立信号增量、后验保护、中介机制或性能改善。

query 首末核完整声明 snapshot/task/RNG及 MJB字节不变，不 restore/step/reset/forward/render。
coordinator 只有开头的官方 reset和正常 prefix steps，不显式 restore/forward/render；官方 reset/
step 内部的正常物理同步不计作额外重建。`mjSTATE_INTEGRATION` 不含 xmat/sensordata 等 derived
buffers；读数依赖本条官方 continuous replay 保留的同步，不能用快照 restore 后手工补 forward替代。

## 闭包与失败

结果逐行保存 source ID、plan/trajectory pin、prefix hash、当前 obs/source snapshot hash、两种
snapshot格式、F64标签/奖励原字节 hex及 JSON值、query前后 snapshot/RNG和模型收据。
原 source snapshot使用 `DL-r5-snapshot-json-array-task-pickle5-sha256-r1`，query使用
`DL-normal-task-query-snapshot-r1`；两种 SHA 不能直接比较，只能通过同 saved snapshot exact及
source ID 关联。

`source_index_row_binding_sha256` 仅 hash固定有序的 source_id/global_unit/plan/trajectory/split/
prefix_typed_sha256；不 hash含瞬时时间的未落盘临时 index对象。原完整库存由已认证 prior report
的外部 raw pin及 `run_inventory_sha256` 绑定。

尾部重新设置显式原四 pin 的 `a.CLI_AUTHORITY`，不调用 fixture 的无 authority分支；再次核五份
外部 pin、完整 metadata/run库存、成功 prior 依赖和当前源码/Git。只有512条全部完成才创建 fresh
外部 JSON。metadata认证仍验证原统计封口关系；本入口不调用 `statistics.analyze`，不以 D/e/Y
选行或生成标签。文件存在或其自报 complete不能代替未来外层实际 exit0及整体≤900秒证据。

## 本轮精确有限验收

测试工作树 ancestor HEAD为 `fef52c4c823f05bde33861bf23959f10960deaf0`；实际被测试的是该
工作树加下列两个当时未提交的 labels文件，不将 ancestor SHA冒称已提交的 labels执行版本。
Windows Python3.12.14；NumPy1.26.4、dm-control1.0.48、MuJoCo3.15.0、ml-dtypes0.5.4。
工作目录 `C:/Users/97370/.codex/worktrees/1f9a/dmr3`，精确命令：

```powershell
& 'C:/Users/97370/AppData/Local/Codex/dl-validation-env/Scripts/python.exe' -m unittest tests.test_dl_normal_labels tests.test_dl_normal_task_query -v
```

本次结果：**12项全通过、无skip，unittest 2.062秒，exit0**，其中5项 labels、7项 query。
测试前后下列实际文件 raw SHA256一致：

| 文件 | SHA256 |
| --- | --- |
| `scripts/dl_normal_labels.py` | `0cfb65b5a40faef155b7179039e8ac7d889738b765cc12a1d175b7545d11afef` |
| `tests/test_dl_normal_labels.py` | `fc9c27d0d212f3bcface2a9daf25b255e34e2e913b1dd67bcf6f861180e75785` |
| `analysis/dl_normal_task_query.py` | `98c5a348e72088cbc7d1b7ba1a59031d372acc14ba44aaa55cc09d59d6552b55` |
| `tests/test_dl_normal_task_query.py` | `2f9c35ce485dc3a1f8ad88d950f91eb9b60871a23b6e11e8adfe7346484d0388` |

真实 CPU 单record fixture从官方 seed reset，生成64步 prefix及另一步对照；RGB/raw均为生成数据，
不是实际 R5。验证当前标签、全状态exact、无restore/render、输入不变、改 raw[t]/后续奖励不改当前
结果，并拒错 ID/clock/static MJB、snapshot/F64单 ULP和 raw/applied错位。
512条 closure测试明确 mock完整来源和 per-record计算，只验证 fixed IDs/fit-held与全有或全无的
协调关系、partial/乱序/末尾失败拒绝及再次认证不漏原四 pin。它不是512次真实 CPU重演。
query测试另覆盖只读禁 step/reset/render/forward、RNG/积分/model写入及 F64不同而F32相同的反例。

本机有限通过不认证真实 R5、跨机器重生成、完整900秒吞吐、真实 NN/特征或方法有效性。
本轮只交付 labels/test/本说明；clean确定提交的 Linux补验和任何实际执行仍待 root处理。
