# DL-NM1 原 replay 显式迁移

2026-10-15，工程 DL-replay-stage-r1。只搬运已认证的历史原字节，不改原库存、
不恢复在线队列/随机流、不运行模型、不占 GPU。独立 C 仍须严格导入全部 NPZ。

## 确定源码与 CPU 验收

实际/测试源码 `e70419bd9e8e3f42dd8cad41d8897decb4fbe455`。
sv3 clean checkout `/data/Policy_Discrepancy/dl-nm1-stage-cpu-code-20261015-e70419b`；
sv2 新隔离源码 `/data/Policy_Discrepancy/dl-nm1-stage-code-20261015-e70419b`。
原项目 checkout 和运行中 R5 的 1584 源码均未移动。

新迁移 helper 纯 stdlib；loader 仅在显式提供 stage 路径与 raw SHA 时允许迁移，
默认原路径/receipt 不变。原 decode、CRC/schema/UUID/16 链/全65跨度验证保留，
拒隐式换目录或重写库存。source 与 materialize 各 900 秒 CPU 全入口，复制失败保留 partial。

clean Linux 18 模块114项：112通过、2原有 optional skip，无新增 skip；
原 stage/旧 probe 的 Linux 进程超时检查均实际执行通过。unittest 208.021秒、
测试进程209.995538248秒、完整入口211.230023558秒/300秒，exit0、owned[]、0GPU。
241源码及7份规范前后真实 Git blobs 一致，父 source_binding 与独审 source_manifest 一致；
另一个 Agent 对本机镜像独立重算全部 blob/日志/终态，无差。它未再次运行测试。

证据：`/data/Policy_Discrepancy/runs/dl-nm1-stage-cpu-20261015-e70419b/attempt01/`；
本机 ignored `analysis/outputs/dl-nm1-stage-cpu-e70419b/attempt01/`。
linux-tests.log SHA `38a841ffec67975b48ccfada75be8119a38f3eb95e0366aee4dc29836ae39b35`。
bundle SHA `c9f6df78b6cc2261525c5fdc00a36e8b7a1981f8ecfec7be1056bcbe14d16f97`，
controller SHA `86af2fa738c30b8fa54eb31b60a438269ebb4587ddf1797255c74d7a287e853c`。

## 真实源认证：已完成

root 用 SSH 实际查询 sv2 主机名为 `lyg0360`；sv3 为 `lyg0326`。
原 replay：
`/data/Policy_Discrepancy/runs/dl-pilot-r3-20261013-13ec72a/dmc_quadruped_walk/N-seed0-attempt01/replay`。
原库存：`/data/Policy_Discrepancy/runs/dl-replay-inventory-20261015-7ac2990/attempt01/inventory.json`，
1,209,489字节，SHA `3339b71ca033d5fc065130751fe163aca05a684b2f14f570ab2018219a3a706d`。

sv2 source capture 严格核全部1280文件、11,421,580,553字节与原库存一致，双遍全 SHA 及两份
原 run 元数据前后相同；没有 NPZ 解码、CRC 复验或模型/环境加载。
输出 `/data/Policy_Discrepancy/runs/dl-nm1-replay-stage-20261015-e70419b-source/source-capture.json`，
579,864字节，SHA `da95f863d950993f08b3ca43f47b5b06c9afe71361e1c988dc9d651ae4ac6e2e`。
内部 entry→child cleanup 17.553213891秒；CLI 打印完整耗时18.172377851秒；
独立外层 SSH 真实 exit0、18.891秒/900秒，20:33:15.816860（北京时间）封口。
内部 JSON 不自证父进程实际退出，外层证据单独保留。

源捕获/原库存已 SCP 到本机后核 raw SHA，再原样复制到 sv3 staging inputs；没有重写原库存。
本机小型证据 `analysis/outputs/dl-nm1-stage-e70419b/`；
外层 source-ssh.log SHA `9b485caeaf3769589bfa99963a3bae88788886dde28d9cbe508e9cf5452cbab6`。

## 目标暂存：进行中

外部 `scp -3 -B -r` 将原 replay 复制到 fresh
`/data/Policy_Discrepancy/runs/dl-nm1-replay-stage-20261015-e70419b/incoming`。
传输自设1800秒有限 I/O 上限、0GPU，不是 source/materialize 各900秒的延长。
20:43快照 incoming 约6.48GB，尚不能称全量复制成功或 C 输入已获资格。
最终目标计划为同级 fresh `replay`；materialize 会逐文件 xb 复制并核 incoming/目标双遍全 SHA，
只有实际 CLI exit0及外部全 wall≤900秒后才接纳 stage-record.json。
失败保留目录与证据，不覆盖/删掉重来。输入与输出收据均放 incoming/replay 外。

还需完成传输、materialize、独立完整 raw 库存复核与 C 严格加载。
迁移成功只说明同字节历史数据可用，不是真实 N1M 校准、正常学习或方法有效证据。
