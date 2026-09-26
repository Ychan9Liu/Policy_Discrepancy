# 项目状态

更新日期：2026-09-27。

## 已验证

- 本机代码仓库：`D:\program\PD\dmr3`；`origin` 为 `Ychan9Liu/Policy_Discrepancy`，`upstream` 为官方 DreamerV3。
- `sv1`、`sv2`、`sv3` 的代码位于 `/data/Policy_Discrepancy/repo`，Python 环境位于 `/data/Policy_Discrepancy/envs/dreamer`。
- 三台机器在提交 `e3f02248693a79dc8b0ebd62c93683888ddaccfe` 上通过了 `debug` 假环境的 CUDA 训练短跑，产生训练损失、日志和检查点。测试目录分别为 `/data/Policy_Discrepancy/runs/smoke-verified-sv1`、`smoke-verified-sv2`、`smoke-verified-sv3`。
- 三台机器的依赖快照一致；见 `env/requirements-dreamer.lock`。
- `sv3` 的 DNS 已写入持久 Netplan 配置，GitHub 和 PyPI 访问已验证；尚未通过重启验证。

## 尚未决定或验证

- 真实科研任务、环境、研究假设、方法细节、baseline、评价指标和训练预算待讨论。
- 真实任务及其额外依赖尚未测试；`debug` 短跑不代表正式任务可运行或效果可复现。
- 正式实验启动、结果汇总和跨服务器分配脚本尚未实现。
- `sv3` 的根分区仍满；用户已决定当前先跳过处理，实验文件继续放在 `/data`。

下一步先确定科研任务与实验协议，再实现配置和脚本。更新状态时只写已观察到的事实，并附提交或运行目录。
