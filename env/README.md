# DreamerV3 环境快照

`requirements-dreamer.lock` 是三台服务器当前 `dreamer` 环境的 `pip freeze` 快照，已核对三台内容一致。它记录软件版本，不包含系统驱动、CUDA 运行时和虚拟环境创建方式。

当前服务器环境路径为 `/data/Policy_Discrepancy/envs/dreamer`，由 Python 3.11 的 `base-py311` 环境以 `venv --system-site-packages` 创建，因此需要保留基础环境。服务器 GPU 训练已验证 JAX 0.4.33。新增或升级依赖时，应在隔离环境验证后更新此快照，并重新检查三台环境是否一致。
