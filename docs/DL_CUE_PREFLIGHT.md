# DL 模型元数据前置与 GPU 资源口径

2026-10-11，记录版本 **`DL-cue-model-metadata-r1`**。这是一项已完成的本机 CPU 只读核查，不是任务线索、物理干预、视觉辨别或方法有效性的验收。候选理论仍为 [DL_CUE_IDENTIFICATION.md](DL_CUE_IDENTIFICATION.md) r0，未冻结任何新实验。

## 1. 实际执行与可追溯证据

查询时仓库 HEAD 为 **`182365de76e5f7742218ef6ebdff417bb790880f`**。查询器位于本机 ignored 目录 `analysis/outputs/dl-cue-preflight/model_geometry_v1.py`，未进入该提交；以其独立 SHA256 绑定执行源码，不能声称该 Git 提交包含查询器。

| 证据 | SHA256 |
| --- | --- |
| `model_geometry_v1.py` | `6309a040da1d00d3b28fe8f63cf0db744aaf6e5fb5b5faf96d27d231739eee1b` |
| `factory-model-geometry-v1.json` | `d2edd481e856d4292a0765b7390ede7cc3f66b0a38411071d96964183764dd1e` |
| `CANDIDATE_NOTES.md`，工程与反例笔记 | `9861929a56d0fb35d09d5abe4670da600da79a04cde40928220dbc1e6c033c4e` |

root 已只读核对源码、收据、三项 hash 与范围，并于 **2026-10-11 21:22:29 北京时间**核验 sv3 证据副本 hash 一致。永久证据位置：

`/data/Policy_Discrepancy/runs/dl-prevalidation-bootstrap-20261010-ae8372e/cue-model-metadata-182365d/`

这里只上传已有证据，未在服务器重跑查询、创建新运行或部署算法；canonical server repo 仍为干净 `d9be5574878d2c3f117cc6667be844f9a527eb84`。sv1／sv2 未同步。查询可用本机受控 Python 执行，必须使用新的输出路径：

```powershell
& "$env:LOCALAPPDATA/Codex/dl-validation-env/Scripts/python.exe" `
  analysis/outputs/dl-cue-preflight/model_geometry_v1.py `
  --output analysis/outputs/dl-cue-preflight/factory-model-geometry-new.json
```

实际依赖为 NumPy 1.26.4、MuJoCo 3.15.0、dm-control 1.0.48，查询器同时核对本机官方源码和 XML hash。它只构造 `quadruped_walk` 的 factory qpos0 模型并做数值查询；construction seed0 不属于训练或科研采样 seed。收据耗时 `.0940000000` 秒，包含构造／查询，不含 Python 导入及最后写盘，不是完整进程 wall time。

reset、Environment／Physics step、render 均设拒绝调用守卫；无外力写入、真实数据／blind、Dreamer／JAX、optimizer 或 GPU。完整 integration、task pickle、task RNG、clock、step_count 与 reset 标志前后逐字节一致。独立 Agent 另审读了数值公式与科学边界；这不是动态轨迹的独立复现。

独立纯数值复核只读取既有 JSON，没有导入查询器或 MuJoCo：焦距、坐标变换与时间尺度一致；假定刚体旋转／trackcom 的投影导数与中心差分最大差约 `8.79e−10`。这个几何对象与下文无约束 articulated 质量矩阵响应不同，不能将两者相乘称实际 pixel/Nm 或反推可见力矩。

## 2. 已验证元数据及不能推出的结论

| 项目 | 已核事实 | 限制 |
| --- | --- | --- |
| 时间与执行器 | 控制步 `.02s`，四个 `.005s` Euler 子步，actuator filter τ `.1s` | 没有执行 transition，不证明新事件的接触或稳定性 |
| 躯干与全模型质量 | torso `67.8584013kg`，全模型 `107.1179058kg` | 工厂模型，不代表自然状态或训练支持 |
| torso 主惯量 | `[1.5322427,1.7643184,2.2108267] kg·m²` | 单体惯量不能给完整关节／接触系统有限位移上界 |
| camera2 | `y`／trackcom，FOV45°；64×64 理想针孔焦距 `77.254834px` | 没有渲染；焦距与标志点导数不证明 RGB 可辨 |
| 世界 y 轴在相机中 | right／up／back 分量 `[0,−.4472136,.8944272]` | 不是相机平面内轴，也不等于任意 yaw 下的 body-local pitch |
| 冻结 q 无约束响应 | `e_yᵀ J_rot M⁻¹ J_rotᵀ e_y=.3702691703 rad/s²/Nm` | 仅在 factory qpos0 的瞬时质量矩阵计算，不含实际接触约束轨迹 |

最后一行对应连续恒定加速度尺度为每 Nm `.00740538 rad/s` 和 `.0000740538 rad`；四子步半隐式恒定加速度尺度为 `.0000925673 rad/Nm`。它们是数值尺度，**不是有限接触响应的上下界，也不能据此选择“可见”的力矩幅度**。投影计算假定全机器人刚性旋转、相机跟随子树 COM；不含实际关节运动、遮挡、轮廓、像素量化或纹理。

外力输入按个体 body COM 施加 Cartesian wrench；运行接口与相机模式的通用含义参见 [MuJoCo simulation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#user-inputs) 和 [camera reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera)。这些在线文档提供接口背景；本次实际版本与数值由上述本机 pin／收据限定。

若以后另立事件对象，工程候选须从可信 `physics[p−1]` 用原 production raw→applied 动作执行到 p，对齐源真实 pre-policy carry[p]；carry 的 prevraw 必须与 action[p−1] 字节一致。不能在 snapshot[p] 额外步进仍拿旧 carry。world-y torque 位于 `xfrc_applied[torso,4]`；候选要求全四子步保持，随后恢复原完整外力数组，并记录每段实际 wrench。只有 stored-applied action 不足以复放外力事件。以上是正确性条件，不是选定幅度或启动协议。

## 3. 当前资源需求

| 工作 | 最低 GPU 数量 | 资源依据与状态 |
| --- | --- | --- |
| 理论、合成统计守卫、现有结果审计、此次元数据查询 | **0** | 当前即可用 CPU 推进；本次实测 GPU 小时为0 |
| 已有 size50m 冻结模型／真实图像诊断 | **1 张独占 A100** | 单卡串行已有实测；原 source／feedback 采样峰约2.5GiB |
| 独立 checkpoint／工程核查并行 | **2 张可选** | 缩短等待；不是同一模型需要两卡，必须先证明工作互不依赖 |
| 当前已分配 sv3 GPU4–7 | **4 张足够覆盖上述并行** | 无需全部启用，不扩大资源；本次没有重新核查实际占用 |
| 后续正常 active-loss 验证／短训练／正式性能比较 | **尚未冻结最终预算** | 既有单步工程采样峰约6.5GiB仅支持该已执行配置，不保证新训练显存或总耗时 |

数量口径是最低一张串行、两张便于并行；更小显存或其他 GPU 型号未经本项目验收。现有四张分配不意味着应同时运行四项，也不证明任意后续配置都已足够。旧六组×三 seed×100k 的约20.4 GPU小时仅为条件短训设计的纯训练吞吐外推，不是当前剩余完整目标的预算；编译、评价、诊断、失败重跑及必要科学设计尚未计入。

本轮没有新 GPU 启动或占用查询，不能报告指定卡当前空闲，也不能由旧快照断言其他项目进程状态。下次实际启动仍须 fresh CUDA／EGL／物理 GPU 占用核验。资源分配继续限定 sv3 GPU4–7。

## 4. 下一必要工作与停止边界

首选在当前完整授权聊天先固定**纯合成 probe 检测范围**：已知线性决策信号、history-only、无信号与 nuisance 对照；固定 seed、维数、聚类、成本、margin、唯一主量和停止规则。可用 CPU，不需要先运行新的物理或模型实验。合成成功不证明真实 q 的有用信息，合成失败仅限制探针的阴性解释。

nested probe 与随机化 cue 仍为两个未采用替代方案。后者缺少独立 RGB 可获取、同历史事件对的有代价动作需求及训练支持证据；本次几何不补齐它们，不冻结 torque、不按 D/K/U 或正号选择幅度／窗口。旧 H100 与 H20 阴性／不确定结果保留，24 blind 未读，gate 与训练未启动。

完整目标仍须第二信号相对 D 的增量、normal active rep 的任务信息损伤／保护、强度匹配与条件 shuffle／reward-only／OOD 替代解释，以及后续学习／性能证据。当前 frozen actor 已有收益只属于即时行为路径的前置，不能当作所有学习路线的必要门。形成可正式实验方法的目标**未完成**。
