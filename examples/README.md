# 轻量复现材料

本目录收录已有实验中的代表性权重、配套配置和网格。四份权重合计约 5.43 MiB，连同网格和配置约 6.26 MiB。

## 已收录文件

| 模型或数据 | 文件 | 大小（约） | 用途 |
| :--- | :--- | :--- | :--- |
| 固定布局 MLP | [checkpoints/mlp/best_model.pt](checkpoints/mlp/best_model.pt) | 0.14 MiB | uniform 布局，6 个传感器，预测 Re 与相位 |
| 动态 DeepSets | [checkpoints/deepsets/best_model.pt](checkpoints/deepsets/best_model.pt) | 0.34 MiB | 动态传感器集合编码器 |
| pPINN | [checkpoints/ppinn/best_ppinn.pt](checkpoints/ppinn/best_ppinn.pt) | 0.32 MiB | 已经验证集选择的参数化流场模型 |
| POD | [checkpoints/pod/pod_model.npz](checkpoints/pod/pod_model.npz) | 4.62 MiB | 使用 60 个候选传感器的 POD 基础模型 |
| 网格 | [data/mesh.h5](data/mesh.h5) | 0.75 MiB | `processed_data_base_60` 的网格与传感器几何 |

配套文件包括：

* MLP：[checkpoint_meta.json](checkpoints/mlp/checkpoint_meta.json)，含传感器索引、归一化状态、结构、训练参数和工况划分。
* DeepSets：[config.json](checkpoints/deepsets/config.json)，含动态编码器与布局配置。
* pPINN：[config.json](checkpoints/ppinn/config.json)，含压力尺度、密度、物理参数及选择信息。
* POD：[eval_config.json](checkpoints/pod/eval_config.json)、[pod_model_info.json](checkpoints/pod/pod_model_info.json)，含模型与评估设置。

## 数据与使用约定

固定 MLP 对应的零基传感器索引为 `[0, 10, 20, 30, 40, 50]`，输入按 `u0,v0,p0,u1,v1,p1,...` 展平，复用检查点中的归一化状态。它与本目录的 60 传感器 POD 模型分别作为各自实现的迁移对照。

pPINN 配套配置采用 `D=0.01`、`rho=1000`、`pressure_scale_mode="rho_uinf2"`。推理时沿用旧模型对坐标、Re、相位和输出尺度的处理。

完整工况数据保留在外部目录。本目录中的 `mesh.h5` 只有网格和几何信息，不包含工况快照；小样本预测还需要符合项目字段约定的 `ReXXX.h5`。默认将完整数据放在项目的 `data/processed_data_base_60/`，或通过 `configs/paths.json`、命令行指定实际目录。迁移对照使用 Re200、Re450 的少量快照。

配置中的相对路径以项目根目录为基准，输出目录使用 `runs/examples/<model>/`。`source_artifacts/` 路径只表示原实验材料来源，并非已收录文件或新的模型加载路径。

当前已完成材料收录与完整性核对。新项目的模型加载、推理接口仍按重建阶段迁移；运行示例和小型工况数据随对应模块补齐。

## 来源与完整性

材料来自原项目的既有 WP-A 实验产物，代码迁移参考版本为 `5d73308`：

[原项目仓库](https://github.com/leecf886/2d_surface_cylinder_flow)

[artifacts.json](artifacts.json) 记录每份文件的相对路径、体积、来源路径和 SHA-256。来源路径相对于原实验数据根目录。

权重与网格按原文件逐字节复制，SHA-256 已核对一致。JSON 配置保留数值参数，将历史机器路径整理为相对路径；清单同时保留配置原件和收录版本的校验值。权重本身未重新序列化，其中的旧路径对象由后续加载适配器处理。

完整 HDF5 工况、训练缓存、全场预测数组和重复运行产物继续外置。`.gitignore` 已为上表中的五个二进制文件设置例外。
