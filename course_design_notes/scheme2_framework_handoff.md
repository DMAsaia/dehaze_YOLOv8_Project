# 方案二多任务 YOLOv8-dehaze 框架交付文档

更新日期：2026-05-20

本文档用于把当前“基于多任务学习的雾天目标检测框架”交接给后续调参、实验分析和报告撰写同学。重点不是介绍论文背景，而是说明当前代码已经完成了什么、怎么复现实验、哪些地方可以继续改。

## 1. 当前交付状态

当前框架已经完成从模型结构到训练验证的完整闭环：

- 基于 YOLOv8n 构建 `YOLOv8-dehaze` 多任务模型。
- 在 YOLO Neck 的 P3 特征后接入轻量 `DehazeHead`。
- 训练阶段同时输出检测预测和去雾图像。
- batch 中已支持读取 `clean_img` 作为去雾监督。
- 已接入辅助 L1 去雾损失：

```python
L_total = L_det + dehaze * L1(dehaze_pred, clean_img)
```

- 已支持 `pretrained=yolov8n.pt`，检测主干和 Detect Head 尽量加载 YOLOv8n 预训练权重，新增 `DehazeHead` 随机初始化。
- 已完成 `dehaze=0`、`dehaze=0.05`、`dehaze=0.10` 消融实验。
- 已实现去雾三联图可视化脚本。

当前推荐默认实验设置：

```text
model=ultralytics/models/v8/yolov8-dehaze.yaml
pretrained=yolov8n.pt
data=datasets/VOC_hazy/VOC_hazy.yaml
imgsz=640
epochs=50
batch=4
seed=0
workers=8
dehaze=0.05
```

## 2. 当前机器环境

本机同环境公平对比实验使用以下环境。后续报告中的主实验表建议以该环境下的结果为准。

| 项目 | 内容 |
|---|---|
| 操作系统 | Windows |
| 项目路径 | `C:/Python/project/ultralytics-yolov8-official` |
| Ultralytics | YOLOv8.0.61 |
| Python | 3.10.20 |
| PyTorch | 2.6.0+cu124 |
| CUDA | cuda:0 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| GPU 显存 | 6141 MiB |
| AMP | True |
| 主要数据集 | `datasets/VOC_hazy/VOC_hazy.yaml` |
| 训练集 | 2501 images |
| 验证集 | 2510 images |
| 验证集实例数 | 6307 instances |
| 类别数 | 20 VOC classes |

备注：

- 组员 A 的外部 baseline 使用不同 Ultralytics、PyTorch 和 GPU 环境，只能作为参考。
- 当前主对比应使用 `baseline_yolov8n_local_50e`、`train8`、`train9` 以及 `train11` 的结果。

## 3. 改动文件清单

| 文件 | 作用 |
|---|---|
| `ultralytics/models/v8/yolov8-dehaze.yaml` | 新增多任务模型配置，在原 YOLOv8 检测头前加入 `DehazeHead`。 |
| `ultralytics/nn/modules.py` | 新增轻量 `DehazeHead` 模块。 |
| `ultralytics/nn/tasks.py` | 支持 `DehazeHead` 解析、训练阶段双输出、预训练 Detect 权重重映射。 |
| `ultralytics/yolo/v8/detect/train.py` | 支持 `clean_img` 预处理、`dehaze_loss` 记录、辅助 L1 loss 计算。 |
| `ultralytics/yolo/data/base.py` | 为数据集样本匹配 clean 图像路径，并加入 label 字典。 |
| `ultralytics/yolo/data/augment.py` | 保证 mosaic、mixup、仿射、翻转、letterbox 等增强同步作用于 hazy 图和 clean 图。 |
| `ultralytics/yolo/data/dataset.py` | batch 格式化时支持 `clean_img`。 |
| `ultralytics/yolo/cfg/default.yaml` | 新增/保留 `dehaze` 超参数，默认值为 `0.05`。 |
| `tools/visualize_dehaze_triplets.py` | 输出 `hazy input / dehaze output / clean target` 三联图。 |
| `course_design_notes/scheme2_framework_guide.md` | 方案二框架导读。 |
| `course_design_notes/scheme2_framework_handoff.md` | 当前交付文档。 |

## 4. 模型结构

当前 `yolov8-dehaze.yaml` 的主体仍是 YOLOv8n 的 Backbone、Neck 和 Detect Head。新增部分是第 22 层：

```yaml
- [15, 1, DehazeHead, [3]]  # 22 Dehaze image from P3/8 feature
- [[15, 18, 21], 1, Detect, [nc]]  # 23 Detect(P3, P4, P5)
```

含义：

- 第 15 层是 YOLO Neck 中的 P3/8 高分辨率特征。
- `DehazeHead` 从 P3 特征恢复一张 3 通道 clean-like 图像。
- `Detect` 仍然接收 P3、P4、P5 三尺度特征，负责目标检测。
- `DehazeHead` 是训练辅助分支，不替代 Detect Head。

当前 `DehazeHead` 是轻量上采样结构：

```text
P3 feature
-> Conv
-> Upsample x2
-> Conv
-> Upsample x2
-> Conv
-> Upsample x2
-> 1x1 Conv
-> Sigmoid
-> dehaze image
```

## 5. Forward 输出行为

训练阶段：

```text
model.training = True
model.return_dehaze = True
输出: (detect_preds, dehaze_img)
```

其中：

- `detect_preds` 交给 YOLO 原始检测损失。
- `dehaze_img` 和 batch 中的 `clean_img` 计算 L1 辅助损失。

验证/预测阶段：

```text
model.eval()
输出: YOLO 原始检测输出
```

注意：

- 正常 `val` / `predict` 不会把去雾图混入检测输出。
- `tools/visualize_dehaze_triplets.py` 为了取出去雾图，会临时设置：

```python
model.eval()
model.training = True
model.return_dehaze = True
```

这只是可视化诊断用法，不代表正式推理流程。

## 6. 预训练权重加载逻辑

推荐训练时使用：

```bash
yolo detect train model=ultralytics/models/v8/yolov8-dehaze.yaml pretrained=yolov8n.pt ...
```

原因：

- 直接从零训练不公平，也不利于收敛。
- 原始 YOLOv8n 的大部分检测特征应该迁移到多任务模型。

已处理的关键问题：

- 原始 YOLOv8n 的 Detect Head 位于 `model.22.*`。
- 当前多任务模型插入了 `DehazeHead`，Detect Head 变为 `model.23.*`。
- 原始同名加载会漏掉 Detect Head 权重。
- 当前 `BaseModel.load()` 已增加重映射逻辑：当检测头索引发生偏移时，将 `model.22.*` 中形状匹配的 Detect 权重迁移到 `model.23.*`。

验证记录：

```text
Transferred 355/375 items from pretrained weights
detect weight changed: True
dehaze layer type: DehazeHead
detect layer type: Detect
```

解释：

- Backbone、Neck、Detect Head 尽量加载 `yolov8n.pt`。
- 新增 `DehazeHead` 无对应预训练参数，保持随机初始化。

## 7. clean 图像数据流

数据集配置中需要能够找到 hazy 图和 clean 图。当前 VOC_hazy 数据已经支持：

```text
batch['img']       -> hazy input
batch['clean_img'] -> clean target
```

训练预处理：

```python
batch['img'] = batch['img'].float() / 255
batch['clean_img'] = batch['clean_img'].float() / 255
```

增强同步：

- mosaic 时同步拼接 `clean_img`。
- mixup 时同步混合 `clean_img`。
- 仿射、翻转、letterbox 时同步变换 `clean_img`。

这样可以保证检测输入图和去雾监督图在空间上对齐。

## 8. dehaze 参数和消融实验

`dehaze` 是去雾辅助损失权重，默认在 `default.yaml` 中：

```yaml
dehaze: 0.05
```

训练损失：

```python
loss[3] = F.l1_loss(dehaze_pred, clean_img) * getattr(self.hyp, 'dehaze', 0.05)
```

重要说明：

- `results.csv` 中的 `train/dehaze_loss` 是已经乘过 `dehaze` 权重后的数值。
- 不同 `dehaze` 权重下的 `train/dehaze_loss` 不能直接横向比较。
- `val/dehaze_loss` 当前为 0，因为 validator 尚未统计 clean target 下的去雾误差。

常用命令：

原始 baseline：

```bash
yolo detect train model=yolov8n.pt data=datasets/VOC_hazy/VOC_hazy.yaml imgsz=640 epochs=50 batch=4 seed=0 workers=8 name=baseline_yolov8n
```

结构消融，有分支但无去雾监督：

```bash
yolo detect train model=ultralytics/models/v8/yolov8-dehaze.yaml pretrained=yolov8n.pt data=datasets/VOC_hazy/VOC_hazy.yaml imgsz=640 epochs=50 batch=4 seed=0 workers=8 dehaze=0 name=ablation_dehaze0_50e
```

当前推荐主实验：

```bash
yolo detect train model=ultralytics/models/v8/yolov8-dehaze.yaml pretrained=yolov8n.pt data=datasets/VOC_hazy/VOC_hazy.yaml imgsz=640 epochs=50 batch=4 seed=0 workers=8 dehaze=0.05
```

较大去雾权重对照：

```bash
yolo detect train model=ultralytics/models/v8/yolov8-dehaze.yaml pretrained=yolov8n.pt data=datasets/VOC_hazy/VOC_hazy.yaml imgsz=640 epochs=50 batch=4 seed=0 workers=8 dehaze=0.10
```

## 9. 已完成实验结果

公平对比应以本机同环境 baseline 为准。组员 A 的其他环境 baseline 可作为外部参考，不建议作为严格对照。

| 组别 | 结果目录 | 设置 | Precision | Recall | mAP50 | mAP50-95 | train/dehaze_loss |
|---|---|---|---:|---:|---:|---:|---:|
| A | `runs/detect/baseline_yolov8n_local_50e` | YOLOv8n baseline | 0.6896 | 0.6024 | 0.6519 | 0.4382 | 0 |
| B | `C:/Users/卢治廷/Desktop/大三下课程/模式识别课设/YOLO v8 dehaze=0 50轮初跑/train11` | YOLOv8-dehaze, `dehaze=0` | 0.6866 | 0.6115 | 0.6588 | 0.4475 | 0 |
| C | `runs/detect/train8` | YOLOv8-dehaze, `dehaze=0.05` | 0.7027 | 0.6141 | 0.6659 | 0.4531 | 0.00662 |
| D | `runs/detect/train9` | YOLOv8-dehaze, `dehaze=0.10` | 0.7081 | 0.6023 | 0.6621 | 0.4490 | 0.01151 |

当前最佳综合指标：

```text
runs/detect/train8
YOLOv8-dehaze + dehaze=0.05
mAP50 = 0.6659
mAP50-95 = 0.4531
```

相对本机 YOLOv8n baseline：

```text
mAP50:    0.6519 -> 0.6659  +0.0140
mAP50-95: 0.4382 -> 0.4531  +0.0149
```

## 10. 当前可以写入报告的结论

可以稳妥表述：

> 在同一套本地修改版 YOLOv8.0.61 环境下，本文重新训练原始 YOLOv8n baseline，并与 YOLOv8-dehaze 多任务模型进行公平对比。实验结果表明，原始 YOLOv8n 在 VOC_hazy 验证集上取得 mAP50=0.6519、mAP50-95=0.4382；加入 DehazeHead 但不使用去雾监督时，mAP50 和 mAP50-95 分别提升至 0.6588 和 0.4475；当去雾辅助损失权重设置为 0.05 时，模型取得最佳综合性能，mAP50=0.6659、mAP50-95=0.4531。相比原始 baseline，mAP50 提升 0.0140，mAP50-95 提升 0.0149。进一步将去雾权重增大到 0.10 后，Precision 略有提升，但 Recall 和 mAP 下降，说明去雾辅助任务与检测主任务之间需要合理权衡，去雾损失权重并非越大越好。

需要避免的表述：

- 不要说“去雾图像质量已经很好”。
- 不要说“模型大幅提升”。
- 不要把组员 A 其他环境 baseline 当成严格公平对照。

更准确的总结：

```text
该框架验证了去雾辅助监督对雾天目标检测有小幅正向作用；
当前更适合作为有效改进原型，而不是最终最优模型。
```

## 11. 去雾三联图可视化

脚本：

```text
tools/visualize_dehaze_triplets.py
```

推荐对当前最佳模型生成三联图：

```bash
python tools/visualize_dehaze_triplets.py ^
  --weights runs/detect/train8_dahaze=0.05/weights/best.pt ^
  --data datasets/VOC_hazy/VOC_hazy.yaml ^
  --split val ^
  --imgsz 640 ^
  --num 12 ^
  --device 0 ^
  --out-dir runs/dehaze_vis/train8_dahaze=0.05
```

已生成目录：

```text
runs/dehaze_vis
runs/dehaze_vis/train8_num30
runs/dehaze_vis/train9
```

报告建议选图方式：

- 选 1 张去雾趋势比较明显的图。
- 选 1 张目标结构保留较好的图。
- 选 1 张存在明显模糊或色偏的失败例。

这样可以客观说明：辅助分支确实在工作，但视觉恢复质量仍有限。

## 12. 已知问题

1. `val/dehaze_loss` 当前为 0  
   验证流程没有统计 clean target 下的去雾误差。如果需要严谨评价去雾质量，建议写离线脚本计算 val 集 L1、PSNR、SSIM。

2. 三联图去雾质量有限  
   当前 `DehazeHead` 很轻量，输出存在模糊、颜色偏移或紫色伪影。它主要作为辅助监督分支，而不是高质量图像复原网络。

3. 提升幅度较小  
   当前最佳 `dehaze=0.05` 相比 baseline 的 mAP50 提升约 1.4 个百分点，mAP50-95 提升约 1.5 个百分点。报告中应写“小幅提升”。

4. 目前主要是单 seed 结果  
   当前核心实验使用 `seed=0`。如果时间允许，可以增加 2 到 3 个 seed 验证稳定性。

5. `dehaze=0` 消融来自 RTX 4080 机器  
   该组结果可用，但 `args.yaml` 中显示的 `model` 是训练后的 `last.pt` 路径。若追求最干净的交付记录，建议在当前本机同环境下重新跑一次命名明确的 `ablation_dehaze0_50e`。

6. 50 epoch 可能尚未完全收敛  
   四组 best mAP 都出现在最后一轮，说明后续可以尝试 100 epoch 或更长训练。

## 13. 后续调参建议

优先级从高到低：

1. 保持框架不变，继续调 `dehaze` 权重  
   建议尝试：

```text
dehaze=0.025
dehaze=0.075
```

2. 延长训练轮数  
   当前 50 epoch 最优点仍在最后一轮，可尝试：

```text
epochs=100
```

3. 补充离线去雾质量指标  
   对 val 集计算：

```text
L1 / PSNR / SSIM
```

用于解释“去雾质量”和“检测性能”不完全一致。

4. 重复 seed  
   建议至少补：

```text
seed=1
seed=2
```

5. 低成本结构改进  
   如果上述都完成后还有时间，再考虑：

- DehazeHead 加 skip connection。
- 融合 P2/P3 多尺度特征。
- 减弱去雾分支对检测特征的梯度干扰。

暂时不建议一上来大改 Backbone 或 Detect Head。当前框架已经能支撑课程设计主结论，大改结构会增加不可控风险。

## 14. 交付检查清单

后续同学继续实验前，建议先确认：

- `yolo detect train model=yolov8n.pt ...` baseline 能正常跑。
- `yolo detect train model=ultralytics/models/v8/yolov8-dehaze.yaml pretrained=yolov8n.pt ...` 能显示预训练权重迁移。
- `results.csv` 中包含 `train/dehaze_loss`。
- `dehaze=0` 时 `train/dehaze_loss=0`。
- `dehaze=0.05` 时 `train/dehaze_loss` 非 0。
- `val/predict` 能正常输出检测结果。
- `tools/visualize_dehaze_triplets.py` 能输出三联图。

只要以上项目成立，说明方案二框架本身是可靠的，后续可以主要围绕调参和实验分析展开。

## 15. 2026-05-20 V2 特征融合实验更新

### 15.1 更新背景

在完成 `baseline_yolov8n_100e` 与 V1 `dehaze0.05_100e` 后，V1 结果显示：仅将 `DehazeHead` 作为辅助去雾输出时，检测性能与 baseline 基本持平，且推理速度明显变慢。进一步检查模型结构可知，V1 中 `Detect` 仍然接收原始 `[P3, P4, P5]`，去雾分支主要通过 L1 去雾损失参与训练，并没有把去雾增强特征显式反馈给检测头。

因此，在原方案二基础上实现 V2 最小结构升级：保留去雾图像辅助输出，同时新增 P3 去雾特征融合模块，使检测头接收融合后的 `P3_fused`。

### 15.2 V2 结构改动

新增配置文件：

```text
ultralytics/models/v8/yolov8-dehaze-v2.yaml
```

V2 的关键结构为：

```yaml
- [15, 1, DehazeFeatureFuse, [3, True, 0.1]]  # 22 fused P3 + dehaze image
- [[22, 18, 21], 1, Detect, [nc]]             # 23 Detect(P3_fused, P4, P5)
```

数据流为：

```text
P3 -> DehazeFeatureFuse -> P3_fused + dehaze_img
Detect([P3_fused, P4, P5])
dehaze_img 与 clean_img 继续计算 L1 去雾辅助损失
```

其中 `DehazeFeatureFuse` 内部采用轻量门控残差融合：

```text
P3_fused = P3 + alpha * sigmoid(gate(dehaze_feat)) * dehaze_feat
```

新增参数：

```text
dehaze_fuse=True/False
dehaze_fuse_alpha=0.1
```

说明：训练开始打印的模型结构表来自 YAML 初始参数，因此即使命令行设置 `dehaze_fuse=False`，结构表中仍可能显示 `DehazeFeatureFuse [64, 3, True, 0.1]`。实际训练时，`ultralytics/yolo/v8/detect/train.py` 会在 `set_model_attributes()` 中将命令行参数写入模块内部。

已通过加载 checkpoint 验证：

```text
dehaze_v2_nofuse_100e: DehazeFeatureFuse.fuse = False
dehaze_v2_fuse0.1_100e: DehazeFeatureFuse.fuse = True
Detect from = [22, 18, 21]
```

因此，`dehaze_v2_nofuse_100e` 的 nofuse 消融是有效的。

### 15.3 V2 100 epoch 实验设置

两组 V2 实验均使用：

```text
model=ultralytics/models/v8/yolov8-dehaze-v2.yaml
pretrained=yolov8n.pt
data=datasets/VOC_hazy/VOC_hazy.yaml
imgsz=640
epochs=100
batch=4
device=0
workers=0
seed=0
optimizer=SGD
dehaze=0.05
```

区别为：

```text
dehaze_v2_nofuse_100e: dehaze_fuse=False, dehaze_fuse_alpha=0.1
dehaze_v2_fuse0.1_100e: dehaze_fuse=True, dehaze_fuse_alpha=0.1
```

结果目录：

```text
runs/detect/dehaze_v2_nofuse_100e
runs/detect/dehaze_v2_fuse0.1_100e
```

### 15.4 100 epoch 结果对比

以下结果以同一代码工程、同一 VOC_hazy 数据集、同一 seed=0 的本机实验为主。baseline、V1、V2 均取对应 `results.csv` / `best.pt` 记录中的验证结果。

| 实验 | 结果目录 | Precision | Recall | mAP50 | mAP50-95 | 备注 |
|---|---|---:|---:|---:|---:|---|
| YOLOv8n baseline 100e | `runs/detect/baseline_yolov8n_100e` | 0.700 | 0.625 | 0.668 | 0.455 | 原始检测基线 |
| V1 dehaze0.05 100e | `runs/detect/dehaze0.05_100e` | 0.705 | 0.619 | 0.669 | 0.450 | 去雾辅助头，不融合特征 |
| V2 nofuse 100e | `runs/detect/dehaze_v2_nofuse_100e` | 0.724 | 0.611 | 0.671 | 0.458 | 使用 V2 模块，但关闭特征残差融合 |
| V2 fuse0.1 100e | `runs/detect/dehaze_v2_fuse0.1_100e` | 0.725 | 0.611 | 0.680 | 0.463 | 当前最佳结果 |

从完整 `results.csv` 读取到的最佳 epoch 信息如下：

| 实验 | best mAP50 epoch | best mAP50 | best mAP50-95 epoch | best mAP50-95 |
|---|---:|---:|---:|---:|
| baseline 100e | 96 | 0.66757 | 96 | 0.45503 |
| V1 dehaze0.05 100e | 97 | 0.66874 | 81 | 0.44996 |
| V2 nofuse 100e | 89 | 0.67523 | 97 | 0.45828 |
| V2 fuse0.1 100e | 99 | 0.67992 | 98 | 0.46301 |

### 15.5 实验结论

1. V2 方向有效。  
   `dehaze_v2_fuse0.1_100e` 相比 `baseline_yolov8n_100e`，mAP50 从 0.668 提升到 0.680，mAP50-95 从 0.455 提升到 0.463，说明去雾增强特征参与检测头输入后，对雾天目标检测有正向作用。

2. 特征融合本身有增益。  
   `dehaze_v2_fuse0.1_100e` 相比 `dehaze_v2_nofuse_100e`，mAP50 提升约 0.009，mAP50-95 提升约 0.005。由于已确认 nofuse 的 `fuse=False` 生效，因此该差异可以作为 P3 去雾特征融合有效的消融证据。

3. V2 nofuse 也优于 V1。  
   即使关闭残差融合，V2 nofuse 的 mAP50-95 仍高于 V1 `dehaze0.05_100e`。这说明新的 `DehazeFeatureFuse` 分支形式和训练行为比原 V1 `DehazeHead` 更稳定，但真正打开融合后性能进一步提升。

4. 当前主要短板是 Recall。  
   V2 fuse0.1 的 Precision 高于 baseline，但 Recall 从 0.625 降到 0.611，说明模型更偏向高置信预测和定位质量，漏检略有增加。后续调参应重点观察 Recall，而不是只看 mAP。

5. V2 fuse0.1 可作为当前方案二主结果。  
   当前最稳妥的报告表述是：V2 将去雾分支从单纯图像辅助监督升级为检测特征融合模块，实验结果表明 P3 去雾增强特征对检测性能具有小幅但稳定的正向作用。

### 15.6 后续实验建议

优先做小消融，不建议立刻大改 Backbone、Detect Head 或加入复杂注意力机制。

建议下一步：

```text
1. 继续测试 dehaze_fuse_alpha=0.05
2. 继续测试 dehaze_fuse_alpha=0.2
3. 可选测试 dehaze=0.025, dehaze_fuse_alpha=0.1
4. 补充 V2 三联图可视化
5. 补充 val 集去雾质量统计：L1 / PSNR / SSIM
```

推荐命令：

```bash
yolo detect train model=ultralytics/models/v8/yolov8-dehaze-v2.yaml pretrained=yolov8n.pt data=datasets/VOC_hazy/VOC_hazy.yaml imgsz=640 epochs=100 batch=4 device=0 workers=0 seed=0 optimizer=SGD dehaze=0.05 dehaze_fuse=True dehaze_fuse_alpha=0.05 name=dehaze_v2_fuse0.05_100e
```

```bash
yolo detect train model=ultralytics/models/v8/yolov8-dehaze-v2.yaml pretrained=yolov8n.pt data=datasets/VOC_hazy/VOC_hazy.yaml imgsz=640 epochs=100 batch=4 device=0 workers=0 seed=0 optimizer=SGD dehaze=0.05 dehaze_fuse=True dehaze_fuse_alpha=0.2 name=dehaze_v2_fuse0.2_100e
```

三联图命令：

```bash
python tools/visualize_dehaze_triplets.py ^
  --weights runs/detect/dehaze_v2_fuse0.1_100e/weights/best.pt ^
  --data datasets/VOC_hazy/VOC_hazy.yaml ^
  --split val ^
  --imgsz 640 ^
  --num 30 ^
  --device 0 ^
  --out-dir runs/dehaze_vis/dehaze_v2_fuse0.1_100e
```

### 15.7 可写入报告的阶段性结论

> 在 V1 多任务去雾辅助检测模型中，去雾分支仅通过 L1 图像恢复损失参与训练，Detect Head 仍然使用原始 P3/P4/P5 特征。为增强去雾任务对检测任务的直接作用，本文进一步设计了 V2 结构，在 P3 层引入 `DehazeFeatureFuse` 模块，将去雾增强特征以门控残差形式融合回检测特征，并令 Detect Head 接收 `P3_fused/P4/P5`。实验结果显示，V2 fuse0.1 在 VOC_hazy 验证集上取得 Precision=0.725、Recall=0.611、mAP50=0.680、mAP50-95=0.463，相比原始 YOLOv8n baseline 和 V1 dehaze0.05 均有提升，说明去雾增强特征参与检测分支对雾天目标检测具有正向作用。但该结构的 Recall 低于 baseline，表明后续仍需通过融合强度和去雾损失权重调节 Precision 与 Recall 的平衡。
