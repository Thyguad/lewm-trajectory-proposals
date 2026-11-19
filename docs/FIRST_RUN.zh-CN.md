# 首次运行

本项目已经包含实验代码，不需要再让 agent 从方案重新实现。先完成数据与环境检查，再运行一个训练种子。没有必要组多卡。

## 1. 配置输入

安装见 README。复制 `configs/cube.yaml` 到 `configs/local.yaml`，填写：

```yaml
dataset: /absolute/path/cube_single_expert.h5
weights: /absolute/path/lewm-cube/weights.pt
device: cuda:0
```

`weights.pt` 同目录必须有官方 `config.json`。保持 `model` 的 192/5/5/5 设置。服务器上通常把 `eval.render_backend` 改为 `egl`；这不是训练必需，只影响环境渲染。

```bash
lewm-proposals doctor --config configs/local.yaml --load-model
lewm-proposals inspect --config configs/local.yaml
```

`doctor` 只确认依赖、输入和可选权重加载；不代表模型已经训练或任务成功。`inspect` 会实际读取轨迹元数据和少量样本，用于确认数据规模与 schema。

## 2. 划分和缓存

```bash
lewm-proposals split --config configs/local.yaml
lewm-proposals encode --config configs/local.yaml
```

`split` 保存固定轨迹编号，默认 2000/200/200。不复制 RGB 大文件。已有划分文件时不会覆盖；正式比较必须复用同一份划分。若要沿用先前手工划分，需要先明确相应 episode 编号，不要假定一个新随机种子与旧测试集天然隔离。

`encode` 只为选中的轨迹缓存表征，按 episode 提交进度。掉线后用相同命令加 `--resume`。权重或数据指纹不同会拒绝混用缓存。缓存完成之前不能训练。

## 3. 先测时间，再完成训练

默认配置为 batch 256、10000 次更新、AdamW 3e-4。可以先暂停在 200 次更新：

```bash
lewm-proposals train --config configs/local.yaml --output runs/train-seed42 --stop-after 200
```

查看 `train.jsonl` 的 elapsed_seconds 和 `status.json` 的 peak_gpu_bytes。预估剩余耗时应区分训练、验证和第一次初始化；若显存不足，减小 batch 后用新的运行目录启动。修改训练配置后不能用旧目录执行 exact resume。

配置不变时继续：

```bash
lewm-proposals train --config configs/local.yaml --output runs/train-seed42 --resume
```

训练过程中只用新生成器，不需要渲染环境或反复运行 CEM。`best.pt` 按固定验证 flow loss 保存；`last.pt` 含优化器和随机数状态，用于恢复。

## 4. 开发集闭环比较

默认 `eval.split: val`，32 个案例，3 个规划种子，3 种方法，共 288 个回合、最多 14400 个实际环境步。若先检查渲染，可将 `eval.cases` 改成 1 并另选输出目录；这种小运行只用于排错。

```bash
lewm-proposals evaluate --config configs/local.yaml \
  --checkpoint runs/train-seed42/best.pt --output runs/val-seed42
```

评估逐回合写入日志；中断后加 `--resume`，已完成回合不会重复。输入权重、案例或配置变动会拒绝恢复。观察 `summary.md` 的成功率和平均/P95 决策耗时，以及原始日志中的初始成功比例、最终距离、生成与 predictor 的一致性。

任何新机制或超参数选择都在 val 完成。需要正式评估时，复制配置，改为 `eval.split: test`、设置最终案例数，使用新输出目录。不同方法必须共享同一固定案例集。

## 5. 独立训练种子

需要估计训练方差时，把 `train.seed` 改成 43、44，分别训练到新的目录。`eval.planner_seeds` 只控制生成或 CEM 的采样，不能代替多个训练种子。

这套流程需要你已有的官方数据和单卡资源。本仓库没有启动远程任务、租卡或完整 GPU 训练；本地工程验证的边界见 VALIDATION.md。
