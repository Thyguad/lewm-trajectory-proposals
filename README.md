# LeWM Trajectory Proposals

独立验证一个第一阶段问题：**冻结 LeWM 后，学习生成候选动作，再由原 predictor 筛选，能否在 Cube 上改善成功率或规划速度？**

本项目包含可执行的数据准备、训练、MPC 评估和汇总代码。研究结果尚未获得；工程检查不代表方法有效。

## 方法与范围

固定当前图像与目标图像，通过冻结的 LeWM 编码器得到 192 维表征。新的联合流模型同时生成 5 个未来状态和 5 个动作块；每个动作块含 5 个不同的五维环境动作，合计 25 步。只有生成器参与训练。

| 方法 | 决策方式 |
|---|---|
| `cem` | 原研究分支的 CEM：300 候选 × 10 轮，30 精英，执行最终分布均值 |
| `flow_direct` | 同一生成器产生一条轨迹，直接执行其动作 |
| `flow_ranked` | 生成 16 条轨迹，由冻结 predictor rollout 后按终点目标距离选一条 |

所有方法执行前 5 个环境步后重新观察，最多执行 50 步。目标取同一原始轨迹未来 25 步的图像。成功定义为最终方块位置距目标不超过 0.04 m；另记曾经达标率。**不检查朝向，不代表完整 OGBench 抓取放置基准。**

生成器是 4 层、宽度 256、8 头、前馈宽度 1024 的 Transformer，实际参数数目为 **3,354,841**。状态和动作分别投影；当前状态、目标及 flow time 作为条件；Euler 生成默认 8 步。状态、动作的 flow matching 损失分别按维度取平均。新生成的状态只用于一致性诊断，筛选必须依据 predictor 对实际待执行动作的预测。

该验证与已有联合状态—动作生成方法接近，**不把这个基础组合宣称为新算法，也不声称复现 WATO 的优化微调**。本项目不包含跨尺度组合、自进化、逆动力学多任务训练或 predictor 微调。

## 安装

推荐 Python 3.11 或 3.12。先为实际硬件安装匹配的 PyTorch，再安装本项目；数据准备和缓存后的生成器训练无需 MuJoCo。

```bash
python -m venv .venv
source .venv/bin/activate

# 仅 CPU 检查可用这一行；CUDA 机器按 PyTorch 官方说明安装对应构建。
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

pip install -e '.[cube,dev,download]'
python -m lewm_proposals --help
```

CPU/CUDA/MPS 都有模型运行路径；CUDA 与 MPS 的实际训练尚需在目标机器验证。单卡即可；代码不启动多卡训练或付费资源。渲染后端独立于模型设备：Linux GPU 常用 `egl`，无显示且需要软件渲染可选择 `osmesa`（需系统库），macOS 通常用 GLFW，按 MuJoCo 要求使用对应启动方式。

## 数据和权重

- 数据：[quentinll/lewm-cube dataset](https://huggingface.co/datasets/quentinll/lewm-cube)，文件 `cube_single_expert.tar.zst`，约 46.2 GB 压缩包。
- 权重：[quentinll/lewm-cube model](https://huggingface.co/quentinll/lewm-cube)，需要同目录的 `weights.pt` 与 `config.json`。
- 已有文件可以直接使用，不复制旧研究仓库，不把数据或权重加入 Git。

所有实验命令只读本地输入，**不会自动下载**。下载是单独的显式操作：

```bash
hf download quentinll/lewm-cube weights.pt config.json --local-dir weights/lewm-cube
hf download quentinll/lewm-cube cube_single_expert.tar.zst --repo-type dataset --local-dir data/downloads
mkdir -p data/cube
tar --zstd -xf data/downloads/cube_single_expert.tar.zst -C data/cube
```

解压后定位 `cube_single_expert.h5`，在配置中填写其实际路径；保留压缩包时需要同时容纳压缩包和解压后的数据。不要在下载前租用闲置 GPU 等待传输。

## 从数据到比较结果

从仓库根目录执行。先复制 `configs/cube.yaml` 为 `configs/local.yaml`，填写绝对 `dataset`、`weights` 路径，并选择 `device` 和 `eval.render_backend`。相对输出路径从当前工作目录解析。

```bash
cp configs/cube.yaml configs/local.yaml
# 编辑 configs/local.yaml 后：
lewm-proposals doctor --config configs/local.yaml --load-model
lewm-proposals inspect --config configs/local.yaml
lewm-proposals split --config configs/local.yaml
lewm-proposals encode --config configs/local.yaml
lewm-proposals train --config configs/local.yaml --output runs/train-seed42
lewm-proposals evaluate --config configs/local.yaml \
  --checkpoint runs/train-seed42/best.pt --output runs/val-seed42
```

首次运行的详细说明见 [FIRST_RUN.zh-CN.md](docs/FIRST_RUN.zh-CN.md)。实验约定见 [PROTOCOL.md](docs/PROTOCOL.md)。

默认从符合长度要求的源轨迹中选择训练 2,000 条、验证 200 条、测试 200 条，其余不使用。可以在配置的 `split.exclude_episodes` 中排除过去手动调参用过的轨迹。划分文件持久保存，数据量不足会明确报错，不会暗中复用验证或测试数据。

缓存包含所选三组轨迹的表征与动作。生成器的表征标准化只拟合训练组；冻结 predictor 的动作标准化沿用原研究代码对**完整原始预训练数据**计算的 StandardScaler，并对所有比较方法固定。测试轨迹仅对新生成器隔离，不能宣称预训练 LeWM 没见过这些轨迹。

中断恢复：

```bash
lewm-proposals encode --config configs/local.yaml --resume
lewm-proposals train --config configs/local.yaml --output runs/train-seed42 --resume
lewm-proposals evaluate --config configs/local.yaml \
  --checkpoint runs/train-seed42/best.pt --output runs/val-seed42 --resume
```

恢复训练要求模型、数据缓存、训练配置及设备类型一致。`--stop-after 200` 可以在保留原定学习率日程的前提下暂停一次训练，用于先测吞吐。更改总训练步数应新建运行目录，不在旧运行上暗中改变日程。

## 输出与解释

- `artifacts/cube-split.json`：完整轨迹划分、固定动作统计、源数据抽样指纹。
- `artifacts/cube-latents.h5`：可恢复的 FP16 表征缓存；未完成缓存不能训练。
- `runs/train-*/`：完整配置、依赖版本、训练/验证日志、`best.pt`、`last.pt`、状态。
- `runs/val-*/episodes.jsonl`：逐案例动作、距离、规划调用数与延迟。
- `summary.json` / `summary.md`：终点成功率、曾经达标率、距离、平均/P95 决策延迟与配对差值。

优先观察成功率—耗时的关系。当前 CEM 与生成式方法的预算不同，不把更少候选下的速度收益解释为等计算量的质量优势。报告记录 predictor 候选转移数和实际调用数，也记录生成器调用数。后续等时间预算比较应单独声明配置。

`planner_seeds` 控制候选采样，不是独立训练种子。需要正式结论时，分别改变 `train.seed` 新建训练运行，并在固定的最终测试案例上比较。不要从 32 个开发案例或训练损失推导论文结论。

## 检查

```bash
ruff check src tests
pytest -q
# 使用真实官方权重的可选数值一致性检查：
LEWM_TEST_WEIGHTS=/absolute/path/weights.pt pytest -q tests/test_official_checkpoint.py
```

工程验证的具体记录见 [VALIDATION.md](docs/VALIDATION.md)。未提供权重时，官方权重检查会明确跳过。

## 来源

模型核心及 Cube 环境来自已检查的 `Thyguad/lewm-research` 提交 `67cf53f26b96033b53b37eb04c89b82c3c88b933` 中的 LeWM / stable-worldmodel 代码，保留在 `vendor/`，不依赖旧仓库的工作区或历史实验。只修改必要的包内导入及最小工具依赖。来源、修改和许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

- [LeWorldModel](https://arxiv.org/abs/2603.19312)
- [Joint World Action Models through Differentiable Trajectory Optimization](https://arxiv.org/abs/2611.17626) — 相关方法与架构规模参照；本项目未实现其 WATO 微调。
