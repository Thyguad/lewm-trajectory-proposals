# 工程验证记录

检查日期：2026-11-19。以下记录只证明实现和接口的检查范围，**不提供研究效果结论**。

## 已执行

1. `ruff check src tests scripts`：通过。
2. 单元与数值检查：**8 passed**，其中包含真实官方 checkpoint 的 rollout 数值对齐。覆盖：
   - 完整轨迹划分不重叠，未来状态与 25 个动作的时间索引正确；
   - 不完整缓存被拒绝，不同身份的缓存不能混合恢复；
   - G 的表征缩放只使用 train 数据；
   - 状态和动作输出头均接收到学习梯度；
   - 同一噪声种子下 K=1 与 K>1 的候选 0 对齐；
   - CPU 中断恢复后的参数与不中断训练逐项一致；
   - 筛选使用冻结 predictor 的 rollout，而非 G 自报的未来状态；
   - 动作归一化、裁剪和 CEM 候选转移预算；
   - 官方权重严格加载、冻结、rollout 与保留的上游 `LeWM.rollout` 数值一致。
3. 官方 Cube checkpoint：编码输出 `(1,192)`，两条候选的预测输出 `(2,5,192)`；所有 LeWM 参数不求梯度。
4. Cube 环境：通过 EGL 渲染得到 `uint8 (224,224,3)`；5 维动作执行成功。
5. `scripts/check_pipeline.py`：实际执行 `doctor → inspect → split → encode → train → evaluate`，三种方法各完成 50 个环境步，生成完整日志和汇总。

## 联通检查的实际规模

联通检查使用新采集的 **3 条、每条 30 步的脚本动作轨迹**，训练/验证/测试各 1 条；G 使用完整 3,354,841 参数架构，但只训练 **2 次更新、batch=2**。评估为 1 个验证案例、1 个规划种子；CEM 4 候选/1 轮，flow 2 个生成步骤，ranked 2 候选。它不是默认研究配置。

这个极小 fixture 的方块几乎没有位移，评估起点已经满足位置阈值。因此其成功率没有研究意义，不能当作三种方法性能相当、或新方法有效的证据。它的目的只是让真实权重、真实 MuJoCo 环境以及所有命令入口连接起来。

## 实际环境

| 项目 | 验证值 |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.5.1+cpu |
| NumPy | 2.5.5 |
| Transformers | 4.57.1 |
| h5py / hdf5plugin | 3.16.0 / 7.1.0 |
| MuJoCo / dm-control / OGBench | 3.3.6 / 1.0.34 / 1.2.0 |
| 模型计算 | CPU |
| 环境渲染 | EGL |
| 新生成器参数 | 3,354,841 |
| 所加载官方 LeWM 总参数 | 18,034,628，全部冻结 |

官方权重 SHA-256：`2839a907362f403f9136383016e91774373a295d958ae75121791f22a9fddf89`。

对应 config.json SHA-256：`4d446944fe28922cc2c5763f43d4ef9132a457bd89e9a0ce5dbceac183994999`。

## 尚未执行

- 完整官方 Cube HDF5 的 2,000 条轨迹训练。
- CUDA/MPS 的训练、显存与吞吐验证。
- 默认 CEM / 16 候选生成器配置下的正式闭环比较。
- 多个独立训练种子和最终测试集评估。

因此没有填入预期成功率、速度倍数或显卡小时实测。执行这些实验时，应保存真实输出，按 PROTOCOL.md 的边界解释结果。

## 重跑联通检查

在安装 `.[cube,dev]` 并准备官方权重后：

```bash
python scripts/check_pipeline.py --weights /absolute/path/weights.pt \
  --output ccfa-workfiles/checks/pipeline --render-backend egl
```

输出目录必须不存在，避免覆盖此前证据。测试只创建很小的本地 fixture，不下载官方数据，不启动远程任务。
