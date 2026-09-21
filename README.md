# WTLRFM-SoS-WFC：乘性流匹配声速图预测 + WFC 波束形成

从 [OpenBreastUS 多角度平面波 IQ 数据集](../../../../data/zhuangyang/openbreast_pw_iq)
的**原始复数 IQ（解析 RF）**预测 **2D 声速图 (SoS map)**，再把预测声速图送进
[WFC 波束形成器](../wfc_dbua_pw) 重建 B-mode 图像。

模型改造自 `/data/zhuangyang/WTLRFM` 的**乘性流匹配**网络
（`ComplexMagPhaseResidualFlowNetwork`，即 `docs/additive_vs_multiplicative_flow.md`
中的"乘性流"）。

```
raw complex IQ [128 阵元, 13 角, 1300 样点]   ← 相位完整保留
        │
        ├─ DAS 相位保持条件（3 组：多速全孔径 / 逐角度 / 子孔径）→ cond [40,128,160]
        │
        ▼
乘性流匹配 v_θ(cond, u_t, t)  ──Euler ODE──►  u_1 = log(c/1500)/0.05
        │                                      c(x,z) = 1500 · exp(0.05 · u_1)
        ▼                                      [128×160 @ 0.3 mm]
WFC 角谱 SSFM 波束形成  ──►  B-mode 图像 [1024×182 @ 0.25 mm]
```

## 1. 相对 WTLRFM 原乘性流改了什么

| | WTLRFM 原乘性流 | 本改造 |
|---|---|---|
| 目标量 | HR 复数 IQ 场 `Z_H` | 声速图 `c(x,z)`（实数、正定） |
| 乘性分解 | `Z_H = M_S · exp(ρ) · exp(i·dφ)` | `c = 1500 · exp(0.05·u)` |
| 冻结结构 `M_S` | 独立训练的包络回归 baseline，**冻结** | **已删除**（见下） |
| 流变量 | 对数幅度残差 `ρ` + 圆上相位残差 `dφ` | 归一化对数声速 `u`（标量场无相位支路） |
| 路径 | 欧氏线性 + S¹ 测地 | 欧氏线性 `u_t=(1-t)u_0+t u_1`，`u_0~N(0,σ_u²)` |
| 损失 | 速度匹配 MSE（仅此项） | 速度匹配 MSE（仅此项） |
| 采样 | Euler ODE + 速度/幅值 clamp | Euler ODE（默认 20 步）+ 速度/`u` clamp |
| 骨干 | WTLR-UNet（多级 Haar DWT + 门控融合） | **同一 WTLR-UNet**（`wtlrfm/` 逐字拷贝） |

**为什么删掉冻结 baseline**：原网络里 `M_S` 是**必需的**——它只预测 `ρ`、`dφ`
两个速度场，本身不产生幅度尺度，必须由外部包络回归网络提供 `M_S`，再乘性合成。
而声速图是**正的实标量**，指数映射本身就是完备参数化：
`c = C_REF · exp(s·u)`。再挂一个冻结网络只会多一个训练阶段、多一个 checkpoint、
多一个偏差来源，没有任何必要。**乘性语义保留在最关键的地方**：流在**对数
（乘性）域**上演化，输出与参考声速**乘性合成**。

其余 WTLRFM 机制照搬：`σ_u` 由 batch 残差 std 的 EMA 自动估计（前 200 batch）
后冻结、作为 buffer 同步进 EMA 副本（260702 修复）；只做速度匹配损失；
Euler ODE 采样带速度/状态 clamp；多样本采样取均值=点估计、std=不确定度。

## 2. 相位信息（重点）

数据是基带复数 IQ，即**解析 RF 信号**，相位信息完整。条件张量由三组
**相位保持**的 DAS 复图像构成（对原始 IQ 的线性泛函，无包络检波）：

| 组 | 内容 | 通道 | 物理含义 |
|---|---|---|---|
| `full` | 全孔径 DAS，假设声速 1450/1500/1550 m/s | 6 | 假设声速偏离真值时整体相干能量下降（全局失配） |
| `angle` | 逐发射角 DAS（全接收孔径，1500 m/s） | 26 | 发射腿 `(x sinθ + z cosθ)/c` 的**角度相关相位** |
| `subap` | 4 个接收子孔径 DAS（1500 m/s） | 8 | 接收孔径上的**到达时间斜率**（局部畸变） |

合计 **40 通道**。两个关键实现细节：

1. **相位严格保留**：幅度用 `asinh(|z|/s)/asinh(3)` 压缩到 ~[0,1]，相位原样保留，
   按 `(m·cosφ, m·sinφ)` 输出。早期版本对 I、Q 分别做 asinh，会**扭曲复数相位**
   ——已修复。
2. **组内共用归一化尺度**：同组内各通道除以同一个标量，因此**通道间的相对幅度/
   相位关系不被破坏**（这正是 SoS 信息所在）；不同组用各自的尺度。

`scripts/check_cache.py` 可出图核验各通道（目标图 / 各速幅度 / 逐角相位 /
子孔径相位差）。

## 3. 数据与网格

| | 范围 | 采样 | 尺寸 |
|---|---|---|---|
| x（横向，相对探头中心） | −19.05 … +19.05 mm（= 38.1 mm 孔径） | 0.30 mm | 128 |
| z（深度，相对探头面） | 0.15 … 47.85 mm（覆盖 3–45 mm 组织 ROI） | 0.30 mm | 160 |

- 128 的横向尺寸是必须的：WTLR 编码器按特征图**宽度**索引小波特征
  （3 级 → 64/32/16），与 UNet 解码器宽度对齐；纵向可任意。
- 目标：`c_true`（仿真网格 600×600 @ 0.1 mm）双线性插值到上表网格；
  归一化 `u = log(c/1500)/0.05`。
- 800 个样本按固定种子 640/80/80 划分（`cache/splits.json`），
  每个样本是独立 phantom，无泄漏；训练集做左右镜像增强。

## 4. 目录结构

```
wtlrfm_sos_wfc/
├── configs/sos_flow.json      条件分组 + WTLR-UNet + 流/训练超参
├── wtlrfm/                    WTLRFM 网络模块（逐字拷贝）
│   ├── wtlr_encoder.py            多级 Haar DWT + 子带编码 + 通道注意力 + 门控融合
│   └── sr3_modules/wtlr_unet.py   WTLR-UNet（+ speckle/kan/liif 依赖）
├── data/
│   ├── geometry.py            网格 / 参考声速 / 归一化 / ROI（全流程唯一来源）
│   ├── das.py                 相位保持 DAS（三组条件）+ 目标插值
│   └── obpw_dataset.py        torch Dataset（读 cache，左右镜像增强）
├── models/sos_mult_flow.py    乘性流匹配（本改造的核心）
├── engine.py                  轻量训练工具（config/EMA/指标/日志/checkpoint）
├── train.py                   训练（单阶段）
├── predict.py                 推理：c_pred/c_std → npz + 面板图
├── wfc_integration/beamform.py  JAX：预测声速图 → WFC 成像（const/pred/true 对比）
└── scripts/
    ├── prepare_cache.py       并行预处理（DAS 三组条件 + 目标插值）
    ├── check_cache.py         条件/目标出图核验
    └── run_end_to_end.sh      一键串起 4 个阶段（跨两个 conda 环境）
```

## 5. 环境

| 阶段 | 环境 | 说明 |
|---|---|---|
| 预处理 / 训练 / 推理 | `py310`（`/home/zhuangyang/miniconda3/envs/py310/bin/python`） | torch 2.6 + pywt |
| WFC 波束形成 | `dbua`（`/home/zhuangyang/miniconda3/envs/dbua/bin/python`） | jax 0.4.30 + optax |

`scripts/run_end_to_end.sh` 用 `PY_TORCH` / `PY_JAX` 环境变量指定解释器。

## 6. 用法

```bash
cd wtlrfm_sos_wfc

# 0) 预处理（36 进程约 20 分钟；单样本 ~11 s CPU）
/home/zhuangyang/miniconda3/envs/py310/bin/python scripts/prepare_cache.py \
    --samples /data/zhuangyang/openbreast_pw_iq/dataset/samples \
    --cache cache --workers 36 --speeds 1450,1500,1550 --n-subap 4

# 0b) 条件核验出图
/home/zhuangyang/miniconda3/envs/py310/bin/python scripts/check_cache.py \
    --cache cache --n 3 --out cache/check.png

# 1) 训练乘性流匹配
/home/zhuangyang/miniconda3/envs/py310/bin/python train.py \
    --config configs/sos_flow.json --cache cache --out out/flow --epochs 300

# 2) 测试集推理（集成均值 + std）
/home/zhuangyang/miniconda3/envs/py310/bin/python predict.py \
    --ckpt out/flow/best.pth --cache cache --split test --out out/pred_flow \
    --n-samples 8 --ode-steps 20

# 3) 用预测声速图做 WFC 波束形成（JAX 环境）
/home/zhuangyang/miniconda3/envs/dbua/bin/python wfc_integration/beamform.py \
    --samples /data/zhuangyang/openbreast_pw_iq/dataset/samples \
    --cmaps out/pred_flow/cmaps --out out/wfc --limit 8
```

或一条命令：`./scripts/run_end_to_end.sh`（`SKIP_CACHE=1 SKIP_TRAIN=1` 可跳过前两步）。

## 7. 训练细节

- 损失：只做速度匹配 `MSE(v_θ, u_1 − u_0)`；Adam lr 5e-5 + cosine，EMA(0.999)，
  梯度裁剪 1.0；每 5 epoch 用 EMA 权重采样 4 次 × 10 步做验证，
  按 val ROI MAE 选最优。
- 指标（`engine.map_metrics`）：全图 MAE、ROI(3–45 mm) MAE、
  ROI 平均声速误差 `|mean_roi(pred) − mean_roi(gt)|`、ROI 内相关系数。
- 参考基线：常数 1500、以及**oracle 常数**（测试集均值，乐观下界）。

## 8. WFC 接入与图像质量核验

`wfc_integration/beamform.py` 直接 `import` 同级 `../wfc_dbua_pw/wfc.py`
（单一物理引擎实现），每个样本输出 3 个声速假设：

| 假设 | 含义 |
|---|---|
| `const` | 均匀 1500 m/s（水） |
| `pred` | **本流水线输出**的 2D 声速图 |
| `true` | 仿真真值图（上界参考） |

每个假设同时保存：`coh`（相干复合 `|Σ_a I_a|`，推荐）、`inc`（非相干
`Σ_a |I_a|`，引擎 `image()` 的默认）、`single`（单角度诊断图）。

**关于"WFC 图像看起来差"的核查结论**（重要）：

1. **引擎与延迟约定正确**：`wfc_integration/verify_psf.py` 用本数据集的精确格式
   （同延迟模型、128 阵元 × 13 角、20 MHz/5 MHz）合成单点散射体，WFC 成像
   峰值与 DAS **完全一致**（4.96/30.00 mm，偏移 0.000 mm，−6 dB 宽度相同）。
   即实现无 bug。
2. **复合方式影响很大**：引擎默认 `image()` 是**非相干**复合（13 个角度幅值
   相加），   在非均匀介质中会糊化。同一批 8 个测试样本、同一 const 假设下
   平均 ipr：相干 3.43 vs 非相干 2.41。集成脚本因此**默认展示相干复合**
   （`*_single.png` 另给单角度诊断图）。
3. **数据本身以镜面反射为主**：数据集 README 明确说明 phantom 是大尺度平滑
   结构（0.25 mm 网格插值），"图像以镜面反射为主"，缺少真实乳腺的亚分辨率
   弥散散斑。因此 WFC 图像呈现的是**镜面结构（皮肤线、层界面）**，看起来
   不像带散斑的常规 B-mode；单角度 WFC 图比 DAS 更干净（孔径外无杂波）。
4. **多次散射/密度对比未建模**：仿真含全波多次散射与密度变化
   （ρ=1000+0.7(c−1500)），而 WFC 是单次散射、只含声速的角谱模型，
   强层界面的混响会在图中形成纵向条纹。这是模型失配，不是实现 bug。

`summary.csv` 给出每个假设的 `coh/inc/single` 的 `ipr`（能量集中度）、
锐度、散斑 SNR，以及 pred 图与 true 图的相关性，用于定量比较。

## 9. 已知局限

1. **物理尺度为推断值**：OpenBreastUS 未给网格间距，0.25 mm 由官方 500 kHz
   样本反推（见数据集 README）；若真实间距不同，声速标签绝对尺度需换算。
2. **2D 仿真、无吸收、无电子噪声**：网络学到的映射限于该仿真分布，
   迁移到真实乳腺数据需重新校准/微调。
3. **声速图可辨识性有限**：13 角平面波 + 38.1 mm 孔径对 2D 声速图的高频
   横向细节不敏感；ROI 平均声速比逐像素结构更可靠（`predict.py` 同时报告）。
4. **条件只覆盖孔径内**：横向超出 ±19.05 mm 的介质不可见，WFC 重采样时
   边界 clamp。
5. **`const best`（测试集均值常数）是 oracle**，仅作参考，不是可用方法。

## 10. 结果记录（2026-09-09 实跑，200 epoch）

命令：`train.py --epochs 200`（640 train / 80 val，batch 8，Adam lr 5e-5 + cosine，
EMA 0.999，A6000 单卡约 85 分钟）。数值来源：
`out/flow/train.log`、`out/pred_flow/summary.csv`、`out/wfc/summary.csv`。

**声速图预测（80 个 test 样本，flow 采样 8 次 × 20 步取均值）**

| 方法 | 全图 MAE | ROI MAE | ROI 平均声速误差 | ROI 相关 |
|---|---|---|---|---|
| **本模型（乘性流）** | **6.57** | **6.51** | **1.14** | **0.896** |
| 常数 1500 m/s | 34.75 | 36.99 | 27.92 | 0.000 |
| oracle 常数 1473.9 m/s | 30.11 | 30.37 | 15.56 | 0.000 |

（单位 m/s。val 最优 `roi_mae=6.74`，epoch 170。）

**WFC 波束形成（8 个 test 样本，相干复合，ROI 内 ipr 集中度；每样本绑定自身 IQ）**

| 声速假设 | 平均 ipr |
|---|---|
| 常数 1500 | 3.43 |
| **预测图** | **3.55**（比常数 +0.127） |
| 真值图 | 3.50（比预测 −0.053） |

`ipr` 是能量集中度代理，真值图并不保证更高；逐样本见 `out/wfc/summary.csv`。
单角度图 `ipr` 增益 pred−const = +0.394。

**已核验**

- WFC 引擎 + 本数据延迟约定：合成点目标峰值落点与 DAS 完全一致
  （0.000 mm 偏差，`wfc_integration/verify_psf.py`）。
- 条件含真实声速信息：相位保持的 40 通道条件经 ridge 回归预测 ROI 平均声速
  （180 训练 / 259 测试）MAE 12.8 m/s、corr +0.83（仅用 `full` 6 通道）。

**未核验 / 待办**

- 跨数据集泛化（真实乳腺 RF、其他探头）未测；
- 2D 声速图的高频细节仍偏平滑（预测的是条件均值），
  逐像素 MAE 6.5 m/s 中约一半来自结构边界；
- WFC 的 `ipr` 只是图像质量代理，绝对数值不宜与真实超声横向比较。

