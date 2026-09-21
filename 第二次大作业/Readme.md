# P2 文档图像频域去噪与 SIFT 特征提取 —— 复现与成果展示方法

配套材料：

| 文件 | 内容 | 是否在提交范围内 |
|---|---|---|
| `doc_image_freq_sift.py` | **作业要求的唯一实现代码**（约 2000 行，含逐段注释）：二维 DFT/IDFT、频域低通/高通/带阻/陷波、空域均值/高斯/中值/拉普拉斯/USM、Canny、Harris、SIFT+匹配、自建数据集、6 组实验、出图出表 | ✅ 提交 |
| `P2_实验报告.docx` | 实验报告：摘要 + 正文 + 频谱图 + 指标表 + 失败案例分析 + 复现说明 | ✅ 提交 |
| `成果说明.md` | 摘要 / 正文 / 复现步骤 + 提交材料清单 + 提交前自查（本系列作业的统一格式） | ✅ 提交 |
| `P2复现与成果展示方法.md` | 本文件：怎么在你自己机器上复现、怎么展示 | 参考 |
| `out/` | 运行产物：`dataset/`（216 张自建图像）、`figures/`（38 张图）、`tables/`（8 个 CSV）、`report_tables.md`、`metrics_all.json` | 附证据 |

> ⚠️ **提交范围提示**：作业要求“实现作业要求的代码只能是一个 .py 文件”，因此**只交 `doc_image_freq_sift.py`**。
> 本目录下的 `make_report.py` 是生成报告的辅助脚本（把 `report_tables.md` 与 `figures/` 拼成 docx），不属于作业实现代码。

---

## 0. 一句话说明两份材料的关系

| 作业要求 | 对应实现 / 产物 |
|---|---|
| 单 .py 实现 DFT/IDFT | `dft2d_direct` / `idft2d_direct`（矩阵法，含可分离性注释）；`fft2c`/`ifft2c`（FFT 加速版）；`dft_selftest` 自检 |
| 频域低通/高通/带阻 | `lp_mask` / `hp_mask` / `band_stop_mask` / `notch_mask`（理想·高斯·巴特沃斯） |
| 空域均值/高斯/中值/拉普拉斯锐化 | `mean_filter` / `gaussian_filter` / `median_filter` / `laplacian_sharpen` / `unsharp_mask` |
| Canny 边缘 | `canny_edge`（四步全自实现） |
| Harris 角点 | `harris_corner` / `harris_top` |
| SIFT 特征提取 | `sift_detect` / `match_descriptors`（ratio + RANSAC） |
| 自建 20 张以上文档图像 | `render_document_page` + `deg_*` + `build_dataset` → 24 页 × 9 变体 = 216 张 |
| 空域 vs 频域去噪 | 实验 E1 → 表 3 / 表 4、图 3~6 |
| 锐化前后边缘与角点、SIFT 匹配率 | 实验 E2 → 表 5、图 7~10 |
| 3 组消融（截止频率/滤波器类型/顺序） | 实验 E3/E4/E5 → 表 6~8、图 11~21 |
| 指标 PSNR/SSIM/边缘F1/角点重复率/SIFT内点 | `psnr` / `ssim` / `edge_prf` / `corner_repeatability` / 匹配内点统计 |
| 失败案例分析 | 实验 E7 → 表 11、图 23，报告第 6 章 |

---

## 1. 环境

```bash
python --version                 # >= 3.9（本机 3.13）
pip install numpy opencv-python scipy matplotlib pillow
```

不需要 GPU、不需要下载任何数据集（数据全部由代码合成），CPU 全流程约 20 分钟。

---

## 2. 三条复现路径

### 2.1 一键全流程（推荐，出报告全部图表）

```bash
python doc_image_freq_sift.py --stage all --out out --n_docs 24
```

产物：

```
out/
├─ dataset/          216 张图像（24 页 × 9 变体）+ manifest.json
├─ figures/          38 张图
├─ tables/           8 个 CSV（e1_raw 逐图原始数据 + e1~e7 聚合均值±std）
├─ report_tables.md  全部指标表（Markdown，报告表 3~11 由此生成）
├─ metrics_all.json  自检结果 + 聚合指标
└─ run_full.log      日志
```

### 2.2 极速自检（4 页 + 半分辨率，约 2 分钟）

```bash
python doc_image_freq_sift.py --stage all --quick --out out_quick
```

> 注意：`--quick` 会把图像缩到 512×384，截止频率 D0 的正确取值随分辨率线性变化，
> **quick 模式的 D0 数值不能写进报告**，只用于确认环境能跑通。

### 2.3 单张真实照片模式（把结论搬到自己的作业照片上）

```bash
python doc_image_freq_sift.py --single 我的作业照片.jpg --out out_single
```

真实照片没有 ground truth，所以不输出 PSNR/SSIM，只输出无参考指标
（Laplacian 方差清晰度、RGB/灰度统计、边缘像素数、Harris 角点数、SIFT 关键点数）
以及一张 2×3 对比图：原图 / 频谱 / Canny 边缘 / 中值 / 频域高斯低通 / USM 锐化 / 角点叠加。

---

## 3. 分步运行

```bash
python doc_image_freq_sift.py --stage data --out out    # 只生成 216 张数据集
python doc_image_freq_sift.py --stage figs --out out    # 只出数据集总览 / 频谱 / SIFT 匹配三张图
python doc_image_freq_sift.py --stage exp  --out out    # 跑 6 组实验 + 出表（不含总览图）
python doc_image_freq_sift.py --stage all --out out --n_docs 12   # 用前 12 页跑（约 10 分钟）
```

---

## 4. 指标口径速查（写报告 / 答辩被问到时用）

| 指标 | 实现 | 口径与坑 |
|---|---|---|
| PSNR | `psnr` | 与干净参考逐像素；文档图上 PSNR 奖励“少动细节”，糊了也可能得高分 |
| SSIM | `ssim` | 自实现，11×11 高斯窗，K1=0.01/K2=0.03；奖励“平滑”，最优点偏向强滤波 |
| 边缘 F1 | `edge_prf` | 真值 = 干净图的 Canny；两侧各膨胀 2 px 容差，避免 1 px 定位差判错 |
| 角点重复率 | `corner_repeatability` | Harris 是相对阈值，必须固定 top-K（本实验 K=500）+ 半径 2.5 px 才可比 |
| SIFT 内点数 | `match_descriptors` | 与同一页的 `view2`（带 2°~5° 旋转、1.0~1.05 缩放）匹配，ratio 0.75 + RANSAC 5 px |
| 振铃指数 | `ringing_index` | 滤波器固有属性：IDFT(H) 的负旁瓣能量占比。理想≈0.47、高斯≈0、巴特沃斯 n=2≈0.11 |

---

## 5. 展示建议（答辩 / 交作业时的讲解顺序）

1. **先放图 2（频谱分析图）**：一张图讲清“噪声铺满全平面、模糊是方向性暗带、光照不匀挤在中心、条纹是孤立亮斑”，
   后面所有方法选择都能从这张图推出来。
2. **再放表 4（各退化下的最优方法）**：说明“脉冲找中值、高斯找轻低通、模糊怎么选都无用”。
3. **重点讲表 6（不同指标的最优 D0 不同）**：这是本实验最有价值的发现——
   同一族滤波器，PSNR 最优与 SSIM 最优的截止频率相差数倍，说明去噪参数必须绑定下游任务。
4. **讲消融 C（表 8）**：顺序上“先去噪再锐化”显著优于“先锐化再去噪”，SIFT 内点数差得最多。
5. **最后放图 22（周期条纹）与图 23（失败案例）**：说明频域方法的结构性优势（定点陷波）
   与能力边界（乘性照度、卷积模糊不是加权能解决的）。

---

## 6. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 图里中文变成方块 | 代码会自动从 `C:\Windows\Fonts` 找 msyh/simhei/simsun；Linux 下装 `wqy-zenhei` 即可 |
| `cv2.error: ... ddepth >= sdepth` | 本代码已在所有滤波入口统一转 float32；若自行改代码，注意 OpenCV 不接受混深度 |
| 频域滤波结果整体偏暗/偏亮 | 检查 `fft2c`/`ifft2c` 是否配对（正变换后 fftshift、逆变换前 ifftshift），以及掩模是否以图像中心为零频 |
| 单张模式报“无法读取图像” | 代码用 `np.fromfile + cv2.imdecode` 读图以兼容中文路径；若仍失败，确认扩展名与实际格式一致 |
| 想换自己的文档照片做数据集 | 把照片放进一个目录，用 `imread_gray` 逐个读入后替换 `load_dataset` 的返回值即可（其余实验代码不用改） |
