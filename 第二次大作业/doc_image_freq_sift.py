# -*- coding: utf-8 -*-
r"""
================================================================================
P2 · 计算摄影中的文档图像频域去噪与 SIFT 特征提取复现            单文件实现
================================================================================
课程：计算摄影   Lab2 · 第 3 周 周四     侧重：模型（复现与实验分析）

本文件是作业要求的**唯一 .py 文件**，自包含完成全部实现与实验：

  ① 二维 DFT / IDFT         —— 矩阵法自实现 + numpy FFT 加速实现，两者互相验证
  ② 频域滤波                —— 理想 / 高斯 / 巴特沃斯 低通·高通，带阻，陷波(notch)
  ③ 空域滤波                —— 均值、高斯、中值、拉普拉斯锐化、USM 非锐化掩模
  ④ 特征提取                —— Canny 边缘（自实现）、Harris 角点（自实现）、SIFT + 匹配
  ⑤ 数据自建                —— 24 张合成文档页（作业/试卷/板书风格）× 6 类退化 + 重拍视图
  ⑥ 实验                    —— 空域 vs 频域去噪 / 锐化前后边缘与角点 / SIFT 匹配率
                               / 3 组消融（截止频率、滤波器类型、去噪→特征顺序）/ 失败案例
  ⑦ 指标                    —— PSNR、SSIM、边缘 F1、角点重复率、SIFT 匹配内点数
  ⑧ 出图出表                —— 频谱图、结果网格、指标柱状图、曲线、CSV + Markdown 表

--------------------------------------------------------------------------------
运行方式（环境需 numpy / opencv-python / scipy / matplotlib / pillow）
--------------------------------------------------------------------------------
    python doc_image_freq_sift.py --stage all          # 建数据 + 全实验 + 出图出表
    python doc_image_freq_sift.py --stage data         # 只生成数据集 out/dataset
    python doc_image_freq_sift.py --stage exp --n_docs 12      # 用前 12 张跑实验
    python doc_image_freq_sift.py --stage exp --quick  # 极速自检（小图 + 少量样本）
    python doc_image_freq_sift.py --single 你的作业.jpg        # 单张照片：指标 + 频谱图

--------------------------------------------------------------------------------
指标 → 作业要求 对照（报告中按此口径写）
--------------------------------------------------------------------------------
    PSNR / SSIM                 去噪/锐化的保真度（越高越好）
    边缘 F1（对 clean 的 Canny） 结构保持能力（容差 2 px 的 P/R/F1）
    角点重复率（Harris 半径匹配）几何特征稳定性（含平均定位误差）
    SIFT 内点数（ratio+RANSAC）  下游几何任务可用性（内点率 = 内点/粗匹配）
================================================================================
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")  # 无界面后端，批量出图不出窗口
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ 全局配置
CFG = {
    "H": 1024,           # 合成文档页高度（--quick 时减半）
    "W": 768,            # 合成文档页宽度
    "N_DOCS": 24,        # 文档页数量（作业要求 >=20）
    "RNG": 20260920,     # 固定随机种子，保证复现
    "EDGE_TOL": 2,       # 边缘 F1 的容差（像素）
    "CORNER_RADIUS": 2.5,  # 角点重复率匹配半径（像素）
    "TOPK_CORNER": 500,  # 角点重复率取响应最强的 top-K 个（可比口径）
}
OUT = "out"              # 输出根目录（main 里可覆盖）

# 中文字体（Windows 常见路径，找不到就退回默认，不影响计算）
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def _setup_cjk_font():
    """matplotlib 中文显示：优先系统中文字体，避免图上出现方块。"""
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                font_manager.fontManager.addfont(p)
                name = font_manager.FontProperties(fname=p).get_name()
                plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
                plt.rcParams["axes.unicode_minus"] = False
                return name
            except Exception:
                pass
    plt.rcParams["axes.unicode_minus"] = False
    return None


CJK_FONT = _setup_cjk_font()
CJK_FONT_FILE = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)


# ==============================================================================
# 1. 基础工具：读写、显示、计时
# ==============================================================================
def imread_gray(path, size=None):
    """读入灰度图，返回 float32 且范围 [0,1]。

    用 np.fromfile + cv2.imdecode 是为了兼容中文路径（cv2.imread 在 Windows
    中文路径下会返回 None）。
    """
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError("无法读取图像: %s" % path)
    if size is not None:  # 统一尺寸，便于逐像素比较
        img = cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    return img.astype(np.float32) / 255.0


def imwrite(path, img):
    """写图，float [0,1] 或 uint8 均可；同样走 imencode 兼容中文路径。"""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 1) * 255.0
        img = img.astype(np.uint8)
    ok, buf = cv2.imencode(os.path.splitext(path)[1], img)
    if not ok:
        raise IOError("写图失败: %s" % path)
    buf.tofile(path)


def to_u8(img):
    """float[0,1] → uint8[0,255]，供 OpenCV 传统算子使用。"""
    return np.clip(img, 0, 1).__mul__(255.0).astype(np.uint8)


def to_f(img):
    """uint8/任意 → float[0,1]。"""
    return np.asarray(img).astype(np.float32) / 255.0


class Timer:
    """with Timer() as t: ...  → t.ms 记录耗时（报告里用于比较各方法代价）。"""

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self.t0) * 1000.0


def ensure_dir(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ==============================================================================
# 2. 二维 DFT / IDFT
#   DFT 定义（教材第 3 讲）：
#       F(u,v) = Σ_x Σ_y f(x,y) · exp[ -j2π(ux/M + vy/N) ]
#       f(x,y) = (1/MN) Σ_u Σ_v F(u,v) · exp[ +j2π(ux/M + vy/N) ]
#   因为指数可分离，二维 DFT 可写成两个一维 DFT 的矩阵乘积：
#       F = W_M · f · W_N^T ,  W_N[k,n] = exp(-j2πkn/N)
#   这正是"二维 DFT 可分离性 → 行列两次一维变换"的实现依据，
#   也是 numpy.fft.fft2 内部的做法（只是 FFT 把 O(N^2) 降到 O(N log N)）。
# ==============================================================================
def dft_matrix(N):
    """返回 N×N 的 DFT 矩阵 W，W[k,n] = exp(-j2πkn/N)。"""
    k = np.arange(N).reshape(-1, 1)      # 行 = 频率索引
    n = np.arange(N).reshape(1, -1)      # 列 = 空间索引
    return np.exp(-2j * np.pi * k * n / N)


def dft2d_direct(f):
    """矩阵法二维 DFT（自实现，不做 FFT 加速）。

    F = W_M · f · W_N^T —— 先对每一列做一维 DFT，再对每一行做一维 DFT。
    复杂度 O(M²N + MN²)，只适合小图/教学验证（报告里用于验证 FFT 的正确性）。
    """
    f = np.asarray(f, dtype=np.complex128)
    M, N = f.shape
    WM, WN = dft_matrix(M), dft_matrix(N)
    return WM @ f @ WN.T


def idft2d_direct(F):
    """矩阵法二维 IDFT：f = (1/MN) · conj(W_M) · F · conj(W_N)^T。

    由 IDFT 定义可直接推出：正变换矩阵的共轭即为逆变换核。
    """
    F = np.asarray(F, dtype=np.complex128)
    M, N = F.shape
    WM, WN = dft_matrix(M), dft_matrix(N)
    return (WM.conj() @ F @ WN.conj().T) / (M * N)


def fft2c(f):
    """中心化二维 DFT（numpy FFT 加速版）：零频搬到图像中心。

    F_shift = fftshift( FFT2(f) )，等价于先对图像做 (-1)^(x+y) 调制再变换，
    这样低频落在矩阵中心，画频谱和写滤波器掩模都直观。
    与 IFFT2 的配对是"正变换后 fftshift、逆变换前 ifftshift"，往返严格还原。
    """
    f = np.asarray(f, dtype=np.float64)
    return np.fft.fftshift(np.fft.fft2(f))


def ifft2c(F):
    """中心化二维 IDFT，返回实数空域图（取实部）。"""
    return np.real(np.fft.ifft2(np.fft.ifftshift(F)))


def spectrum_mag(f, log=True):
    """频谱幅度图（可视用）：|F(u,v)|，log 压缩动态范围。

    注意两点（报告里会写）：
      · 幅度谱必须做 log(1+|F|) 才看得见细节，否则被直流分量淹没；
      · 频谱图关于中心共轭对称，所以翻转 180° 是同一张图的"倒立版"。
    """
    F = fft2c(f)
    mag = np.abs(F)
    if log:
        mag = np.log1p(mag)
    m = mag.max()
    return (mag / m).astype(np.float32) if m > 0 else mag.astype(np.float32)


def radial_freq_grid(shape):
    """返回以频谱中心为原点的频率半径矩阵 D(u,v) 与两个正交频率坐标。

    D(u,v) = sqrt((u-M/2)² + (v-N/2)²)，单位是"频率格数"。
    截止频率 D0 就是这张掩模上半径的阈值——本实验扫的就是这个量。
    """
    M, N = shape
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    VV, UU = np.meshgrid(v, u)          # 注意 meshgrid 的 x/y 对应列/行
    D = np.sqrt(UU.astype(np.float64) ** 2 + VV.astype(np.float64) ** 2)
    return D, UU, VV


# ==============================================================================
# 3. 频域滤波器（低通 / 高通 / 带阻 / 陷波）
#   低通：H_LP(u,v) = 1 当 D <= D0 —— 保留低频、去掉高频（去噪、抗混叠、抗摩尔纹）
#   高通：H_HP = 1 - H_LP        —— 保留边缘细节（锐化、边缘提取）
#   带阻：H_BS = 1 - H_BP        —— 抑制某个环带（周期性条纹/纹理）
#   陷波：H_NT 在指定频点挖坑    —— 精确打掉正弦干扰的共轭频率对
#   三种"形状"的区别只在过渡带：
#       ideal      : 0/1 硬截断 → 最陡，但空域 sinc 旁瓣 → 振铃(Gibbs)
#       gaussian   : exp(-D²/2D0²) → 最平滑，无振铃，但过渡带宽、边缘更糊
#       butterworth: 1/(1+(D/D0)^2n) → 阶数 n 可调，n↑ 趋近理想、振铃↑
# ==============================================================================
def lp_mask(shape, D0, kind="ideal", order=2):
    """低通滤波器频域响应 H（中心化，与 fft2c 的输出对齐）。"""
    D, _, _ = radial_freq_grid(shape)
    if kind == "ideal":
        return (D <= D0).astype(np.float64)
    if kind == "gaussian":
        return np.exp(-(D ** 2) / (2.0 * D0 ** 2))
    if kind == "butterworth":
        # n 阶巴特沃斯：n=1 很平缓，n=2 常用，n>=4 基本等于理想但开始振铃
        return 1.0 / (1.0 + (D / max(D0, 1e-6)) ** (2 * order))
    raise ValueError("未知低通类型: %s" % kind)


def hp_mask(shape, D0, kind="ideal", order=2):
    """高通 = 1 - 低通（同一 D0/形状）。"""
    return 1.0 - lp_mask(shape, D0, kind, order)


def band_stop_mask(shape, D0, width, kind="gaussian"):
    """环带阻滤波器：D0 为中心半径，width 为带宽半宽。

    实现：外径低通 ∩ (内径低通取反) → 只留环带，再 1-环带 得到带阻。
    """
    outer = lp_mask(shape, D0 + width, kind)
    inner = lp_mask(shape, max(D0 - width, 1e-6), kind)
    band = np.clip(outer - inner, 0, 1)
    return 1.0 - band


def notch_mask(shape, spots, radius=6.0, kind="gaussian"):
    """陷波滤波器：在给定频点集合上挖掉（含共轭对称点）。

    spots: [(u, v, ...)] 相对于**中心**的偏移（相对坐标，不含 M//2 偏移）。
    用于消除拍摄屏幕/网格产生的正弦条纹——它在频谱上是几对亮斑。
    实际使用时把频谱图上的亮斑坐标传进来即可。
    """
    M, N = shape
    H = np.ones((M, N), dtype=np.float64)
    uu = np.arange(M) - M // 2
    vv = np.arange(N) - N // 2
    VV, UU = np.meshgrid(vv, uu)
    for (du, dv) in spots:
        for s in (1, -1):  # 实信号的频谱共轭对称，正负频点要一起打掉
            d2 = (UU - s * du) ** 2 + (VV - s * dv) ** 2
            if kind == "gaussian":
                H *= 1.0 - np.exp(-d2 / (2.0 * radius ** 2))
            else:  # ideal 陷波
                H *= 1.0 - (d2 <= radius ** 2)
    return H


def find_spectrum_peaks(f, center_exclude=12, topk=4, min_dist=14, k=6.0, win=31):
    """自动检测周期性噪声在频谱上的孤立亮斑（带阻/陷波的靶点坐标）。

    为什么不能"直接取最亮的点"：文档内容的频谱也有一堆强峰（笔画、表格线、
    甚至程式的水平轴亮带），取全局最大往往挑到文字而不是干扰。
    这里用**局部突出度（prominence）**：
        prom(u,v) = log|F| - boxblur(log|F|, 31)   —— 减去 31×31 的局部均值
    周期性条纹在频域是"孤立、极窄、且旁边很干净"的点，prom 很高；
    而文字/表格的强峰往往成片成带，局部均值本来就高，prom 反而低。
    再叠加"按半径归一化"的 z 分数做二次确认，最后用 min_dist 抑制近邻重复，
    取前 topk 组（实信号频谱共轭对称，正负频点会成对出现，这本身就是
    判定"这是正弦干扰"的证据）。
    """
    F = np.log1p(np.abs(fft2c(f))).astype(np.float32)
    prom = F - cv2.blur(F, (win, win))                    # 局部突出度
    M, N = F.shape
    D, _, _ = radial_freq_grid((M, N))
    prom[D < center_exclude] = 0                          # 排除低频（文档内容的主场）
    # 二次确认：按半径归一化的 z 分数（同半径环上离群程度）
    ring = np.round(D).astype(np.int32)
    cnt = np.bincount(ring.ravel()).astype(np.float64)
    s = np.bincount(ring.ravel(), weights=F.ravel())
    s2 = np.bincount(ring.ravel(), weights=(F ** 2).ravel())
    mu = s / np.maximum(cnt, 1)
    sd = np.sqrt(np.maximum(s2 / np.maximum(cnt, 1) - mu ** 2, 0))
    z = (F - mu[ring]) / np.maximum(sd[ring], 1e-6)
    mx = cv2.dilate(prom, np.ones((min_dist, min_dist), np.uint8))
    thr = float(prom.mean() + k * prom.std())
    cand = np.argwhere((prom >= mx - 1e-6) & (prom > thr) & (z > 3.0))
    cand = sorted([(prom[y, x], y, x) for y, x in cand], reverse=True)
    picks = []
    for _, y, x in cand:
        du, dv = int(y - M // 2), int(x - N // 2)
        if all((du - a) ** 2 + (dv - b) ** 2 > min_dist ** 2 for a, b in picks):
            picks.append((du, dv))
        if len(picks) >= topk:
            break
    return picks[:topk]


def freq_filter(img, H):
    """频域滤波统一入口：g = IDFT[ H(u,v) · DFT[f] ]。

    注意 fftshift/ifftshift 的配对：fft2c 内部已做中心化，所以掩模 H 直接
    以图像中心为零频，乘法是逐像素的 Hadamard 积——滤波器设计好不好，
    全在 H 上，这也是频域法相对空域卷积最大的优势（先看频谱再定点设计）。
    """
    return np.clip(ifft2c(fft2c(img) * H), 0, 1).astype(np.float32)


def freq_filter_direct(img, H):
    """用自实现的矩阵法 DFT 做同样的滤波（教学验证用，慢，只在小图上跑）。

    报告里用它说明："频域滤波 = 乘掩模" 这一结论与 FFT/DFT 实现无关。
    """
    F = dft2d_direct(img)
    return np.clip(np.real(idft2d_direct(F * H)), 0, 1).astype(np.float32)


# ==============================================================================
# 4. 空域滤波器
#   与频域滤波的对应关系（报告里重点写）：
#       空域均值核  ←→ 频域 sinc 型低通（有旁瓣，会漏高频）
#       空域高斯核  ←→ 频域高斯低通（唯一"核与频谱都是高斯"的特例，无振铃）
#       空域中值    ←→ 非线性，无频域乘性对应，对脉冲噪声特别有效
#       空域拉普拉斯 ←→ 频域 -(u²+v²) 型高通（二阶微分锐化）
#   → 卷积定理 f*g ↔ F·G 说明：空域小核卷积 == 频域乘该核的频谱（线性时不变时成立）
# ==============================================================================
def mean_filter(img, k=3):
    """均值滤波：核内平均（= 空域盒式卷积）。OpenCV 内部用积分图，O(1)/像素。"""
    return cv2.blur(np.asarray(img, dtype=np.float32), (k, k))


def gaussian_filter(img, k=5, sigma=1.0):
    """高斯滤波：按距离加权，比均值核更保边（中心权重高、高频泄漏更少）。"""
    return cv2.GaussianBlur(np.asarray(img, dtype=np.float32), (k, k), sigma,
                            borderType=cv2.BORDER_REFLECT)


def median_filter(img, k=3):
    """中值滤波：非线性，把孤立极值（椒盐）替换为邻域中位数，不产生新灰度。"""
    return cv2.medianBlur(to_u8(img), k).astype(np.float32) / 255.0


def laplacian_sharpen(img, alpha=1.0, ksize=3):
    """拉普拉斯锐化：g = f + alpha·∇²f。

    拉普拉斯是二阶微分，对细节/边缘响应强、对平滑区响应弱，
    相当于频域上乘 -(u²+v²)（高通）。alpha 越大边缘越"硬"，
    同时噪声也被一起放大——这是"先锐化再去噪"顺序失败的根本原因。
    """
    img = np.asarray(img, dtype=np.float32)
    lap = cv2.Laplacian(img, cv2.CV_32F, ksize=ksize)
    return np.clip(img + alpha * lap, 0, 1).astype(np.float32)


def unsharp_mask(img, k=7, sigma=2.0, amount=1.0):
    """USM 非锐化掩模：g = f + amount·(f - blur(f))。

    本质是"原图减去低频"得到高频，再按比例加回去 → 与频域高通增强同构
    （f + a·HP(f) 就是频域写法），区别只是 HP 的形状由空域核决定。
    """
    img = np.asarray(img, dtype=np.float32)
    blur = cv2.GaussianBlur(img, (k, k), sigma)
    return np.clip(img + amount * (img - blur), 0, 1).astype(np.float32)


# ==============================================================================
# 5. 特征提取：Canny 边缘（自实现）、Harris 角点（自实现）、SIFT
# ==============================================================================
def canny_edge(img, sigma=1.4, low=0.06, high=0.16, return_stage=False):
    """Canny 边缘检测（自实现，按教材四步）。

    步骤：
      1) 高斯平滑               —— 抑制噪声（Canny 的第一条准则）
      2) Sobel 求梯度幅值/方向   —— 边缘 = 灰度变化最快的方向
      3) 非极大值抑制(NMS)       —— 把"脊"细化成单像素宽；按方向量化成 4 个
                                   主方向，与两侧相邻像素比较，只保留局部极大
      4) 双阈值 + 滞后连接       —— 强边缘直接保留，弱边缘只有在与强边缘连通
                                   时才保留（用形态学膨胀迭代逼近连通性，向量化实现）
    返回 0/1 的 uint8 边缘图（1=边缘）。
    """
    f = img.astype(np.float32)
    # (1) 高斯平滑
    g = cv2.GaussianBlur(f, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)
    # (2) 梯度（Sobel 3×3 是梯度算子的可分离近似）
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = np.rad2deg(np.arctan2(gy, gx)) % 180.0     # 方向折叠到 [0,180)
    # (3) 非极大值抑制：向量化实现（4 个方向桶，各取两侧邻居比较）
    H, W = mag.shape
    pad = np.pad(mag, 1, mode="edge")
    c = pad[1:-1, 1:-1]
    n, s, e, w = pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, 2:], pad[1:-1, :-2]
    ne, nw, se, sw = pad[:-2, 2:], pad[:-2, :-2], pad[2:, 2:], pad[2:, :-2]
    b0 = (ang < 22.5) | (ang >= 157.5)               # 水平梯度 → 比上下
    b1 = (ang >= 22.5) & (ang < 67.5)                # 45° → 比反对角
    b2 = (ang >= 67.5) & (ang < 112.5)               # 垂直梯度 → 比左右
    b3 = (ang >= 112.5) & (ang < 157.5)              # 135° → 比主对角
    keep = np.zeros_like(mag, dtype=bool)
    keep |= b0 & (c >= n) & (c >= s)
    keep |= b1 & (c >= nw) & (c >= se)
    keep |= b2 & (c >= e) & (c >= w)
    keep |= b3 & (c >= ne) & (c >= sw)
    nms = np.where(keep, mag, 0.0)
    # (4) 双阈值 + 滞后连接
    hi = high * nms.max() if nms.max() > 0 else 1.0
    lo = low * nms.max() if nms.max() > 0 else 0.0
    strong = (nms >= hi)
    weak = (nms >= lo) & (~strong)
    # 滞后连接：弱边缘只有"连着强边缘"才保留。等价于在弱边缘图上找连通域，
    # 只保留含强边缘的那些连通域（用一次连通域标记，比逐次膨胀快一个量级）。
    cand = (strong | weak).astype(np.uint8)
    n_lab, labels = cv2.connectedComponents(cand, connectivity=8)
    if n_lab > 1:
        keep_lab = np.zeros(n_lab, np.uint8)
        keep_lab[np.unique(labels[strong])] = 1
        keep_lab[0] = 0                       # 0 是背景
        edges = keep_lab[labels].astype(np.uint8)
    else:
        edges = np.zeros_like(cand)
    if return_stage:
        return edges, dict(mag=mag, nms=nms, strong=strong.astype(np.uint8),
                           weak=weak.astype(np.uint8))
    return edges


def harris_corner(img, k=0.04, win=3, ksize=3, thresh_ratio=0.01, nms_size=7, topk=None):
    """Harris 角点检测（自实现）。

    原理：在窗口内求二阶矩矩阵 M = Σ w·[[Ix², IxIy], [IxIy, Iy²]]，
          角点响应 R = det(M) - k·trace(M)²；
    判读：平坦区 R≈0；直线边缘 R<0；角点（两方向都变化大）R>0。
    实现：
      1) Sobel 求 Ix, Iy
      2) 高斯窗加权累加得到 Ix², Iy², IxIy 的局部和（= M 的分量）
      3) 算 R，阈值 = thresh_ratio × R.max()
      4) 3×3 邻域非极大值抑制（膨胀比较），最后可选按 R 排序取前 topk 个
    返回 (N×2 的 (x,y) 角点数组, R 图)。返回坐标是为了和 clean 图做重复率比较。
    """
    f = img.astype(np.float32)
    Ix = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=ksize)
    Iy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=ksize)
    Ixx = cv2.GaussianBlur(Ix * Ix, (win, win), 0)
    Iyy = cv2.GaussianBlur(Iy * Iy, (win, win), 0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (win, win), 0)
    det = Ixx * Iyy - Ixy * Ixy
    tr = Ixx + Iyy
    R = det - k * tr * tr
    R[R < thresh_ratio * R.max()] = 0
    if nms_size > 0:  # 非极大值抑制：只留局部峰
        mx = cv2.dilate(R, np.ones((nms_size, nms_size), np.uint8))
        R[R < mx] = 0
    ys, xs = np.nonzero(R)
    if topk is not None and len(ys) > topk:  # 只保留响应最强的前 topk 个
        idx = np.argsort(R[ys, xs])[::-1][:topk]
        ys, xs = ys[idx], xs[idx]
    return np.stack([xs, ys], axis=1).astype(np.float32), R


def harris_top(img, topk=500, **kw):
    """取响应最强的 top-k 个角点。

    用途：做**角点重复率**时不能直接用"阈值冒出来的全部角点"，因为 Harris 的
    阈值是相对量（thresh_ratio × R.max），图像一模糊 R.max 就变，点数会
    剧烈波动，重复率会被噪声化的点数稀释。标准做法（特征评价惯例）是
    在两边各取响应最强的 top-k 点再比——这样比的是"最好的那批特征"是否还在原处。
    """
    pts, R = harris_corner(img, **kw)
    if len(pts) > topk:
        vals = R[pts[:, 1].astype(int), pts[:, 0].astype(int)]
        pts = pts[np.argsort(vals)[::-1][:topk]]
    return pts


_SIFT = None


def get_sift(nfeatures=600, contrast_threshold=0.03, edge_threshold=12, sigma=1.6):
    """惰性创建 SIFT 检测器（构造有开销，实验里复用同一个实例）。

    参数说明（都是有意的选择，报告里会交代）：
      nfeatures=600  只保留对比度最强的 600 个关键点。拍照文档的纸纹会产生
                     成千上万个低质量关键点，截断后匹配更稳定、也更快；
      contrastThreshold=0.03  略高于默认 0.04 的邻域，用来滤掉纸面噪声点；
      edgeThreshold=12 抑制强边缘上的不稳点（文档里长直线很多）。

    说明：SIFT 采用 OpenCV 的标准实现（DoG 金字塔极值检测 + 亚像素精修 +
    梯度方向直方图主方向 + 4×4×8=128 维描述子）；从零复现 SIFT 属于另一个
    量级的工程，本实验的"复现"目标是**成像链路 + 频域去噪 → 特征的因果链**，
    因此 SIFT 直接调用成熟实现，但检测参数、匹配、评价、消融全部自己写。
    """
    global _SIFT
    if _SIFT is None:
        _SIFT = cv2.SIFT_create(nfeatures=nfeatures,
                                contrastThreshold=contrast_threshold,
                                edgeThreshold=edge_threshold,
                                sigma=sigma)
    return _SIFT


def sift_detect(img, mask=None):
    """SIFT 检测：返回关键点列表与 (N×128) 描述子矩阵。"""
    kps, des = get_sift().detectAndCompute(to_u8(img), mask)
    return (kps or []), (des if des is not None else np.zeros((0, 128), np.float32))


def match_descriptors(ka, da, kb, db, ratio=0.75):
    """由两组（关键点, 描述子）做匹配：Lowe ratio test + RANSAC 单应校验。

    返回 dict:  pa/pb  内点坐标对; n_match 粗匹配数（ratio test 后）;
                n_inlier 内点数（内点 = 满足同一单应的匹配）;
                inlier_ratio; kps_a/kps_b 关键点数量。
    为什么必须加 RANSAC：文档图像纹理少、重复笔画多（同一个字反复出现），
    ratio test 后仍有大量误配，只有几何一致性校验后的内点数才是
    "下游任务（拼接/配准/多帧超分）真正可用的特征数"。
    拆出这个函数是为了让实验能复用缓存的描述子（clean / view2 反复用到）。
    """
    res = dict(n_ka=len(ka), n_kb=len(kb), n_match=0, n_inlier=0,
               inlier_ratio=float("nan"), pa=[], pb=[], kps_a=ka, kps_b=kb)
    if len(ka) < 2 or len(kb) < 2:
        return res
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(da, db, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    res["n_match"] = len(good)
    if len(good) < 4:
        return res
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # RANSAC 估计单应，返回内点掩模（>=4 对才可解）
    Hm, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if mask is None:
        return res
    mask = mask.ravel().astype(bool)
    res["n_inlier"] = int(mask.sum())
    res["inlier_ratio"] = float(mask.sum()) / float(len(good))
    res["pa"] = src[mask].reshape(-1, 2)
    res["pb"] = dst[mask].reshape(-1, 2)
    res["matches"] = [g for g, ok in zip(good, mask) if ok]
    return res


def sift_match(img_a, img_b, ratio=0.75):
    """SIFT 匹配（检测 + 匹配一步到位，单张照片模式用）。"""
    ka, da = sift_detect(img_a)
    kb, db = sift_detect(img_b)
    return match_descriptors(ka, da, kb, db, ratio)


# ==============================================================================
# 6. 质量与几何指标
# ==============================================================================
def psnr(ref, x, data_range=1.0):
    """峰值信噪比：PSNR = 10·log10(MAX² / MSE)。"""
    mse = float(np.mean((ref.astype(np.float64) - x.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10((data_range ** 2) / mse)


def ssim(ref, x, data_range=1.0, sigma=1.5, K1=0.01, K2=0.03):
    """结构相似度 SSIM（自实现，11×11 高斯窗，与 Wang 2004 原版一致）。

    SSIM = [(2μxμy+C1)(2σxy+C2)] / [(μx²+μy²+C1)(σx²+σy²+C2)]
    C1=(K1·L)², C2=(K2·L)² 是为了避免分母为 0。
    注意：SSIM 对"模糊"比 PSNR 敏感——一张糊掉但均值正确的图 PSNR 可能不低，
    但 SSIM 会掉，本实验里这一点在频域低通上表现得尤其明显。
    """
    a = ref.astype(np.float32)
    b = x.astype(np.float32)
    C1, C2 = (K1 * data_range) ** 2, (K2 * data_range) ** 2

    def blur(z):  # 11×11 高斯窗（win=11 是 Wang 2004 原文的推荐设置）
        return cv2.GaussianBlur(z, (11, 11), sigma, borderType=cv2.BORDER_REFLECT)

    mu_a, mu_b = blur(a), blur(b)
    saa = blur(a * a) - mu_a * mu_a
    sbb = blur(b * b) - mu_b * mu_b
    sab = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + C1) * (2 * sab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2)
    return float(np.mean(num / den))


def edge_prf(pred, gt, tol=2):
    """边缘 P/R/F1（对真值边缘图）。

    直接用像素级比较会因 1 px 的定位差把正确边缘判成错，所以两边都先做
    容差膨胀：预测边缘落在真值边缘 tol 邻域内即算 TP。返回 (P, R, F1)。
    """
    p = (pred > 0).astype(np.uint8)
    g = (gt > 0).astype(np.uint8)
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    gd = cv2.dilate(g, k)
    pd = cv2.dilate(p, k)
    tp = int((p & gd).sum())
    prec = tp / max(int(p.sum()), 1)
    rec = int((g & pd).sum()) / max(int(g.sum()), 1)
    f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
    return float(prec), float(rec), float(f1)


def corner_repeatability(pts_a, pts_b, radius=2.5):
    """角点重复率：以 A 的角点为基准，在半径 radius 内能在 B 中找到对应者
    的比例；同时给出平均定位误差（越接近 0 越好）。

    另外给一个对称口径 repeat_sym：A→B 与 B→A 重复率的调和平均，
    避免"A 少 B 多"时单侧指标虚高。返回 dict。
    """
    ra = np.asarray(pts_a, dtype=np.float64).reshape(-1, 2)
    rb = np.asarray(pts_b, dtype=np.float64).reshape(-1, 2)
    out = dict(n_a=len(ra), n_b=len(rb), repeat=0.0, repeat_sym=0.0, loc_err=float("nan"))

    def _one(a, b):
        if len(a) == 0 or len(b) == 0:
            return 0.0, float("nan")
        d, _ = cKDTree(b).query(a)
        ok = d <= radius
        return float(ok.sum()) / float(len(a)), (float(d[ok].mean()) if ok.any() else float("nan"))

    r_ab, e_ab = _one(ra, rb)
    r_ba, _ = _one(rb, ra)
    out["repeat"] = r_ab
    out["repeat_sym"] = 0.0 if r_ab + r_ba == 0 else 2 * r_ab * r_ba / (r_ab + r_ba)
    out["loc_err"] = e_ab
    return out


def ringing_index(H):
    """振铃（Gibbs）指数 —— 由滤波器本身决定，与图像无关。

    做法：求滤波器 H 的空域脉冲响应 h = IDFT(H)（即"点扩散函数"），
          统计 h 中**负值部分**的能量占比  sum(|h|, h<0) / sum(|h|)。
    物理含义：
      · 理想低通 H 是硬截断 → h 是 jinc 函数，一圈圈正负交替的旁瓣，
        负旁瓣就是空域上的"亮边/暗边过冲"，负能量占比高 → 振铃强；
      · 高斯低通的 H 也是高斯 → h 仍是高斯（恒为正）→ 指数 ≈ 0，无振铃；
      · 巴特沃斯阶数 n 越大 → 过渡带越陡 → 负旁瓣越多 → 指数单调上升。
    该指数只与滤波器形状和 D0 有关，是"设计滤波器时就该看的指标"；
    图像域的实际观感在 fig06b 的局部放大图里对照验证（两者结论一致）。
    """
    h = np.real(np.fft.ifft2(np.fft.ifftshift(H.astype(np.complex128))))
    tot = float(np.abs(h).sum())
    if tot <= 1e-12:
        return 0.0
    return float(np.abs(h[h < 0]).sum() / tot)


def evaluate(ref, x, gt_edge=None, ref_corners=None, ref_for_sift=None, name=""):
    """一次性算出某项结果的全部指标（报告表格的一行）。

    ref        干净参考图（PSNR/SSIM 基准，也是真值边缘/角点的来源）
    x          待评价的结果图
    gt_edge    预先算好的干净图 Canny 边缘（避免重复计算）
    ref_corners 预先算好的干净图 Harris 角点
    ref_for_sift 另一视图（用于 SIFT 匹配），为 None 时跳过匹配指标
    """
    row = dict(name=name)
    row["psnr"] = psnr(ref, x)
    row["ssim"] = ssim(ref, x)
    e = canny_edge(x)
    p, r, f1 = edge_prf(e, gt_edge, tol=CFG["EDGE_TOL"])
    row.update(edge_p=p, edge_r=r, edge_f1=f1, n_edge=int(e.sum()))
    cpts_all, R = harris_corner(x)                     # 全部角点：反映"结构丰富度"
    n_corner = int(len(cpts_all))
    if n_corner > CFG["TOPK_CORNER"]:                  # top-K：用于重复率（可比口径）
        vals = R[cpts_all[:, 1].astype(int), cpts_all[:, 0].astype(int)]
        cpts_top = cpts_all[np.argsort(vals)[::-1][:CFG["TOPK_CORNER"]]]
    else:
        cpts_top = cpts_all
    rep = corner_repeatability(ref_corners, cpts_top, CFG["CORNER_RADIUS"])
    row.update(n_corner=n_corner, corner_repeat=rep["repeat"],
               corner_repeat_sym=rep["repeat_sym"], corner_loc_err=rep["loc_err"])
    if ref_for_sift is not None:
        m = sift_match(x, ref_for_sift)
        row.update(sift_kp=len(m["kps_a"]), sift_match=m["n_match"],
                   sift_inlier=m["n_inlier"], sift_inlier_ratio=m["inlier_ratio"])
    return row


# ==============================================================================
# 7. 数据自建：合成"拍照文档图像"
#   为什么自建而不是直接拍照：本实验要比较 PSNR/SSIM/边缘 F1/角点重复率，
#   必须有一张"干净参考图"（ground truth）。真实拍照拿不到逐像素真值，
#   因此采用"合成干净页 → 施加可控退化"的两段式，退化模型对齐第 3 讲成像链路：
#       I_obs = PSF ⊛ ( L · R ) · g + n
#     · PSF         = 拍摄模糊（散焦/运动）     → 频域乘 OTF，高频被压制
#     · L·R 照度分  = 纸张反射率 × 光源分布     → 低频乘性场（光照不匀）
#     · n           = 传感器噪声               → 高斯/椒盐/斑点
#     · 周期条纹     = 屏幕摩尔纹/网格/banner   → 频域上几对孤立亮斑
#   共 24 张干净页（作业/试卷/板书 各 8 张）× 6 类退化 + 1 张"重拍视图"，
#   合计 24 × 9 = 216 张，满足作业"20 张以上、含噪声/模糊/光照不均"的要求。
# ==============================================================================
_PAGE_WORDS = list(
    "计算 摄影 图像 频域 空域 卷积 傅里叶 变换 频谱 高通 低通 滤波 采样 重建 "
    "照度 反射 文档 识别 特征 提取 边缘 角点 匹配 描述 子 尺度 空间 梯度 方向 "
    "噪声 模糊 抖动 曝光 白平衡 校正 阈值 掩模 截止 频率 振铃 过冲 分辨率 像素 "
    "实验 结果 分析 对比 方法 模型 参数 消融 指标 结论 问题 误差 视觉 质量 评价".split()
)

_DOC_STYLES = ["作业", "试卷", "板书"]


def _rand_text(rng, n_chars):
    """从词库里随机拼一段中文（长度按字数控制），让每页内容不同。"""
    s = ""
    while len(s) < n_chars:
        s += _PAGE_WORDS[int(rng.integers(len(_PAGE_WORDS)))]
    return s[:n_chars]


def _pil_font(size, bold=False):
    """取中文字体；msyhbd/simhei 优先作为粗体。"""
    cands = [r"C:\Windows\Fonts\msyhbd.ttc", r"C:\Windows\Fonts\simhei.ttf"] if bold \
        else [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simsun.ttc"]
    for p in cands + _FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                idx = 0
                return ImageFont.truetype(p, size, index=idx)
            except Exception:
                continue
    return ImageFont.load_default()


def render_document_page(seed, size=None, style=None):
    """合成一张"干净"文档页（float32 [0,1]）。

    三种版式分别对应作业/试卷/板书：
      作业  —— 白纸黑字，标题 + 分节 + 手写式批注行
      试卷  —— 白纸，含选择题方框、表格、装订线、页码
      板书  —— 深色黑板 + 浅色粉笔字（值域整体反转，考验频域方法的普适性）
    同时叠加纸张纹理（低频渐晕 + 高频细粒噪点），保证"clean"也不是理想阶跃图。
    """
    H = size[0] if size else CFG["H"]
    W = size[1] if size else CFG["W"]
    rng = np.random.default_rng(seed)
    style = style or _DOC_STYLES[seed % len(_DOC_STYLES)]
    board = (style == "板书")

    bg = 42 if board else 255
    fg = 226 if board else 28
    if board:  # 黑板：带轻微色斑
        base = np.full((H, W), bg, np.float32)
        low = rng.normal(0, 10, (H // 24 + 1, W // 24 + 1)).astype(np.float32)
        base += cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC)
        page = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "L")
    else:
        page = Image.new("L", (W, H), bg)

    d = ImageDraw.Draw(page)
    f_title = _pil_font(max(26, H // 30), bold=True)
    f_sec = _pil_font(max(20, H // 42), bold=True)
    f_body = _pil_font(max(16, H // 52))
    f_small = _pil_font(max(12, H // 72))

    # ---- 页眉（把"课程/学号"写上去，制造稳定的短竖笔画，利于角点检测）
    d.line([(40, 46), (W - 40, 46)], fill=fg, width=2)
    d.text((40, 18), "计算摄影 Lab2 · %s · No.%03d" % (style, seed), font=f_small, fill=fg)
    d.text((W - 190, 18), "姓名：同学%d" % (seed % 40), font=f_small, fill=fg)

    # ---- 标题
    title = {"作业": "第3讲 二维DFT 与 频域滤波 作业", "试卷": "数字图像处理 期中测试卷",
             "板书": "板书：卷积定理与频域去噪"}[style]
    d.text((W // 2 - 230, 62), title, font=f_title, fill=fg)
    y = 118

    # ---- 正文：分节 + 若干文字行
    n_sec = 4 if style != "板书" else 5
    for s in range(n_sec):
        if y > H - 240:
            break
        d.text((42, y), "%d.%d  %s" % (s // 3 + 1, s % 3 + 1,
                                       _rand_text(rng, 8)), font=f_sec, fill=fg)
        y += 34
        for _ in range(int(rng.integers(2, 4))):
            if y > H - 240:
                break
            n = int(rng.integers(22, 30))
            d.text((52, y), _rand_text(rng, n), font=f_body, fill=fg)
            y += 26
        y += 6

    # ---- 表格（给规整的横竖直线 —— 频域上的方向性高频，很适合做滤波对比）
    if style != "板书":
        ty = y + 8
        tw, th = W - 150, 26
        cols = [0, 150, 300, 450, tw]
        for r in range(4):
            d.line([(42, ty + r * th), (42 + tw, ty + r * th)], fill=fg, width=2)
        for c in cols:
            d.line([(42 + c, ty), (42 + c, ty + 3 * th)], fill=fg, width=2)
        for r in range(3):
            for c in range(4):
                d.text((50 + cols[c], ty + r * th + 4), _rand_text(rng, 4), font=f_small, fill=fg)
        y = ty + 3 * th + 14

    # ---- 公式区（用 ASCII 近似，制造细密笔画）
    for k in range(2):
        d.text((60, y + k * 24), "F(u,v) = SUM_x SUM_y f(x,y) e^{-j2*pi*(ux/M+vy/N)}",
               font=f_small, fill=fg)
    y += 60

    # ---- 习题/答题框（方框 = 规整直角，Harris 会给出强角点，便于观察重复率）
    if style == "试卷":
        for k in range(3):
            box_y = min(y + k * 34, H - 90)
            d.rectangle([42, box_y, 60, box_y + 24], outline=fg, width=2)
            d.text((72, box_y + 2), _rand_text(rng, int(rng.integers(16, 26))), font=f_small, fill=fg)

    # ---- 页脚：页码 + 印章（圆形 → 天然的"角点/斑点"富集区）
    d.line([(40, H - 52), (W - 40, H - 52)], fill=fg, width=2)
    d.text((W // 2, H - 40), "- %d -" % (seed % 30 + 1), font=f_small, fill=fg)
    if style != "板书":
        cx, cy, r = W - 110, H - 110, 42
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fg, width=3)
        d.text((cx - 30, cy - 8), "已阅 %d" % (seed % 9 + 1), font=f_small, fill=fg)

    img = np.asarray(page).astype(np.float32) / 255.0
    # ---- 纸张纹理：低频渐晕 + 高频细粒（真实拍照纸面从来不是纯白）
    yy = np.linspace(0, 1, H, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, W, dtype=np.float32)[None, :]
    vign = 1.0 - 0.05 * ((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
    img = img * vign + rng.normal(0, 0.004, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


# ---------------------------------------------------------------- 退化算子
def deg_gaussian_noise(img, sigma=0.055, seed=0):
    """加性高斯噪声（传感器读出/热噪声），σ 是相对满量程的比例。"""
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, sigma, img.shape).astype(np.float32), 0, 1)


def deg_salt_pepper(img, ratio=0.02, seed=0):
    """椒盐噪声（坏点/传输错误）：随机把像素打成 0 或 1。"""
    rng = np.random.default_rng(seed)
    out = img.copy()
    m = rng.random(img.shape)
    out[m < ratio / 2] = 0.0
    out[m > 1 - ratio / 2] = 1.0
    return out


def deg_speckle(img, sigma=0.35, seed=0):
    """乘性斑点噪声（相干成像/低照度）：I' = I·(1 + n)。"""
    rng = np.random.default_rng(seed)
    return np.clip(img * (1.0 + rng.normal(0, sigma, img.shape).astype(np.float32)), 0, 1)


def deg_motion_blur(img, length=15, angle=30.0):
    """匀速直线运动模糊：空域是等权线段核 → 频域为 sinc 条带（有周期性零点）。

    频域零点是关键：这些频率的信息被彻底抹掉，所以"低通去噪"救不回来，
    报告里把它作为频域方法的失败案例之一。
    """
    ker = np.zeros((length, length), np.float32)
    ker[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    ker = cv2.warpAffine(ker, M, (length, length))
    ker /= max(ker.sum(), 1e-6)
    return np.clip(cv2.filter2D(img, -1, ker, borderType=cv2.BORDER_REFLECT), 0, 1)


def deg_defocus_blur(img, radius=3):
    """散焦模糊：用圆盘（pillbox）核，频域近似 jinc，高频整体衰减。"""
    r = max(1, int(radius))
    k = np.zeros((2 * r + 1, 2 * r + 1), np.float32)
    cv2.circle(k, (r, r), r, 1.0, -1)
    k /= k.sum()
    return np.clip(cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT), 0, 1)


def deg_uneven_illumination(img, strength=0.55, seed=0, mode=None):
    """光照不匀：乘一个低频函数（灯偏侧/手挡光/暗角）。

    这是**低频乘性**退化 —— 用低通/高通都无法消除（低通会把它一起保留，
    高通会让它变成整体亮度漂移），必须用照度-反射分解或同态滤波。
    本实验用它说明"频域滤波 ≠ 万能"，与 P1 的物理先验呼应。
    """
    H, W = img.shape
    rng = np.random.default_rng(seed)
    mode = mode or ["diag", "radial", "spot"][seed % 3]
    if mode == "diag":
        yy = np.linspace(1.0, 1.0 - strength, H, dtype=np.float32)[:, None]
        xx = np.linspace(1.0, 1.0 - 0.4 * strength, W, dtype=np.float32)[None, :]
        field = yy * xx
    elif mode == "radial":  # 暗角
        yy = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
        xx = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
        r = np.sqrt(yy ** 2 + xx ** 2) / math.sqrt(2)
        field = 1.0 - strength * r ** 2
    else:  # 局部阴影斑（手/身体遮挡）
        low = 1.0 - strength * rng.random((6, 6)).astype(np.float32)
        field = cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC)
    return np.clip(img * np.clip(field, 0.15, 1.2), 0, 1)


def deg_periodic_noise(img, f0=0.055, angle=30.0, amp=0.10, harmonics=2, seed=0):
    """周期性正弦干扰（拍屏幕的摩尔纹 / 网格 / 条纹光源）。

    频域上表现为**若干对孤立亮斑**，正是带阻/陷波滤波器的用武之地：
    低通能压掉但会糊掉文字，陷波只打那几个频点，笔画几乎无损。
    """
    H, W = img.shape
    th = math.radians(angle)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    proj = xx * math.cos(th) + yy * math.sin(th)
    pat = np.zeros_like(proj)
    for h in range(1, harmonics + 1):
        pat += (amp / h) * np.sin(2 * math.pi * f0 * h * proj)
    return np.clip(img + pat, 0, 1)


def deg_jpeg(img, quality=35):
    """JPEG 压缩伪影（8×8 DCT 块效应）：高频被量化丢弃，文字边缘出现方块。"""
    ok, buf = cv2.imencode(".jpg", to_u8(img), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return to_f(cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE))


def geometric_perturb(img, angle=3.0, scale=1.03, dx=12, dy=-8, border=0.0):
    """几何扰动（旋转+缩放+平移）：模拟"同一页再拍一张"的另一视角。

    SIFT 的匹配内点数要比较才有意义，所以必须构造**几何不同**的两张图，
    否则单应退化成恒等，RANSAC 会把全部粗匹配都判为内点，指标失去区分度。
    """
    H, W = img.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    return np.clip(cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=border), 0, 1)


def make_view2(clean, seed):
    """重拍视图：轻微几何变化 + 光照不匀 + 中等高斯噪声 + 轻微散焦。

    它同时是 SIFT 匹配的"目标视图"，也保证匹配任务不是-trivial 的。
    """
    v = geometric_perturb(clean, angle=float(np.random.default_rng(seed).uniform(2, 5)),
                          scale=float(np.random.default_rng(seed + 1).uniform(1.0, 1.05)),
                          dx=int(np.random.default_rng(seed + 2).integers(-18, 18)),
                          dy=int(np.random.default_rng(seed + 3).integers(-18, 18)))
    v = deg_uneven_illumination(v, strength=0.35, seed=seed)
    v = deg_defocus_blur(v, 1)
    v = deg_gaussian_noise(v, 0.025, seed)
    return v


# 数据集变体清单：名字 → 生成函数（lambda 里带不同参数）
def variant_specs(seed):
    """返回 {变体名: 无参可调用} —— 一次定义数据集包含哪些退化。"""
    return {
        "clean": lambda: None,                                   # 干净参考（不退化）
        "noise": lambda: ("deg_gaussian_noise", dict(sigma=0.25, seed=seed)),      # 高斯噪声（强）
        "sp": lambda: ("deg_salt_pepper", dict(ratio=0.02, seed=seed)),            # 椒盐
        "speckle": lambda: ("deg_speckle", dict(sigma=0.35, seed=seed)),           # 乘性斑点
        "motion": lambda: ("deg_motion_blur", dict(length=15, angle=30.0)),        # 运动模糊
        "defocus": lambda: ("deg_defocus_blur", dict(radius=3)),                   # 散焦
        "illum": lambda: ("deg_uneven_illumination", dict(strength=0.55, seed=seed)),  # 光照不匀
        "periodic": lambda: ("deg_periodic_noise", dict(f0=0.055, angle=30.0, amp=0.10, seed=seed)),
        "jpeg": lambda: ("deg_jpeg", dict(quality=30)),                            # 压缩伪影
    }


_DEG_TABLE = {
    "deg_gaussian_noise": deg_gaussian_noise,
    "deg_salt_pepper": deg_salt_pepper,
    "deg_speckle": deg_speckle,
    "deg_motion_blur": deg_motion_blur,
    "deg_defocus_blur": deg_defocus_blur,
    "deg_uneven_illumination": deg_uneven_illumination,
    "deg_periodic_noise": deg_periodic_noise,
    "deg_jpeg": deg_jpeg,
}


def build_dataset(out_dir, n_docs=None, size=None, save_images=True):
    """生成自建数据集，返回 manifest（含每个变体的文件路径）。

    manifest 结构:  {doc_id: {"style":.., "clean":路径, "noise":路径, ..., "view2": 路径}}
    同时把 clean / view2 也落盘，方便核查；报告中的表格都是在这批图上跑的。
    """
    n_docs = n_docs or CFG["N_DOCS"]
    img_dir = os.path.join(out_dir, "dataset")
    ensure_dir(img_dir)
    manifest = {}
    for i in range(n_docs):
        seed = CFG["RNG"] + i * 17
        style = _DOC_STYLES[i % len(_DOC_STYLES)]
        clean = render_document_page(seed, size=size)
        rec = {"style": style, "seed": seed, "shape": list(clean.shape)}
        if save_images:
            p = os.path.join(img_dir, "doc%02d_clean.png" % i)
            imwrite(p, clean)
            rec["clean"] = p
        else:
            rec["clean"] = None
        for vname, spec in variant_specs(seed).items():
            if vname == "clean":
                continue
            fn, kw = spec()
            deg = _DEG_TABLE[fn](clean, **kw)
            rec[vname] = None
            if save_images:
                p = os.path.join(img_dir, "doc%02d_%s.png" % (i, vname))
                imwrite(p, deg)
                rec[vname] = p
            else:
                rec[vname] = deg  # 不落盘时直接把数组放进内存
        v2 = make_view2(clean, seed)
        if save_images:
            p = os.path.join(img_dir, "doc%02d_view2.png" % i)
            imwrite(p, v2)
            rec["view2"] = p
        else:
            rec["view2"] = v2
        manifest["doc%02d" % i] = rec
    meta = dict(n_docs=n_docs, shape=[int(CFG["H"]), int(CFG["W"])],
                strategies=list(variant_specs(0).keys()) + ["view2"],
                styles=_DOC_STYLES, seed=CFG["RNG"])
    with open(os.path.join(img_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, docs={k: v["style"] for k, v in manifest.items()}),
                  f, ensure_ascii=False, indent=2)
    return manifest, meta


def load_dataset(out_dir, n_docs=None):
    """从磁盘读入数据集（不落盘模式则由 build_dataset 直接返回内存数组）。"""
    img_dir = os.path.join(out_dir, "dataset")
    names = sorted(f for f in os.listdir(img_dir) if f.startswith("doc") and f.endswith("_clean.png"))
    if n_docs:
        names = names[:n_docs]
    manifest = {}
    for c in names:
        doc = c.split("_")[0]
        rec = {"clean": imread_gray(os.path.join(img_dir, c))}
        for v in ["noise", "sp", "speckle", "motion", "defocus", "illum", "periodic", "jpeg", "view2"]:
            p = os.path.join(img_dir, "%s_%s.png" % (doc, v))
            rec[v] = imread_gray(p) if os.path.exists(p) else None
        rec["style"] = _DOC_STYLES[int(doc[3:]) % len(_DOC_STYLES)]
        manifest[doc] = rec
    return manifest


# ==============================================================================
# 8. 出图与出表工具
# ==============================================================================
def make_grid(items, path, ncols=4, suptitle=None, cmap="gray", vmin=0, vmax=1,
              cell=(2.75, 3.05), dpi=125, title_size=8, suptitle_size=12):
    """通用图版：items=[(图像, 标题), ...] → 网格图存到 path。

    统一走这个函数是为了让报告里所有对比图版式一致（同一行 = 同一输入，
    同一列 = 同一方法），读者可以横向比效果、纵向比方法。
    """
    n = max(len(items), 1)
    nrows = int(math.ceil(n / float(ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell[0] * ncols, cell[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (im, title) in zip(axes, items):
        if isinstance(im, str):        # 容错：万一 (标题, 图) 写反了，自动交换
            im, title = title, im
        ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=title_size)
        ax.axis("off")
    for ax in axes[len(items):]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_size)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if suptitle else None)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def make_bars(groups, series, path, ylabel, suptitle=None, ylim=None, rotation=20):
    """分组柱状图：groups=组名列表; series={系列名: [每组值]}。"""
    x = np.arange(len(groups))
    k = len(series)
    w = 0.8 / max(k, 1)
    fig, ax = plt.subplots(figsize=(1.35 * len(groups) + 2.2, 4.0))
    for i, (name, vals) in enumerate(series.items()):
        ax.bar(x + (i - (k - 1) / 2.0) * w, vals, w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8, rotation=rotation, ha="right")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3, ls=":")
    ax.legend(fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    if suptitle:
        ax.set_title(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def make_curves(xs, series, path, xlabel, ylabel, suptitle=None, xlog=False):
    """折线/曲线族：xs=x 轴; series={系列名: [y...]}。"""
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for name, ys in series.items():
        ax.plot(xs, ys, marker="o", ms=4, lw=1.6, label=name)
    if xlog:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(alpha=0.3, ls=":")
    ax.legend(fontsize=8)
    if suptitle:
        ax.set_title(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def draw_matches(img_a, img_b, pa, pb, path, title="", max_lines=60, dpi=125):
    """画 SIFT 内点连线图（左 A 右 B，只画 RANSAC 内点）。"""
    H = max(img_a.shape[0], img_b.shape[0])
    Wa, Wb = img_a.shape[1], img_b.shape[1]
    canvas = np.zeros((H, Wa + Wb, 3), np.float32)
    a3 = cv2.cvtColor(to_u8(img_a), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    b3 = cv2.cvtColor(to_u8(img_b), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    canvas[:a3.shape[0], :Wa] = a3
    canvas[:b3.shape[0], Wa:] = b3
    rng = np.random.default_rng(7)
    for (x1, y1), (x2, y2) in list(zip(pa, pb))[:max_lines]:
        c = tuple(float(v) for v in rng.random(3) * 0.6 + 0.4)
        cv2.line(canvas, (int(x1), int(y1)), (int(x2) + Wa, int(y2)), c, 1, cv2.LINE_AA)
        cv2.circle(canvas, (int(x1), int(y1)), 2, c, -1)
        cv2.circle(canvas, (int(x2) + Wa, int(y2)), 2, c, -1)
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.imshow(np.clip(canvas, 0, 1))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def md_table(rows, cols, headers=None, group_col=None):
    """把 dict 列表转成 Markdown 表格（写进报告的就是这张表）。"""
    headers = headers or cols
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                if v != v:  # NaN
                    v = "—"
                elif abs(v) >= 1000:
                    v = "%.0f" % v
                elif abs(v) >= 10:
                    v = "%.2f" % v
                else:
                    v = "%.3f" % v
            cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_csv(rows, cols, path):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (("%.4f" % r[k]) if isinstance(r.get(k), float) else r.get(k, ""))
                        for k in cols})


def summarize(rows, group_keys, metric_keys):
    """把逐图结果行按 group_keys 聚合为 mean/std（报告表格一律给 mean±std）。"""
    buckets = {}
    for r in rows:
        key = tuple(r[k] for k in group_keys)
        buckets.setdefault(key, []).append(r)
    out = []
    for key, rs in buckets.items():
        row = {k: v for k, v in zip(group_keys, key)}
        row["n"] = len(rs)
        for m in metric_keys:
            vals = np.array([r[m] for r in rs if r.get(m) is not None and r[m] == r[m]], dtype=float)
            row[m] = float(vals.mean()) if len(vals) else float("nan")
            row[m + "_std"] = float(vals.std()) if len(vals) else float("nan")
        out.append(row)
    return out


def fmt_pm(row, key, nd=2):
    """mean ± std 字符串（NaN 显示为 —）。"""
    v, s = row.get(key), row.get(key + "_std", 0.0)
    if v is None or v != v:
        return "—"
    return ("%." + str(nd) + "f ± %." + str(nd) + "f") % (v, s if s == s else 0.0)


# ==============================================================================
# 9. 实验
#   统一约定：
#     · 参考图 = 同一页的 clean（PSNR/SSIM/边缘 F1/角点重复率的真值来源）
#     · SIFT 匹配的"另一视图" = 同一页的 view2（含小角度旋转/缩放/平移 + 光照
#       不匀 + 轻微散焦 + 噪声），不这样做的话单应退化成恒等，内点数没有区分度
#     · 所有表格都是"逐图算 → 再对图片求均值±标准差"，不是先平均图再算指标
# ==============================================================================
_SIFT_CACHE = {}

def sift_cached(key, img):
    """缓存 SIFT 结果（clean / view2 在一轮实验里会被反复用到）。"""
    if key not in _SIFT_CACHE:
        _SIFT_CACHE[key] = sift_detect(img)
    return _SIFT_CACHE[key]


def match_to_view2(doc, img, view2, ratio=0.75):
    """把结果图 img 与同一页的另一视图 view2 做 SIFT 匹配（ratio + RANSAC）。"""
    ka, da = sift_detect(img)
    kb, db = sift_cached("v2:" + doc, view2)
    return match_descriptors(ka, da, kb, db, ratio)


def _refs(doc_rec, doc):
    """取一张图的评价基准（干净边缘图、干净 top-K 角点、干净 SIFT 缓存）。"""
    clean = doc_rec["clean"]
    if "gt_edge" not in doc_rec:
        doc_rec["gt_edge"] = canny_edge(clean)
        doc_rec["gt_corner"] = harris_top(clean, CFG["TOPK_CORNER"])
        sift_cached("clean:" + doc, clean)
    return doc_rec["gt_edge"], doc_rec["gt_corner"]


def eval_result(doc, doc_rec, img, name):
    """对一张结果图算全套指标（含对 view2 的 SIFT 匹配）。"""
    gt_edge, gt_corner = _refs(doc_rec, doc)
    row = evaluate(doc_rec["clean"], img, gt_edge=gt_edge, ref_corners=gt_corner,
                   ref_for_sift=None, name=name)
    m = match_to_view2(doc, img, doc_rec["view2"])
    row.update(doc=doc, style=doc_rec["style"], sift_kp=len(m["kps_a"]),
               sift_match=m["n_match"], sift_inlier=m["n_inlier"],
               sift_inlier_ratio=m["inlier_ratio"])
    return row


# ---------------------------------------------------------------- 方法集合
def denoise_methods():
    """E1 用：空域（均值/高斯/中值）与频域（理想/高斯/巴特沃斯低通）去噪方法集。

    D0 的取值不是随手写的：干净的 768×1024 文档页上，一个 2~3 px 宽的笔画
    对应的频率约在 D≈150~250 格（频率格数 = 周期/像素 × 对应维度），
    所以 D0=90 会糊掉笔画，D0=150 是折中区，D0=240 基本只压噪声。
    报告里 E3 的 D0 扫描就是把这套对应关系量化出来。
    """
    return {
        "① 不处理(基线)": lambda im: im,
        "② 空域均值 3×3": lambda im: mean_filter(im, 3),
        "③ 空域均值 5×5": lambda im: mean_filter(im, 5),
        "④ 空域高斯 3×3 σ=0.8": lambda im: gaussian_filter(im, 3, 0.8),
        "⑤ 空域高斯 5×5 σ=1.0": lambda im: gaussian_filter(im, 5, 1.0),
        "⑥ 空域高斯 7×7 σ=1.6": lambda im: gaussian_filter(im, 7, 1.6),
        "⑦ 空域中值 3×3": lambda im: median_filter(im, 3),
        "⑧ 空域中值 5×5": lambda im: median_filter(im, 5),
        "⑨ 频域理想LPF D0=150": lambda im: freq_filter(im, lp_mask(im.shape, 150, "ideal")),
        "⑩ 频域高斯LPF D0=90": lambda im: freq_filter(im, lp_mask(im.shape, 90, "gaussian")),
        "⑪ 频域高斯LPF D0=150": lambda im: freq_filter(im, lp_mask(im.shape, 150, "gaussian")),
        "⑫ 频域高斯LPF D0=240": lambda im: freq_filter(im, lp_mask(im.shape, 240, "gaussian")),
    }


def sharpen_methods():
    """E2 用：空域锐化（拉普拉斯/USM）与频域高通增强（1+β·HP）。"""
    return {
        "① 原始(不锐化)": lambda im: im,
        "② 拉普拉斯锐化 α=0.5": lambda im: laplacian_sharpen(im, 0.5),
        "③ 拉普拉斯锐化 α=1.0": lambda im: laplacian_sharpen(im, 1.0),
        "④ USM 7×7 σ=2.0 a=1.0": lambda im: unsharp_mask(im, 7, 2.0, 1.0),
        "⑤ USM 15×15 σ=4.0 a=1.5": lambda im: unsharp_mask(im, 15, 4.0, 1.5),
        "⑥ 频域高通增强 β=0.5": lambda im: freq_filter(im, 1.0 + 0.5 * hp_mask(im.shape, 20, "gaussian")),
        "⑦ 频域高通增强 β=1.0": lambda im: freq_filter(im, 2.0 * hp_mask(im.shape, 20, "gaussian") + 1e-3),
    }


DEG_ORDER = ["noise", "sp", "speckle", "motion", "defocus", "illum", "periodic", "jpeg"]
DEG_CN = {
    "noise": "高斯噪声 σ=0.25", "sp": "椒盐 2%", "speckle": "乘性斑点 σ=0.35",
    "motion": "运动模糊 L=15/30°", "defocus": "散焦 r=3", "illum": "光照不匀 0.55",
    "periodic": "周期条纹 0.055@30°", "jpeg": "JPEG q=30", "clean": "干净参考",
}


# ---------------------------------------------------------------- E1
def exp1_denoise(ds, out_dir, figs, degs=("noise", "sp", "speckle", "motion")):
    """E1：空域 vs 频域去噪（含 PSNR/SSIM/边缘F1/角点重复率/SIFT内点）。

    预期与解释（报告里逐条对应）：
      · 加性高斯噪声：线性滤波（均值/高斯/频域高斯 LPF）有效，且"越平滑越
        去噪但越糊"——PSNR 峰值往往出现在中等强度处；
      · 椒盐噪声：空域中值 >> 频域低通。中值在排序域取中间值，能把孤立极值
        整颗剔掉；而频域低通是线性加权和，脉冲会被"抹开"成一片涟漪，
        所以 PSNR 提升有限、边缘 F1 反而被拖低；
      · 运动模糊：低通只会更糊。它的信息损失在频域是**零点**，不是"高频太强"，
        所以任何低通都无法恢复（这是频域方法最典型的失败）。
    """
    rows, fig_items = [], {}
    methods = denoise_methods()
    for deg in degs:
        for doc, rec in ds.items():
            if rec.get(deg) is None:
                continue
            for mname, fn in methods.items():
                with Timer() as t:
                    out = fn(rec[deg])
                row = eval_result(doc, rec, out, mname)
                row.update(deg=deg, deg_cn=DEG_CN[deg], method=mname, time_ms=t.ms)
                rows.append(row)
                if doc == "doc00":
                    fig_items.setdefault(deg, []).append(
                        (out, "%s\nPSNR %.2f  SSIM %.3f" % (mname, row["psnr"], row["ssim"])))
        print("  [E1] %s 完成" % DEG_CN[deg])

    metrics = ["psnr", "ssim", "edge_f1", "n_edge", "corner_repeat", "sift_inlier", "time_ms"]
    agg = summarize(rows, ["deg", "deg_cn", "method"], metrics)

    # 图 3：高斯噪声与椒盐下的各方法结果网格
    for deg in degs:
        items = fig_items.get(deg)
        if not items:
            continue
        src = ds["doc00"].get(deg)
        make_grid([(src, "输入：%s" % DEG_CN[deg])] + items,
                  os.path.join(figs, "fig03_denoise_grid_%s.png" % deg),
                  ncols=4, suptitle="E1 空域 vs 频域去噪（doc00 · %s）" % DEG_CN[deg])

    # 图 4：指标柱状图（按退化分组，只挑代表性方法，避免柱子太密）
    pick = ["① 不处理(基线)", "③ 空域均值 5×5", "⑤ 空域高斯 5×5 σ=1.0",
            "⑥ 空域高斯 7×7 σ=1.6", "⑧ 空域中值 5×5",
            "⑩ 频域高斯LPF D0=90", "⑪ 频域高斯LPF D0=150", "⑫ 频域高斯LPF D0=240"]
    for metric, path, ylab in [("psnr", "fig04a_denoise_psnr.png", "PSNR (dB)"),
                               ("ssim", "fig04b_denoise_ssim.png", "SSIM"),
                               ("edge_f1", "fig04c_denoise_edgef1.png", "边缘 F1"),
                               ("sift_inlier", "fig04d_denoise_sift.png", "SIFT 内点数")]:
        series = {}
        groups = [DEG_CN[d] for d in degs]
        for m in pick:
            series[m] = [next((r[metric] for r in agg if r["deg"] == d and r["method"] == m),
                              float("nan")) for d in degs]
        make_bars(groups, series, os.path.join(figs, path), ylab,
                  suptitle="E1 去噪指标对比（%s，24 张平均）" % metric)
    return rows, agg


# ---------------------------------------------------------------- E2
def exp2_sharpen(ds, out_dir, figs, degs=("motion", "defocus", "noise")):
    """E2：锐化前后 边缘数 / 角点数 / SIFT 内点数的变化。

    核心现象：锐化把高频抬起来，边缘数和角点数**单调上升**，但 PSNR/SSIM
    同时下降；如果输入本身有噪声，拉普拉斯会把噪声也放大（α 越大越明显），
    于是"边缘更多"并不等于"信息更多"——所以要看几何指标（角点重复率、
    SIFT 内点）而不只是数量。
    """
    rows, fig_items = [], {}
    methods = sharpen_methods()
    for deg in degs:
        for doc, rec in ds.items():
            if rec.get(deg) is None:
                continue
            for mname, fn in methods.items():
                with Timer() as t:
                    out = fn(rec[deg])
                row = eval_result(doc, rec, out, mname)
                row.update(deg=deg, deg_cn=DEG_CN[deg], method=mname, time_ms=t.ms)
                rows.append(row)
                if doc == "doc00":
                    e = canny_edge(out)
                    cpts, _ = harris_corner(out)
                    vis = cv2.cvtColor(to_u8(out), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
                    for x, y in cpts[:400]:
                        cv2.circle(vis, (int(x), int(y)), 2, (1, 0.2, 0.2), -1)
                    fig_items.setdefault(deg, []).extend([
                        (out, "%s\nPSNR %.2f 边缘 %d" % (mname, row["psnr"], row["n_edge"])),
                        (e, "%s\n边缘图（Canny）" % mname),
                        (vis, "%s\nHarris 角点 %d 个" % (mname, row["n_corner"])),
                    ])
        print("  [E2] %s 完成" % DEG_CN[deg])

    metrics = ["psnr", "ssim", "n_edge", "edge_f1", "n_corner", "corner_repeat", "sift_inlier"]
    agg = summarize(rows, ["deg", "deg_cn", "method"], metrics)
    for deg in degs:
        items = fig_items.get(deg)
        if items:
            make_grid([(ds["doc00"][deg], "输入：%s" % DEG_CN[deg])] + items,
                      os.path.join(figs, "fig07_sharpen_%s.png" % deg),
                      ncols=4, suptitle="E2 锐化前后：边缘与角点（doc00 · %s）" % DEG_CN[deg])
    # 指标趋势柱状：边缘数 / 角点数 / SIFT 内点 / PSNR
    for metric, path, ylab in [("n_edge", "fig07b_sharpen_nedge.png", "边缘像素数"),
                               ("n_corner", "fig07c_sharpen_ncorner.png", "Harris 角点数"),
                               ("sift_inlier", "fig07d_sharpen_sift.png", "SIFT 内点数"),
                               ("psnr", "fig07e_sharpen_psnr.png", "PSNR (dB)")]:
        series, groups = {}, [DEG_CN[d] for d in degs]
        for m in methods:
            series[m] = [next((r[metric] for r in agg if r["deg"] == d and r["method"] == m), float("nan"))
                         for d in degs]
        make_bars(groups, series, os.path.join(figs, path), ylab,
                  suptitle="E2 锐化对%s的影响" % ylab)
    return rows, agg


# ---------------------------------------------------------------- E3 消融 A
def exp3_cutoff(ds, out_dir, figs, deg="noise", d0_list=(30, 60, 110, 170, 250, 350, 480)):
    """消融 A：截止频率 D0 的影响（滤波器形状 × D0 网格）。

    预期与实测（报告里逐条对照）：
      · D0 太小（30~60）→ 笔画被当高频切掉，边缘 F1 与 SIFT 内点骤降，
        但 SSIM 反而很高（因为 SSIM 奖励"平滑"，糊成一片也算结构相似）；
      · D0 太大（380~500）→ 噪声几乎原样保留，PSNR 高但 SSIM 低；
      · 存在折中区，而且**不同指标的最优 D0 不同**：PSNR 偏好大 D0（保守，
        少动细节），SSIM 偏好小 D0（激进，平滑优先），边缘 F1 / SIFT 内点
        居中。这一条是本次实验最有价值的发现，说明"去噪最优"必须先声明
        用哪个指标、面向哪一步下游任务。
    """
    rows = []
    for kind, order in [("ideal", 2), ("gaussian", 2), ("butterworth", 2)]:
        for d0 in d0_list:
            H = None
            ring = float("nan")
            for doc, rec in ds.items():
                x = rec.get(deg)
                if x is None:
                    continue
                if H is None:                       # 同一 D0/形状下掩模只算一次
                    H = lp_mask(x.shape, d0, kind, order)
                    ring = ringing_index(H)         # 振铃指数只跟 H 有关
                out = freq_filter(x, H)
                row = eval_result(doc, rec, out, "%s D0=%d" % (kind, d0))
                row.update(deg=deg, kind=kind, D0=d0, ringing=ring)
                rows.append(row)
    agg = summarize(rows, ["kind", "D0"], ["psnr", "ssim", "edge_f1", "sift_inlier",
                                          "corner_repeat", "ringing"])
    for metric, ylab in [("psnr", "PSNR (dB)"), ("ssim", "SSIM"), ("edge_f1", "边缘 F1"),
                         ("sift_inlier", "SIFT 内点数"), ("ringing", "振铃像素占比"), ("corner_repeat", "角点重复率")]:
        series = {}
        for kind in ["ideal", "gaussian", "butterworth"]:
            rs = sorted([r for r in agg if r["kind"] == kind], key=lambda r: r["D0"])
            series[{"ideal": "理想LPF", "gaussian": "高斯LPF", "butterworth": "巴特沃斯 n=2"}[kind]] = \
                [r[metric] for r in rs]
        make_curves(list(d0_list), series, os.path.join(figs, "fig05_cutoff_%s.png" % metric),
                    "截止频率 D0（频率格数）", ylab,
                    suptitle="消融A 截止频率扫描：%s（%s，24 张平均）" % (ylab, DEG_CN[deg]))
    # 视觉对照：同一张图在不同 D0 下的结果
    rec0 = ds["doc00"]
    gt_edge, _ = _refs(rec0, "doc00")      # 确保干净边缘图已算好
    items = [(rec0[deg], "输入：%s" % DEG_CN[deg])]
    for d0 in [30, 60, 120, 240, 400]:
        out = freq_filter(rec0[deg], lp_mask(rec0[deg].shape, d0, "gaussian"))
        items.append((out, "高斯LPF D0=%d\nPSNR %.2f 边缘F1 %.3f" % (
            d0, psnr(rec0["clean"], out), edge_prf(canny_edge(out), gt_edge)[2])))
    for d0 in [60, 150, 400]:
        out = freq_filter(rec0[deg], lp_mask(rec0[deg].shape, d0, "ideal"))
        items.append((out, "理想LPF D0=%d\n振铃指数 %.3f" % (d0, ringing_index(lp_mask(rec0[deg].shape, d0, "ideal")))))
    make_grid(items, os.path.join(figs, "fig05b_cutoff_visual.png"), ncols=5,
              suptitle="消融A 视觉对照：D0 与滤波器形状（doc00 · %s）" % DEG_CN[deg])
    return rows, agg


# ---------------------------------------------------------------- E4 消融 B
def exp4_filter_type(ds, out_dir, figs, deg="noise", D0=150):
    """消融 B：滤波器类型（理想 / 高斯 / 巴特沃斯阶数 / 带阻 / 高通组合）。

    预期：三种低通在 PSNR/SSIM 上差距不大，但**振铃指数**差异明显——理想
    LPF 的硬截断在空域等于乘 jinc 型核（点扩散函数带一圈圈正负旁瓣），
    文字边缘两侧出现亮暗条纹；高斯核的频谱也是高斯（唯一自傅里叶的函数族），
    过渡带最平滑、几乎没有负旁瓣。工程结论：文档图像优先用高斯或低阶
    巴特沃斯；只有需要极陡的频带切割（打掉条纹/摩尔纹）时才用理想型，
    而且那时要接受振铃代价（可用"理想带阻 + 事后中值"折中）。
    """
    cands = [
        ("理想LPF", lambda im: freq_filter(im, lp_mask(im.shape, D0, "ideal"))),
        ("高斯LPF", lambda im: freq_filter(im, lp_mask(im.shape, D0, "gaussian"))),
        ("巴特沃斯 n=1", lambda im: freq_filter(im, lp_mask(im.shape, D0, "butterworth", 1))),
        ("巴特沃斯 n=2", lambda im: freq_filter(im, lp_mask(im.shape, D0, "butterworth", 2))),
        ("巴特沃斯 n=4", lambda im: freq_filter(im, lp_mask(im.shape, D0, "butterworth", 4))),
        ("巴特沃斯 n=8", lambda im: freq_filter(im, lp_mask(im.shape, D0, "butterworth", 8))),
        ("带阻 D0=150 w=30", lambda im: freq_filter(im, band_stop_mask(im.shape, 150, 30, "gaussian"))),
        ("理想带阻 D0=150 w=30", lambda im: freq_filter(im, band_stop_mask(im.shape, 150, 30, "ideal"))),
        ("1-低通(纯高通)", lambda im: freq_filter(im, hp_mask(im.shape, D0, "gaussian"))),
        ("低通+0.6×高通(带通)", lambda im: freq_filter(
            im, lp_mask(im.shape, D0, "gaussian") * (1.0 + 0.6 * hp_mask(im.shape, 20, "gaussian")))),
    ]
    # 与上面一一对应的"掩模生成函数"：振铃指数是滤波器的固有属性，只需 H 就能算
    mask_of = {
        "理想LPF": lambda sh: lp_mask(sh, D0, "ideal"),
        "高斯LPF": lambda sh: lp_mask(sh, D0, "gaussian"),
        "巴特沃斯 n=1": lambda sh: lp_mask(sh, D0, "butterworth", 1),
        "巴特沃斯 n=2": lambda sh: lp_mask(sh, D0, "butterworth", 2),
        "巴特沃斯 n=4": lambda sh: lp_mask(sh, D0, "butterworth", 4),
        "巴特沃斯 n=8": lambda sh: lp_mask(sh, D0, "butterworth", 8),
        "带阻 D0=150 w=30": lambda sh: band_stop_mask(sh, 150, 30, "gaussian"),
        "理想带阻 D0=150 w=30": lambda sh: band_stop_mask(sh, 150, 30, "ideal"),
        "1-低通(纯高通)": lambda sh: hp_mask(sh, D0, "gaussian"),
        "低通+0.6×高通(带通)": lambda sh: lp_mask(sh, D0, "gaussian") * (1.0 + 0.6 * hp_mask(sh, 20, "gaussian")),
    }
    rows, items, masks = [], [], []
    for name, fn in cands:
        ring = ringing_index(mask_of[name](ds["doc00"][deg].shape))
        for doc, rec in ds.items():
            x = rec.get(deg)
            if x is None:
                continue
            out = fn(x)
            row = eval_result(doc, rec, out, name)
            row.update(deg=deg, method=name, ringing=ring)
            rows.append(row)
        out0 = fn(ds["doc00"][deg])
        items.append((out0, "%s\nPSNR %.2f 振铃指数 %.3f" % (name, psnr(ds["doc00"]["clean"], out0), ring)))
    # 掩模与结果并排 + 局部放大（看振铃）
    H, W = ds["doc00"][deg].shape
    zoom = (slice(120, 300), slice(60, 320))
    for name, _ in cands[:6]:
        masks.append((mask_of[name]((H, W)), "H(u,v)：%s" % name))
    make_grid([(ds["doc00"][deg], "输入：%s" % DEG_CN[deg])] + items,
              os.path.join(figs, "fig06_filter_type.png"), ncols=4,
              suptitle="消融B 滤波器类型对比（doc00 · %s · D0=%d）" % (DEG_CN[deg], D0))
    make_grid(masks + [(np.clip(ds["doc00"]["clean"][zoom], 0, 1), "局部放大区：干净图"),
                       (np.clip(freq_filter(ds["doc00"][deg], lp_mask((H, W), D0, "ideal"))[zoom], 0, 1),
                        "理想LPF 放大（振铃）"),
                       (np.clip(freq_filter(ds["doc00"][deg], lp_mask((H, W), D0, "gaussian"))[zoom], 0, 1),
                        "高斯LPF 放大（无振铃）")],
              os.path.join(figs, "fig06b_ringing_masks.png"), ncols=5,
              suptitle="消融B 滤波器响应 H(u,v) 与振铃局部放大")
    agg = summarize(rows, ["method"], ["psnr", "ssim", "edge_f1", "sift_inlier", "ringing", "corner_repeat"])
    return rows, agg


# ---------------------------------------------------------------- E5 消融 C
def exp5_order(ds, out_dir, figs):
    """消融 C：去噪 → 特征 的顺序（本实验最有工程价值的一组）。

    输入固定为"退化=运动模糊+高斯噪声"的合成拍照图（贴近真实拍糊 + 噪点）。
    pipeline 设计：
      ① 直接提特征            基线
      ② 仅锐化                边缘多但噪声被放大
      ③ 仅去噪                干净但笔画变软
      ④ 锐化→去噪             先放大噪声再糊掉，最差
      ⑤ 去噪→锐化             先抑噪再抬高频，通常最好（尤其看 SIFT 内点）
      ⑥ 去噪→锐化→轻度再滤波   收尾抑制锐化引入的过冲
      ⑦ 去噪→USM              USM 不引入二阶过冲，通常比拉普拉斯温和
    评价：不只比 PSNR，重点比**边缘 F1 / 角点重复率 / SIFT 内点数**，
    因为下游 OCR / 拼接用的是特征，而不是像素误差。
    """
    lpf = lambda im, d0=40: freq_filter(im, lp_mask(im.shape, d0, "gaussian"))

    def _lap(im, a=0.8):
        return laplacian_sharpen(im, a)

    pipes = {
        "① 直接特征(不去噪不锐化)": lambda im: im,
        "② 仅锐化(拉普拉斯 α=0.8)": _lap,
        "③ 仅去噪(高斯LPF D0=40)": lpf,
        "④ 先锐化→后去噪": lambda im: lpf(_lap(im)),
        "⑤ 先去噪→后锐化": lambda im: _lap(lpf(im)),
        "⑥ 去噪→锐化→轻度再滤波": lambda im: lpf(_lap(lpf(im)), 70),
        "⑦ 去噪→USM": lambda im: unsharp_mask(lpf(im), 15, 4.0, 1.2),
        "⑧ 去噪→双边风格(中值+USM)": lambda im: unsharp_mask(median_filter(lpf(im), 3), 9, 3.0, 1.0),
    }
    rows, match_vis = [], {}
    for name, fn in pipes.items():
        for doc, rec in ds.items():
            base = rec.get("motion")
            if base is None:
                continue
            # 在运动模糊之上再叠一层高斯噪声，模拟"手抖 + 高 ISO"
            seed = CFG["RNG"] + int(doc[3:]) * 17
            x = deg_gaussian_noise(base, 0.03, seed)
            out = fn(x)
            row = eval_result(doc, rec, out, name)
            row.update(deg="motion+noise", method=name)
            rows.append(row)
            if doc == "doc00":
                m = match_to_view2(doc, out, rec["view2"])
                match_vis[name] = (out, m)
    # 出图：各流水线的匹配内点连线（挑 4 条最有代表性的）
    for i, name in enumerate(["① 直接特征(不去噪不锐化)", "② 仅锐化(拉普拉斯 α=0.8)",
                              "③ 仅去噪(高斯LPF D0=40)", "⑤ 先去噪→后锐化"]):
        out, m = match_vis[name]
        draw_matches(out, ds["doc00"]["view2"], np.asarray(m["pa"]), np.asarray(m["pb"]),
                     os.path.join(figs, "fig08_order_%d.png" % i),
                     title="消融C 顺序：%s（粗匹配 %d，内点 %d）" % (name, m["n_match"], m["n_inlier"]))
    agg = summarize(rows, ["method"], ["psnr", "ssim", "edge_f1", "n_edge", "n_corner",
                                       "corner_repeat", "sift_match", "sift_inlier"])
    for metric, path, ylab in [("sift_inlier", "fig08b_order_sift.png", "SIFT 内点数"),
                               ("edge_f1", "fig08c_order_edgef1.png", "边缘 F1"),
                               ("corner_repeat", "fig08d_order_corner.png", "角点重复率"),
                               ("psnr", "fig08e_order_psnr.png", "PSNR (dB)")]:
        make_bars([r["method"] for r in agg], {ylab: [r[metric] for r in agg]},
                  os.path.join(figs, path), ylab,
                  suptitle="消融C 去噪→特征顺序（运动模糊+噪声，24 张平均）", rotation=25)
    return rows, agg


# ---------------------------------------------------------------- E6 周期噪声
def exp6_periodic(ds, out_dir, figs):
    """E6（附加）：周期条纹（拍屏摩尔纹/网格）—— 低通 vs 带阻 vs 陷波。

    三者的取舍：低通能压条纹但连带砍掉文字高频；带阻按半径切一条环带；
    陷波先用 find_spectrum_peaks 把频谱上的孤立亮斑找出来，只打那几个频点，
    笔画几乎无损。这就是"先看频谱、再定点设计"的完整演示。
    """
    rows = []
    auto_spots = None
    for doc, rec in ds.items():
        x = rec.get("periodic")
        if x is None:
            continue
        if doc == "doc00":
            auto_spots = find_spectrum_peaks(x, center_exclude=10, topk=4, min_dist=12)
        spots = auto_spots or [(42, 24), (-42, -24)]
        cands = [
            ("① 不处理", lambda im: im,
             lambda sh: np.ones(sh)),
            ("② 高斯LPF D0=35", lambda im: freq_filter(im, lp_mask(im.shape, 35, "gaussian")),
             lambda sh: lp_mask(sh, 35, "gaussian")),
            ("③ 高斯LPF D0=120", lambda im: freq_filter(im, lp_mask(im.shape, 120, "gaussian")),
             lambda sh: lp_mask(sh, 120, "gaussian")),
            ("④ 高斯带阻 D0=46 w=12", lambda im: freq_filter(im, band_stop_mask(im.shape, 46, 12, "gaussian")),
             lambda sh: band_stop_mask(sh, 46, 12, "gaussian")),
            ("⑤ 理想带阻 D0=46 w=12", lambda im: freq_filter(im, band_stop_mask(im.shape, 46, 12, "ideal")),
             lambda sh: band_stop_mask(sh, 46, 12, "ideal")),
            ("⑥ 陷波(自动检测亮斑 r=10)", lambda im: freq_filter(im, notch_mask(im.shape, spots, 10, "gaussian")),
             lambda sh: notch_mask(sh, spots, 10, "gaussian")),
        ]
        for name, fn, mfn in cands:
            out = fn(x)
            row = eval_result(doc, rec, out, name)
            row.update(deg="periodic", method=name, ringing=ringing_index(mfn(x.shape)))
            rows.append(row)
    agg = summarize(rows, ["method"], ["psnr", "ssim", "edge_f1", "sift_inlier", "corner_repeat"])
    # 图 9：频谱 + 亮斑标记 + 各方法结果
    x = ds["doc00"]["periodic"]
    spec = spectrum_mag(x)
    Hs = [(spec, "输入频谱(log)")]
    if auto_spots:
        Hs.append((spec, "自动检测到的亮斑：%s" % str(auto_spots)))
    masks = [(band_stop_mask(x.shape, 46, 12, "gaussian"), "带阻 H(u,v) D0=46 w=12"),
             (notch_mask(x.shape, auto_spots or [(28, 37)], 10, "gaussian"), "陷波 H(u,v) r=10")]
    outs = [(ds["doc00"]["clean"], "干净参考"),
            (x, "输入：周期条纹"),
            (freq_filter(x, lp_mask(x.shape, 35, "gaussian")), "高斯LPF D0=35（糊）"),
            (freq_filter(x, band_stop_mask(x.shape, 46, 12, "gaussian")), "带阻 D0=46 w=12"),
            (freq_filter(x, notch_mask(x.shape, auto_spots or [(28, 37)], 10, "gaussian")), "陷波（亮斑定点）")]
    make_grid(Hs + masks + outs, os.path.join(figs, "fig09_periodic_notch.png"), ncols=5,
              suptitle="E6 周期条纹：频谱亮斑检测 → 带阻/陷波 vs 低通")
    return rows, agg


# ---------------------------------------------------------------- E7 失败案例
def exp7_failures(ds, out_dir, figs):
    """E7：失败案例分析（报告专章）。

    四个案例都坚持"给现象 + 给频域解释 + 给补救方向"：
      F1 理想LPF 截止过低      → 笔画断裂，边缘 F1 / SIFT 崩；解释：文字墨迹的
                                 能量集中在高频（2~3 px 笔画），硬切等于删掉笔画
      F2 频域低通去椒盐        → 脉冲被线性加权"抹开"成涟漪，反而制造伪边缘
      F3 光照不匀              → 乘性低频，任何低通/高通都无效；补救是先做
                                 低频照度归一化（呼应 P1 的物理先验）
      F4 运动模糊              → 频谱有方向性零点，信息已被销毁，只能靠反卷积
      F5 JPEG 块效应           → 8×8 块边界是规则高频，低通"压掉了噪点也压掉了笔画"，
                                 对块效应几乎无效
    """
    rec = ds["doc00"]
    cases = []

    # F1 截止过低
    x = rec["noise"]
    for d0 in (15, 45):
        out = freq_filter(x, lp_mask(x.shape, d0, "ideal"))
        e = canny_edge(out)
        cases.append(("F1 理想LPF D0=%d（切掉笔画）" % d0, out, e, rec))

    # F2 频域低通去椒盐 vs 中值
    x = rec["sp"]
    cases.append(("F2a 椒盐：频域高斯LPF D0=120", freq_filter(x, lp_mask(x.shape, 120, "gaussian")),
                  canny_edge(freq_filter(x, lp_mask(x.shape, 120, "gaussian"))), rec))
    cases.append(("F2b 椒盐：空域中值 3×3（正解）", median_filter(x, 3), canny_edge(median_filter(x, 3)), rec))

    # F3 光照不匀 + 照度归一化补救
    x = rec["illum"]
    norm = np.clip(x / np.clip(cv2.GaussianBlur(x, (0, 0), 25), 1e-3, 1), 0, 1)
    cases.append(("F3a 光照不匀：频域高斯LPF D0=60（无效）",
                  freq_filter(x, lp_mask(x.shape, 60, "gaussian")),
                  canny_edge(freq_filter(x, lp_mask(x.shape, 60, "gaussian"))), rec))
    cases.append(("F3b 补救：低频照度归一化 + LPF D0=150", freq_filter(norm, lp_mask(norm.shape, 150, "gaussian")),
                  canny_edge(freq_filter(norm, lp_mask(norm.shape, 150, "gaussian"))), rec))
    # F4 运动模糊：频谱零点
    x = rec["motion"]
    cases.append(("F4a 运动模糊：低通 D0=60（更糊）", freq_filter(x, lp_mask(x.shape, 60, "gaussian")),
                  canny_edge(freq_filter(x, lp_mask(x.shape, 60, "gaussian"))), rec))
    cases.append(("F4b 运动模糊：仅锐化 USM a=1.2", unsharp_mask(x, 15, 4.0, 1.2),
                  canny_edge(unsharp_mask(x, 15, 4.0, 1.2)), rec))
    # F5 JPEG 块效应
    x = rec["jpeg"]
    cases.append(("F5a JPEG q=30：频域LPF D0=120", freq_filter(x, lp_mask(x.shape, 120, "gaussian")),
                  canny_edge(freq_filter(x, lp_mask(x.shape, 120, "gaussian"))), rec))
    cases.append(("F5b JPEG q=30：中值 3×3（部分去块）", median_filter(x, 3), canny_edge(median_filter(x, 3)), rec))

    items, rows = [], []
    for title, out, e, r in cases:
        row = eval_result("doc00", r, out, title)
        rows.append(row)
        items.append((out, "%s\nPSNR %.2f 边缘F1 %.3f" % (title, row["psnr"], row["edge_f1"])))
        items.append((e, "边缘图（%s）" % title))
    # 运动模糊频谱 + 输入原图放前面
    items = [(rec["motion"], "输入：运动模糊 L=15/30°"),
             (spectrum_mag(rec["motion"]), "运动模糊频谱：可见方向性零点带"),
             (rec["illum"], "输入：光照不匀"),
             (rec["jpeg"], "输入：JPEG q=30")] + items
    make_grid(items, os.path.join(figs, "fig10_failure_cases.png"), ncols=4,
              suptitle="E7 失败案例：频域滤波在文档图像上的边界（doc00）")
    return rows


# ==============================================================================
# 10. 频域基础验证 + 频谱/数据集总览图 + SIFT 匹配可视化
# ==============================================================================
def dft_selftest():
    """自检：① 自实现矩阵法 DFT 与 numpy FFT 是否一致；② 正逆变换是否可还原。

    报告里用它证明"频域滤波的实现是可信的"——两条路径的相对误差在 1e-12
    量级（浮点精度极限），说明后面的所有频域结论与实现无关，只与滤波器设计有关。
    """
    rng = np.random.default_rng(0)
    f = rng.random((32, 48)).astype(np.float64)
    Fd = dft2d_direct(f)                                    # 自实现：未中心化
    Ff = np.fft.fft2(f)                                     # numpy：未中心化
    err_fwd = float(np.max(np.abs(Fd - Ff)) / np.max(np.abs(Ff)))
    back = np.real(idft2d_direct(Fd))                       # 正逆往返
    err_rt = float(np.max(np.abs(back - f)))
    # 频移公式验证：空间域 (-1)^(x+y) 调制 == 频域中心化（零频搬到中心）
    mod = (-1.0) ** (np.arange(f.shape[0])[:, None] + np.arange(f.shape[1])[None, :])
    shift_formula = dft2d_direct(f * mod)
    F_center = np.fft.fftshift(Ff)
    shift_ok = float(np.max(np.abs(shift_formula - F_center)) / np.max(np.abs(F_center)))
    # 可分离性验证：先列后行的一维变换 == 二维一次变换（同一件事的两种写法）
    sep = dft_matrix(f.shape[0]) @ f @ dft_matrix(f.shape[1]).T
    err_sep = float(np.max(np.abs(sep - Fd)) / np.max(np.abs(Fd)))
    # 中心化 FFT 与中心化自实现 DFT 的一致性（fft2c 的正确性）
    err_center = float(np.max(np.abs(fft2c(f) - F_center)) / np.max(np.abs(F_center)))
    out = dict(rel_err_direct_vs_fft=err_fwd, round_trip_max_abs_err=err_rt,
               rel_err_fftshift_formula=shift_ok, rel_err_separability=err_sep,
               rel_err_fft2c=err_center, shape=list(f.shape))
    print("[自检] 矩阵法DFT vs numpy FFT 相对误差 = %.3e；DFT↔IDFT 往返最大误差 = %.3e"
          % (err_fwd, err_rt))
    print("[自检] 频移公式(-1)^(x+y) 相对误差 = %.3e；可分离性验证相对误差 = %.3e；fft2c 一致性 = %.3e"
          % (shift_ok, err_sep, err_center))
    return out


def fig_dataset_overview(ds, figs):
    """图 1：数据集总览 —— 三种版式的干净页 + 一张页的 8 类退化。"""
    docs = list(ds.keys())
    picks = {s: next((d for d in docs if ds[d]["style"] == s), docs[0]) for s in _DOC_STYLES}
    items = [(ds[picks[s]]["clean"], "干净页（%s）" % s) for s in _DOC_STYLES]
    d0 = docs[0]
    for v in DEG_ORDER:
        if ds[d0].get(v) is not None:
            items.append((ds[d0][v], "退化：%s" % DEG_CN[v]))
    items.append((ds[d0]["view2"], "重拍视图 view2（旋转/缩放 + 光照不匀 + 噪声）"))
    make_grid(items, os.path.join(figs, "fig01_dataset_overview.png"), ncols=4,
              suptitle="自建文档图像数据集（%d 页 × 9 变体 = %d 张）" % (len(docs), len(docs) * 9))


def fig_spectrum_analysis(ds, figs, doc="doc00"):
    """图 2：频谱分析 —— 不同退化在频域长什么样，以及四类滤波器的 H(u,v)。

    这是把"空域现象"和"频域设计"连起来的一张图：
      高斯噪声 → 频谱整体抬升、白噪声的"雪花"铺满全平面（各频率同量级）
      模糊     → 高频被压制，频谱向中心收缩，暗环/零点带
      光照不匀 → 只在极低频（中心）出现巨大能量，直流分量异常突出
      周期条纹 → 中心之外出现对称的孤立亮斑（陷波/带阻的靶点）
    """
    rec = ds[doc]
    H, W = rec["clean"].shape
    items = [
        (rec["clean"], "干净页：能量集中在文字笔画的高频"),
        (spectrum_mag(rec["clean"]), "干净页频谱（log）"),
        (rec["noise"], "退化：%s" % DEG_CN["noise"]),
        (spectrum_mag(rec["noise"]), "噪声频谱：全平面抬升（白噪声）"),
        (rec["motion"], "退化：%s" % DEG_CN["motion"]),
        (spectrum_mag(rec["motion"]), "模糊频谱：高频衰减 + 方向性零点带"),
        (rec["illum"], "退化：%s" % DEG_CN["illum"]),
        (spectrum_mag(rec["illum"]), "光照不匀频谱：能量全挤在中心低频"),
        (rec["periodic"], "退化：%s" % DEG_CN["periodic"]),
        (spectrum_mag(rec["periodic"]), "周期条纹频谱：对称孤立亮斑 ← 陷波靶点"),
        (lp_mask((H, W), 150, "gaussian"), "低通 H(u,v) 高斯 D0=150"),
        (hp_mask((H, W), 150, "gaussian"), "高通 H(u,v) 高斯 D0=150"),
        (band_stop_mask((H, W), 46, 12, "gaussian"), "带阻 H(u,v) D0=46 w=12"),
        (notch_mask((H, W), find_spectrum_peaks(rec["periodic"], 12, 3, 14) or [(28, 37)], 10),
         "陷波 H(u,v)（亮斑处挖坑）"),
    ]
    make_grid(items, os.path.join(figs, "fig02_spectrum_analysis.png"), ncols=4,
              suptitle="频域视角：退化的频谱特征 与 四类滤波器响应 H(u,v)（%s）" % doc)


def fig_sift_detail(ds, figs, doc="doc00"):
    """图 11：SIFT 匹配细节（干净页 ↔ 重拍视图），只画 RANSAC 内点。"""
    rec = ds[doc]
    m = match_to_view2(doc, rec["clean"], rec["view2"])
    draw_matches(rec["clean"], rec["view2"], np.asarray(m["pa"]), np.asarray(m["pb"]),
                 os.path.join(figs, "fig11_sift_matching.png"),
                 title="SIFT 匹配：clean ↔ view2（关键点 %d/%d，粗匹配 %d，RANSAC 内点 %d，内点率 %.2f）"
                       % (m["n_ka"], m["n_kb"], m["n_match"], m["n_inlier"], m["inlier_ratio"]))
    return m


# ==============================================================================
# 11. 汇总成表（CSV + Markdown）
# ==============================================================================
def agg_md(agg, group_keys, headers, metrics, nd=2):
    """把聚合结果渲染成 Markdown 表（mean ± std）。"""
    cols = list(group_keys) + list(metrics)
    rows = []
    for r in agg:
        d = {k: r[k] for k in group_keys}
        for m in metrics:
            d[m] = fmt_pm(r, m, nd if m in ("psnr", "ssim", "edge_f1", "corner_repeat",
                                            "sift_inlier_ratio") else 0)
        rows.append(d)
    return md_table(rows, cols, headers=[headers.get(c, c) for c in cols])


def write_tables(res, out_dir, figs):
    """所有实验结果 → CSV（原始 + 聚合）+ Markdown（可直接贴进报告）。"""
    tab = os.path.join(out_dir, "tables")
    ensure_dir(tab)
    M = ["psnr", "ssim", "edge_f1", "n_edge", "corner_repeat", "n_corner", "sift_match", "sift_inlier", "time_ms"]
    hdr = {"deg_cn": "退化类型", "method": "方法", "psnr": "PSNR(dB)", "ssim": "SSIM",
           "edge_f1": "边缘F1", "n_edge": "边缘像素数", "corner_repeat": "角点重复率",
           "n_corner": "角点数", "sift_match": "SIFT粗匹配", "sift_inlier": "SIFT内点",
           "time_ms": "耗时(ms)", "kind": "滤波器", "D0": "截止频率D0", "ringing": "振铃占比"}

    write_csv(res["e1_rows"], ["doc", "style", "deg", "deg_cn", "method"] + M, os.path.join(tab, "e1_raw.csv"))
    write_csv(res["e1_agg"], ["deg", "deg_cn", "method"] + M + [m + "_std" for m in M],
              os.path.join(tab, "e1_denoise.csv"))
    write_csv(res["e2_agg"], ["deg", "deg_cn", "method"] + M + [m + "_std" for m in M],
              os.path.join(tab, "e2_sharpen.csv"))
    write_csv(res["e3_agg"], ["kind", "D0", "psnr", "ssim", "edge_f1", "sift_inlier",
                              "corner_repeat", "ringing"], os.path.join(tab, "e3_cutoff.csv"))
    write_csv(res["e4_agg"], ["method", "psnr", "ssim", "edge_f1", "sift_inlier", "ringing",
                              "corner_repeat"], os.path.join(tab, "e4_filter_type.csv"))
    write_csv(res["e5_agg"], ["method"] + M, os.path.join(tab, "e5_order.csv"))
    write_csv(res["e6_agg"], ["method", "psnr", "ssim", "edge_f1", "sift_inlier",
                              "corner_repeat"], os.path.join(tab, "e6_periodic.csv"))
    write_csv(res["e7_rows"], ["name", "psnr", "ssim", "edge_f1", "n_edge", "n_corner",
                               "corner_repeat", "sift_inlier"], os.path.join(tab, "e7_failures.csv"))

    md = ["# P2 实验数据表（自动生成，无需手抄）", ""]

    md += ["## 表 1 空域 vs 频域去噪（%d 张文档平均±std）" % res["n_docs"], "",
           agg_md(res["e1_agg"], ["deg_cn", "method"], hdr, ["psnr", "ssim", "edge_f1", "corner_repeat", "sift_inlier", "time_ms"]), ""]

    md += ["## 表 2 各退化下的最优方法（按指标自动挑）", ""]
    best = []
    for deg in sorted(set(r["deg"] for r in res["e1_agg"])):
        sub = [r for r in res["e1_agg"] if r["deg"] == deg]
        if not sub:
            continue
        bp = max(sub, key=lambda r: r["psnr"])
        bf = max(sub, key=lambda r: r["edge_f1"])
        bs = max(sub, key=lambda r: (r["sift_inlier"] if r["sift_inlier"] == r["sift_inlier"] else -1))
        best.append(dict(deg=DEG_CN.get(deg, deg),
                         best_psnr="%s (%.2f)" % (bp["method"], bp["psnr"]),
                         best_edge="%s (%.3f)" % (bf["method"], bf["edge_f1"]),
                         best_sift="%s (%.1f)" % (bs["method"], bs["sift_inlier"])))
    md += [md_table(best, ["deg", "best_psnr", "best_edge", "best_sift"],
                    headers=["退化类型", "PSNR 最优", "边缘F1 最优", "SIFT内点 最优"]), ""]

    md += ["## 表 3 锐化前后：边缘 / 角点 / SIFT（%d 张平均）" % res["n_docs"], "",
           agg_md(res["e2_agg"], ["deg_cn", "method"], hdr,
                  ["n_edge", "edge_f1", "n_corner", "corner_repeat", "sift_inlier", "psnr", "ssim"]), ""]

    md += ["## 表 4 消融A：截止频率 D0 扫描（高斯噪声）", "",
           agg_md(res["e3_agg"], ["kind", "D0"], hdr,
                  ["psnr", "ssim", "edge_f1", "sift_inlier", "ringing"]), ""]

    md += ["## 表 5 消融B：滤波器类型（高斯噪声，D0=35）", "",
           agg_md(res["e4_agg"], ["method"], hdr,
                  ["psnr", "ssim", "edge_f1", "sift_inlier", "ringing"]), ""]

    md += ["## 表 6 消融C：去噪→特征 顺序（运动模糊+噪声）", "",
           agg_md(res["e5_agg"], ["method"], hdr,
                  ["psnr", "ssim", "edge_f1", "n_edge", "corner_repeat", "sift_match", "sift_inlier"]), ""]

    md += ["## 表 7 周期条纹：低通 / 带阻 / 陷波", "",
           agg_md(res["e6_agg"], ["method"], hdr, ["psnr", "ssim", "edge_f1", "sift_inlier"]), ""]

    md += ["## 表 8 失败案例指标", ""]
    md += [md_table([{k: r[k] for k in ["name", "psnr", "ssim", "edge_f1", "n_edge", "n_corner", "sift_inlier"]}
                     for r in res["e7_rows"]],
                    ["name", "psnr", "ssim", "edge_f1", "n_edge", "n_corner", "sift_inlier"],
                    headers=["案例", "PSNR(dB)", "SSIM", "边缘F1", "边缘像素数", "角点数", "SIFT内点"]), ""]

    with open(os.path.join(out_dir, "report_tables.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    # 全量原始数据 + 自检结果
    with open(os.path.join(out_dir, "metrics_all.json"), "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in res.items() if not k.endswith("_rows")}, f,
                  ensure_ascii=False, indent=2, default=str)
    return tab


# ==============================================================================
# 12. 单张照片模式（拿自己的作业照片跑一遍，验证结论可迁移）
# ==============================================================================
def run_single(path, out_dir):
    """对任意一张真实照片：出指标 + 频谱 + 去噪/锐化对比图。

    注意真实照片没有 ground truth，因此 PSNR/SSIM 无法计算，
    只输出无参考指标（Laplacian 方差清晰度、边缘数、角点数、SIFT 关键点数、
    频谱能量分布），并给出可视化对比 —— 这正是把实验结论搬到真实数据的用法。
    """
    ensure_dir(out_dir)
    img = imread_gray(path, size=(CFG["H"], CFG["W"]) if CFG["H"] else None)
    lap = cv2.Laplacian(img, cv2.CV_32F).var()
    e = canny_edge(img)
    cpts, _ = harris_corner(img)
    kps, _ = sift_detect(img)
    info = dict(file=os.path.basename(path), shape=list(img.shape),
                laplacian_var=float(lap), gray_mean=float(img.mean()),
                gray_std=float(img.std()), n_edge=int(e.sum()), n_corner=int(len(cpts)),
                n_sift_kp=len(kps))
    print(json.dumps(info, ensure_ascii=False, indent=2))
    items = [(img, "输入照片"), (spectrum_mag(img), "频谱（log）"),
             (canny_edge(img), "Canny 边缘 %d px" % e.sum()),
             (median_filter(img, 3), "中值 3×3"),
             (freq_filter(img, lp_mask(img.shape, 40, "gaussian")), "频域高斯LPF D0=40"),
             (unsharp_mask(img, 15, 4.0, 1.0), "USM 锐化")]
    vis = cv2.cvtColor(to_u8(img), cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    for x, y in cpts[:400]:
        cv2.circle(vis, (int(x), int(y)), 2, (1, 0.2, 0.2), -1)
    items.append((vis, "Harris 角点 %d 个" % len(cpts)))
    make_grid(items, os.path.join(out_dir, "single_report.png"), ncols=4,
              suptitle="单张照片复现：%s" % info["file"])
    with open(os.path.join(out_dir, "single_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    return info


# ==============================================================================
# 13. 主流程
# ==============================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="P2 文档图像频域去噪与 SIFT 特征提取复现（单文件）",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", default="all", choices=["all", "data", "exp", "figs"],
                    help="all=全流程；data=只建数据集；exp=跑实验+出表；figs=只出总览图")
    ap.add_argument("--out", default=OUT, help="输出目录（默认 out/）")
    ap.add_argument("--n_docs", type=int, default=None, help="数据集文档页数（默认 24）")
    ap.add_argument("--exp_docs", type=int, default=16,
                    help="参与指标统计的文档页数（默认 16，其余页仍会生成、仅不进入均值）")
    ap.add_argument("--quick", action="store_true", help="极速自检：4 页 + 半分辨率")
    ap.add_argument("--single", default=None, help="单张照片模式：给定图片路径")
    ap.add_argument("--seed", type=int, default=CFG["RNG"], help="随机种子")
    args = ap.parse_args(argv)

    out_dir = args.out
    figs = os.path.join(out_dir, "figures")
    ensure_dir(out_dir, figs)
    if args.quick:
        CFG.update(H=384, W=512, N_DOCS=4)
    if args.n_docs:
        CFG["N_DOCS"] = args.n_docs
    CFG["RNG"] = args.seed

    if args.single:
        run_single(args.single, os.path.join(out_dir, "single"))
        return 0

    t_all = time.perf_counter()
    print("=" * 78)
    print("P2 文档图像频域去噪与 SIFT 特征提取复现" + ("（quick 模式）" if args.quick else ""))
    print("=" * 78)

    # ---- 0) 频域实现自检
    selftest = dft_selftest()

    # ---- 1) 数据
    print("[数据] 合成 %d 张文档页 × 9 变体 ..." % CFG["N_DOCS"])
    manifest, meta = build_dataset(out_dir, n_docs=CFG["N_DOCS"], size=(CFG["H"], CFG["W"]))
    ds_all = load_dataset(out_dir, n_docs=CFG["N_DOCS"])
    # 参与指标统计的子集：指标里最贵的是 SIFT 匹配（每张约 0.4 s），
    # 16 页已经能让均值的标准误差足够小；其余页仍生成在 dataset/ 里，
    # 并作为"留出样本"用于人工核查（--exp_docs 可调）。
    n_exp = min(args.exp_docs or len(ds_all), len(ds_all))
    ds = dict(list(ds_all.items())[:n_exp])
    print("[数据] 数据集 %d 张图像；指标统计使用前 %d 页（留出 %d 页）"
          % (len(ds_all) * 9, n_exp, len(ds_all) - n_exp))
    if args.stage == "data":
        print("[数据] 完成，见 %s/dataset" % out_dir)
        return 0

    # ---- 2) 总览图（数据集 + 频谱）
    fig_dataset_overview(ds_all, figs)
    fig_spectrum_analysis(ds, figs)
    msift = fig_sift_detail(ds, figs)
    print("[图] 总览 / 频谱 / SIFT 匹配 已出图")

    if args.stage == "figs":
        return 0

    # ---- 3) 实验
    print("[实验] E1 空域 vs 频域去噪 ...")
    e1_rows, e1_agg = exp1_denoise(ds, out_dir, figs)
    print("[实验] E2 锐化前后 边缘/角点/SIFT ...")
    e2_rows, e2_agg = exp2_sharpen(ds, out_dir, figs)
    print("[实验] E3 消融A 截止频率 D0 扫描 ...")
    e3_rows, e3_agg = exp3_cutoff(ds, out_dir, figs)
    print("[实验] E4 消融B 滤波器类型 ...")
    e4_rows, e4_agg = exp4_filter_type(ds, out_dir, figs)
    print("[实验] E5 消融C 去噪→特征 顺序 ...")
    e5_rows, e5_agg = exp5_order(ds, out_dir, figs)
    print("[实验] E6 周期条纹：低通/带阻/陷波 ...")
    e6_rows, e6_agg = exp6_periodic(ds, out_dir, figs)
    print("[实验] E7 失败案例 ...")
    e7_rows = exp7_failures(ds, out_dir, figs)

    # ---- 4) 出表
    res = dict(selftest=selftest, n_docs=len(ds), n_docs_dataset=len(ds_all),
               n_images=len(ds_all) * 9, quick=bool(args.quick),
               e1_rows=e1_rows, e1_agg=e1_agg, e2_rows=e2_rows, e2_agg=e2_agg,
               e3_rows=e3_rows, e3_agg=e3_agg, e4_rows=e4_rows, e4_agg=e4_agg,
               e5_rows=e5_rows, e5_agg=e5_agg, e6_rows=e6_rows, e6_agg=e6_agg,
               e7_rows=e7_rows, sift_detail=dict(n_ka=msift["n_ka"], n_kb=msift["n_kb"],
                                                 n_match=msift["n_match"], n_inlier=msift["n_inlier"],
                                                 inlier_ratio=msift["inlier_ratio"]))
    write_tables(res, out_dir, figs)

    print("-" * 78)
    print("完成，用时 %.1f s" % (time.perf_counter() - t_all))
    print("  数据：%s/dataset/（%d 张）" % (out_dir, len(ds_all) * 9))
    print("  图  ：%s/figures/" % out_dir)
    print("  表  ：%s/tables/ + report_tables.md" % out_dir)
    # 打印几条关键结论，便于直接核对
    try:
        nz = [r for r in e1_agg if r["deg"] == "noise"]
        sp = [r for r in e1_agg if r["deg"] == "sp"]
        if nz:
            b = max(nz, key=lambda r: r["psnr"])
            print("  · 高斯噪声 PSNR 最优：%s（%.2f dB）" % (b["method"], b["psnr"]))
        if sp:
            b = max(sp, key=lambda r: r["psnr"])
            print("  · 椒盐噪声 PSNR 最优：%s（%.2f dB）" % (b["method"], b["psnr"]))
        b = max(e5_agg, key=lambda r: (r["sift_inlier"] if r["sift_inlier"] == r["sift_inlier"] else -1))
        print("  · 消融C 中 SIFT 内点最多：%s（%.1f）" % (b["method"], b["sift_inlier"]))
    except Exception as e:
        print("  (结论摘要跳过：%s)" % e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
