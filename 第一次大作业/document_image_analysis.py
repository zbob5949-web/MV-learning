"""拍照文档图像基础分析：颜色空间、直方图与质量指标。

本脚本是综述《物理先验驱动的拍照文档图像偏色与光照不匀分析》的配套实现，
直接对应论文以下三节的指标定义，供复现与数据收集方案使用：

    2.1 颜色空间      -> RGB 三通道均值、灰度加权亮度
    2.2 基础统计特征  -> 均值/标准差(对比度)、Laplacian 方差(清晰度)、背景偏色
    2.4 物理先验      -> 成像方程 I(x)=L(x)R(x)+n(x) 中低频照度 L 的不均匀度

用法：
    python document_image_analysis.py 输入图.jpg --out_dir out_demo
    python document_image_analysis.py 输入图.jpg --out_dir out_demo --figures

产出（均写入 --out_dir，同时把 JSON 打印到终端）：
    metrics.json         9 项质量指标
    gray_histogram.png   灰度直方图（纯 Pillow 绘制，无额外依赖）
    report_figure.png    可直接插入报告的 2x2 展示图（仅在给出 --figures 时生成）

依赖：numpy、pillow（必需）；matplotlib（仅 --figures 需要，缺失时自动跳过）。
刻意不依赖 scipy/opencv，故 Laplacian 与低频照度均用 numpy/Pillow 自行实现。
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------
# 指标 -> 论文章节 对照表（写报告/答辩时按此对照）
#
#   width / height                -> 2.1  数字图像表示（像素阵列尺寸）
#   rgb_mean                      -> 2.1  RGB 颜色空间三通道均值
#   gray_mean                     -> 2.2  平均亮度（近似曝光水平）
#   gray_std_contrast             -> 2.2  对比度（亮度标准差）
#   laplacian_variance_sharpness  -> 2.2  清晰度（Laplacian 响应方差 VoL）
#   lab_background_a_offset       -> 2.2  偏色轴 a*≈R-G（红-绿）
#   lab_background_b_offset       -> 2.2  偏色轴 b*≈B-(R+G)/2（黄-蓝）
#   illumination_nonuniformity    -> 2.4  低频照度 L 的相对动态范围
# ---------------------------------------------------------------------------

# 灰度加权系数取 ITU-R BT.601（与论文 2.1 节的 Y=0.299R+0.587G+0.114B 一致）
GRAY_WEIGHTS = (0.299, 0.587, 0.114)

# 偏色统计只取"近白像素"的灰度分位阈值：>60% 分位视为纸张背景，
# 避免深色文字与彩色插图主导均值（论文 2.2 节"尽量在去除文字和插图影响后估计"）。
BRIGHT_PERCENTILE = 60

# 低频照度的缩略倍数：1/20 缩略再放大 ≈ 低通滤波，用于粗略分离照度分量 L。
ILLUMINATION_DOWNSCALE = 20


def laplacian_variance(gray: np.ndarray) -> float:
    """清晰度指标：Laplacian 响应的方差（Variance of Laplacian, VoL）。

    对应论文 2.2 节"清晰度可用拉普拉斯方差，高值通常意味着边缘更锐利"。
    采用标准 4-邻域离散 Laplacian 核：

        [ 0  1  0]
        [ 1 -4  1]
        [ 0  1  0]

    为免引入 scipy/opencv 依赖，卷积用数组切片直接展开；同时丢弃最外一圈
    像素，从而避开边界环绕（np.roll）产生的虚假响应。
    """
    inner = gray[1:-1, 1:-1]
    lap = (gray[:-2, 1:-1] + gray[2:, 1:-1]      # 上邻域 + 下邻域
           + gray[1:-1, :-2] + gray[1:-1, 2:]    # 左邻域 + 右邻域
           - 4.0 * inner)                        # 中心像素 ×(-4)
    return float(lap.var())


def background_color_offsets(rgb: np.ndarray, gray: np.ndarray):
    """估计近白纸张背景上的偏色，对应论文 2.2 节。

    物理依据（论文 2.4 节）：纸张反射率 R 在大面积区域近似平滑且接近中性，
    因此背景像素的 RGB 通道差异主要来自光源色温，而非物体本色。

    返回两个近似 Lab 对立轴的偏移量：
        a* ≈ R - G                  红-绿轴，正值偏红、负值偏青
        b* ≈ B - (R + G) / 2        黄-蓝轴，正值偏黄、负值偏蓝
    """
    bright = gray > np.percentile(gray, BRIGHT_PERCENTILE)
    # 极端情况下（如图片几乎全黑）近白像素为空，退化为全图统计以免崩溃
    sample = rgb[bright] if bright.any() else rgb.reshape(-1, 3)
    a_offset = float((sample[:, 0] - sample[:, 1]).mean())
    b_offset = float((sample[:, 2] - (sample[:, 0] + sample[:, 1]) / 2.0).mean())
    return a_offset, b_offset


def estimate_illumination(gray: np.ndarray) -> np.ndarray:
    """估计低频照度分量 L(x)，对应论文 2.4 节成像方程 I(x)=L(x)R(x)+n(x)。

    做法：先把灰度图缩小 ILLUMINATION_DOWNSCALE 倍再放大回原尺寸，
    等效于一次廉价的低通滤波，留下的即是缓慢变化的照度 L；
    文字/线条等高频的反射率 R 结构在这一步被抹平。
    返回值与 gray 同尺寸，供不均匀度计算与伪彩可视化共用。
    """
    h, w = gray.shape
    small = Image.fromarray(gray.astype(np.uint8)).resize(
        (max(2, w // ILLUMINATION_DOWNSCALE), max(2, h // ILLUMINATION_DOWNSCALE))
    )
    return np.asarray(small.resize((w, h)), dtype=np.float32)


def illumination_nonuniformity(illum: np.ndarray) -> float:
    """照度不均匀度：(Lmax - Lmin) / Lmean，即低频照度的相对动态范围。

    对应论文 2.4 节"以 L 的空间梯度或低频动态范围量化光照不匀"：
    均匀白光下该值接近 0，存在单侧阴影时会显著增大。
    """
    return float((illum.max() - illum.min()) / max(illum.mean(), 1.0))


def analyze(path: Path) -> dict:
    """读入一张图片，计算全部 9 项质量指标。

    返回的 dict 可直接 json.dumps 落盘为 metrics.json。
    """
    # 统一转 RGB 读取：PIL 会按文件内的色彩信息正确解释通道顺序与位深
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    # 灰度化：Y = 0.299R + 0.587G + 0.114B（丢失色温信息，但利于阈值化与清晰度评价）
    gray = (GRAY_WEIGHTS[0] * rgb[..., 0]
            + GRAY_WEIGHTS[1] * rgb[..., 1]
            + GRAY_WEIGHTS[2] * rgb[..., 2]).astype(np.float32)

    a_offset, b_offset = background_color_offsets(rgb, gray)
    illum = estimate_illumination(gray)

    return {
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "rgb_mean": [round(float(x), 3) for x in rgb.reshape(-1, 3).mean(axis=0)],
        "gray_mean": round(float(gray.mean()), 3),
        "gray_std_contrast": round(float(gray.std()), 3),
        "laplacian_variance_sharpness": round(laplacian_variance(gray), 3),
        "lab_background_a_offset": round(a_offset, 3),
        "lab_background_b_offset": round(b_offset, 3),
        "illumination_nonuniformity": round(illumination_nonuniformity(illum), 4),
    }


def draw_histogram(gray: np.ndarray, out_path: Path) -> None:
    """用 Pillow 画灰度直方图并存为 PNG（不依赖 matplotlib）。

    直方图对应论文 2.2 节：峰值位置反映整体曝光，峰值宽度反映对比度。
    """
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    hist = hist / max(hist.max(), 1)          # 归一化到 [0,1]，便于缩放绘图
    canvas = Image.new("RGB", (512, 300), "white")
    draw = ImageDraw.Draw(canvas)
    # 逐 bin 连成折线；y 轴翻转（图像坐标原点在左上角）
    for i in range(1, 256):
        draw.line((i * 2 - 2, 299 - int(hist[i - 1] * 280),
                   i * 2, 299 - int(hist[i] * 280)), fill=(49, 91, 120), width=1)
    canvas.save(out_path)


def draw_report_figure(path: Path, gray: np.ndarray, illum: np.ndarray,
                       metrics: dict, out_path: Path) -> bool:
    """生成可直接插入报告的 2x2 展示图，呼应论文各节的论证。

    四宫格：
        左上 输入照片缩略图        -> 2.1 数据来源
        右上 灰度直方图            -> 2.2 基础统计特征
        左下 低频照度 L 伪彩热力图  -> 2.4 物理先验（标注 L 的极值位置）
        右下 指标卡                -> 全指标汇总

    matplotlib 为可选依赖；未安装或缺少中文字体时不影响主流程，返回 False。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")                 # 无 GUI 后端，适配脚本/CI 环境
        import matplotlib.pyplot as plt
    except ImportError:
        print("[提示] 未安装 matplotlib，已跳过 --figures 展示图生成")
        return False

    # 大图先缩略：matplotlib 渲染数千万像素会显著拖慢速度并占用大量内存
    thumb = Image.open(path).convert("RGB")
    thumb.thumbnail((1200, 1200))
    # 照度图本身已由 1/20 缩略得到（分辨率极低），再缩到固定尺寸几乎无损
    illum_small = np.asarray(Image.fromarray(illum.astype(np.uint8)).resize((480, 320)),
                             dtype=np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # 左上：输入照片
    axes[0][0].imshow(thumb)
    axes[0][0].set_title("Input photo (downscaled)")
    axes[0][0].axis("off")

    # 右上：灰度直方图（对应 2.2 节）
    # 先用 numpy 统计频数再画柱状，避免把上千万像素直接交给 plt.hist
    hist, edges = np.histogram(gray, bins=256, range=(0, 256))
    axes[0][1].bar(edges[:-1], hist, width=1.0, color=(49 / 255, 91 / 255, 120 / 255))
    axes[0][1].set_title("Gray histogram")
    axes[0][1].set_xlabel("gray level")
    axes[0][1].set_ylabel("pixel count")

    # 左下：低频照度伪彩图，并标出极暗/极亮位置（对应 2.4 节）
    im = axes[1][0].imshow(illum_small, cmap="inferno")
    for label, idx, value, color in (
            ("Lmin", np.unravel_index(np.argmin(illum_small), illum_small.shape), illum.min(), "cyan"),
            ("Lmax", np.unravel_index(np.argmax(illum_small), illum_small.shape), illum.max(), "lime")):
        axes[1][0].plot(idx[1], idx[0], marker="+", color=color, markersize=14,
                        markeredgewidth=2, label="%s≈%.1f" % (label, value))
    axes[1][0].set_title("Low-frequency illumination L (1/%d thumbnail)" % ILLUMINATION_DOWNSCALE)
    axes[1][0].legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=axes[1][0], fraction=0.046)

    # 右下：指标卡
    axes[1][1].axis("off")
    lines = ["width x height : %d x %d" % (metrics["width"], metrics["height"]),
             "rgb_mean       : %s" % (metrics["rgb_mean"],),
             "gray_mean      : %s" % metrics["gray_mean"],
             "gray_std       : %s" % metrics["gray_std_contrast"],
             "laplacian_var  : %s" % metrics["laplacian_variance_sharpness"],
             "a* offset      : %s" % metrics["lab_background_a_offset"],
             "b* offset      : %s" % metrics["lab_background_b_offset"],
             "nonuniformity  : %s" % metrics["illumination_nonuniformity"]]
    axes[1][1].text(0.02, 0.95, "\n".join(lines), va="top", ha="left",
                    family="monospace", fontsize=11, transform=axes[1][1].transAxes)
    axes[1][1].set_title("Metrics (paper 2.1 / 2.2 / 2.4)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="拍照文档图像质量指标分析（P1 配套代码）")
    ap.add_argument("image", type=Path, help="输入图片路径")
    ap.add_argument("--out_dir", type=Path, default=Path("analysis_out"),
                    help="产物输出目录，默认 analysis_out")
    ap.add_argument("--figures", action="store_true",
                    help="额外生成可插入报告的 2x2 展示图 report_figure.png（需 matplotlib）")
    args = ap.parse_args()

    metrics = analyze(args.image)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 产物 1：指标 JSON
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 产物 2：灰度直方图（用于复现步骤的最小展示包）
    gray = np.asarray(Image.open(args.image).convert("L"))
    draw_histogram(gray, args.out_dir / "gray_histogram.png")

    # 产物 3（可选）：报告插图
    if args.figures:
        illum = estimate_illumination(gray.astype(np.float32))
        draw_report_figure(args.image, gray, illum, metrics, args.out_dir / "report_figure.png")

    # 终端同步打印，便于直接粘贴进报告或日志
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
