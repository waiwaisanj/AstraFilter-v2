
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
import os
import sys

# 修正：用绝对路径加 modules 目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES_DIR = os.path.join(BASE_DIR, "modules")
if MODULES_DIR not in sys.path:
    sys.path.insert(0, MODULES_DIR)

st.set_page_config(page_title="AstraFilter", page_icon="🔭", layout="wide")

st.title("🔭 AstraFilter")
st.subheader("快速移动天体在线检测工具")

st.markdown("上传天文图像，AstraFilter 会自动检测其中的条纹候选体（可能是快速移动小行星的轨迹），并给出每个候选体的位置、长度、方向和置信度。")

with st.sidebar:
    st.header("检测参数")
    method = st.selectbox("检测方法", ["traditional (传统图像处理)", "unet (U-Net 神经网络)", "both (两者结合)"])
    n_sigma = st.slider("亮度阈值 (sigma)", 3.0, 8.0, 5.0, 0.5)
    min_length = st.slider("最小条纹长度 (像素)", 5, 30, 10)
    min_aspect = st.slider("最小长宽比", 2.0, 6.0, 3.0, 0.5)
    min_linearity = st.slider("最小线性度 (PCA)", 1.5, 5.0, 2.5, 0.5)

st.markdown("---")
st.header("上传图像")

uploaded_file = st.file_uploader("选择 FITS 文件或普通图像（PNG/JPG）", type=["fits", "fz", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    data = None
    if file_ext in ("fits", "fz"):
        from astropy.io import fits
        try:
            with fits.open(BytesIO(uploaded_file.read())) as hdul:
                for hdu in hdul:
                    if hdu.data is not None and len(hdu.data.shape) == 2:
                        data = np.nan_to_num(hdu.data.copy(), nan=0.0)
                        break
            st.success("已读取 FITS 文件")
        except Exception as e:
            st.error("读取 FITS 失败: " + str(e))
    else:
        try:
            img = Image.open(uploaded_file).convert("L")
            data = np.array(img).astype(np.float32)
            st.success("已读取图像")
        except Exception as e:
            st.error("读取图像失败: " + str(e))

    if data is not None:
        st.write("图像尺寸: " + str(data.shape))
        st.write("像素范围: " + str(round(float(data.min()), 2)) + " ~ " + str(round(float(data.max()), 2)))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("原始图像")
            med = np.nanmedian(data)
            std = np.nanstd(data)
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(data, cmap="gray", vmin=med-2*std, vmax=med+5*std)
            ax.axis("off")
            st.pyplot(fig)

        if st.button("开始检测", type="primary"):
            with st.spinner("检测中..."):
                try:
                    from detector import StripeDetector
                    detector = StripeDetector(method="traditional")
                    candidates = detector.detect_traditional(data, n_sigma=n_sigma, min_length=min_length, min_aspect=min_aspect, min_linearity=min_linearity)
                except Exception as e:
                    st.error("检测失败: " + str(e))
                    st.exception(e)
                    candidates = []

            st.markdown("---")
            st.header("检测结果")

            if len(candidates) == 0:
                st.warning("没有检测到候选体。可以尝试降低亮度阈值或最小长度。")
            else:
                st.success("检测到 " + str(len(candidates)) + " 个候选体")

                with col2:
                    st.subheader("检测结果可视化")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    med = np.nanmedian(data)
                    std = np.nanstd(data)
                    ax.imshow(data, cmap="gray", vmin=med-2*std, vmax=med+5*std)
                    for c in candidates:
                        rect = Rectangle((c["x"]-c["length"]/2, c["y"]-c["length"]/2), c["length"], c["length"], linewidth=2, edgecolor="lime", facecolor="none")
                        ax.add_patch(rect)
                    ax.axis("off")
                    st.pyplot(fig)

                df = pd.DataFrame(candidates)
                st.subheader("候选体列表")
                st.dataframe(df)

                csv = df.to_csv(index=False)
                st.download_button("下载 CSV 报告", csv, "astra_filter_candidates.csv", "text/csv")

                st.subheader("候选体局部放大图")
                n_show = min(6, len(candidates))
                cols_ui = st.columns(3)
                for i, c in enumerate(candidates[:n_show]):
                    box = max(40, c["length"])
                    x1 = max(0, int(c["x"]) - box)
                    x2 = min(data.shape[1], int(c["x"]) + box)
                    y1 = max(0, int(c["y"]) - box)
                    y2 = min(data.shape[0], int(c["y"]) + box)
                    patch = data[y1:y2, x1:x2]
                    fig, ax = plt.subplots(figsize=(3, 3))
                    med = np.nanmedian(patch)
                    std = np.nanstd(patch)
                    ax.imshow(patch, cmap="gray", vmin=med-2*std, vmax=med+5*std)
                    ax.set_title("候选 " + str(i+1) + " (线性度 " + str(round(c["linearity"], 1)) + ")", fontsize=10)
                    ax.axis("off")
                    with cols_ui[i % 3]:
                        st.pyplot(fig)

else:
    st.markdown("---")
    st.subheader("使用说明")
    st.markdown("1. 上传一个天文图像文件（FITS 或 PNG/JPG）")
    st.markdown("2. 在左侧调整检测参数")
    st.markdown("3. 点击「开始检测」")
    st.markdown("4. 查看检测到的候选体，下载 CSV 报告")

    st.markdown("---")
    st.subheader("支持的文件格式")
    st.markdown("- **FITS** (`.fits`, `.fits.fz`)：天文标准格式")
    st.markdown("- **PNG/JPG**：普通图像，会自动转为灰度处理")

    st.markdown("---")
    st.subheader("检测原理")
    st.markdown("AstraFilter 使用传统图像处理和 PCA 线性度分析来检测条纹候选体。它先找出比背景亮 5 个标准差的像素，然后用连通域分析找出所有独立目标，最后根据长宽比和 PCA 线性度筛选出细长的条纹。")

    st.markdown("---")
    st.subheader("应用场景")
    st.markdown("- 快速检查一张天文图像是否有条纹目标")
    st.markdown("- 批量处理多张 ZTF 图像，找出 FMO 候选体")
    st.markdown("- 教学演示，帮助理解快速移动天体的检测原理")
