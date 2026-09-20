
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
from scipy import ndimage
import requests
from astropy.io import fits
from astropy.wcs import WCS

st.set_page_config(page_title="AstraFilter", page_icon="🔭", layout="wide")


# ===== 无障碍主题配置 =====
THEMES = {
    "default": {
        "name": "默认 (深色背景)",
        "bg": "#0e1117",
        "fg": "#fafafa",
        "accent": "#ff4b4b",
        "box_color": "#00ff00",
        "colormap": "gray",
    },
    "high_contrast": {
        "name": "高对比度 (黑白)",
        "bg": "#000000",
        "fg": "#ffffff",
        "accent": "#ffff00",
        "box_color": "#ffff00",
        "colormap": "gray",
    },
    "colorblind_friendly": {
        "name": "色盲友好 (蓝橙)",
        "bg": "#1a1a1a",
        "fg": "#e0e0e0",
        "accent": "#ff8c00",
        "box_color": "#00bfff",
        "colormap": "cividis",
    },
    "light": {
        "name": "浅色 (白底黑字)",
        "bg": "#ffffff",
        "fg": "#000000",
        "accent": "#0050b3",
        "box_color": "#0050b3",
        "colormap": "gray",
    },
}

FONT_SIZES = {
    "小": "14px",
    "中": "16px",
    "大": "20px",
    "超大": "26px",
}


def apply_accessibility_theme():
    """应用无障碍主题"""
    if "theme_key" not in st.session_state:
        st.session_state["theme_key"] = "default"
    if "font_size" not in st.session_state:
        st.session_state["font_size"] = "中"
    if "voice_enabled" not in st.session_state:
        st.session_state["voice_enabled"] = False

    theme = THEMES[st.session_state["theme_key"]]
    font_size = FONT_SIZES[st.session_state["font_size"]]

    css = f"""
    <style>
    .stApp {{
        background-color: {theme["bg"]};
        color: {theme["fg"]};
        font-size: {font_size};
    }}
    .stMarkdown, .stText, p, div, span, label {{
        color: {theme["fg"]} !important;
        font-size: {font_size};
    }}
    h1, h2, h3, h4 {{
        color: {theme["accent"]} !important;
    }}
    .stButton > button {{
        background-color: {theme["accent"]};
        color: {theme["bg"]};
        font-size: {font_size};
        font-weight: bold;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {theme["fg"]};
        font-size: {font_size};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return theme


theme = apply_accessibility_theme()


def detect_traditional(image, n_sigma=5, min_length=10, min_linearity=3.0):
    med = np.nanmedian(image)
    std = np.nanstd(image)
    img = np.nan_to_num(image, nan=0.0)
    binary = img > (med + n_sigma * std)
    labeled, num = ndimage.label(binary, structure=np.ones((3, 3)))
    candidates = []
    for i in range(1, num + 1):
        ys, xs = np.where(labeled == i)
        n_pix = len(ys)
        if n_pix < 8 or n_pix > 500:
            continue
        pts = np.column_stack([xs, ys]).astype(float)
        pts_centered = pts - pts.mean(axis=0)
        cov = np.cov(pts_centered.T)
        if cov.size != 4:
            continue
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        if eigvals[1] < 1e-6:
            linearity = 999.0
        else:
            linearity = float(np.sqrt(eigvals[0] / eigvals[1]))
        principal_length = 4.0 * np.sqrt(eigvals[0])
        if principal_length < min_length:
            continue
        if linearity < min_linearity:
            continue
        candidates.append({
            "x": int(xs.mean()),
            "y": int(ys.mean()),
            "length": int(principal_length),
            "linearity": float(linearity),
            "n_pixels": int(n_pix),
        })
    candidates = sorted(candidates, key=lambda c: -c["linearity"])
    kept = []
    for c in candidates:
        if all(np.sqrt((c["x"]-k["x"])**2 + (c["y"]-k["y"])**2) > 50 for k in kept):
            kept.append(c)
    return kept


def query_mpchecker(ra, dec, radius_arcmin=5, date_str="20231001"):
    url = "https://www.minorplanetcenter.net/cgi-bin/checkmp.cgi"
    params = {
        "year": date_str[:4], "month": date_str[4:6], "day": date_str[6:8],
        "which": "pos", "ra": ra, "dec": dec,
        "r": str(radius_arcmin), "limit": "24.0", "oc": "500",
        "sort": "d", "mot": "h", "tmot": "s", "pdes": "u",
        "needed": "f", "ps": "n", "type": "p",
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        if "No known minor planets" in r.text:
            return {"status": "clear", "message": "该坐标附近无已知天体"}
        return {"status": "found", "message": "MPChecker 返回结果，可能有已知天体"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_audio_description(candidates, data_shape):
    """生成音频化的检测结果描述"""
    if len(candidates) == 0:
        return "未检测到候选体。"

    h, w = data_shape
    parts = [f"检测到 {len(candidates)} 个候选体。"]

    for i, c in enumerate(candidates):
        # 位置描述
        x_ratio = c["x"] / w
        y_ratio = c["y"] / h

        if x_ratio < 0.33:
            x_desc = "左侧"
        elif x_ratio < 0.67:
            x_desc = "中央"
        else:
            x_desc = "右侧"

        if y_ratio < 0.33:
            y_desc = "上方"
        elif y_ratio < 0.67:
            y_desc = "中部"
        else:
            y_desc = "下方"

        # 线性度描述
        lin = c["linearity"]
        if lin > 10:
            lin_desc = "非常细长"
        elif lin > 5:
            lin_desc = "较细长"
        else:
            lin_desc = "一般"

        parts.append(
            f"候选体 {i+1}: 位于{y_desc}{x_desc}，长度 {c['length']} 像素，线性度 {lin:.1f}，{lin_desc}。"
        )

    return " ".join(parts)


# ===== 页面 =====
st.title("🔭 AstraFilter")
st.subheader("无障碍天文图像分析平台")

# 无障碍设置面板
with st.sidebar:
    st.header("♿ 无障碍设置")

    theme_names = {k: v["name"] for k, v in THEMES.items()}
    theme_key = st.selectbox(
        "配色主题",
        options=list(THEMES.keys()),
        format_func=lambda k: theme_names[k],
        index=list(THEMES.keys()).index(st.session_state.get("theme_key", "default")),
    )
    if theme_key != st.session_state.get("theme_key"):
        st.session_state["theme_key"] = theme_key
        st.rerun()

    font_key = st.selectbox(
        "字体大小",
        options=list(FONT_SIZES.keys()),
        index=list(FONT_SIZES.keys()).index(st.session_state.get("font_size", "中")),
    )
    if font_key != st.session_state.get("font_size"):
        st.session_state["font_size"] = font_key
        st.rerun()

    voice_enabled = st.checkbox("启用语音描述", value=st.session_state.get("voice_enabled", False))
    st.session_state["voice_enabled"] = voice_enabled

    st.markdown("---")
    st.header("检测参数")
    n_sigma = st.slider("亮度阈值 (sigma)", 3.0, 8.0, 5.0, 0.5)
    min_length = st.slider("最小条纹长度 (像素)", 5, 50, 15)
    min_linearity = st.slider("最小线性度 (PCA)", 1.5, 10.0, 3.0, 0.5)


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🔍 检测", "🌌 星图", "📊 图像信息", "✅ 验证", "♿ 无障碍", "📚 学习"])

with tab1:
    uploaded_file = st.file_uploader("上传 FITS 或普通图像", type=["fits", "fz", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        file_ext = uploaded_file.name.split(".")[-1].lower()
        data = None
        header = None
        wcs = None

        if file_ext in ("fits", "fz"):
            try:
                with fits.open(BytesIO(uploaded_file.read())) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None and len(hdu.data.shape) == 2:
                            data = np.nan_to_num(hdu.data.copy(), nan=0.0)
                            header = hdu.header
                            try:
                                wcs = WCS(header)
                            except Exception:
                                wcs = None
                            break
                st.success("已读取 FITS 文件")
            except Exception as e:
                st.error("读取失败: " + str(e))
        else:
            try:
                img = Image.open(uploaded_file).convert("L")
                data = np.array(img).astype(np.float32)
                st.success("已读取图像")
            except Exception as e:
                st.error("读取失败: " + str(e))

        if data is not None:
            st.session_state["data"] = data
            st.session_state["header"] = header
            st.session_state["wcs"] = wcs

            st.write("图像尺寸: " + str(data.shape))

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("原始图像")
                med = np.nanmedian(data)
                std = np.nanstd(data)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(data, cmap=theme["colormap"], vmin=med-2*std, vmax=med+5*std)
                ax.axis("off")
                st.pyplot(fig)

            if st.button("开始检测", type="primary"):
                with st.spinner("检测中..."):
                    candidates = detect_traditional(data, n_sigma=n_sigma, min_length=min_length, min_linearity=min_linearity)
                st.session_state["candidates"] = candidates

                if st.session_state.get("voice_enabled"):
                    desc = generate_audio_description(candidates, data.shape)
                    st.session_state["audio_desc"] = desc

            candidates = st.session_state.get("candidates", [])

            st.markdown("---")
            st.header("检测结果")

            if len(candidates) == 0:
                st.warning("没有检测到候选体。")
            else:
                st.success("检测到 " + str(len(candidates)) + " 个候选体")

                with col2:
                    st.subheader("检测结果可视化")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    med = np.nanmedian(data)
                    std = np.nanstd(data)
                    ax.imshow(data, cmap=theme["colormap"], vmin=med-2*std, vmax=med+5*std)
                    for c in candidates:
                        rect = Rectangle((c["x"]-c["length"]/2, c["y"]-c["length"]/2), c["length"], c["length"],
                                        linewidth=3, edgecolor=theme["box_color"], facecolor="none")
                        ax.add_patch(rect)
                    ax.axis("off")
                    st.pyplot(fig)

                df = pd.DataFrame(candidates)
                st.subheader("候选体列表")
                st.dataframe(df)

                csv = df.to_csv(index=False)
                st.download_button("下载 CSV 报告", csv, "astra_filter_candidates.csv", "text/csv")

                if st.session_state.get("voice_enabled") and "audio_desc" in st.session_state:
                    st.markdown("---")
                    st.subheader("🔊 语音描述")
                    st.info(st.session_state["audio_desc"])
                    st.markdown("如果你使用屏幕阅读器，可以直接朗读上面的文本。")

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
                    ax.imshow(patch, cmap=theme["colormap"], vmin=med-2*std, vmax=med+5*std)
                    ax.set_title("Candidate " + str(i+1), fontsize=12)
                    ax.axis("off")
                    with cols_ui[i % 3]:
                        st.pyplot(fig)


with tab2:
    st.header("候选体在天球上的位置")
    data = st.session_state.get("data")
    wcs = st.session_state.get("wcs")
    candidates = st.session_state.get("candidates", [])

    if data is None:
        st.info("请先在检测标签页上传图像。")
    elif wcs is None:
        st.warning("图像没有 WCS 信息，无法转换为天球坐标。")
    elif len(candidates) == 0:
        st.info("还没有检测到候选体。")
    else:
        coords = []
        for c in candidates:
            try:
                ra, dec = wcs.all_pix2world(c["x"], c["y"], 0)
                coords.append({"ra": float(ra), "dec": float(dec), "x": c["x"], "y": c["y"]})
            except Exception:
                continue

        if coords:
            df_coords = pd.DataFrame(coords)
            st.subheader("候选体天球坐标")
            st.dataframe(df_coords)

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111, projection="aitoff")
            ax.grid(True)
            ra_rad = [np.radians(c["ra"] - 180) if c["ra"] > 180 else np.radians(c["ra"]) for c in coords]
            dec_rad = [np.radians(c["dec"]) for c in coords]
            ax.scatter(ra_rad, dec_rad, s=100, c=theme["box_color"], marker="*", zorder=5)
            ax.set_title("Candidate positions", color=theme["fg"])
            st.pyplot(fig)


with tab3:
    st.header("图像元数据")
    header = st.session_state.get("header")
    if header is None:
        st.info("请上传 FITS 文件。")
    else:
        keys = ["TELESCOP", "INSTRUME", "FILTER", "EXPTIME", "MJD-OBS", "DATE-OBS", "AIRMASS", "SEEING"]
        for k in keys:
            if k in header:
                st.write("**" + k + "**: " + str(header[k]))

        st.markdown("---")
        st.subheader("完整头部信息")
        header_dict = {k: str(header[k]) for k in header.keys()}
        df_h = pd.DataFrame(list(header_dict.items()), columns=["Keyword", "Value"])
        st.dataframe(df_h)


with tab4:
    st.header("候选体验证")
    candidates = st.session_state.get("candidates", [])
    wcs = st.session_state.get("wcs")

    if len(candidates) == 0:
        st.info("请先检测候选体。")
    elif wcs is None:
        st.warning("需要 WCS 信息。")
    else:
        if st.button("开始 MPChecker 查询"):
            for i, c in enumerate(candidates):
                try:
                    ra, dec = wcs.all_pix2world(c["x"], c["y"], 0)
                    st.write("**候选体 " + str(i+1) + "**: RA=" + str(round(float(ra), 5)) + ", Dec=" + str(round(float(dec), 5)))
                    with st.spinner("查询中..."):
                        result = query_mpchecker(float(ra), float(dec))
                    if result["status"] == "clear":
                        st.success("  " + result["message"])
                    elif result["status"] == "found":
                        st.warning("  " + result["message"])
                    else:
                        st.error("  失败: " + result["message"])
                except Exception as e:
                    st.error("出错: " + str(e))


with tab5:
    st.header("♿ 无障碍功能说明")
    st.markdown("""
AstraFilter 为色盲和视觉障碍用户提供了以下功能：

### 配色主题
- **默认**：深色背景，适合夜间使用
- **高对比度**：黑白配色，对比度最高，适合低视力用户
- **色盲友好**：蓝橙配色（cividis 色谱），红绿色盲用户可区分
- **浅色**：白底黑字，适合强光环境

### 字体大小
- 小 / 中 / 大 / 超大，可在左侧栏调整

### 语音描述
- 开启后，检测结果会生成文字描述，说明每个候选体的位置、长度和线性度
- 兼容屏幕阅读器（NVDA、JAWS、VoiceOver）

### 音频化检测结果
- 候选体位置用文字描述："左侧"、"中央"、"右侧"、"上方"、"下方"
- 线性度用文字描述："非常细长"、"较细长"、"一般"

### 设计理念
传统天文软件假设用户能看见屏幕上的彩色图像。对于色盲和视障用户，这些软件几乎不可用。AstraFilter 通过多主题配色、可调字体、语音描述和音频化检测结果，让天文图像分析对所有人开放。
    """)


with tab6:
    st.header("什么是快速移动天体 (FMO)？")
    st.markdown("""
快速移动天体（FMO）是指在天球上运动速率超过 **0.5 度/天** 的太阳系小天体。

包括：
- 近地小行星（NEA）— 行星防御的重点监测对象
- 主带小行星 — 运动速率通常小于 0.5 度/天
- 人造卫星和空间碎片

在 ZTF 巡天望远镜的 30 秒曝光中，FMO 会留下一条细长的条纹，而恒星和星系呈现为点源。
    """)

    st.markdown("---")
    st.header("AstraFilter 的检测原理")
    st.markdown("""
**第一步：亮度阈值** — 找出比背景亮度高 5 个标准差的像素。

**第二步：8 连通域分析** — 把相邻的亮像素连成独立目标（含对角线相邻）。

**第三步：PCA 分析** — 计算每个目标的主方向长度和线性度。

**第四步：筛选** — 保留长度 > 15 像素、线性度 > 3.0 的目标。
    """)

    st.markdown("---")
    st.header("相关资源和工具")
    st.markdown("""
- ZTF 数据存档（IRSA）：https://irsa.ipac.caltech.edu/Missions/ztf.html
- MPChecker：https://www.minorplanetcenter.net/cgi-bin/checkmp.cgi
- SkyBoT：https://ssp.imcce.fr/forms/skybot
- JPL Horizons：https://ssd.jpl.nasa.gov/horizons/
- DeepStreaks：https://github.com/dmitryduev/DeepStreaks
    """)
