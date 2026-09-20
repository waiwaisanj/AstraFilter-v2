
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
            return {"status": "clear", "message": "该坐标附近 5 角分内无已知天体"}
        return {"status": "found", "message": "MPChecker 有返回结果，可能有已知天体"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


st.title("🔭 AstraFilter")
st.subheader("快速移动天体在线检测与分析平台")

st.markdown("上传天文图像，AstraFilter 会自动检测条纹候选体（可能是快速移动小行星的轨迹），并提供候选体验证、星图定位、元数据分析和科普讲解。")

# ===== 侧边栏 =====
with st.sidebar:
    st.header("检测参数")
    n_sigma = st.slider("亮度阈值 (sigma)", 3.0, 8.0, 5.0, 0.5)
    min_length = st.slider("最小条纹长度 (像素)", 5, 50, 15)
    min_linearity = st.slider("最小线性度 (PCA)", 1.5, 10.0, 3.0, 0.5)

# ===== 主区域 Tabs =====
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 检测", "🌌 星图", "📊 图像信息", "✅ 验证", "📚 学习"])

# ---- Tab 1: 检测 ----
with tab1:
    uploaded_file = st.file_uploader("上传 FITS 或普通图像 (PNG/JPG)", type=["fits", "fz", "png", "jpg", "jpeg"], key="upload")

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
                st.error("读取 FITS 失败: " + str(e))
        else:
            try:
                img = Image.open(uploaded_file).convert("L")
                data = np.array(img).astype(np.float32)
                st.success("已读取图像")
            except Exception as e:
                st.error("读取图像失败: " + str(e))

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
                ax.imshow(data, cmap="gray", vmin=med-2*std, vmax=med+5*std)
                ax.axis("off")
                st.pyplot(fig)

            if st.button("开始检测", type="primary"):
                with st.spinner("检测中..."):
                    candidates = detect_traditional(data, n_sigma=n_sigma, min_length=min_length, min_linearity=min_linearity)
                st.session_state["candidates"] = candidates

            candidates = st.session_state.get("candidates", [])

            st.markdown("---")
            st.header("检测结果")

            if len(candidates) == 0:
                st.warning("没有检测到候选体。可尝试降低亮度阈值或最小长度。")
            else:
                st.success("检测到 " + str(len(candidates)) + " 个候选体")

                with col2:
                    st.subheader("结果可视化")
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
                    ax.set_title("Candidate " + str(i+1) + " (L=" + str(c["length"]) + ")", fontsize=10)
                    ax.axis("off")
                    with cols_ui[i % 3]:
                        st.pyplot(fig)


# ---- Tab 2: 星图 ----
with tab2:
    st.header("候选体在天球上的位置")
    st.markdown("如果上传的 FITS 文件包含 WCS 信息，AstraFilter 会把候选体的像素坐标转换为天球坐标，并显示它们在天空中的位置。")

    data = st.session_state.get("data")
    wcs = st.session_state.get("wcs")
    candidates = st.session_state.get("candidates", [])

    if data is None:
        st.info("请先在检测标签页上传图像。")
    elif wcs is None:
        st.warning("上传的图像没有 WCS 信息（可能是 PNG/JPG），无法转换为天球坐标。请上传 ZTF 或其他望远镜的 FITS 文件。")
    elif len(candidates) == 0:
        st.info("还没有检测到候选体。")
    else:
        # 计算候选体的天球坐标
        coords = []
        for c in candidates:
            try:
                ra, dec = wcs.all_pix2world(c["x"], c["y"], 0)
                coords.append({"ra": float(ra), "dec": float(dec), "x": c["x"], "y": c["y"], "length": c["length"]})
            except Exception:
                continue

        if len(coords) == 0:
            st.warning("无法转换坐标。")
        else:
            # 显示表格
            df_coords = pd.DataFrame(coords)
            st.subheader("候选体天球坐标")
            st.dataframe(df_coords)

            # 画天球投影
            st.subheader("天球投影")
            st.markdown("下图是候选体在赤道坐标系中的位置（使用 Aitoff 投影）。")

            fig = plt.figure(figsize=(10, 5))
            ax = fig.add_subplot(111, projection="aitoff")
            ax.grid(True)

            # 把所有候选体标在图上
            ras = [c["ra"] for c in coords]
            decs = [c["dec"] for c in coords]
            ra_rad = [np.radians(r - 180) if r > 180 else np.radians(r) for r in ras]
            dec_rad = [np.radians(d) for d in decs]

            ax.scatter(ra_rad, dec_rad, s=80, c="red", marker="*", zorder=5)
            for i, (r, d) in enumerate(zip(ra_rad, dec_rad)):
                ax.text(r, d, str(i+1), fontsize=10, color="blue")

            ax.set_title("Candidate positions (Aitoff projection)")
            st.pyplot(fig)


# ---- Tab 3: 图像信息 ----
with tab3:
    st.header("图像元数据")
    header = st.session_state.get("header")

    if header is None:
        st.info("请上传 FITS 文件以查看元数据。")
    else:
        # 显示关键信息
        st.subheader("关键观测参数")
        keys = ["TELESCOP", "INSTRUME", "FILTER", "EXPTIME", "OBSJD", "MJD-OBS", "DATE-OBS", "AIRMASS", "SEEING"]
        for k in keys:
            if k in header:
                st.write("**" + k + "**: " + str(header[k]))

        st.markdown("---")
        st.subheader("完整头部信息")
        header_dict = {k: str(header[k]) for k in header.keys()}
        df_h = pd.DataFrame(list(header_dict.items()), columns=["Keyword", "Value"])
        st.dataframe(df_h)

        # WCS 信息
        wcs = st.session_state.get("wcs")
        if wcs is not None:
            st.markdown("---")
            st.subheader("WCS (World Coordinate System)")
            st.write("CRVAL (参考点天球坐标): " + str(wcs.wcs.crval))
            st.write("CRPIX (参考点像素坐标): " + str(wcs.wcs.crpix))
            st.write("CTYPE (投影类型): " + str(wcs.wcs.ctype))


# ---- Tab 4: 验证 ----
with tab4:
    st.header("候选体验证")
    st.markdown("把候选体的坐标输入 MPChecker 数据库，检查是否有已知天体。")

    candidates = st.session_state.get("candidates", [])
    wcs = st.session_state.get("wcs")

    if len(candidates) == 0:
        st.info("请先在检测标签页检测候选体。")
    elif wcs is None:
        st.warning("上传的图像没有 WCS 信息，无法获取候选体的天球坐标。")
    else:
        st.markdown("点击下方按钮，对每个候选体查询 MPChecker。")

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
                        st.error("  查询失败: " + result["message"])
                except Exception as e:
                    st.error("候选体 " + str(i+1) + " 出错: " + str(e))


# ---- Tab 5: 学习 ----
with tab5:
    st.header("什么是快速移动天体 (FMO)？")
    st.markdown("""
快速移动天体（Fast-Moving Object, FMO）是指在天球上运动速率超过 **0.5 度/天** 的太阳系小天体。

它们包括：
- **近地小行星（NEA）** — 轨道与地球轨道相交或接近，是行星防御的重点监测对象
- **主带小行星** — 运动速率通常小于 0.5 度/天，不形成明显条纹
- **人造卫星和空间碎片** — 轨道周期短，运动速率高，但通常更亮

在 ZTF 这类宽视场巡天望远镜的 30 秒曝光中，FMO 会留下一条**细长的条纹**，而静止的恒星和星系则呈现为点源。这让条纹成为 FMO 的独特标志。
    """)

    st.markdown("---")
    st.header("AstraFilter 的检测原理")
    st.markdown("""
AstraFilter 使用传统图像处理和 PCA（主成分分析）来检测条纹：

**第一步：亮度阈值**
找出比背景亮度高 5 个标准差的像素。

**第二步：8 连通域分析**
把相邻的亮像素连成一个独立目标。8 连通意味着对角线相邻的像素也算作同一目标，这对检测斜向条纹很关键。

**第三步：PCA 分析**
对每个目标内的像素做 PCA，计算：
- **主方向长度** — 目标在最长方向上的长度
- **线性度** — 第一主成分与第二主成分的比值，越大表示越像一条直线

**第四步：筛选**
保留长度 > 15 像素、线性度 > 3.0 的目标，作为候选体。
    """)

    st.markdown("---")
    st.header("相关资源和工具")
    st.markdown("""
- **ZTF 官方数据存档（IRSA）**：[https://irsa.ipac.caltech.edu/Missions/ztf.html](https://irsa.ipac.caltech.edu/Missions/ztf.html)
- **MPChecker（小行星中心）**：[https://www.minorplanetcenter.net/cgi-bin/checkmp.cgi](https://www.minorplanetcenter.net/cgi-bin/checkmp.cgi)
- **SkyBoT（IMCCE）**：[https://ssp.imcce.fr/forms/skybot](https://ssp.imcce.fr/forms/skybot)
- **JPL Horizons**：[https://ssd.jpl.nasa.gov/horizons/](https://ssd.jpl.nasa.gov/horizons/)
- **DeepStreaks（ZTF FMO 检测系统）**：[https://github.com/dmitryduev/DeepStreaks](https://github.com/dmitryduev/DeepStreaks)
    """)

    st.markdown("---")
    st.header("引用与致谢")
    st.markdown("""
- Duev et al. (2019), DeepStreaks: identifying fast-moving objects in the Zwicky Transient Facility data with deep learning
- Ye et al. (2019), ZTF Moving Object Pipeline System (MOPS)
- ZTF 数据来自 NASA/IPAC 红外科学档案
    """)
