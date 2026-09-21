
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

# 可选依赖（用 try-except 包裹，防止构建环境缺失导致崩溃）
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from astroquery.vizier import Vizier
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False
    Vizier = None


st.set_page_config(page_title="AstraFilter", page_icon="🔭", layout="wide")

# ===== 移动端响应式 CSS =====
st.markdown("""
<style>
/* 手机端单列布局 */
@media (max-width: 768px) {
    .stApp {
        padding: 0.5rem;
    }
    .stColumns > div {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }
    .stButton > button {
        width: 100% !important;
        padding: 1rem !important;
        font-size: 1.1rem !important;
    }
    .stSlider {
        padding: 0.5rem 0 !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.8rem !important;
        padding: 0.5rem 0.3rem !important;
    }
}

/* 触控友好：增大点击目标 */
.stButton > button {
    min-height: 44px;
    border-radius: 8px;
}

/* 文件上传区更大，方便手指点击 */
[data-testid="stFileUploader"] {
    padding: 1rem;
    border-radius: 8px;
}

/* 让 dataframe 在小屏幕上可横向滚动 */
.dataframe {
    overflow-x: auto;
    display: block;
}

/* 图片自适应宽度 */
.stImage img {
    max-width: 100% !important;
    height: auto !important;
}
</style>
""", unsafe_allow_html=True)



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




# ============ 多语言支持 ============
LANG_STRINGS = {
    "zh": {
        "app_title": "AstraFilter",
        "app_subtitle": "快速移动天体检测与验证平台",
        "tab_home": "📰 首页",
        "tab_detect": "🔍 检测",
        "tab_sky": "🌌 星图",
        "tab_meta": "📊 图像信息",
        "tab_verify": "✅ 验证",
        "tab_a11y": "♿ 无障碍",
        "tab_photo": "📷 星空照片",
        "tab_learn": "📚 学习",
        "home_title": "欢迎使用 AstraFilter",
        "home_desc": "AstraFilter 是一个免费、开源、可离线工作的天文图像分析平台。它可以检测天文图像中的快速移动天体（FMO），提供 3D 星图、多源验证、无障碍支持和科普教学。",
        "news_title": "最新天文与航天新闻",
        "news_loading": "正在加载最新新闻...",
        "news_refresh": "刷新新闻",
        "news_source": "来源",
        "news_read_more": "阅读原文",
        "quick_start": "快速开始",
        "quick_start_desc": "点击上方「检测」标签，上传一张 FITS 或 PNG 图像即可开始。",
        "features": "主要功能",
        "feature_1": "条纹检测 — 上传图像，自动识别 FMO 条纹",
        "feature_2": "3D 星图 — 用离线星表显示候选体附近的恒星",
        "feature_3": "多源验证 — MPChecker、SkyBoT、JPL Horizons",
        "feature_4": "无障碍 — 四种主题、语音描述、色盲友好",
        "feature_5": "星空照片 — 手机拍的星空照片自动识别亮星",
        "feature_6": "科普教学 — FMO 原理、检测方法讲解",
        "language": "语言",
    },
    "en": {
        "app_title": "AstraFilter",
        "app_subtitle": "Fast-Moving Object Detection and Validation Platform",
        "tab_home": "📰 Home",
        "tab_detect": "🔍 Detection",
        "tab_sky": "🌌 Sky Map",
        "tab_meta": "📊 Metadata",
        "tab_verify": "✅ Verification",
        "tab_a11y": "♿ Accessibility",
        "tab_photo": "📷 Photo",
        "tab_learn": "📚 Learn",
        "home_title": "Welcome to AstraFilter",
        "home_desc": "AstraFilter is a free, open-source, offline-capable platform for astronomical image analysis. It detects Fast-Moving Objects (FMOs) in astronomical images and offers 3D sky maps, multi-source validation, accessibility support, and educational content.",
        "news_title": "Latest Astronomy and Space News",
        "news_loading": "Loading latest news...",
        "news_refresh": "Refresh News",
        "news_source": "Source",
        "news_read_more": "Read more",
        "quick_start": "Quick Start",
        "quick_start_desc": "Click the Detection tab above and upload a FITS or PNG image to begin.",
        "features": "Features",
        "feature_1": "Streak Detection — Auto-detect FMO streaks in images",
        "feature_2": "3D Sky Map — Show nearby stars using offline catalog",
        "feature_3": "Multi-source Validation — MPChecker, SkyBoT, JPL Horizons",
        "feature_4": "Accessibility — 4 themes, voice description, colorblind-friendly",
        "feature_5": "Star Photo — Auto-detect bright stars in phone photos",
        "feature_6": "Education — FMO principles and detection methods",
        "language": "Language",
    },
}

if "lang" not in st.session_state:
    st.session_state["lang"] = "zh"


def t(key):
    return LANG_STRINGS[st.session_state["lang"]].get(key, key)


# ============ 实时天文新闻 ============
@st.cache_data(ttl=3600)
def fetch_astronomy_news():
    """获取最新的天文和航天新闻，缓存 1 小时"""
    news = []

    # 源 1：Spaceflight News API（稳定、免费、无限制）
    try:
        r = requests.get(
            "https://api.spaceflightnewsapi.net/v4/articles/",
            params={"limit": 6, "ordering": "-published_at"},
            timeout=10,
        )
        if r.status_code == 200:
            for art in r.json().get("results", []):
                news.append({
                    "title": art.get("title", ""),
                    "source": art.get("news_site", "Spaceflight News"),
                    "url": art.get("url", ""),
                    "published": art.get("published_at", "")[:10],
                    "summary": (art.get("summary", "") or "")[:200],
                })
    except Exception:
        pass

    # 源 2：NASA 每日天文图片（APOD）
    try:
        r = requests.get(
            "https://api.nasa.gov/planetary/apod",
            params={"api_key": "DEMO_KEY"},
            timeout=10,
        )
        if r.status_code == 200:
            data = r.json()
            news.append({
                "title": data.get("title", "NASA APOD"),
                "source": "NASA APOD",
                "url": data.get("url", "https://apod.nasa.gov/"),
                "published": data.get("date", ""),
                "summary": (data.get("explanation", "") or "")[:200],
            })
    except Exception:
        pass

    return news


# ============ 离线星表 ============
import os as _os

_stars_cache = None


def load_bright_stars():
    global _stars_cache
    if _stars_cache is not None:
        return _stars_cache
    try:
        csv_path = _os.path.join(
            _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
            "data", "bright_stars.csv"
        )
        if not _os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        _stars_cache = df
        return df
    except Exception:
        return None


def query_stars_offline(ra, dec, radius_deg=1.0, max_mag=7.0):
    df = load_bright_stars()
    if df is None or len(df) == 0:
        return None
    ra_r = np.radians(df["ra"].values)
    dec_r = np.radians(df["dec"].values)
    ra_c = np.radians(ra)
    dec_c = np.radians(dec)
    cos_dist = np.sin(dec_c) * np.sin(dec_r) + np.cos(dec_c) * np.cos(dec_r) * np.cos(ra_r - ra_c)
    cos_dist = np.clip(cos_dist, -1, 1)
    dist_deg = np.degrees(np.arccos(cos_dist))
    mask = (dist_deg < radius_deg) & (df["mag"] < max_mag)
    result = df[mask].copy()
    if len(result) == 0:
        return None
    result = result.rename(columns={"ra": "RAmdeg", "dec": "DEmdeg", "mag": "VTmag"})
    return result


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






# ===== 离线星表 =====
import os as _os

_stars_cache = None

def load_bright_stars():
    """加载离线亮星表"""
    global _stars_cache
    if _stars_cache is not None:
        return _stars_cache
    try:
        csv_path = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data", "bright_stars.csv")
        if not _os.path.exists(csv_path):
            return None
        df = pd.read_csv(csv_path)
        _stars_cache = df
        return df
    except Exception:
        return None


def query_stars_offline(ra, dec, radius_deg=1.0, max_mag=7.0):
    """用离线星表查询天区内的恒星"""
    df = load_bright_stars()
    if df is None or len(df) == 0:
        return None

    ra_r = np.radians(df["ra"].values)
    dec_r = np.radians(df["dec"].values)
    ra_c = np.radians(ra)
    dec_c = np.radians(dec)

    cos_dist = np.sin(dec_c) * np.sin(dec_r) + np.cos(dec_c) * np.cos(dec_r) * np.cos(ra_r - ra_c)
    cos_dist = np.clip(cos_dist, -1, 1)
    dist_deg = np.degrees(np.arccos(cos_dist))

    mask = (dist_deg < radius_deg) & (df["mag"] < max_mag)
    result = df[mask].copy()
    if len(result) == 0:
        return None

    result = result.rename(columns={"ra": "RAmdeg", "dec": "DEmdeg", "mag": "VTmag"})
    return result


def query_stars_around(ra, dec, radius_deg=1.0, max_mag=12):
    if not ASTROQUERY_AVAILABLE:
        return None
    
    """查询天区附近的恒星，用 Vizier 的 Tycho-2 星表"""
    try:
        vizier = Vizier(
            columns=["RAmdeg", "DEmdeg", "VTmag", "HIP", "TYC"],
            row_limit=500,
            column_filters={"VTmag": "<" + str(max_mag)},
        )
        catalog_list = vizier.query_region(
            SkyCoord(ra=ra, dec=dec, unit="deg"),
            radius=radius_deg * u.deg,
            catalog="I/259/tyc2",
        )
        if len(catalog_list) == 0:
            return None
        stars = catalog_list[0]
        return stars
    except Exception as e:
        return None


def make_3d_sky(stars, candidates=None, ra_center=None, dec_center=None):
    """用 matplotlib 生成 3D 星空图（不依赖 plotly）"""
    if stars is None or len(stars) == 0:
        return None

    try:
        ra_rad = np.radians(np.array(stars["RAmdeg"], dtype=float))
        dec_rad = np.radians(np.array(stars["DEmdeg"], dtype=float))
        mags = np.array(stars["VTmag"], dtype=float)
    except Exception:
        return None

    # 转换到 3D 坐标
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)

    # 用星等确定大小（越亮越大）
    sizes = np.clip((14 - mags) * 8, 3, 60)

    fig = plt.figure(figsize=(10, 10), facecolor="black")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("black")

    # 画恒星
    ax.scatter(x, y, z, s=sizes, c=mags, cmap="cividis", alpha=0.8, edgecolors="none")

    # 画候选体
    if candidates is not None:
        cand_ra = np.radians(np.array([c["ra"] for c in candidates]))
        cand_dec = np.radians(np.array([c["dec"] for c in candidates]))
        cx = np.cos(cand_dec) * np.cos(cand_ra)
        cy = np.cos(cand_dec) * np.sin(cand_ra)
        cz = np.sin(cand_dec)
        ax.scatter(cx, cy, cz, s=200, c="red", marker="*", edgecolors="yellow", linewidths=2, zorder=10)

    # 画球面网格
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="gray", alpha=0.15, linewidth=0.5)

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_axis_off()

    plt.tight_layout()
    return fig



def detect_bright_stars(image, n_sigma=10, min_pixels=2, max_pixels=500, max_stars=20):
    """检测图像中的亮星"""
    med = np.median(image)
    std = np.std(image)
    binary = image > (med + n_sigma * std)
    labeled, num = ndimage.label(binary, structure=np.ones((3, 3)))
    stars = []
    for i in range(1, num + 1):
        ys, xs = np.where(labeled == i)
        n = len(ys)
        if n < min_pixels or n > max_pixels:
            continue
        brightness = float(np.sum(image[ys, xs]) - med * n)
        stars.append({
            "x": int(xs.mean()),
            "y": int(ys.mean()),
            "brightness": brightness,
            "n_pixels": n,
        })
    stars = sorted(stars, key=lambda s: -s["brightness"])
    return stars[:max_stars]


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


tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([t("tab_home"), t("tab_detect"), t("tab_sky"), t("tab_meta"), t("tab_verify"), t("tab_a11y"), t("tab_photo"), t("tab_learn")])

with tab0:
    # 顶部语言切换
    lang_options = ["中文", "English"]
    current_idx = 0 if st.session_state["lang"] == "zh" else 1
    lang_col1, lang_col2 = st.columns([4, 1])
    with lang_col2:
        chosen = st.selectbox(t("language"), lang_options, index=current_idx, key="lang_selector")
        new_lang = "zh" if chosen == "中文" else "en"
        if new_lang != st.session_state["lang"]:
            st.session_state["lang"] = new_lang
            st.rerun()

    st.markdown("---")
    st.subheader(t("home_title"))
    st.markdown(t("home_desc"))

    st.markdown("---")
    st.subheader("📰 " + t("news_title"))

    col_a, col_b = st.columns([4, 1])
    with col_b:
        if st.button("🔄 " + t("news_refresh")):
            st.cache_data.clear()

    with st.spinner(t("news_loading")):
        news = fetch_astronomy_news()

    if len(news) == 0:
        st.warning("无法获取最新新闻，请检查网络连接。")
    else:
        for i, item in enumerate(news):
            with st.container():
                c1, c2 = st.columns([4, 1])
                with c1:
                    st.markdown("### " + item["title"])
                    st.caption(t("news_source") + ": " + item["source"] + " | " + item["published"])
                    if item.get("summary"):
                        st.write(item["summary"])
                with c2:
                    st.link_button(t("news_read_more"), item["url"])
                st.markdown("---")

    st.subheader("🚀 " + t("quick_start"))
    st.markdown(t("quick_start_desc"))

    st.markdown("---")
    st.subheader(t("features"))
    st.markdown("- " + t("feature_1"))
    st.markdown("- " + t("feature_2"))
    st.markdown("- " + t("feature_3"))
    st.markdown("- " + t("feature_4"))
    st.markdown("- " + t("feature_5"))
    st.markdown("- " + t("feature_6"))


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
    st.header("星空浏览器")
    st.markdown("显示候选体附近的已知恒星。可以拖动旋转 3D 星空视图。")

    data = st.session_state.get("data")
    wcs = st.session_state.get("wcs")
    candidates = st.session_state.get("candidates", [])

    if data is None:
        st.info("请先在检测标签页上传图像。")
    elif wcs is None:
        st.warning("图像没有 WCS 信息，无法转换为天球坐标。请上传 FITS 文件。")
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

        if not coords:
            st.warning("无法转换坐标。")
        else:
            st.subheader("候选体天球坐标")
            df_coords = pd.DataFrame(coords)
            st.dataframe(df_coords)

            # ===== 查询已知恒星 =====
            ra_center = float(np.mean([c["ra"] for c in coords]))
            dec_center = float(np.mean([c["dec"] for c in coords]))

            st.markdown("---")
            st.subheader("虚拟天文台查询")
            st.write("中心坐标: RA=" + str(round(ra_center, 4)) + "°, Dec=" + str(round(dec_center, 4)) + "°")

            radius = st.slider("查询半径 (度)", 0.1, 5.0, 1.0, 0.1)
            max_mag = st.slider("最大星等", 8, 15, 12)

            if st.button("查询附近恒星"):
                with st.spinner("查询中..."):
                    # 优先用离线星表
                    stars = query_stars_offline(ra_center, dec_center, radius, max_mag)
                    if stars is None or len(stars) == 0:
                        # 备用：在线查询
                        stars = query_stars_around(ra_center, dec_center, radius, max_mag)
                if stars is None or len(stars) == 0:
                    st.warning("没有找到恒星，可尝试增大查询半径或提高最大星等。")
                else:
                    st.success("找到 " + str(len(stars)) + " 颗恒星")
                    st.session_state["stars"] = stars
                    st.session_state["ra_center"] = ra_center
                    st.session_state["dec_center"] = dec_center

            # ===== 显示 3D 星图 =====
            if "stars" in st.session_state:
                stars = st.session_state["stars"]
                ra_c = st.session_state.get("ra_center", ra_center)
                dec_c = st.session_state.get("dec_center", dec_center)

                st.markdown("---")
                st.subheader("3D 星空视图")
                st.markdown("**在手机上拖动可以旋转视角。这个视图可以在 VR 眼镜（如 Google Cardboard）中查看。**")

                fig3d = make_3d_sky(stars, candidates=coords, ra_center=ra_c, dec_center=dec_c)
                if fig3d is not None:
                    st.pyplot(fig3d)

                # 显示 2D Aitoff 投影
                st.markdown("---")
                st.subheader("2D 天球投影")
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111, projection="aitoff")
                ax.grid(True)
                ra_rad = [np.radians(c["ra"] - 180) if c["ra"] > 180 else np.radians(c["ra"]) for c in coords]
                dec_rad = [np.radians(c["dec"]) for c in coords]
                ax.scatter(ra_rad, dec_rad, s=100, c="red", marker="*", zorder=5)
                ax.set_title("Candidate positions (Aitoff projection)")
                st.pyplot(fig)

                # 恒星列表
                st.markdown("---")
                st.subheader("附近恒星列表（前 50 颗）")
                st.dataframe(stars[:50].to_pandas())


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
    st.header("📷 星空照片亮星检测")
    st.markdown("上传一张星空照片（手机拍摄的照片也可以），系统会自动检测照片里的亮星，并在照片上标注它们的位置。这个功能可以帮助你识别照片中的星座和主要恒星。")

    photo_file = st.file_uploader("选择星空照片", type=["png", "jpg", "jpeg", "fits", "fz"], key="photo_upload")

    if photo_file is not None:
        # 读取图像
        file_ext = photo_file.name.split(".")[-1].lower()
        photo_data = None

        if file_ext in ("fits", "fz"):
            try:
                with fits.open(BytesIO(photo_file.read())) as hdul:
                    for hdu in hdul:
                        if hdu.data is not None and len(hdu.data.shape) == 2:
                            photo_data = np.nan_to_num(hdu.data.copy(), nan=0.0)
                            break
            except Exception as e:
                st.error("读取失败: " + str(e))
        else:
            try:
                img = Image.open(photo_file).convert("L")
                photo_data = np.array(img).astype(np.float32)
            except Exception as e:
                st.error("读取失败: " + str(e))

        if photo_data is not None:
            st.write("图像尺寸: " + str(photo_data.shape))

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("原始照片")
                med = np.median(photo_data)
                std = np.std(photo_data)
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(photo_data, cmap="gray", vmin=med-2*std, vmax=med+5*std)
                ax.axis("off")
                st.pyplot(fig)

            st.markdown("---")
            st.subheader("检测参数")
            threshold = st.slider("亮星阈值 (sigma)", 5, 30, 10, 1)
            max_stars = st.slider("最多检测亮星数", 5, 50, 20, 1)

            if st.button("检测亮星", type="primary"):
                with st.spinner("检测中..."):
                    stars = detect_bright_stars(photo_data, n_sigma=threshold, max_stars=max_stars)
                st.session_state["bright_stars"] = stars
                st.session_state["photo_data"] = photo_data

            stars = st.session_state.get("bright_stars", [])

            if len(stars) > 0:
                st.success("检测到 " + str(len(stars)) + " 颗亮星")

                with col2:
                    st.subheader("标注结果")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    med = np.median(photo_data)
                    std = np.std(photo_data)
                    ax.imshow(photo_data, cmap="gray", vmin=med-2*std, vmax=med+5*std)
                    for i, s in enumerate(stars):
                        ax.plot(s["x"], s["y"], "r+", markersize=15, markeredgewidth=2)
                        ax.text(s["x"]+10, s["y"]-10, str(i+1), color="lime", fontsize=14, weight="bold")
                    ax.axis("off")
                    st.pyplot(fig)

                st.markdown("---")
                st.subheader("亮星列表")
                df_stars = pd.DataFrame(stars)
                st.dataframe(df_stars)

                st.markdown("---")
                st.subheader("亮星几何模式")
                st.markdown("下面是亮星在照片中的相对位置。你可以把这个模式和大范围星图对照，判断拍摄的天区。")

                fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
                ax.set_facecolor("black")
                ax.invert_yaxis()
                xs = [s["x"] for s in stars]
                ys = [s["y"] for s in stars]
                sizes = [s["n_pixels"] * 8 + 20 for s in stars]
                ax.scatter(xs, ys, s=sizes, c="yellow", edgecolors="orange", linewidth=1)
                for i, s in enumerate(stars):
                    ax.text(s["x"]+15, s["y"]-15, str(i+1), color="white", fontsize=12)
                ax.set_aspect("equal")
                ax.axis("off")
                st.pyplot(fig)

                st.markdown("---")
                st.info("提示：如果知道照片的大致拍摄方向（比如北极星方向），可以用「星图」标签页里的 3D 星空视图对照。你也可以用 Astrometry.net 的在线服务做完整的星空识别。")




with tab7:
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
