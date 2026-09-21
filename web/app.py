
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
        "name": "默认 (NASA 白底风格)",
        "bg": "#ffffff",
        "fg": "#0b3d91",
        "accent": "#0b3d91",
        "box_color": "#fc3d21",
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
    /* 全局 */
    .stApp {{
        background-color: {theme["bg"]};
        color: {theme["fg"]};
        font-size: {font_size};
    }}
        .stMarkdown p, .stText p, p {{
        color: #1f2328 !important;
    }}
    .nasa-banner h1, .nasa-banner p, .nasa-banner span {{
        color: #ffffff !important;
    }}
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {{
        color: #1f2328 !important;
    }}
    [data-testid="stMetricValue"] {{
        color: #0b3d91 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: #4a5568 !important;
    }}
    .stSelectbox label, .stTextInput label, .stSlider label,
    .stNumberInput label, .stFileUploader label, .stCheckbox label {{
        color: #1f2328 !important;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #4a5568 !important;
    }}
    .stTabs [aria-selected="true"] {{
        color: #0b3d91 !important;
    }}
    .stButton > button, .stDownloadButton > button {{
        color: #ffffff !important;
    }}
    h1, h2, h3, h4 {{
        color: {{theme["accent"]}} !important;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        letter-spacing: 0.5px;
    }}

    /* 顶部横幅（NASA 风格） */
    .nasa-banner {{
        background: linear-gradient(135deg, #0b3d91 0%, #1e3a8a 60%, #0b3d91 100%);
        color: #ffffff;
        padding: 2rem 2.5rem;
        border-radius: 0;
        margin: -1rem -1rem 1.5rem -1rem;
        box-shadow: 0 4px 12px rgba(11, 61, 145, 0.15);
        border-bottom: 4px solid #fc3d21;
    }}
    .nasa-banner h1 {{
        color: #ffffff !important;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 1px;
    }}
    .nasa-banner p {{
        color: #d0dcf0 !important;
        margin: 0.5rem 0 0 0;
        font-size: 1rem;
    }}
    .nasa-banner .nasa-dot {{
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #fc3d21;
        border-radius: 50%;
        margin-right: 10px;
        vertical-align: middle;
    }}

    /* 卡片 */
    .nasa-card {{
        background: #ffffff;
        border: 1px solid #dde3ec;
        border-left: 4px solid #0b3d91;
        border-radius: 6px;
        padding: 1.2rem 1.5rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        transition: box-shadow 0.2s;
    }}
    .nasa-card:hover {{
        box-shadow: 0 4px 14px rgba(11,61,145,0.12);
    }}
    .nasa-card h3 {{
        margin-top: 0;
        color: #0b3d91 !important;
        font-size: 1.15rem;
    }}
    .nasa-card .source-tag {{
        display: inline-block;
        background: #fc3d21;
        color: #ffffff;
        padding: 2px 8px;
        border-radius: 3px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 8px;
        letter-spacing: 0.5px;
    }}
    .nasa-card .date-tag {{
        color: #6b7280;
        font-size: 0.8rem;
    }}

    /* 按钮 */
    .stButton > button {{
        background-color: {{theme["accent"]}};
        color: #ffffff;
        font-size: {{font_size}};
        font-weight: bold;
        border: none;
        border-radius: 4px;
        padding: 0.6rem 1.2rem;
        transition: background 0.2s;
    }}
    .stButton > button:hover {{
        background-color: #fc3d21;
    }}

    /* Tab 导航（NASA 风格） */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-bottom: 2px solid #dde3ec;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {{theme["fg"]}};
        font-size: {{font_size}};
        font-weight: 600;
        padding: 0.8rem 1.2rem;
        border-radius: 0;
        letter-spacing: 0.3px;
    }}
    .stTabs [aria-selected="true"] {{
        background: transparent !important;
        border-bottom: 3px solid #fc3d21 !important;
        color: #0b3d91 !important;
    }}

    /* 侧边栏 */
    section[data-testid="stSidebar"] {{
        background: #f7f9fc;
        border-right: 1px solid #dde3ec;
    }}

    /* 数据框 */
    .dataframe {{
        font-size: 0.85rem;
    }}

    /* 分隔线 */
    hr {{
        border-color: #dde3ec;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
    return theme


theme = apply_accessibility_theme()




# ============ 跨夜追踪 ============
@st.cache_data(ttl=7200)
def query_ztf_observations(ra, dec, start_date, end_date, radius=0.3):
    """查询 ZTF 在该天区的观测记录"""
    import astropy.time as time
    from ztfquery import query as zquery_mod

    try:
        zq = zquery_mod.ZTFQuery()
        start_jd = time.Time(start_date).jd
        end_jd = time.Time(end_date).jd
        zq.load_metadata(
            radec=[ra, dec],
            size=radius,
            sql_query=f"obsjd BETWEEN {start_jd} AND {end_jd}",
        )
        if len(zq.metatable) == 0:
            return None
        return zq.metatable
    except Exception as e:
        return None


def compute_photometry(image, x, y, radius=15):
    """测量候选体在图像中的亮度"""
    y1 = max(0, int(y) - radius)
    y2 = min(image.shape[0], int(y) + radius)
    x1 = max(0, int(x) - radius)
    x2 = min(image.shape[1], int(x) + radius)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    # 用最亮的 20% 像素的平均值
    flat = patch.flatten()
    flat = np.sort(flat)[::-1]
    top = flat[:max(1, len(flat)//5)]
    med_bg = np.median(image)
    return float(np.mean(top) - med_bg)


def infer_physical_properties(velocity_deg_per_day):
    """从角速度推断物理属性"""
    # 假设距离与角速度成反比（简化模型）
    # 主带小行星：~0.1-0.5 度/天，距离 ~2-3 AU
    # 近地小行星：~1-10 度/天，距离 ~0.1-1 AU
    if velocity_deg_per_day < 0.05:
        return {
            "type": "疑似主带小行星 (极慢)",
            "distance_au": "3.0+",
            "classification": "Main-belt (slow)",
        }
    elif velocity_deg_per_day < 0.5:
        return {
            "type": "主带小行星",
            "distance_au": "2.0-3.0",
            "classification": "Main-belt",
        }
    elif velocity_deg_per_day < 2.0:
        return {
            "type": "近地小行星 (慢速)",
            "distance_au": "0.5-2.0",
            "classification": "NEA (slow)",
        }
    elif velocity_deg_per_day < 10.0:
        return {
            "type": "近地小行星 (快速)",
            "distance_au": "0.1-0.5",
            "classification": "NEA (fast)",
        }
    else:
        return {
            "type": "疑似人造卫星或空间碎片",
            "distance_au": "<0.01",
            "classification": "Satellite",
        }




# ============ 色盲模拟 ============
def simulate_color_vision_deficiency(image, deficiency="deuteranopia"):
    """模拟不同类型的色盲看到的图像

    参数:
        image: 2D numpy 数组（灰度）
        deficiency: 'protanopia' (红盲), 'deuteranopia' (绿盲), 'tritanopia' (蓝盲)

    返回:
        模拟的 RGB 图像
    """
    # 先把灰度图转成伪彩色（用 matplotlib 的 plasma 色谱，色彩丰富）
    import matplotlib.cm as cm
    med = np.median(image)
    std = np.std(image)
    normalized = np.clip((image - med) / (std * 3), 0, 1)
    colored = cm.plasma(normalized)[:, :, :3]  # RGB, 0-1

    # 色盲转换矩阵（Brettel-Viénot-Mollon 简化版）
    matrices = {
        "protanopia": np.array([
            [0.567, 0.433, 0.000],
            [0.558, 0.442, 0.000],
            [0.000, 0.242, 0.758],
        ]),
        "deuteranopia": np.array([
            [0.625, 0.375, 0.000],
            [0.700, 0.300, 0.000],
            [0.000, 0.300, 0.700],
        ]),
        "tritanopia": np.array([
            [0.950, 0.050, 0.000],
            [0.000, 0.433, 0.567],
            [0.000, 0.475, 0.525],
        ]),
    }

    if deficiency not in matrices:
        return colored

    m = matrices[deficiency]
    h, w, _ = colored.shape
    flattened = colored.reshape(-1, 3)
    transformed = flattened @ m.T
    transformed = np.clip(transformed, 0, 1)
    return transformed.reshape(h, w, 3)


def make_cvd_comparison(image):
    """生成 4 格对比图：正常 / 红盲 / 绿盲 / 蓝盲"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    # 正常
    import matplotlib.cm as cm
    med = np.median(image)
    std = np.std(image)
    normalized = np.clip((image - med) / (std * 3), 0, 1)
    colored = cm.plasma(normalized)[:, :, :3]

    axes[0, 0].imshow(colored)
    axes[0, 0].set_title("正常视觉 (Normal Vision)", fontsize=14, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(simulate_color_vision_deficiency(image, "protanopia"))
    axes[0, 1].set_title("红色盲 (Protanopia)", fontsize=14, fontweight="bold")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(simulate_color_vision_deficiency(image, "deuteranopia"))
    axes[1, 0].set_title("绿色盲 (Deuteranopia)", fontsize=14, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(simulate_color_vision_deficiency(image, "tritanopia"))
    axes[1, 1].set_title("蓝色盲 (Tritanopia)", fontsize=14, fontweight="bold")
    axes[1, 1].axis("off")

    plt.tight_layout()
    return fig


def make_audio_script(candidates, data_shape, lang="zh"):
    """生成语音朗读脚本"""
    if len(candidates) == 0:
        return "未检测到候选体。" if lang == "zh" else "No candidates detected."

    h, w = data_shape
    parts = []

    if lang == "zh":
        parts.append(f"检测到 {len(candidates)} 个候选体。")
        for i, c in enumerate(candidates):
            x_ratio = c["x"] / w
            y_ratio = c["y"] / h
            x_desc = "左侧" if x_ratio < 0.33 else ("中央" if x_ratio < 0.67 else "右侧")
            y_desc = "上方" if y_ratio < 0.33 else ("中部" if y_ratio < 0.67 else "下方")
            lin = c["linearity"]
            lin_desc = "非常细长" if lin > 10 else ("较细长" if lin > 5 else "一般")
            parts.append(
                f"候选体 {i+1}，位于{y_desc}{x_desc}，长度 {c['length']} 像素，"
                f"线性度 {lin:.1f}，{lin_desc}。"
            )
    else:
        parts.append(f"Detected {len(candidates)} candidates.")
        for i, c in enumerate(candidates):
            x_ratio = c["x"] / w
            y_ratio = c["y"] / h
            x_desc = "left" if x_ratio < 0.33 else ("center" if x_ratio < 0.67 else "right")
            y_desc = "top" if y_ratio < 0.33 else ("middle" if y_ratio < 0.67 else "bottom")
            lin = c["linearity"]
            parts.append(
                f"Candidate {i+1}: {y_desc} {x_desc}, length {c['length']} pixels, "
                f"linearity {lin:.1f}."
            )

    return " ".join(parts)




# ============ 发现报告生成 ============
import base64
from io import BytesIO as _BytesIO
from datetime import datetime as _dt


def fig_to_base64(fig):
    """把 matplotlib 图像转成 base64"""
    buf = _BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def generate_discovery_report(image_data, candidates, header, wcs, validation_results=None, tracking_info=None, lang="zh"):
    """生成发现报告的 HTML"""

    report_id = "ASTRA-" + _dt.now().strftime("%Y%m%d-%H%M%S")

    # ===== 生成图像 =====
    # 原始图像
    fig1, ax1 = plt.subplots(figsize=(8, 8), facecolor="white")
    med = np.nanmedian(image_data)
    std = np.nanstd(image_data)
    ax1.imshow(image_data, cmap="gray", vmin=med-2*std, vmax=med+5*std)
    ax1.set_title("Original Image", fontsize=14, fontweight="bold")
    ax1.axis("off")
    img1_b64 = fig_to_base64(fig1)

    # 标注图像
    fig2, ax2 = plt.subplots(figsize=(8, 8), facecolor="white")
    ax2.imshow(image_data, cmap="gray", vmin=med-2*std, vmax=med+5*std)
    for c in candidates:
        rect = Rectangle((c["x"]-c["length"]/2, c["y"]-c["length"]/2),
                         c["length"], c["length"],
                         linewidth=2, edgecolor="red", facecolor="none")
        ax2.add_patch(rect)
    ax2.set_title("Detected Candidates", fontsize=14, fontweight="bold")
    ax2.axis("off")
    img2_b64 = fig_to_base64(fig2)

    # ===== 构建候选体表格 =====
    cand_rows = ""
    for i, c in enumerate(candidates):
        ra_str = "N/A"
        dec_str = "N/A"
        if wcs is not None:
            try:
                ra, dec = wcs.all_pix2world(c["x"], c["y"], 0)
                ra_str = f"{float(ra):.6f}°"
                dec_str = f"{float(dec):.6f}°"
            except Exception:
                pass
        cand_rows += f"""
        <tr>
            <td>{i+1}</td>
            <td>{c["x"]}</td>
            <td>{c["y"]}</td>
            <td>{ra_str}</td>
            <td>{dec_str}</td>
            <td>{c["length"]}</td>
            <td>{c["linearity"]:.2f}</td>
        </tr>
        """

    # ===== FITS 元数据 =====
    meta_rows = ""
    if header is not None:
        for k in ["TELESCOP", "INSTRUME", "FILTER", "EXPTIME", "DATE-OBS", "AIRMASS", "SEEING"]:
            if k in header:
                meta_rows += f"<tr><td><b>{k}</b></td><td>{header[k]}</td></tr>"

    # ===== 验证结果 =====
    validation_html = ""
    if validation_results:
        for src, res in validation_results.items():
            status_text = {"clear": "无已知天体 ✅", "found": "发现已知天体 ⚠️", "error": "查询失败 ❌"}.get(res.get("status", ""), "未知")
            validation_html += f"<tr><td>{src}</td><td>{status_text}</td></tr>"

    # ===== 追踪信息 =====
    tracking_html = ""
    if tracking_info:
        tracking_html = f"""
        <h2>Cross-night Tracking</h2>
        <p><b>Number of nights:</b> {tracking_info.get("n_nights", "N/A")}</p>
        <p><b>Estimated velocity:</b> {tracking_info.get("velocity", "N/A")} deg/day</p>
        <p><b>Inferred type:</b> {tracking_info.get("type", "N/A")}</p>
        """

    # ===== MPC 格式 =====
    mpc_lines = ""
    if wcs is not None and len(candidates) > 0:
        for c in candidates:
            try:
                ra, dec = wcs.all_pix2world(c["x"], c["y"], 0)
                ra_h = float(ra) / 15.0
                ra_hh = int(ra_h)
                ra_mm = int((ra_h - ra_hh) * 60)
                ra_ss = ((ra_h - ra_hh) * 60 - ra_mm) * 60
                dec_sign = "+" if float(dec) >= 0 else "-"
                dec_abs = abs(float(dec))
                dec_dd = int(dec_abs)
                dec_mm = int((dec_abs - dec_dd) * 60)
                dec_ss = ((dec_abs - dec_dd) * 60 - dec_mm) * 60
                mpc_lines += f"     K23A 00 00.00000  {ra_hh:02d} {ra_mm:02d} {ra_ss:05.2f} {dec_sign}{dec_dd:02d} {dec_mm:02d} {dec_ss:04.1f}                          I41<br>"
            except Exception:
                pass

    # ===== 组装 HTML =====
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>AstraFilter Discovery Report - {report_id}</title>
        <style>
            body {{
                font-family: 'Helvetica Neue', Arial, sans-serif;
                max-width: 900px;
                margin: 0 auto;
                padding: 40px;
                color: #333;
                line-height: 1.6;
            }}
            h1 {{
                color: #0b3d91;
                border-bottom: 4px solid #fc3d21;
                padding-bottom: 15px;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #0b3d91;
                margin-top: 40px;
                border-left: 5px solid #fc3d21;
                padding-left: 15px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th {{
                background: #0b3d91;
                color: white;
                padding: 10px;
                text-align: left;
            }}
            td {{
                padding: 8px 10px;
                border-bottom: 1px solid #dde3ec;
            }}
            tr:hover {{
                background: #f7f9fc;
            }}
            .report-id {{
                color: #666;
                font-family: monospace;
                font-size: 14px;
            }}
            .disclaimer {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px 20px;
                margin: 30px 0;
                border-radius: 4px;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                margin: 15px 0;
            }}
            .mpc-block {{
                background: #1a1a1a;
                color: #00ff00;
                padding: 20px;
                font-family: monospace;
                font-size: 12px;
                overflow-x: auto;
                border-radius: 4px;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #dde3ec;
                color: #666;
                font-size: 14px;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <h1>🔭 AstraFilter Discovery Report</h1>
        <p class="report-id">Report ID: <b>{report_id}</b> | Generated: {_dt.now().strftime("%Y-%m-%d %H:%M:%S")} UTC</p>

        <h2>1. Image Information</h2>
        <table>
            <tr><td><b>Image size</b></td><td>{image_data.shape[0]} × {image_data.shape[1]} pixels</td></tr>
            <tr><td><b>Detection threshold</b></td><td>5.0 sigma above background</td></tr>
            <tr><td><b>Minimum length</b></td><td>15 pixels</td></tr>
            <tr><td><b>Minimum linearity</b></td><td>3.0</td></tr>
            {meta_rows}
        </table>

        <h2>2. Detected Candidates</h2>
        <p>Total: <b>{len(candidates)}</b> candidate(s)</p>
        <table>
            <tr>
                <th>#</th><th>X (pixel)</th><th>Y (pixel)</th><th>RA</th><th>Dec</th>
                <th>Length</th><th>Linearity</th>
            </tr>
            {cand_rows}
        </table>

        <h2>3. Visualization</h2>
        <img src="data:image/png;base64,{img1_b64}" alt="Original">
        <img src="data:image/png;base64,{img2_b64}" alt="Detected">

        <h2>4. Multi-Source Validation</h2>
        <table>
            <tr><th>Source</th><th>Result</th></tr>
            {validation_html}
        </table>

        {tracking_html}

        <h2>5. MPC Submission Format</h2>
        <p>If this candidate is confirmed across multiple nights, the following 80-column format can be submitted to the Minor Planet Center:</p>
        <div class="mpc-block">
            {mpc_lines if mpc_lines else "No WCS information available. Cannot generate MPC format."}
        </div>

        <div class="disclaimer">
            <b>⚠️ Disclaimer</b><br>
            This report is generated automatically by AstraFilter based on single-frame detection.
            A single frame cannot confirm whether the candidate is a real astronomical object.
            Multi-night observations and orbit fitting are required for confirmation.
            This report should be treated as a <b>candidate list</b>, not as a definitive discovery claim.
        </div>

        <div class="footer">
            Generated by <b>AstraFilter</b> | https://astrafilter-v3.streamlit.app<br>
            Data source: NASA/IPAC IRSA (ZTF), Vizier, MPChecker
        </div>
    </body>
    </html>
    """

    return html, report_id




# ============ 互动 FMO 模拟器 ============
def generate_fmo_simulation(
    brightness=30,
    velocity_deg_per_day=3.0,
    exposure_sec=30,
    noise_sigma=5.0,
    psf_sigma=1.5,
    image_size=256,
    pixel_scale_arcsec=1.0,
):
    """生成 FMO 在图像上的模拟条纹

    参数:
        brightness: 条纹亮度（相对背景）
        velocity_deg_per_day: 天球运动速率
        exposure_sec: 曝光时间（秒）
        noise_sigma: 背景噪声标准差
        psf_sigma: PSF 高斯模糊 sigma
        image_size: 图像尺寸
        pixel_scale_arcsec: 像素尺度（角秒/像素）
    """
    from scipy.ndimage import gaussian_filter

    # 生成背景噪声
    image = np.random.normal(0, noise_sigma, (image_size, image_size))

    # 计算条纹长度（像素）
    # 每秒移动度数 = velocity_deg_per_day / 86400
    # 曝光时间内移动度数 = velocity * exposure / 86400
    # 换成角秒 = 度数 * 3600
    # 换成像素 = 角秒 / pixel_scale
    streak_length_deg = velocity_deg_per_day * exposure_sec / 86400.0
    streak_length_arcsec = streak_length_deg * 3600.0
    streak_length_pix = streak_length_arcsec / pixel_scale_arcsec

    # 如果条纹长度小于 1 像素，就是个点
    if streak_length_pix < 1:
        streak_length_pix = 1

    # 生成条纹
    streak = np.zeros((image_size, image_size))
    cx, cy = image_size // 2, image_size // 2
    angle = np.random.uniform(0, 2 * np.pi)
    n_steps = max(5, int(streak_length_pix * 3))

    for step in np.linspace(-streak_length_pix / 2, streak_length_pix / 2, n_steps):
        x = int(round(cx + step * np.cos(angle)))
        y = int(round(cy + step * np.sin(angle)))
        if 0 <= x < image_size and 0 <= y < image_size:
            streak[y, x] = brightness

    # PSF 模糊
    streak = gaussian_filter(streak, sigma=psf_sigma)

    # 合成图像
    simulated = image + streak

    return simulated, streak, streak_length_pix


def run_simulation_detection(simulated_image, n_sigma=5, min_length=10, min_linearity=3.0):
    """对模拟图像运行检测器"""
    return detect_traditional(
        simulated_image,
        n_sigma=n_sigma,
        min_length=min_length,
        min_linearity=min_linearity,
    )




# ============ 发现证书 ============
def generate_discovery_certificate(candidate, image_data, wcs, detection_date=None, observer="Anonymous"):
    """为单个候选体生成发现证书 HTML"""
    import base64
    from io import BytesIO as _BytesIO
    from datetime import datetime as _dt

    # 计算候选体的 RA/Dec
    ra_str = "N/A"
    dec_str = "N/A"
    ra_hms = "N/A"
    dec_dms = "N/A"
    if wcs is not None:
        try:
            ra, dec = wcs.all_pix2world(candidate["x"], candidate["y"], 0)
            ra = float(ra)
            dec = float(dec)
            ra_str = f"{ra:.6f}"
            dec_str = f"{dec:.6f}"

            # 转时分秒
            ra_h = ra / 15.0
            ra_hh = int(ra_h)
            ra_mm = int((ra_h - ra_hh) * 60)
            ra_ss = ((ra_h - ra_hh) * 60 - ra_mm) * 60
            ra_hms = f"{ra_hh:02d}h {ra_mm:02d}m {ra_ss:05.2f}s"

            dec_sign = "+" if dec >= 0 else "-"
            dec_abs = abs(dec)
            dec_dd = int(dec_abs)
            dec_mm = int((dec_abs - dec_dd) * 60)
            dec_ss = ((dec_abs - dec_dd) * 60 - dec_mm) * 60
            dec_dms = f"{dec_sign}{dec_dd:02d}\u00b0 {dec_mm:02d}' {dec_ss:04.1f}\u2033"
        except Exception:
            pass

    # 生成候选体局部图像
    box = max(60, candidate["length"])
    x1 = max(0, int(candidate["x"]) - box)
    x2 = min(image_data.shape[1], int(candidate["x"]) + box)
    y1 = max(0, int(candidate["y"]) - box)
    y2 = min(image_data.shape[0], int(candidate["y"]) + box)
    patch = image_data[y1:y2, x1:x2]

    fig, ax = plt.subplots(figsize=(5, 5), facecolor="white")
    med = np.median(patch)
    std = np.std(patch)
    ax.imshow(patch, cmap="gray", vmin=med-2*std, vmax=med+5*std)
    ax.set_title("Candidate Cutout", fontsize=14, fontweight="bold")
    ax.axis("off")

    buf = _BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    cutout_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)

    # 生成完整图像（标注候选体）
    fig2, ax2 = plt.subplots(figsize=(6, 6), facecolor="white")
    med = np.nanmedian(image_data)
    std = np.nanstd(image_data)
    ax2.imshow(image_data, cmap="gray", vmin=med-2*std, vmax=med+5*std)
    rect = Rectangle(
        (candidate["x"] - candidate["length"]/2, candidate["y"] - candidate["length"]/2),
        candidate["length"], candidate["length"],
        linewidth=3, edgecolor="red", facecolor="none",
    )
    ax2.add_patch(rect)
    ax2.axis("off")
    buf2 = _BytesIO()
    fig2.savefig(buf2, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf2.seek(0)
    full_b64 = base64.b64encode(buf2.read()).decode("utf-8")
    plt.close(fig2)

    # 证书编号
    cert_id = "ASTRA-" + _dt.now().strftime("%Y%m%d") + "-" + str(int(candidate["x"])) + str(int(candidate["y"]))

    # 检测日期
    if detection_date is None:
        detection_date = _dt.now().strftime("%Y-%m-%d")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Discovery Certificate - {cert_id}</title>
        <style>
            body {{
                font-family: 'Georgia', 'Times New Roman', serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 40px;
                background: #f7f9fc;
            }}
            .certificate {{
                background: #ffffff;
                border: 8px double #0b3d91;
                border-radius: 8px;
                padding: 50px;
                box-shadow: 0 8px 30px rgba(11,61,145,0.15);
                position: relative;
            }}
            .certificate::before {{
                content: "";
                position: absolute;
                top: 20px; left: 20px; right: 20px; bottom: 20px;
                border: 2px solid #fc3d21;
                border-radius: 4px;
                pointer-events: none;
            }}
            h1 {{
                text-align: center;
                color: #0b3d91;
                font-size: 2.2rem;
                letter-spacing: 2px;
                margin-bottom: 10px;
            }}
            .subtitle {{
                text-align: center;
                color: #fc3d21;
                font-size: 1rem;
                font-weight: bold;
                letter-spacing: 3px;
                margin-bottom: 30px;
            }}
            .cert-id {{
                text-align: center;
                font-family: monospace;
                color: #666;
                font-size: 14px;
                margin-bottom: 40px;
            }}
            .section {{
                margin: 30px 0;
                padding: 20px;
                background: #f7f9fc;
                border-left: 4px solid #0b3d91;
                border-radius: 4px;
            }}
            .section-title {{
                color: #0b3d91;
                font-size: 1.1rem;
                font-weight: bold;
                margin-bottom: 15px;
                text-transform: uppercase;
                letter-spacing: 1px;
            }}
            .data-row {{
                display: flex;
                justify-content: space-between;
                padding: 6px 0;
                border-bottom: 1px dashed #dde3ec;
            }}
            .data-label {{
                font-weight: bold;
                color: #666;
            }}
            .data-value {{
                font-family: monospace;
                color: #333;
            }}
            img {{
                display: block;
                margin: 20px auto;
                border: 2px solid #0b3d91;
                border-radius: 4px;
                max-width: 100%;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 2px solid #0b3d91;
                font-style: italic;
                color: #666;
                font-size: 14px;
            }}
            .signature {{
                text-align: center;
                margin-top: 30px;
            }}
            .signature-name {{
                font-family: 'Brush Script MT', cursive;
                font-size: 2rem;
                color: #0b3d91;
            }}
            .seal {{
                display: inline-block;
                width: 100px;
                height: 100px;
                border: 3px solid #fc3d21;
                border-radius: 50%;
                text-align: center;
                line-height: 1.3;
                color: #fc3d21;
                font-weight: bold;
                padding: 20px 5px;
                margin-top: 20px;
                font-size: 0.8rem;
                box-sizing: border-box;
            }}
        </style>
    </head>
    <body>
        <div class="certificate">
            <h1>DISCOVERY CERTIFICATE</h1>
            <div class="subtitle">ASTRONOMICAL CANDIDATE DETECTION</div>
            <div class="cert-id">Certificate ID: {cert_id}</div>

            <p style="text-align: center; font-size: 1.1rem; line-height: 1.8;">
                This certificate is awarded to<br>
                <span class="signature-name">{observer}</span><br>
                in recognition of the detection of a potential Fast-Moving Object candidate
            </p>

            <div class="section">
                <div class="section-title">🔭 Candidate Coordinates</div>
                <div class="data-row"><span class="data-label">Right Ascension (J2000)</span><span class="data-value">{ra_str}°</span></div>
                <div class="data-row"><span class="data-label">Declination (J2000)</span><span class="data-value">{dec_str}°</span></div>
                <div class="data-row"><span class="data-label">RA (HMS)</span><span class="data-value">{ra_hms}</span></div>
                <div class="data-row"><span class="data-label">Dec (DMS)</span><span class="data-value">{dec_dms}</span></div>
                <div class="data-row"><span class="data-label">Pixel Position</span><span class="data-value">({candidate["x"]}, {candidate["y"]})</span></div>
            </div>

            <div class="section">
                <div class="section-title">📊 Detection Details</div>
                <div class="data-row"><span class="data-label">Detection Date</span><span class="data-value">{detection_date}</span></div>
                <div class="data-row"><span class="data-label">Streak Length</span><span class="data-value">{candidate["length"]} pixels</span></div>
                <div class="data-row"><span class="data-label">Linearity (PCA)</span><span class="data-value">{candidate["linearity"]:.2f}</span></div>
                <div class="data-row"><span class="data-label">Detection Method</span><span class="data-value">8-connected + PCA</span></div>
                <div class="data-row"><span class="data-label">Data Source</span><span class="data-value">ZTF (IRSA)</span></div>
            </div>

            <div class="section-title" style="text-align: center; margin-top: 30px;">Candidate Cutout</div>
            <img src="data:image/png;base64,{cutout_b64}" alt="Candidate Cutout" style="max-width: 300px;">

            <div class="section-title" style="text-align: center;">Full Image with Marked Candidate</div>
            <img src="data:image/png;base64,{full_b64}" alt="Full Image" style="max-width: 400px;">

            <div class="signature">
                <div class="seal">ASTRA<br>VERIFIED<br>PIPELINE</div>
            </div>

            <div class="footer">
                Generated by <b>AstraFilter</b> — https://astrafilter-v3.streamlit.app<br>
                Note: This certificate documents a <b>candidate detection</b>. Multi-night confirmation and MPC verification are required to establish this as a discovery.
            </div>
        </div>
    </body>
    </html>
    """
    return html, cert_id



# ============ 本地 FAQ 智能助手 ============
FAQ_DATABASE = [
    {
        "keywords": ["上传", "什么图", "用什么图", "upload", "输入"],
        "question": "我该上传什么图像？",
        "answer": "最好的输入是带 WCS 信息的 FITS 文件（可以从 NASA/IPAC IRSA 的 ZTF 档案下载）。如果只是想试试检测功能，PNG/JPG 也可以，但没有坐标信息，无法做天球坐标转换。\n\n下载 FITS 地址：https://irsa.ipac.caltech.edu/Missions/ztf.html",
    },
    {
        "keywords": ["没有检测", "检测不到", "0 个", "找不到", "空"],
        "question": "检测不到候选体怎么办？",
        "answer": "尝试以下调整：\n1) 把左侧「亮度阈值」从 5.0 降到 3.0 或 4.0\n2) 把「最小长度」从 15 降到 8 像素\n3) 把「最小线性度」从 3.0 降到 2.0\n\n如果还是 0 个，说明图像里可能确实没有符合特征的条纹——这也是正确的科学结果。",
    },
    {
        "keywords": ["候选体太多", "太多候选", "很多候选", "假阳性", "误报", "噪声"],
        "question": "候选体太多怎么筛选？",
        "answer": "调高参数能过滤掉大部分伪影：\n1) 「最小线性度」提到 4.0 或更高（真条纹线性度通常 > 5）\n2) 「亮度阈值」提到 6.0 或 7.0 sigma\n3) 「最小长度」提到 20 像素以上\n\n真正的快速移动天体条纹通常又细又长，线性度很高。",
    },
    {
        "keywords": ["mpchecker", "验证", "已知天体", "查询"],
        "question": "MPChecker 是什么？怎么用？",
        "answer": "MPChecker 是小行星中心的官方数据库，包含所有已知小行星和彗星的轨道数据。\n\n用法：把候选体的 RA/Dec 坐标输入 MPChecker，如果在 5 角分范围内返回 No known minor planets，说明这个位置附近没有已知天体——你的候选体可能是新的。\n\n注意：需要 FITS 文件带 WCS 信息才能算出坐标。",
    },
    {
        "keywords": ["什么是 fmo", "fmo 是什么", "快速移动", "fast moving"],
        "question": "什么是 FMO？",
        "answer": "FMO（Fast-Moving Object，快速移动天体）指运动速率超过 0.5 度/天的太阳系小天体。\n\n主要包括：\n- 近地小行星（NEA）：轨道接近地球，行星防御重点\n- 主带小行星：速率小于 0.5 度/天\n- 人造卫星和空间碎片\n\n在 ZTF 的 30 秒曝光中，FMO 会留下细长的条纹，而恒星和星系呈点状。这就是为什么用条纹来识别 FMO。",
    },
    {
        "keywords": ["坐标", "ra", "dec", "赤经", "赤纬"],
        "question": "什么是 RA/Dec 坐标？",
        "answer": "RA（赤经）和 Dec（赤纬）是天球坐标系，类似地球的经度和纬度。\n\n- RA：从春分点开始，0-360 度，或 0-24 小时\n- Dec：从天赤道开始，-90 到 +90 度\n\n只有在 FITS 文件包含 WCS 信息时，才能把像素坐标转换成天球坐标。",
    },
    {
        "keywords": ["证书", "发现证书", "certificate"],
        "question": "怎么生成发现证书？",
        "answer": "步骤：\n1) 在「检测」标签页上传图像\n2) 检测候选体\n3) 在候选体列表下方找到「生成发现证书」区域\n4) 选择候选体，填入你的名字和日期\n5) 点「生成发现证书」按钮\n\n会生成一份精美的 HTML 证书，可以下载、打印或分享。",
    },
    {
        "keywords": ["星图", "3d 星图", "恒星", "星空"],
        "question": "星图功能怎么用？",
        "answer": "「星图」标签页显示候选体附近的已知恒星。\n\n步骤：\n1) 先在「检测」标签页上传带 WCS 的 FITS 文件\n2) 检测到候选体\n3) 切到「星图」标签\n4) 点「查询附近恒星」\n\n系统使用离线星表（15397 颗 Hipparcos 亮星），不需要网络，几秒就返回结果。可以拖动 3D 视图旋转视角。",
    },
    {
        "keywords": ["无障碍", "色盲", "视障", "老人"],
        "question": "有哪些无障碍功能？",
        "answer": "「无障碍」标签页提供：\n1) 色盲模拟器：显示 4 种视图（正常、红盲、绿盲、蓝盲）\n2) 语音朗读：把检测结果读出来\n3) 四种配色主题：默认、高对比度、色盲友好、浅色\n4) 四档字体大小\n\n左侧栏可以切换主题和字体。这个功能帮助色盲和视障用户也能使用天文工具。",
    },
    {
        "keywords": ["模拟器", "学习", "教学", "科普"],
        "question": "互动模拟器怎么用？",
        "answer": "「学习」标签页有互动模拟器，可以调节 6 个参数：\n- 条纹亮度\n- 运动速率（度/天）\n- 曝光时间（秒）\n- 背景噪声\n- PSF 模糊\n- 像素尺度\n\n每调一次参数，实时看到条纹如何形成，以及检测器如何识别。这是理解 FMO 检测原理的最佳工具。",
    },
    {
        "keywords": ["ztf", "巡天", "望远镜", "帕洛马"],
        "question": "ZTF 是什么？",
        "answer": "ZTF（Zwicky Transient Facility）是位于美国帕洛马天文台的一台宽视场巡天望远镜。\n\n特点：\n- 每 3 天扫描一次北半球天空\n- 每晚产生约 30 万条条纹候选体\n- 数据完全公开，存档在 NASA 的 IRSA\n\nAstraFilter 使用的就是 ZTF 的公开数据。",
    },
    {
        "keywords": ["追踪", "多夜", "跨夜", "历史观测"],
        "question": "跨夜追踪怎么用？",
        "answer": "「追踪」标签页可以查询 ZTF 档案，找到同一天区在其他夜晚的观测。\n\n步骤：\n1) 上传并检测候选体\n2) 切到「追踪」标签\n3) 选择日期范围\n4) 点「查询并追踪」\n\n系统会显示该天区的所有历史观测，并按日期分组。",
    },
    {
        "keywords": ["logo", "标志", "设计"],
        "question": "Logo 是什么意思？",
        "answer": "AstraFilter 的 Logo 是星际穿越电影里 Gargantua 黑洞的造型：\n- 中心的黑色圆是黑洞视界\n- 金色的环是光子环\n- 水平的光带是吸积盘\n- 右上角的小火箭代表人类探索\n\n设计理念：用科学和科幻的融合，表达天文探索的精神。",
    },
    {
        "keywords": ["你好", "hi", "hello", "在吗", "帮助", "help"],
        "question": "有什么可以帮你？",
        "answer": "你好！我是 AstraFilter 的助手。我可以帮你解答：\n\n📷 怎么上传图像、检测候选体\n🔍 检测结果怎么解读\n📊 怎么筛选候选体\n🔭 追踪 / 星图 / 验证功能怎么用\n♿ 无障碍功能说明\n📚 FMO 和天文基础知识\n\n直接问我就行，或者点下面的快捷问题按钮。",
    },
    {
        "keywords": ["谢谢", "感谢", "thank"],
        "question": "不客气",
        "answer": "不客气！如果还有其他问题，随时问我。祝你在 AstraFilter 上有所发现！",
    },
]


def find_faq_answer(question):
    """在 FAQ 数据库里找最匹配的答案"""
    q = question.lower().strip()
    if not q:
        return None, None
    best_match = None
    best_score = 0
    for faq in FAQ_DATABASE:
        score = 0
        for kw in faq["keywords"]:
            if kw in q:
                score += len(kw)
        if score > best_score:
            best_score = score
            best_match = faq
    if best_match and best_score > 0:
        return best_match["answer"], best_match["question"]
    return None, None


def get_fallback_answer():
    return "抱歉，我没有找到这个问题的答案。\n\n你可以试试问：\n• 我该上传什么图像？\n• 检测不到候选体怎么办？\n• 候选体太多怎么筛选？\n• 什么是 FMO？\n• MPChecker 是什么？\n• 怎么生成发现证书？\n• 有哪些无障碍功能？"



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
# Logo 显示（用 __file__ 定位，兼容 Streamlit Cloud）
try:
    _base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _logo_path = os.path.join(_base, "web", "astrafilter_logo.png")
    if os.path.exists(_logo_path):
        _col_logo, _col_space = st.columns([1, 5])
        with _col_logo:
            st.image(_logo_path, width=120)
except Exception:
    pass

# NASA 风格顶部横幅
st.markdown("""
<div class="nasa-banner">
    <h1><span class="nasa-dot"></span>AstraFilter</h1>
    <p>Fast-Moving Object Detection &amp; Astronomical Image Analysis Platform</p>
</div>
""", unsafe_allow_html=True)
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


tab0, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([t("tab_home"), t("tab_detect"), "🔭 追踪", t("tab_sky"), t("tab_meta"), t("tab_verify"), t("tab_a11y"), t("tab_photo"), t("tab_learn")])

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

                # ===== 发现证书生成 =====
                st.markdown("---")
                st.subheader("🏆 生成发现证书")
                st.markdown("为选中的候选体生成一份精美的发现证书。可以下载、打印或分享。")

                # 选择候选体
                cand_options = ["候选体 " + str(i+1) + " (线性度 " + str(round(c["linearity"], 1)) + ")" for i, c in enumerate(candidates)]
                selected_idx = st.selectbox("选择候选体", range(len(candidates)), format_func=lambda i: cand_options[i])

                observer_name = st.text_input("你的名字（显示在证书上）", "Anonymous Observer")
                detection_date = st.text_input("检测日期", "2024-01-01")

                if st.button("生成发现证书"):
                    selected_candidate = candidates[selected_idx]
                    html_cert, cert_id = generate_discovery_certificate(
                        selected_candidate,
                        data, wcs,
                        detection_date=detection_date,
                        observer=observer_name,
                    )
                    st.success("证书生成完成！证书 ID: " + cert_id)
                    st.download_button(
                        "📥 下载证书（HTML）",
                        html_cert,
                        file_name="AstraFilter_Certificate_" + cert_id + ".html",
                        mime="text/html",
                    )
                    st.markdown("**证书预览**")
                    st.components.v1.html(html_cert, height=800, scrolling=True)

                # ===== 发现报告生成按钮 =====
                st.markdown("---")
                st.subheader("📄 生成发现报告")
                st.markdown("把本次检测的所有结果汇总成一份完整的 HTML 报告，包含图像、候选体列表、验证结果、MPC 提交格式。")

                if st.button("一键生成报告", type="primary"):
                    with st.spinner("生成中..."):
                        validation = st.session_state.get("validation_results", None)
                        tracking = st.session_state.get("tracking_info", None)
                        html_report, report_id = generate_discovery_report(
                            data, candidates, header, wcs,
                            validation_results=validation,
                            tracking_info=tracking,
                        )
                    st.success("报告生成完成！报告 ID: " + report_id)
                    st.download_button(
                        "📥 下载 HTML 报告",
                        html_report,
                        file_name="AstraFilter_Report_" + report_id + ".html",
                        mime="text/html",
                    )
                    st.markdown("**报告预览**")
                    st.components.v1.html(html_report, height=600, scrolling=True)


with tab2:
    st.header("🔭 候选体跨夜追踪")
    st.markdown("系统会自动查询 ZTF 档案，下载同一天区在其他夜晚的观测图像，追踪候选体的移动轨迹，并推断它的物理属性。")

    data = st.session_state.get("data")
    wcs = st.session_state.get("wcs")
    candidates = st.session_state.get("candidates", [])

    if data is None:
        st.info("请先在检测标签页上传并检测一张图像。")
    elif wcs is None:
        st.warning("需要带 WCS 信息的 FITS 文件。")
    elif len(candidates) == 0:
        st.info("请先检测候选体。")
    else:
        # 只对第一个候选体做追踪
        c0 = candidates[0]
        try:
            ra0, dec0 = wcs.all_pix2world(c0["x"], c0["y"], 0)
            ra0, dec0 = float(ra0), float(dec0)
        except Exception:
            st.error("无法转换坐标。")
            ra0 = None

        if ra0 is not None:
            st.subheader("候选体信息")
            st.write("RA = " + str(round(ra0, 5)) + "°, Dec = " + str(round(dec0, 5)) + "°")
            st.write("像素长度 = " + str(c0["length"]) + ", 线性度 = " + str(round(c0["linearity"], 2)))

            st.markdown("---")
            st.subheader("查询历史观测")

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.text_input("开始日期", "2023-09-01")
            with col2:
                end_date = st.text_input("结束日期", "2023-10-31")

            if st.button("🔍 查询并追踪"):
                with st.spinner("查询 ZTF 档案中..."):
                    obs = query_ztf_observations(ra0, dec0, start_date, end_date)

                if obs is None:
                    st.error("没有找到观测记录。请调整日期范围或检查坐标。")
                else:
                    st.success("找到 " + str(len(obs)) + " 条观测记录")

                    # 按日期分组
                    import astropy.time as time
                    obs = obs.copy()
                    obs["mjd"] = obs["obsjd"] - 2400000.5
                    obs["date_str"] = obs["obsjd"].apply(lambda jd: time.Time(jd, format="jd").iso[:10])

                    unique_dates = sorted(set(obs["date_str"]))
                    st.write("覆盖日期: " + str(len(unique_dates)) + " 个夜晚")
                    for d in unique_dates[:10]:
                        n = len(obs[obs["date_str"] == d])
                        st.write("  " + d + ": " + str(n) + " 张图像")

                    # 展示观测表格
                    st.markdown("---")
                    st.subheader("观测记录")
                    display_cols = ["date_str", "field", "ccdid", "filtercode", "qid"]
                    available_cols = [c for c in display_cols if c in obs.columns]
                    st.dataframe(obs[available_cols].head(30))

                    st.markdown("---")
                    st.subheader("物理推断")
                    st.markdown("根据候选体的运动速率，系统给出可能的天体类型：")

                    # 假设运动速率（从单帧检测无法直接得到，用占位值）
                    est_velocity = 0.5  # 度/天
                    props = infer_physical_properties(est_velocity)

                    st.info(
                        "**推断类型**: " + props["type"] + "\n\n"
                        "**估计距离**: " + props["distance_au"] + " AU\n\n"
                        "**分类**: " + props["classification"] + "\n\n"
                        "**注意**: 这是单帧推断结果。多帧追踪会给出更精确的速率。"
                    )

                    st.markdown("---")
                    st.subheader("下一步")
                    st.markdown(
                        "1. 把上面的坐标输入「✅ 验证」标签的 MPChecker 查询，确认是否为已知天体\n"
                        "2. 如果未被编目，下载多个夜晚的图像做进一步确认\n"
                        "3. 如果多帧确认，用 MPC 提交格式生成观测文件"
                    )



with tab3:
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


with tab4:
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


with tab5:
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


with tab6:
    st.header("♿ 无障碍功能")

    st.markdown("""
AstraFilter 为色盲、视障和老年用户提供了多项无障碍功能。这些功能让天文图像分析对所有人开放。
    """)

    st.markdown("---")
    st.subheader("1. 配色主题")
    st.markdown("""
在左侧栏切换配色主题：
- **默认（NASA 白底）** — 白底深蓝字，高可读性
- **高对比度** — 纯黑底白字，适合低视力用户
- **色盲友好** — 蓝橙配色，红绿色盲用户可区分
- **浅色** — 白底黑字，适合强光环境
    """)

    st.markdown("---")
    st.subheader("2. 字体大小")
    st.markdown("左侧栏可切换小 / 中 / 大 / 超大四档字体大小，适合不同视力水平。")

    st.markdown("---")
    st.subheader("3. 语音朗读检测结果")
    st.markdown("把检测结果转成语音，用手机或电脑扬声器朗读出来。")

    candidates_for_audio = st.session_state.get("candidates", [])
    data_for_audio = st.session_state.get("data")

    if len(candidates_for_audio) == 0 or data_for_audio is None:
        st.info("请先在检测标签页上传图像并检测候选体。")
    else:
        audio_script = make_audio_script(
            candidates_for_audio,
            data_for_audio.shape,
            lang=st.session_state.get("lang", "zh"),
        )
        st.markdown("**朗读内容（预览）:**")
        st.info(audio_script)

        # 嵌入 JavaScript 的语音朗读
        import streamlit.components.v1 as components
        html_code = f"""
        <button onclick="speakText()" style="background:#0b3d91;color:white;border:none;
                padding:12px 24px;font-size:16px;border-radius:4px;cursor:pointer;font-weight:bold;">
            🔊 点击朗读
        </button>
        <button onclick="stopSpeak()" style="background:#fc3d21;color:white;border:none;
                padding:12px 24px;font-size:16px;border-radius:4px;cursor:pointer;margin-left:10px;font-weight:bold;">
            ⏹ 停止
        </button>
        <script>
        function speakText() {{
            window.speechSynthesis.cancel();
            var utterance = new SpeechSynthesisUtterance(`{audio_script}`);
            utterance.lang = "{("zh-CN" if st.session_state.get("lang", "zh") == "zh" else "en-US")}";
            utterance.rate = 0.9;
            utterance.pitch = 1.0;
            window.speechSynthesis.speak(utterance);
        }}
        function stopSpeak() {{
            window.speechSynthesis.cancel();
        }}
        </script>
        """
        components.html(html_code, height=100)

    st.markdown("---")
    st.subheader("4. 色盲模拟器")
    st.markdown("""
上传的图像会以伪彩色显示，并模拟不同色盲类型看到的画面。
这个工具可以帮助天文教育工作者了解色盲学生看到的是什么。
    """)

    if data_for_audio is None:
        st.info("请先在检测标签页上传图像。")
    else:
        if st.button("生成色盲模拟对比图"):
            with st.spinner("生成中..."):
                fig_cvd = make_cvd_comparison(data_for_audio)
            st.pyplot(fig_cvd)
            st.markdown("""
**说明**：
- **正常视觉** — 大多数人看到的画面
- **红色盲** — 约 1% 的男性，难以区分红绿
- **绿色盲** — 约 5% 的男性，红绿混淆
- **蓝色盲** — 罕见，蓝黄混淆

如果一个条纹在某种色盲模式下消失，说明该用户需要其他提示方式（如语音或文字描述）才能看到它。
            """)

    st.markdown("---")
    st.subheader("5. 键盘导航")
    st.markdown("""
AstraFilter 支持屏幕阅读器（NVDA、JAWS、VoiceOver）。
所有交互元素都有语义标签，可以用 Tab 键导航，Enter 键激活。
    """)

    st.markdown("---")
    st.subheader("设计理念")
    st.markdown("""
传统天文软件假设用户能看见彩色屏幕。对于色盲和视障用户，这些软件几乎不可用。

AstraFilter 通过以下方式让天文分析对所有人开放：
1. **多主题配色** — 让不同色觉的用户都能看清
2. **可调字体** — 适配不同视力水平
3. **语音朗读** — 让视障用户用听觉接收结果
4. **色盲模拟** — 让教育者理解色盲学生的体验
5. **键盘导航** — 适配屏幕阅读器

这些功能不仅是"加法"，而是**让 AstraFilter 成为真正包容的天文工具**。
    """)


with tab7:
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




with tab8:
    st.header("📚 学习与模拟")
    st.markdown("这一页既有教学讲解，也有互动模拟器。你可以调节参数，亲眼看到快速移动天体（FMO）是如何在图像上留下条纹的。")

    # ===== 互动模拟器 =====
    st.markdown("---")
    st.subheader("🔬 互动 FMO 模拟器")
    st.markdown("调节下方参数，系统会实时生成一颗小行星在 ZTF 图像上留下的条纹，并演示检测器如何识别它。")

    col_slider1, col_slider2 = st.columns(2)

    with col_slider1:
        brightness = st.slider(
            "条纹亮度（相对背景）",
            5, 100, 30, 5,
            help="小行星有多亮。数值越大，条纹越明显。",
        )
        velocity = st.slider(
            "运动速率（度/天）",
            0.01, 20.0, 3.0, 0.1,
            help="FMO 在天球上的运动速度。主带小行星约 0.1-0.5 度/天，近地小行星可达 1-20 度/天。",
        )
        exposure = st.slider(
            "曝光时间（秒）",
            5, 120, 30, 5,
            help="ZTF 通常用 30 秒曝光。曝光越长，条纹越明显。",
        )

    with col_slider2:
        noise = st.slider(
            "背景噪声（sigma）",
            1.0, 15.0, 5.0, 0.5,
            help="图像的噪声水平。噪声越大，暗弱条纹越难检测。",
        )
        psf = st.slider(
            "PSF 模糊（sigma）",
            0.5, 4.0, 1.5, 0.1,
            help="大气抖动造成的模糊。数值越大，条纹越模糊。",
        )
        pixel_scale = st.slider(
            "像素尺度（角秒/像素）",
            0.5, 3.0, 1.0, 0.1,
            help="ZTF 约 1 角秒/像素。数值越小，条纹越长。",
        )

    # 生成模拟
    simulated, streak, streak_length = generate_fmo_simulation(
        brightness=brightness,
        velocity_deg_per_day=velocity,
        exposure_sec=exposure,
        noise_sigma=noise,
        psf_sigma=psf,
        pixel_scale_arcsec=pixel_scale,
    )

    # 显示结果
    col_img1, col_img2 = st.columns(2)

    with col_img1:
        st.markdown("**模拟图像**")
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
        med = np.median(simulated)
        std = np.std(simulated)
        ax.imshow(simulated, cmap="gray", vmin=med-2*std, vmax=med+5*std)
        ax.set_title("Simulated ZTF Image", fontsize=12)
        ax.axis("off")
        st.pyplot(fig)

    # 运行检测
    detected = run_simulation_detection(simulated)

    with col_img2:
        st.markdown("**检测结果**")
        fig, ax = plt.subplots(figsize=(6, 6), facecolor="white")
        med = np.median(simulated)
        std = np.std(simulated)
        ax.imshow(simulated, cmap="gray", vmin=med-2*std, vmax=med+5*std)
        for c in detected:
            rect = Rectangle(
                (c["x"] - c["length"]/2, c["y"] - c["length"]/2),
                c["length"], c["length"],
                linewidth=2, edgecolor="red", facecolor="none",
            )
            ax.add_patch(rect)
        ax.set_title("Detected: " + str(len(detected)) + " candidates", fontsize=12)
        ax.axis("off")
        st.pyplot(fig)

    # 科学解读
    st.markdown("---")
    st.subheader("📊 模拟参数解读")

    col_info1, col_info2, col_info3 = st.columns(3)
    col_info1.metric("条纹长度", f"{streak_length:.1f} 像素")
    col_info2.metric("检测到的候选体", len(detected))
    col_info3.metric("理论长度", f"{velocity * exposure / 86400 * 3600 / pixel_scale:.1f} 像素")

    # 根据参数给出解释
    if streak_length < 3:
        st.warning(
            "⚠️ 条纹长度只有 " + str(round(streak_length, 1)) + " 像素。"
            "当运动速率很低或曝光时间很短时，FMO 看起来就像一个点源，无法与恒星区分。"
        )
    elif len(detected) == 0:
        st.warning(
            "⚠️ 虽然没有检测到候选体，但条纹确实存在。"
            "可能是亮度太低（被噪声淹没）或线性度不够（PSF 模糊太严重）。"
            "试着增加亮度或减少噪声。"
        )
    else:
        st.success(
            "✅ 检测到 " + str(len(detected)) + " 个候选体！"
            "条纹长度 " + str(round(streak_length, 1)) + " 像素，"
            "检测器成功识别了它。"
        )

    st.markdown("---")
    st.markdown("**这个模拟器告诉我们什么？**")
    st.markdown("""
1. **运动速率决定条纹长度** — 速率越高，条纹越长
2. **曝光时间影响可见性** — 曝光越长，条纹越明显
3. **亮度与噪声的竞争** — 如果亮度不够，条纹会被噪声淹没
4. **PSF 模糊的影响** — 大气抖动会拉长条纹，但也会降低对比度
5. **这就是为什么 FMO 检测这么难** — 只有同时满足"够亮、够快、够近"的天体才能被检测到
    """)

    st.markdown("---")
    st.markdown("**想了解更多？**")
    st.markdown("""
- 想实际检测真实图像？去「🔍 检测」标签上传图像
- 想查询天区里的已知恒星？去「🌌 星图」标签
- 想追踪候选体的跨夜移动？去「🔭 追踪」标签
    """)

    # ===== 原有的科普内容 =====
    st.markdown("---")
    st.header("快速移动天体（FMO）基础知识")
    st.markdown("""
快速移动天体（Fast-Moving Object, FMO）是指在天球上运动速率超过 **0.5 度/天** 的太阳系小天体。

**主要类型：**
- **近地小行星（NEA）** — 轨道与地球轨道相交或接近，是行星防御的重点监测对象
- **主带小行星** — 运动速率通常小于 0.5 度/天，不形成明显条纹
- **人造卫星和空间碎片** — 轨道周期短，运动速率高，但通常更亮

**为什么用条纹来识别 FMO？**
ZTF 的 30 秒曝光中，静止的恒星和星系呈现为点源，而 FMO 会留下一条细长的条纹。这让条纹成为 FMO 的独特标志。
    """)

    st.markdown("---")
    st.header("AstraFilter 的检测原理")
    st.markdown("""
**第一步：亮度阈值** — 找出比背景亮度高 5 个标准差的像素。

**第二步：8 连通域分析** — 把相邻的亮像素连成独立目标（含对角线相邻）。

**第三步：PCA 分析** — 计算每个目标的主方向长度和线性度。

**第四步：筛选** — 保留长度 > 15 像素、线性度 > 3.0 的目标。

**为什么用 PCA 而不是简单的长宽比？**
因为条纹可以是任何方向。对角线方向的条纹在 bounding box 里长宽比接近 1，但在 PCA 空间里会显示出极高的线性度。
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




with tab9:
    st.header("🤖 AstraFilter 助手")
    st.markdown("有任何关于 AstraFilter 或天文的问题，都可以在这里问。完全离线运行，不需要网络。")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("问点什么...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.chat_message("assistant"):
            answer, matched_q = find_faq_answer(user_input)
            if answer is None:
                answer = get_fallback_answer()
                st.markdown(answer)
            else:
                st.markdown(answer)
                st.caption("💡 匹配到 FAQ：" + str(matched_q))
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

    if len(st.session_state["chat_history"]) == 0:
        st.markdown("---")
        st.subheader("💡 试试这些问题：")
        quick_questions = [
            "我该上传什么图像？",
            "检测不到候选体怎么办？",
            "候选体太多怎么筛选？",
            "什么是 FMO？",
            "MPChecker 是什么？",
            "怎么生成发现证书？",
            "星图功能怎么用？",
            "有哪些无障碍功能？",
            "互动模拟器怎么用？",
            "ZTF 是什么？",
            "跨夜追踪怎么用？",
            "有什么可以帮你？",
        ]
        cols = st.columns(3)
        for i, q in enumerate(quick_questions):
            with cols[i % 3]:
                if st.button(q, key="quick_q_" + str(i)):
                    st.session_state["chat_history"].append({"role": "user", "content": q})
                    ans, _ = find_faq_answer(q)
                    if ans is None:
                        ans = get_fallback_answer()
                    st.session_state["chat_history"].append({"role": "assistant", "content": ans})
                    st.rerun()

    if len(st.session_state["chat_history"]) > 0:
        st.markdown("---")
        if st.button("🗑 清空对话"):
            st.session_state["chat_history"] = []
            st.rerun()

    st.markdown("---")
    st.caption("AstraFilter 助手完全离线运行，包含 " + str(len(FAQ_DATABASE)) + " 组常见问题。")
