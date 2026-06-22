"""
=============================================================================
English OCR 手写识别翻译助手 - 全局配置文件
=============================================================================
功能: 集中管理 API 密钥、模型参数、预处理阈值、路径等配置项

便携设计说明:
  - 所有模型文件存放在 ./models/ 目录下（项目文件夹内）
  - API Key 保存在 ./api_key.txt 中（不会被 Git 上传）
  - 整个文件夹拷贝到任何电脑都能直接运行
  - 首次运行自动创建所需目录

模型目录结构:
  models/
  ├── easyocr/          # EasyOCR 英文识别模型 (~68MB)
  └── huggingface/      # Sentence-Transformers 嵌入模型 (~80MB, 可选)
=============================================================================
"""

import os
from pathlib import Path
import ssl

# ---- Fix SSL cert verification (cloud servers often lack CA certs) ----
ssl._create_default_https_context = ssl._create_unverified_context
os.environ.setdefault("PYTHONHTTPSVERIFY", "0")

# ==================== 项目根目录 ====================
ROOT_DIR = Path(__file__).parent.absolute()

# ==================== 便携模型目录 ====================
# 所有模型文件存放在项目文件夹内的 models/ 目录
# 拷贝到新电脑时，整个文件夹一起拷贝即可
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ---- HuggingFace 缓存重定向到本地 ----
HF_CACHE_DIR = MODELS_DIR / "huggingface"
HF_CACHE_DIR.mkdir(exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")

# ---- EasyOCR 模型目录重定向到本地 ----
EASYOCR_MODEL_DIR = MODELS_DIR / "easyocr"
EASYOCR_MODEL_DIR.mkdir(exist_ok=True)
os.environ["EASYOCR_MODULE_PATH"] = str(EASYOCR_MODEL_DIR)

# ---- HuggingFace 镜像（国内加速，可选） ----
# 取消注释下面这行即可使用国内镜像下载模型
# os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ==================== API Key 持久化 ====================
# 优先级: 环境变量 > api_key.txt 文件 > config.py 默认值
# api_key.txt 在项目文件夹内，不会被 Git 追踪

API_KEY_FILE = ROOT_DIR / "api_key.txt"

def load_api_key() -> str:
    """从环境变量或本地文件加载 API Key"""
    # 优先用环境变量
    env_key = os.environ.get("LLM_API_KEY", "")
    if env_key and env_key != "your-api-key-here":
        return env_key

    # 其次读取项目内的 api_key.txt
    if API_KEY_FILE.exists():
        try:
            with open(API_KEY_FILE, "r", encoding="utf-8") as f:
                saved_key = f.read().strip()
                if saved_key:
                    return saved_key
        except Exception:
            pass

    # 都没配就先返回占位符
    return "your-api-key-here"

def save_api_key(key: str) -> bool:
    """保存 API Key 到本地文件（持久化）"""
    try:
        with open(API_KEY_FILE, "w", encoding="utf-8") as f:
            f.write(key.strip())
        # 更新全局变量
        global LLM_API_KEY
        LLM_API_KEY = key.strip()
        # 同时写入 .gitignore 确保不提交
        _ensure_gitignore()
        return True
    except Exception:
        return False

def _ensure_gitignore():
    """确保 api_key.txt 和 models/ 被 gitignore"""
    gitignore_path = ROOT_DIR / ".gitignore"
    entries = {"api_key.txt", "models/", ".env", "__pycache__/", "*.pyc"}
    existing = set()
    if gitignore_path.exists():
        with open(gitignore_path, "r", encoding="utf-8") as f:
            existing = {line.strip() for line in f}
    missing = entries - existing
    if missing:
        with open(gitignore_path, "a", encoding="utf-8") as f:
            for entry in sorted(missing):
                f.write(f"\n{entry}")

LLM_API_KEY = load_api_key()

# ==================== LLM API 配置 ====================
# ---------- 大模型提供商配置 ----------
# 支持 OpenAI 兼容接口，修改 base_url 即可切换提供商
LLM_PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek V3",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "description": "DeepSeek V3，性价比最高，中文强（推荐）",
    },
    "deepseek-r1": {
        "name": "DeepSeek R1 (推理)",
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-reasoner",
        "description": "DeepSeek R1 推理模型，复杂语法分析",
    },
    "qwen": {
        "name": "通义千问 (Qwen)",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": "qwen-turbo",
        "description": "阿里云通义千问，中文翻译能力强",
    },
    "glm": {
        "name": "智谱 GLM",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "model": "glm-4-flash",
        "description": "智谱 GLM-4 系列，性价比高",
    },
    "openai": {
        "name": "OpenAI GPT",
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "description": "OpenAI GPT-4o-mini，综合能力强",
    },
}

# 当前使用的提供商
DEFAULT_PROVIDER = "deepseek"

# LLM 调用参数
LLM_MAX_TOKENS = 2048
LLM_TEMPERATURE = 0.3
LLM_TIMEOUT = 60
LLM_MAX_RETRIES = 3

# ==================== OCR 配置 ====================
OCR_CONFIDENCE_THRESHOLD = 0.3
OCR_LANGUAGES = ["en"]
OCR_GPU_ENABLED = True           # 是否启用 GPU 加速（无 GPU 自动回退 CPU）

# ==================== 图像预处理配置 ====================
PREPROCESSING = {
    "denoise_strength": 10,
    "clahe_clip_limit": 2.0,
    "clahe_grid_size": (8, 8),
    "binary_block_size": 15,
    "binary_C": 10,
    "deskew_max_angle": 45,
    "min_contour_area": 50,
}

# ==================== 知识库配置 ====================
KB_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KB_RETRIEVAL_TOP_K = 5
KB_DIR = ROOT_DIR / "knowledge"
KB_INDEX_PATH = KB_DIR / "faiss_index.bin"
KB_META_PATH = KB_DIR / "index_metadata.json"

# ==================== 文本清洗配置 ====================
OCR_CONFUSION_PAIRS = {
    "rn": "m", "cl": "d", "vv": "w",
    "1": "l", "0": "o", "5": "s",
    "ri": "n",
}

# ==================== 前端配置 ====================
GRADIO_TITLE = "英语手写 OCR 识别翻译助手"
GRADIO_THEME = "soft"
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False
