"""全局配置"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 图片库目录
IMAGE_LIBRARY_DIR = PROJECT_ROOT / "image_library"

# 上传临时目录
UPLOAD_DIR = PROJECT_ROOT / "uploads"

# 数据库路径
DB_PATH = PROJECT_ROOT / "image_library.db"

# FAISS 索引路径
FAISS_INDEX_PATH = PROJECT_ROOT / "faiss_index.bin"

# FAISS ID 映射路径
FAISS_ID_MAP_PATH = PROJECT_ROOT / "faiss_id_map.json"

# Chinese-CLIP 模型配置
CLIP_MODEL_NAME = "RN50"          # 可选: RN50, ViT-B-16, ViT-L-14, ViT-L-14-336, ViT-H-14
CLIP_DEVICE = "cpu"                # cpu 或 cuda
CLIP_DOWNLOAD_ROOT = str(PROJECT_ROOT / "models")

# 向量维度 (RN50=1024, ViT-B-16=512, ViT-L-14=768, ViT-H-14=1024)
EMBEDDING_DIM = 1024

# 默认阈值
DEFAULT_SIMILARITY_THRESHOLD = 0.85

# 图片哈希
HASH_SIZE = 16  # 感知哈希尺寸

# 支持的图片格式
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}

# 确保目录存在
IMAGE_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
os.makedirs(CLIP_DOWNLOAD_ROOT, exist_ok=True)
