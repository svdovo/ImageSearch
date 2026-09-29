"""图片处理模块 - 向量化、哈希计算、元数据提取"""
import io
import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
import imagehash

from config import (
    CLIP_MODEL_NAME, CLIP_DEVICE, CLIP_DOWNLOAD_ROOT,
    EMBEDDING_DIM, HASH_SIZE, SUPPORTED_EXTENSIONS
)

# 延迟加载，避免启动时卡住
_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def _load_clip_model():
    """延迟加载 Chinese-CLIP 模型"""
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is not None:
        return

    import torch
    import cn_clip.clip as clip
    from cn_clip.clip import load_from_name

    print(f"[ImageProcessor] 正在加载 Chinese-CLIP 模型 ({CLIP_MODEL_NAME})...")
    _clip_model, _clip_preprocess = load_from_name(
        CLIP_MODEL_NAME, device=CLIP_DEVICE,
        download_root=CLIP_DOWNLOAD_ROOT, use_modelscope=True
    )
    _clip_model.eval()
    _clip_tokenizer = clip
    print(f"[ImageProcessor] Chinese-CLIP 模型加载完成")


def compute_image_hash(image: Image.Image) -> Tuple[str, str, str]:
    """计算图片的三种感知哈希"""
    phash = str(imagehash.phash(image, hash_size=HASH_SIZE))
    dhash = str(imagehash.dhash(image, hash_size=HASH_SIZE))
    ahash = str(imagehash.average_hash(image, hash_size=HASH_SIZE))
    return phash, dhash, ahash


def compute_embedding(image: Image.Image) -> np.ndarray:
    """使用 Chinese-CLIP 计算图片向量"""
    import torch

    _load_clip_model()

    processed = _clip_preprocess(image).unsqueeze(0).to(CLIP_DEVICE)
    with torch.no_grad():
        features = _clip_model.encode_image(processed)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten().astype(np.float32)


def compute_text_embedding(text: str) -> np.ndarray:
    """使用 Chinese-CLIP 计算文本向量"""
    import torch

    _load_clip_model()

    tokens = _clip_tokenizer.tokenize([text]).to(CLIP_DEVICE)
    with torch.no_grad():
        features = _clip_model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().flatten().astype(np.float32)


def get_image_metadata(filepath: str) -> dict:
    """提取图片元数据"""
    img = Image.open(filepath)
    width, height = img.size
    file_size = Path(filepath).stat().st_size

    # 提取 EXIF 信息
    exif_data = {}
    try:
        exif = img.getexif()
        if exif:
            # 获取相机型号、拍摄时间等
            for tag_id in [0x010F, 0x0110, 0x9003, 0x0132]:  # Make, Model, DateTimeOriginal, DateTime
                val = exif.get(tag_id)
                if val:
                    tag_names = {0x010F: "camera_make", 0x0110: "camera_model",
                                 0x9003: "datetime_original", 0x0132: "datetime"}
                    exif_data[tag_names.get(tag_id, f"tag_{tag_id}")] = str(val)
    except Exception:
        pass

    return {
        "width": width,
        "height": height,
        "file_size": file_size,
        "format": img.format or "unknown",
        "mode": img.mode,
        "exif": exif_data
    }


def process_image(filepath: str) -> Optional[dict]:
    """
    处理单张图片：提取元数据、计算哈希和向量
    返回: {filename, filepath, width, height, file_size, phash, dhash, ahash, embedding}
    """
    try:
        filepath = str(Path(filepath).resolve())
        if not Path(filepath).exists():
            return None

        ext = Path(filepath).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return None

        img = Image.open(filepath).convert("RGB")
        metadata = get_image_metadata(filepath)

        phash, dhash, ahash = compute_image_hash(img)
        embedding = compute_embedding(img)

        return {
            "filename": Path(filepath).name,
            "filepath": filepath,
            "width": metadata["width"],
            "height": metadata["height"],
            "file_size": metadata["file_size"],
            "phash": phash,
            "dhash": dhash,
            "ahash": ahash,
            "embedding": embedding.tobytes(),
            "exif": metadata.get("exif", {})
        }
    except Exception as e:
        print(f"[ImageProcessor] 处理图片失败 {filepath}: {e}")
        return None


def hamming_distance(hash1: str, hash2: str) -> int:
    """计算两个十六进制哈希的汉明距离"""
    try:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2
    except Exception:
        return 64  # 最大距离
