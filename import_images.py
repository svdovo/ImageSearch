"""导入已有图片到数据库和FAISS索引"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from pathlib import Path
from config import IMAGE_LIBRARY_DIR, SUPPORTED_EXTENSIONS
from database import Database
from image_processor import process_image
from vector_search import VectorSearchEngine
import numpy as np

def import_existing_images():
    """导入image_library目录中已有的图片"""
    Database.init_db()
    db = Database()
    engine = VectorSearchEngine()

    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(IMAGE_LIBRARY_DIR.glob(f"*{ext}"))
        image_files.extend(IMAGE_LIBRARY_DIR.glob(f"*{ext.upper()}"))

    print(f"找到 {len(image_files)} 张图片文件")

    success = 0
    skipped = 0
    failed = 0
    batch_items = []

    for i, img_path in enumerate(image_files):
        if (i + 1) % 50 == 0:
            print(f"  处理进度: {i + 1}/{len(image_files)}")

        # 检查是否已存在
        if db.get_image_by_path(str(img_path.resolve())):
            skipped += 1
            continue

        result = process_image(str(img_path))
        if not result:
            failed += 1
            continue

        image_id = db.add_image(
            filename=result["filename"],
            filepath=result["filepath"],
            width=result["width"],
            height=result["height"],
            file_size=result["file_size"],
            phash=result["phash"],
            dhash=result["dhash"],
            ahash=result["ahash"],
            embedding=result["embedding"]
        )

        embedding = np.frombuffer(result["embedding"], dtype=np.float32)
        batch_items.append((image_id, embedding))
        success += 1

    # 批量加入FAISS
    if batch_items:
        engine.add_batch(batch_items)

    print(f"\n导入完成:")
    print(f"  成功: {success}")
    print(f"  跳过(已存在): {skipped}")
    print(f"  失败: {failed}")
    print(f"  FAISS索引: {engine.count} 条向量")


if __name__ == "__main__":
    import_existing_images()
