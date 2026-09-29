"""FastAPI 主应用 - 图片素材库以图搜图与去重系统"""
import os
import io
import uuid
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import (
    PROJECT_ROOT, IMAGE_LIBRARY_DIR, UPLOAD_DIR,
    DEFAULT_SIMILARITY_THRESHOLD, SUPPORTED_EXTENSIONS
)
from database import Database
from image_processor import (
    process_image, compute_embedding, compute_text_embedding,
    compute_image_hash, hamming_distance
)
from vector_search import VectorSearchEngine

app = FastAPI(title="图片素材库 - 以图搜图与去重系统", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "frontend")), name="static")
app.mount("/images", StaticFiles(directory=str(IMAGE_LIBRARY_DIR)), name="images")

# 全局初始化
db = None
engine = None


@app.on_event("startup")
async def startup():
    global db, engine
    Database.init_db()
    db = Database()
    engine = VectorSearchEngine()
    print(f"[App] 系统启动完成, 当前索引 {engine.count} 条向量")


# ==================== 功能1: 图像向量化与索引构建 ====================

@app.post("/api/upload")
async def upload_image(file: UploadFile = File(...)):
    """上传单张图片并入库（功能1）"""
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"不支持的图片格式: {ext}")

    # 保存文件
    unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
    dest = IMAGE_LIBRARY_DIR / unique_name
    content = await file.read()
    dest.write_bytes(content)

    # 处理图片
    result = process_image(str(dest))
    if not result:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "图片处理失败")

    # 存入数据库
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

    # 加入向量索引
    embedding = np.frombuffer(result["embedding"], dtype=np.float32)
    engine.add(image_id, embedding)

    return {
        "success": True,
        "image_id": image_id,
        "filename": result["filename"],
        "width": result["width"],
        "height": result["height"],
        "phash": result["phash"],
        "message": "图片已成功入库"
    }


@app.post("/api/batch-upload")
async def batch_upload(files: List[UploadFile] = File(...)):
    """批量上传图片（功能5的一部分）"""
    results = {"success": 0, "failed": 0, "skipped": 0, "images": []}

    batch_items = []  # (image_id, embedding) for FAISS batch add

    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            results["failed"] += 1
            continue

        unique_name = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        dest = IMAGE_LIBRARY_DIR / unique_name
        content = await file.read()
        dest.write_bytes(content)

        result = process_image(str(dest))
        if not result:
            dest.unlink(missing_ok=True)
            results["failed"] += 1
            continue

        if db.get_image_by_path(result["filepath"]):
            results["skipped"] += 1
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

        results["success"] += 1
        results["images"].append({
            "image_id": image_id,
            "filename": result["filename"],
            "width": result["width"],
            "height": result["height"]
        })

    # 批量加入FAISS索引
    if batch_items:
        engine.add_batch(batch_items)

    return results


@app.post("/api/import-folder")
async def import_folder(folder_path: str = Form(...)):
    """从文件夹批量导入图片（功能5）"""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise HTTPException(400, "文件夹路径不存在")

    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))

    if not image_files:
        raise HTTPException(400, "文件夹中没有找到支持的图片文件")

    results = {"success": 0, "failed": 0, "skipped": 0, "total": len(image_files), "images": []}
    batch_items = []

    for img_path in image_files:
        if db.get_image_by_path(str(img_path.resolve())):
            results["skipped"] += 1
            continue

        # 复制到图片库
        unique_name = f"{uuid.uuid4().hex[:8]}_{img_path.name}"
        dest = IMAGE_LIBRARY_DIR / unique_name
        shutil.copy2(str(img_path), str(dest))

        result = process_image(str(dest))
        if not result:
            dest.unlink(missing_ok=True)
            results["failed"] += 1
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

        results["success"] += 1
        results["images"].append({
            "image_id": image_id,
            "filename": result["filename"]
        })

    if batch_items:
        engine.add_batch(batch_items)

    return results


# ==================== 搜索接口 ====================

@app.post("/api/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Form(20),
    threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD)
):
    """以图搜图"""
    # 保存临时文件
    temp_path = UPLOAD_DIR / f"temp_search_{uuid.uuid4().hex[:8]}{Path(file.filename).suffix}"
    content = await file.read()
    temp_path.write_bytes(content)

    try:
        result = process_image(str(temp_path))
        if not result:
            raise HTTPException(400, "搜索图片处理失败")

        embedding = np.frombuffer(result["embedding"], dtype=np.float32)
        search_results = engine.search(embedding, top_k=top_k, threshold=threshold)

        # 补充图片信息
        images = []
        for image_id, score in search_results:
            img_info = db.get_image(image_id)
            if img_info:
                # 计算哈希距离作为辅助判断
                phash_dist = hamming_distance(result["phash"], img_info["phash"])
                images.append({
                    "image_id": image_id,
                    "filename": img_info["filename"],
                    "filepath": img_info["filepath"],
                    "width": img_info["width"],
                    "height": img_info["height"],
                    "file_size": img_info["file_size"],
                    "similarity": round(score, 4),
                    "phash_distance": phash_dist,
                    "is_near_duplicate": phash_dist < 10 and score > 0.9
                })

        return {
            "query_hash": result["phash"],
            "total_results": len(images),
            "threshold": threshold,
            "results": images
        }
    finally:
        temp_path.unlink(missing_ok=True)


@app.post("/api/search/text")
async def search_by_text(
    text: str = Form(...),
    top_k: int = Form(20),
    threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD)
):
    """以文搜图"""
    embedding = compute_text_embedding(text)
    search_results = engine.search(embedding, top_k=top_k, threshold=threshold)

    images = []
    for image_id, score in search_results:
        img_info = db.get_image(image_id)
        if img_info:
            images.append({
                "image_id": image_id,
                "filename": img_info["filename"],
                "filepath": img_info["filepath"],
                "width": img_info["width"],
                "height": img_info["height"],
                "file_size": img_info["file_size"],
                "similarity": round(score, 4)
            })

    return {
        "query_text": text,
        "total_results": len(images),
        "threshold": threshold,
        "results": images
    }


# ==================== 功能2: 近重复判定与归并 ====================

@app.post("/api/detect-duplicates")
async def detect_duplicates(threshold: float = Form(DEFAULT_SIMILARITY_THRESHOLD)):
    """
    检测所有近重复图片并分组（功能2）
    使用向量相似度 + 感知哈希双重判定
    """
    all_images = db.get_all_active_images()
    if len(all_images) < 2:
        return {"groups": [], "message": "图片数量不足，无法检测重复"}

    # 构建并查集
    parent = {img["id"]: img["id"] for img in all_images}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # 两两比较（对于大量图片可优化为基于哈希的候选筛选）
    pairs = []
    n = len(all_images)

    # 先用哈希快速筛选候选对
    hash_groups = {}
    for img in all_images:
        h = img["phash"][:8]  # 取前8位作为粗筛
        if h not in hash_groups:
            hash_groups[h] = []
        hash_groups[h].append(img)

    # 对候选对进行精确比较
    compared = set()
    for h, group in hash_groups.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                img_a, img_b = group[i], group[j]
                pair_key = (min(img_a["id"], img_b["id"]), max(img_a["id"], img_b["id"]))
                if pair_key in compared:
                    continue
                compared.add(pair_key)

                # 哈希距离
                phash_dist = hamming_distance(img_a["phash"], img_b["phash"])
                if phash_dist > 15:  # 哈希距离太大，跳过
                    continue

                # 向量相似度
                emb_a = np.frombuffer(img_a["embedding"], dtype=np.float32)
                emb_b = np.frombuffer(img_b["embedding"], dtype=np.float32)
                similarity = float(np.dot(emb_a, emb_b))

                if similarity >= threshold or phash_dist < 5:
                    union(img_a["id"], img_b["id"])
                    pairs.append({
                        "image_a": img_a["id"],
                        "image_b": img_b["id"],
                        "similarity": round(similarity, 4),
                        "phash_distance": phash_dist
                    })

    # 收集分组
    groups_map = {}
    for img in all_images:
        root = find(img["id"])
        if root not in groups_map:
            groups_map[root] = []
        groups_map[root].append(img)

    # 只保留有重复的组
    duplicate_groups = []
    for root, members in groups_map.items():
        if len(members) < 2:
            continue

        # 选择代表图（分辨率最高的）
        members.sort(key=lambda x: x["width"] * x["height"], reverse=True)
        representative = members[0]

        # 计算组内相似度
        rep_emb = np.frombuffer(representative["embedding"], dtype=np.float32)
        member_scores = []
        for m in members:
            if m["id"] == representative["id"]:
                member_scores.append((m["id"], 1.0))
            else:
                m_emb = np.frombuffer(m["embedding"], dtype=np.float32)
                sim = float(np.dot(rep_emb, m_emb))
                member_scores.append((m["id"], round(sim, 4)))

        # 存入数据库
        group_id = db.create_group(
            group_name=f"重复组_{len(duplicate_groups) + 1}",
            representative_id=representative["id"],
            threshold=threshold,
            members=member_scores
        )

        duplicate_groups.append({
            "group_id": group_id,
            "representative_id": representative["id"],
            "representative_filename": representative["filename"],
            "member_count": len(members),
            "members": [
                {
                    "image_id": mid,
                    "filename": m["filename"],
                    "filepath": m["filepath"],
                    "width": m["width"],
                    "height": m["height"],
                    "file_size": m["file_size"],
                    "similarity": score
                }
                for mid, score in member_scores
            ]
        })

    # 按组大小排序
    duplicate_groups.sort(key=lambda x: x["member_count"], reverse=True)

    return {
        "total_groups": len(duplicate_groups),
        "total_duplicate_images": sum(g["member_count"] for g in duplicate_groups),
        "threshold": threshold,
        "groups": duplicate_groups
    }


# ==================== 功能5: 批量导入与索引增量更新 ====================

@app.get("/api/library/stats")
async def get_stats():
    """获取素材库统计信息（功能5）"""
    stats = db.stats()
    stats["index_count"] = engine.count
    return stats


@app.delete("/api/images/{image_id}")
async def delete_image(image_id: int):
    """删除图片并更新索引（功能5 - 增量更新）"""
    img_info = db.get_image(image_id)
    if not img_info:
        raise HTTPException(404, "图片不存在")

    # 从FAISS索引移除
    engine.remove(image_id)

    # 从数据库标记删除
    db.delete_image(image_id)

    # 删除文件
    filepath = Path(img_info["filepath"])
    if filepath.exists():
        filepath.unlink()

    return {"success": True, "message": f"已删除图片: {img_info['filename']}"}


@app.post("/api/reindex")
async def reindex():
    """重建索引（功能5 - 全量重建）"""
    global engine
    from vector_search import VectorSearchEngine

    # 删除旧索引
    from config import FAISS_INDEX_PATH, FAISS_ID_MAP_PATH
    FAISS_INDEX_PATH.unlink(missing_ok=True)
    FAISS_ID_MAP_PATH.unlink(missing_ok=True)

    # 重建
    engine = VectorSearchEngine()
    all_images = db.get_all_active_images()

    batch_items = []
    for img in all_images:
        embedding = np.frombuffer(img["embedding"], dtype=np.float32)
        batch_items.append((img["id"], embedding))

    if batch_items:
        engine.add_batch(batch_items)

    return {
        "success": True,
        "reindexed": len(batch_items),
        "message": f"索引重建完成，共 {len(batch_items)} 条"
    }


# ==================== 功能6: 阈值调节与效果对比 ====================

@app.post("/api/threshold-compare")
async def threshold_compare(
    file: UploadFile = File(...),
    thresholds: str = Form("0.70,0.75,0.80,0.85,0.90,0.95")
):
    """
    同一张查询图在不同阈值下的结果对比（功能6）
    返回各阈值下的搜索结果，便于并排比较
    """
    threshold_list = [float(t.strip()) for t in thresholds.split(",")]

    # 处理查询图
    temp_path = UPLOAD_DIR / f"temp_compare_{uuid.uuid4().hex[:8]}{Path(file.filename).suffix}"
    content = await file.read()
    temp_path.write_bytes(content)

    try:
        result = process_image(str(temp_path))
        if not result:
            raise HTTPException(400, "图片处理失败")

        embedding = np.frombuffer(result["embedding"], dtype=np.float32)

        # 先获取所有候选（用最低阈值）
        all_candidates = engine.search_all_above_threshold(embedding, threshold=min(threshold_list))

        comparisons = []
        for t in sorted(threshold_list):
            filtered = [(iid, score) for iid, score in all_candidates if score >= t]

            images = []
            for image_id, score in filtered[:30]:  # 每组最多30条
                img_info = db.get_image(image_id)
                if img_info:
                    phash_dist = hamming_distance(result["phash"], img_info["phash"])
                    images.append({
                        "image_id": image_id,
                        "filename": img_info["filename"],
                        "filepath": img_info["filepath"],
                        "width": img_info["width"],
                        "height": img_info["height"],
                        "similarity": round(score, 4),
                        "phash_distance": phash_dist
                    })

            comparisons.append({
                "threshold": t,
                "result_count": len(images),
                "results": images
            })

        return {
            "query_filename": file.filename,
            "query_hash": result["phash"],
            "total_candidates": len(all_candidates),
            "comparisons": comparisons
        }
    finally:
        temp_path.unlink(missing_ok=True)


# ==================== 功能7: 重复组管理 ====================

@app.get("/api/groups")
async def list_groups():
    """获取所有重复组（功能7）"""
    groups = db.get_all_groups()
    return {"total": len(groups), "groups": groups}


@app.get("/api/groups/{group_id}")
async def get_group(group_id: int):
    """获取单个重复组详情（功能7）"""
    group = db.get_group(group_id)
    if not group:
        raise HTTPException(404, "重复组不存在")
    return group


@app.delete("/api/groups/{group_id}")
async def delete_group(group_id: int):
    """删除重复组（功能7）"""
    db.delete_group(group_id)
    return {"success": True, "message": "重复组已删除"}


@app.post("/api/groups/{group_id}/keep")
async def keep_representative(group_id: int, image_id: int = Form(...)):
    """
    保留指定图片，删除组内其他图片（功能7）
    自动推荐保留画质最高的一张
    """
    group = db.get_group(group_id)
    if not group:
        raise HTTPException(404, "重复组不存在")

    kept_count = 0
    deleted_count = 0

    for member in group["members"]:
        if member["image_id"] == image_id:
            continue  # 保留这张

        # 删除其他成员
        img_info = db.get_image(member["image_id"])
        if img_info:
            engine.remove(member["image_id"])
            db.delete_image(member["image_id"])
            filepath = Path(img_info["filepath"])
            if filepath.exists():
                filepath.unlink()
            deleted_count += 1

    # 删除组
    db.delete_group(group_id)

    return {
        "success": True,
        "kept_image_id": image_id,
        "deleted_count": deleted_count,
        "message": f"保留1张，删除{deleted_count}张重复图片"
    }


@app.post("/api/groups/{group_id}/auto-clean")
async def auto_clean_group(group_id: int):
    """
    自动清理重复组 - 保留画质最高的，删除其余（功能7）
    """
    group = db.get_group(group_id)
    if not group:
        raise HTTPException(404, "重复组不存在")

    # 代表图就是画质最高的（已在创建时按分辨率排序）
    representative_id = group["representative_id"]

    deleted_count = 0
    for member in group["members"]:
        if member["image_id"] == representative_id:
            continue

        img_info = db.get_image(member["image_id"])
        if img_info:
            engine.remove(member["image_id"])
            db.delete_image(member["image_id"])
            filepath = Path(img_info["filepath"])
            if filepath.exists():
                filepath.unlink()
            deleted_count += 1

    db.delete_group(group_id)

    return {
        "success": True,
        "kept_image_id": representative_id,
        "deleted_count": deleted_count,
        "message": f"自动清理完成，保留画质最高的图片，删除{deleted_count}张"
    }


@app.post("/api/groups/batch-auto-clean")
async def batch_auto_clean():
    """批量自动清理所有重复组（功能7）"""
    groups = db.get_all_groups()
    total_deleted = 0
    total_groups = len(groups)

    for group in groups:
        representative_id = group["representative_id"]
        for member in group["members"]:
            if member["image_id"] == representative_id:
                continue
            img_info = db.get_image(member["image_id"])
            if img_info:
                engine.remove(member["image_id"])
                db.delete_image(member["image_id"])
                filepath = Path(img_info["filepath"])
                if filepath.exists():
                    filepath.unlink()
                total_deleted += 1
        db.delete_group(group["id"])

    return {
        "success": True,
        "cleaned_groups": total_groups,
        "total_deleted": total_deleted,
        "message": f"批量清理完成，处理{total_groups}个重复组，删除{total_deleted}张图片"
    }


# ==================== 功能10: 素材库浏览与导出 ====================

@app.get("/api/library")
async def list_library(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    keyword: str = Query("")
):
    """浏览素材库（功能10）"""
    try:
        offset = (page - 1) * page_size
        images, total = db.list_images(offset=offset, limit=page_size, keyword=keyword)

        # 添加图片URL并移除二进制embedding字段
        for img in images:
            img["url"] = f"/images/{Path(img['filepath']).name}"
            img.pop("embedding", None)  # 移除二进制数据

        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": (total + page_size - 1) // page_size,
            "images": images
        }
    except Exception as e:
        import traceback
        print(f"[ERROR] list_library: {e}")
        traceback.print_exc()
        raise HTTPException(500, str(e))


@app.get("/api/images/{image_id}")
async def get_image_info(image_id: int):
    """获取单张图片详情（功能10）"""
    img_info = db.get_image(image_id)
    if not img_info:
        raise HTTPException(404, "图片不存在")
    img_info["url"] = f"/images/{Path(img_info['filepath']).name}"
    img_info.pop("embedding", None)
    return img_info


@app.delete("/api/library/clear")
async def clear_library():
    """清空素材库（功能10）"""
    all_images = db.get_all_active_images()
    for img in all_images:
        filepath = Path(img["filepath"])
        if filepath.exists():
            filepath.unlink()
        db.delete_image(img["id"])

    # 重建空索引
    from config import FAISS_INDEX_PATH, FAISS_ID_MAP_PATH
    FAISS_INDEX_PATH.unlink(missing_ok=True)
    FAISS_ID_MAP_PATH.unlink(missing_ok=True)
    global engine
    from vector_search import VectorSearchEngine
    engine = VectorSearchEngine()

    return {"success": True, "message": f"已清空素材库，删除{len(all_images)}张图片"}


@app.post("/api/export")
async def export_images(image_ids: str = Form(...)):
    """
    导出选中的图片为ZIP文件（功能10）
    image_ids: 逗号分隔的图片ID列表
    """
    ids = [int(x.strip()) for x in image_ids.split(",") if x.strip()]
    if not ids:
        raise HTTPException(400, "未选择任何图片")

    # 创建ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for image_id in ids:
            img_info = db.get_image(image_id)
            if img_info:
                filepath = Path(img_info["filepath"])
                if filepath.exists():
                    zf.write(str(filepath), img_info["filename"])

    zip_buffer.seek(0)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=exported_images.zip"}
    )


@app.post("/api/export/manifest")
async def export_manifest(image_ids: str = Form(...)):
    """
    导出选中图片的清单（JSON）（功能10）
    """
    ids = [int(x.strip()) for x in image_ids.split(",") if x.strip()]
    manifest = []
    for image_id in ids:
        img_info = db.get_image(image_id)
        if img_info:
            manifest.append({
                "image_id": image_id,
                "filename": img_info["filename"],
                "filepath": img_info["filepath"],
                "width": img_info["width"],
                "height": img_info["height"],
                "file_size": img_info["file_size"],
                "phash": img_info["phash"],
                "import_time": img_info["import_time"]
            })

    return {"total": len(manifest), "images": manifest}


# ==================== 前端路由 ====================

@app.get("/")
async def serve_frontend():
    """提供前端页面"""
    return FileResponse(str(PROJECT_ROOT / "frontend" / "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
