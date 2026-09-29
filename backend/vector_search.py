"""向量搜索模块 - FAISS 索引管理"""
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

import faiss

from config import FAISS_INDEX_PATH, FAISS_ID_MAP_PATH, EMBEDDING_DIM


class VectorSearchEngine:
    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.id_map: dict = {}  # faiss_id -> image_id
        self.reverse_map: dict = {}  # image_id -> faiss_id
        self._load_or_create()

    def _load_or_create(self):
        """加载已有索引或创建新索引"""
        if FAISS_INDEX_PATH.exists() and FAISS_ID_MAP_PATH.exists():
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(FAISS_ID_MAP_PATH, "r", encoding="utf-8") as f:
                self.id_map = json.load(f)
            self.reverse_map = {v: k for k, v in self.id_map.items()}
            print(f"[VectorSearch] 加载已有索引: {self.index.ntotal} 条向量")
        else:
            # 使用内积索引（配合归一化向量 = 余弦相似度）
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.id_map = {}
            self.reverse_map = {}
            print("[VectorSearch] 创建新索引")

    def _save(self):
        """保存索引和映射"""
        faiss.write_index(self.index, str(FAISS_INDEX_PATH))
        with open(FAISS_ID_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(self.id_map, f)

    def add(self, image_id: int, embedding: np.ndarray) -> int:
        """添加一条向量到索引，返回 faiss_id"""
        vec = embedding.reshape(1, -1).astype(np.float32)
        # 确保归一化
        faiss.normalize_L2(vec)

        faiss_id = self.index.ntotal
        self.index.add(vec)
        self.id_map[str(faiss_id)] = image_id
        self.reverse_map[image_id] = faiss_id
        self._save()
        return faiss_id

    def add_batch(self, items: List[Tuple[int, np.ndarray]]):
        """批量添加向量"""
        if not items:
            return

        vectors = []
        for image_id, embedding in items:
            vec = embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(vec)
            vectors.append(vec)

            faiss_id = self.index.ntotal + len(vectors) - 1
            self.id_map[str(faiss_id)] = image_id
            self.reverse_map[image_id] = faiss_id

        batch = np.vstack(vectors)
        self.index.add(batch)
        self._save()
        print(f"[VectorSearch] 批量添加 {len(items)} 条向量")

    def remove(self, image_id: int):
        """从索引中移除一条向量（通过重建索引实现）"""
        if image_id not in self.reverse_map:
            return

        faiss_id_to_remove = self.reverse_map[image_id]

        # 收集所有需要保留的向量
        kept_vectors = []
        new_id_map = {}
        new_reverse_map = {}
        new_faiss_id = 0

        for old_faiss_id_str, img_id in self.id_map.items():
            old_faiss_id = int(old_faiss_id_str)
            if old_faiss_id == faiss_id_to_remove:
                continue

            vec = self.index.reconstruct(old_faiss_id).reshape(1, -1)
            kept_vectors.append(vec)
            new_id_map[str(new_faiss_id)] = img_id
            new_reverse_map[img_id] = new_faiss_id
            new_faiss_id += 1

        # 重建索引
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if kept_vectors:
            batch = np.vstack(kept_vectors)
            self.index.add(batch)

        self.id_map = new_id_map
        self.reverse_map = new_reverse_map
        self._save()
        print(f"[VectorSearch] 移除 image_id={image_id}")

    def search(self, query_embedding: np.ndarray, top_k: int = 20,
               threshold: float = 0.0) -> List[Tuple[int, float]]:
        """
        搜索相似向量
        返回: [(image_id, similarity_score), ...]
        """
        if self.index.ntotal == 0:
            return []

        vec = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)

        actual_k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(vec, actual_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if dist < threshold:
                continue
            image_id = self.id_map.get(str(int(idx)))
            if image_id is not None:
                results.append((int(image_id), float(dist)))

        return results

    def search_all_above_threshold(self, query_embedding: np.ndarray,
                                    threshold: float = 0.85) -> List[Tuple[int, float]]:
        """搜索所有超过阈值的相似向量"""
        if self.index.ntotal == 0:
            return []

        vec = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)

        # 搜索全部
        distances, indices = self.index.search(vec, self.index.ntotal)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if dist < threshold:
                break  # FAISS返回结果按距离降序，后面都更小
            image_id = self.id_map.get(str(int(idx)))
            if image_id is not None:
                results.append((int(image_id), float(dist)))

        return results

    @property
    def count(self) -> int:
        return self.index.ntotal
