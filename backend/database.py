"""SQLite 数据库管理 - 图片元数据与重复组"""
import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

from config import DB_PATH


class Database:
    _local = threading.local()

    @classmethod
    def _get_conn(cls) -> sqlite3.Connection:
        if not hasattr(cls._local, "conn") or cls._local.conn is None:
            cls._local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
            cls._local.conn.row_factory = sqlite3.Row
            cls._local.conn.execute("PRAGMA journal_mode=WAL")
        return cls._local.conn

    @classmethod
    def init_db(cls):
        conn = cls._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                filepath TEXT NOT NULL UNIQUE,
                width INTEGER,
                height INTEGER,
                file_size INTEGER,
                phash TEXT,
                dhash TEXT,
                ahash TEXT,
                embedding BLOB,
                import_time TEXT DEFAULT (datetime('now','localtime')),
                status TEXT DEFAULT 'active'
            );

            CREATE TABLE IF NOT EXISTS duplicate_groups (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_name TEXT,
                representative_id INTEGER,
                threshold REAL DEFAULT 0.85,
                created_time TEXT DEFAULT (datetime('now','localtime')),
                FOREIGN KEY (representative_id) REFERENCES images(id)
            );

            CREATE TABLE IF NOT EXISTS group_members (
                group_id INTEGER NOT NULL,
                image_id INTEGER NOT NULL,
                similarity REAL,
                PRIMARY KEY (group_id, image_id),
                FOREIGN KEY (group_id) REFERENCES duplicate_groups(id),
                FOREIGN KEY (image_id) REFERENCES images(id)
            );

            CREATE INDEX IF NOT EXISTS idx_images_filepath ON images(filepath);
            CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash);
            CREATE INDEX IF NOT EXISTS idx_images_status ON images(status);
        """)
        conn.commit()

    @classmethod
    def add_image(cls, filename: str, filepath: str, width: int, height: int,
                  file_size: int, phash: str, dhash: str, ahash: str,
                  embedding: bytes) -> int:
        conn = cls._get_conn()
        cursor = conn.execute(
            """INSERT OR IGNORE INTO images
               (filename, filepath, width, height, file_size, phash, dhash, ahash, embedding)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (filename, filepath, width, height, file_size, phash, dhash, ahash, embedding)
        )
        conn.commit()
        return cursor.lastrowid

    @classmethod
    def get_image(cls, image_id: int) -> Optional[dict]:
        conn = cls._get_conn()
        row = conn.execute("SELECT * FROM images WHERE id = ? AND status = 'active'", (image_id,)).fetchone()
        return dict(row) if row else None

    @classmethod
    def get_image_by_path(cls, filepath: str) -> Optional[dict]:
        conn = cls._get_conn()
        row = conn.execute("SELECT * FROM images WHERE filepath = ? AND status = 'active'", (filepath,)).fetchone()
        return dict(row) if row else None

    @classmethod
    def list_images(cls, offset: int = 0, limit: int = 50, keyword: str = "") -> tuple:
        conn = cls._get_conn()
        if keyword:
            rows = conn.execute(
                "SELECT * FROM images WHERE status='active' AND filename LIKE ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (f"%{keyword}%", limit, offset)
            ).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM images WHERE status='active' AND filename LIKE ?",
                (f"%{keyword}%",)
            ).fetchone()[0]
        else:
            rows = conn.execute(
                "SELECT * FROM images WHERE status='active' ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset)
            ).fetchall()
            total = conn.execute(
                "SELECT COUNT(*) FROM images WHERE status='active'"
            ).fetchone()[0]
        return [dict(r) for r in rows], total

    @classmethod
    def get_all_active_images(cls) -> list:
        conn = cls._get_conn()
        rows = conn.execute("SELECT * FROM images WHERE status='active' ORDER BY id").fetchall()
        return [dict(r) for r in rows]

    @classmethod
    def delete_image(cls, image_id: int):
        conn = cls._get_conn()
        conn.execute("UPDATE images SET status='deleted' WHERE id = ?", (image_id,))
        conn.execute("DELETE FROM group_members WHERE image_id = ?", (image_id,))
        conn.commit()

    @classmethod
    def delete_group(cls, group_id: int):
        conn = cls._get_conn()
        conn.execute("DELETE FROM group_members WHERE group_id = ?", (group_id,))
        conn.execute("DELETE FROM duplicate_groups WHERE id = ?", (group_id,))
        conn.commit()

    @classmethod
    def create_group(cls, group_name: str, representative_id: int, threshold: float,
                     members: list) -> int:
        """members: [(image_id, similarity), ...]"""
        conn = cls._get_conn()
        cursor = conn.execute(
            "INSERT INTO duplicate_groups (group_name, representative_id, threshold) VALUES (?, ?, ?)",
            (group_name, representative_id, threshold)
        )
        group_id = cursor.lastrowid
        for image_id, similarity in members:
            conn.execute(
                "INSERT INTO group_members (group_id, image_id, similarity) VALUES (?, ?, ?)",
                (group_id, image_id, similarity)
            )
        conn.commit()
        return group_id

    @classmethod
    def get_all_groups(cls) -> list:
        conn = cls._get_conn()
        groups = conn.execute("""
            SELECT g.*, i.filename as rep_filename, i.filepath as rep_filepath,
                   COUNT(gm.image_id) as member_count
            FROM duplicate_groups g
            LEFT JOIN images i ON g.representative_id = i.id
            LEFT JOIN group_members gm ON g.id = gm.group_id
            GROUP BY g.id
            ORDER BY g.created_time DESC
        """).fetchall()
        result = []
        for g in groups:
            gd = dict(g)
            members = conn.execute("""
                SELECT gm.*, i.filename, i.filepath, i.width, i.height, i.file_size
                FROM group_members gm
                JOIN images i ON gm.image_id = i.id
                WHERE gm.group_id = ?
                ORDER BY gm.similarity DESC
            """, (g["id"],)).fetchall()
            gd["members"] = [dict(m) for m in members]
            result.append(gd)
        return result

    @classmethod
    def get_group(cls, group_id: int) -> Optional[dict]:
        conn = cls._get_conn()
        g = conn.execute("""
            SELECT g.*, i.filename as rep_filename, i.filepath as rep_filepath
            FROM duplicate_groups g
            LEFT JOIN images i ON g.representative_id = i.id
            WHERE g.id = ?
        """, (group_id,)).fetchone()
        if not g:
            return None
        gd = dict(g)
        members = conn.execute("""
            SELECT gm.*, i.filename, i.filepath, i.width, i.height, i.file_size
            FROM group_members gm
            JOIN images i ON gm.image_id = i.id
            WHERE gm.group_id = ?
            ORDER BY gm.similarity DESC
        """, (group_id,)).fetchall()
        gd["members"] = [dict(m) for m in members]
        return gd

    @classmethod
    def remove_member_from_group(cls, group_id: int, image_id: int):
        conn = cls._get_conn()
        conn.execute("DELETE FROM group_members WHERE group_id=? AND image_id=?", (group_id, image_id))
        conn.commit()

    @classmethod
    def stats(cls) -> dict:
        conn = cls._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM images WHERE status='active'").fetchone()[0]
        groups = conn.execute("SELECT COUNT(*) FROM duplicate_groups").fetchone()[0]
        total_size = conn.execute("SELECT COALESCE(SUM(file_size),0) FROM images WHERE status='active'").fetchone()[0]
        return {"total_images": total, "total_groups": groups, "total_size_mb": round(total_size / 1024 / 1024, 2)}
