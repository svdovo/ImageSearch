# 图片素材库 - 以图搜图与去重系统

基于 Chinese-CLIP 和 FAISS 的图片素材库管理系统，支持以图搜图、近重复图片检测与去重。

## 功能特性

| 功能 | 说明 |
|------|------|
| **图像向量化与索引构建** | Chinese-CLIP (RN50) 编码 + FAISS 索引 |
| **近重复判定与归并** | 感知哈希 + 向量相似度双重判定 |
| **批量导入与增量更新** | 支持单张/批量上传、文件夹导入 |
| **阈值调节与效果对比** | 多阈值并排对比搜索结果 |
| **重复组管理** | 自动/手动清理重复图片 |
| **素材库浏览与导出** | 网格浏览、搜索过滤、ZIP导出 |

## 技术栈

- **后端**: Python + FastAPI
- **前端**: Vue 3 + TailwindCSS
- **向量化**: Chinese-CLIP (RN50)
- **向量搜索**: FAISS
- **感知哈希**: imagehash
- **数据库**: SQLite

## 快速开始

### 1. 环境要求

- Python 3.10+
- pip

### 2. 安装依赖

```bash
cd image_search_app

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 安装 Chinese-CLIP（需要源码）
pip install ../Chinese-CLIP-master --no-deps
```

### 3. 生成测试图片（可选）

```bash
python generate_test_images.py
```

### 4. 导入图片

```bash
python import_images.py
```

### 5. 启动服务

```bash
# 方式1：使用启动脚本
start.bat

# 方式2：手动启动
cd backend
python app.py
```

### 6. 访问

打开浏览器访问 **http://localhost:8000**

## 项目结构

```
image_search_app/
├── backend/              # 后端代码
│   ├── app.py           # FastAPI 主应用
│   ├── config.py        # 全局配置
│   ├── database.py      # SQLite 数据库
│   ├── image_processor.py  # 图片处理
│   └── vector_search.py    # FAISS 向量搜索
├── frontend/
│   └── index.html       # Vue3 前端
├── image_library/       # 图片存储
├── models/              # 模型文件（自动下载）
├── generate_test_images.py  # 测试图片生成
├── import_images.py     # 图片导入脚本
├── requirements.txt     # 依赖列表
└── start.bat           # 启动脚本
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/upload | 上传单张图片 |
| POST | /api/batch-upload | 批量上传 |
| POST | /api/import-folder | 文件夹导入 |
| POST | /api/search/image | 以图搜图 |
| POST | /api/search/text | 以文搜图 |
| POST | /api/detect-duplicates | 检测重复 |
| POST | /api/threshold-compare | 阈值对比 |
| GET | /api/groups | 获取重复组 |
| GET | /api/library | 浏览素材库 |
| POST | /api/export | 导出ZIP |

## 注意事项

- 首次启动会自动下载 Chinese-CLIP 模型（约300MB）
- 图片库和模型文件较大，已被 .gitignore 排除
- 建议将模型文件放在本地，避免重复下载
