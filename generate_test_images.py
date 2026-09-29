"""生成测试图片 - 用于演示以图搜图与去重功能"""
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "image_library")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色主题
THEMES = [
    ("风景", [(135, 206, 235), (34, 139, 34), (255, 255, 224)]),   # 蓝天绿地
    ("海洋", [(0, 105, 148), (0, 191, 255), (255, 255, 255)]),      # 蓝色系
    ("日落", [(255, 140, 0), (255, 69, 0), (255, 215, 0)]),         # 暖色系
    ("森林", [(0, 100, 0), (34, 139, 34), (144, 238, 144)]),        # 绿色系
    ("城市", [(128, 128, 128), (169, 169, 169), (211, 211, 211)]),  # 灰色系
    ("花卉", [(255, 105, 180), (255, 20, 147), (255, 182, 193)]),   # 粉色系
    ("沙漠", [(210, 180, 140), (245, 222, 179), (188, 170, 130)]),  # 沙色系
    ("星空", [(25, 25, 112), (0, 0, 139), (72, 61, 139)]),          # 深蓝系
]

LABELS = {
    "风景": ["山脉", "湖泊", "草原", "峡谷", "瀑布", "田野", "丘陵", "河流"],
    "海洋": ["海滩", "海浪", "珊瑚", "海岛", "港口", "灯塔", "渔船", "海豚"],
    "日落": ["晚霞", "夕阳", "余晖", "黄昏", "落日", "彩霞", "暮色", "金光"],
    "森林": ["松树", "竹林", "树林", "苔藓", "小径", "蘑菇", "藤蔓", "溪流"],
    "城市": ["高楼", "街道", "桥梁", "广场", "地铁", "公园", "建筑", "夜景"],
    "花卉": ["玫瑰", "牡丹", "樱花", "荷花", "菊花", "郁金香", "兰花", "梅花"],
    "沙漠": ["沙丘", "骆驼", "绿洲", "戈壁", "仙人掌", "日落", "星空", "遗迹"],
    "星空": ["银河", "流星", "星云", "月亮", "星座", "极光", "星轨", "深空"],
}


def create_gradient_image(width, height, colors):
    """创建渐变背景"""
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        ratio = y / height
        r = int(colors[0][0] * (1 - ratio) + colors[1][0] * ratio)
        g = int(colors[0][1] * (1 - ratio) + colors[1][1] * ratio)
        b = int(colors[0][2] * (1 - ratio) + colors[1][2] * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    return img


def add_shapes(img, colors, count=5):
    """添加随机几何图形"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(count):
        shape_type = random.choice(["circle", "rectangle", "triangle"])
        color = random.choice(colors + [(255, 255, 255)])
        x1 = random.randint(0, w // 2)
        y1 = random.randint(0, h // 2)
        x2 = x1 + random.randint(20, w // 3)
        y2 = y1 + random.randint(20, h // 3)
        if shape_type == "circle":
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=None)
        elif shape_type == "rectangle":
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=None)
        else:
            draw.polygon([(x1, y2), (x2, y2), (x1 + (x2 - x1) // 2, y1)], fill=color)
    return img


def add_text(img, text, theme):
    """添加文字标签"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    # 尝试使用系统字体
    font_paths = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    font = None
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, 24)
                break
            except:
                pass
    if font is None:
        font = ImageFont.load_default()

    # 文字位置
    x = random.randint(10, w - 150)
    y = random.randint(10, h - 40)
    draw.text((x, y), f"{theme}·{text}", fill=(255, 255, 255), font=font)
    return img


def create_image(theme_name, label, size=(400, 300), variation=0):
    """创建一张主题图片"""
    theme = next((t for t in THEMES if t[0] == theme_name), THEMES[0])
    colors = theme[1]

    # 根据variation调整颜色
    if variation > 0:
        colors = [(min(255, c + random.randint(-20, 20)) for c in color) for color in colors]
        colors = [tuple(c) for c in colors]

    img = create_gradient_image(size[0], size[1], colors)
    img = add_shapes(img, colors, count=random.randint(3, 8))

    # 添加一些噪声/纹理
    if variation >= 2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 2.0)))

    img = add_text(img, label, theme_name)
    return img


def generate_test_images():
    """生成测试图片集"""
    print("开始生成测试图片...")
    count = 0

    for theme_name, _ in THEMES:
        labels = LABELS[theme_name]
        for label in labels:
            # 原始图片
            img = create_image(theme_name, label)
            filename = f"{theme_name}_{label}.png"
            img.save(os.path.join(OUTPUT_DIR, filename))
            count += 1

            # 变体1: 不同尺寸（模拟裁剪/缩放）
            sizes = [(300, 225), (500, 375), (350, 263)]
            for i, size in enumerate(sizes):
                img_v = create_image(theme_name, label, size=size, variation=1)
                filename_v = f"{theme_name}_{label}_v{i+1}_{size[0]}x{size[1]}.png"
                img_v.save(os.path.join(OUTPUT_DIR, filename_v))
                count += 1

            # 变体2: 加模糊/噪声（模拟不同处理）
            img_v2 = create_image(theme_name, label, variation=2)
            filename_v2 = f"{theme_name}_{label}_blur.png"
            img_v2.save(os.path.join(OUTPUT_DIR, filename_v2))
            count += 1

    print(f"共生成 {count} 张测试图片，保存在: {OUTPUT_DIR}")
    return count


if __name__ == "__main__":
    generate_test_images()
