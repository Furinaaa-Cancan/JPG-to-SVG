# 🎨 艺术风格SVG生成器

将照片转换为各种现代艺术风格的SVG矢量图。

## 📁 核心脚本

| 脚本 | 功能 |
|------|------|
| `picasso_masterpiece.py` | **毕加索名作风格**（Guernica、Weeping Woman等10大名作） |
| `picasso_figurative.py` | 毕加索具象风格（保留人物特征） |
| `cubism_batch.py` | 立体主义批量生成（25种变体） |
| `duchamp_batch.py` | 杜尚《下楼梯的裸女》风格 |
| `massive_art_generator.py` | 大规模多风格生成器 |
| `modern_art_styles.py` | 风格定义库 |
| `sd_to_svg.py` | SD风格化+SVG转换 |
| `svg_enhancer.py` | SVG后处理增强 |

## 📂 输出目录

```
output/
├── cubism/     # 立体主义 (30张)
├── duchamp/    # 杜尚风格 (28张)
├── popart/     # 波普艺术
└── lowpoly/    # 低多边形
```

---

## 📋 运行前准备

### 1. 确保已安装Python环境
打开终端，输入以下命令检查：
```bash
python --version
```
应该显示 Python 3.x.x

### 2. 安装必要的库
在终端中运行：
```bash
pip install torch diffusers pillow opencv-python numpy svgwrite
```

### 3. 准备输入图片
把你想要转换的照片放到 `01_input` 文件夹里

---

## 🚀 如何运行

### 第一步：打开代码文件
用任意编辑器打开 `06_style_art/massive_art_generator.py`

### 第二步：修改配置（在文件最底部的 main 函数里）

找到这几行代码（大约在第380行左右）：

```python
def main():
    input_image = "/Volumes/Seagate/SAM3/01_input/Ladygaga_2.jpg"  # ← 改成你的图片路径
    output_dir = "/Volumes/Seagate/SAM3/06_style_art/output/massive_art_hq"  # ← 改成你想保存的位置
    START_FROM = 1  # ← 从第几张开始生成（1表示从头开始）
```

**需要修改的地方：**
- `input_image`：你的输入图片的完整路径
- `output_dir`：生成的SVG保存到哪里
- `START_FROM`：从第几张开始（如果中断了可以从断点继续）

### 第三步：运行程序
在终端中进入项目目录，然后运行：
```bash
cd /你的项目路径/SAM3
python 06_style_art/massive_art_generator.py
```

---

## ⏱️ 运行时间

- 每张图大约需要 1-3 分钟
- 200张图总共需要 3-10 小时（取决于电脑性能）
- 程序会自动保存进度，中断后可以继续

---

## 📁 输出文件说明

运行完成后，输出文件夹里会有：

| 文件类型 | 说明 |
|---------|------|
| `art_v001_cubism_analytical.svg` | SVG矢量图（主要输出） |
| `art_v001_cubism_analytical_preview.png` | PNG预览图（方便查看） |
| `seed_log.json` | 种子日志（重要！用于复现） |

---

## 🔄 如果想重新生成某一张图

### 方法1：查看日志找到种子
打开 `seed_log.json`，找到你想重新生成的那张图，记下它的参数：
```json
{
  "version": 42,
  "style_key": "pop_art_warhol",
  "seed": 1234567890,
  "strength": 0.7,
  "num_colors": 240,
  ...
}
```

### 方法2：用种子重新生成
在Python中运行：
```python
from massive_art_generator import MassiveArtGenerator

generator = MassiveArtGenerator()
generator.load_sd()

# 用保存的种子重新生成
generator.regenerate_single(
    image_path="你的图片路径",
    output_dir="输出目录",
    version=42,           # 版本号
    seed=1234567890,      # 从日志里复制的种子
    style_key="pop_art_warhol",  # 风格名
    strength=0.8,         # 可以调整
    num_colors=300,       # 可以增加颜色数量
    simplify=0.0001       # 可以降低这个值让细节更多
)
```

---

## 🎭 包含的艺术风格（35种）

### 立体主义系列
- 分析立体主义 (毕加索、布拉克)
- 综合立体主义 (毕加索、格里斯)
- 奥菲斯主义 (德劳内夫妇)

### 未来主义系列
- 意大利未来主义 (波丘尼、巴拉)
- 俄罗斯立体未来主义 (马列维奇)

### 表现主义系列
- 德国表现主义 (基希纳、诺尔德)
- 抽象表现主义 (德库宁、克莱因)
- 蒙克表现主义 (蒙克)

### 几何抽象系列
- 至上主义 (马列维奇)
- 新造型主义 (蒙德里安)
- 构成主义 (利西茨基)

### 超现实主义系列
- 达利超现实主义
- 米罗超现实主义
- 马格利特超现实主义

### 色彩系列
- 野兽派 (马蒂斯)
- 色域绘画 (罗斯科)

### 波普艺术系列
- 沃霍尔波普
- 利希滕斯坦波普

### 当代艺术系列
- 行动绘画 (波洛克)
- 欧普艺术 (瓦萨雷利)
- 新表现主义 (巴斯奎特)
- 点彩派 (修拉)
- 新艺术运动 (穆夏、克里姆特)
- 包豪斯 (康定斯基、克利)
- 极简主义 (贾德)
- 概念艺术 (勒维特)
- 数字故障艺术
- 街头艺术 (班克斯、KAWS)
- 照相写实主义 (克洛斯)
- 大地艺术 (史密森)
- 英国青年艺术家 (赫斯特)
- 动态艺术 (考尔德)
- 贫穷艺术
- 激浪派 (小野洋子、白南准)

---

## ❓ 常见问题

### Q: 程序中断了怎么办？
A: 修改 `START_FROM` 为下一个版本号，重新运行即可继续。比如已经生成到 v050，就设置 `START_FROM = 51`

### Q: 生成的文件太小怎么办？
A: 可以调整参数：
- 增加 `num_colors`（比如改成 350）
- 降低 `simplify`（比如改成 0.00005）

### Q: 内存不够怎么办？
A: 程序会自动清理内存。如果还是不够，可以降低分辨率（在代码里把 1536 改成 1024）

### Q: 想用同样的种子生成不同风格？
A: 可以！用 `regenerate_single()` 方法，保持 seed 不变，换一个 style_key 就行

---

## 📞 需要帮助？

如果遇到问题，可以把错误信息发给我，我来帮你解决！
