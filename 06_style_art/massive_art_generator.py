"""
大规模艺术风格SVG生成器 - 200张超高质量版
包含30+种当代/现代艺术流派
目标：每张SVG 80MB+
"""

import torch
import gc
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
import svgwrite
import json
from datetime import datetime
import random
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp


class MassiveArtGenerator:
    """大规模多风格艺术生成器 - 超高质量版"""

    # 30种当代/现代艺术流派
    ART_STYLES = {
        # ===== 立体主义系列 =====
        "cubism_analytical": {
            "name": "分析立体主义",
            "artists": ["Pablo Picasso", "Georges Braque"],
            "prompt": (
                "analytical cubism masterpiece, portrait fragmented into intersecting geometric planes, "
                "multiple simultaneous viewpoints, monochromatic earth tones brown gray ochre beige, "
                "shattered crystalline forms, intellectual deconstruction of form, "
                "overlapping transparent angular planes, spatial ambiguity, "
                "1910 Paris avant-garde revolutionary painting, museum quality fine art, "
                "extremely detailed brushwork, high resolution masterpiece"
            ),
            "negative": "bright colors, smooth, realistic, photographic, soft edges, blurry, low quality"
        },
        "cubism_synthetic": {
            "name": "综合立体主义",
            "artists": ["Pablo Picasso", "Juan Gris", "Fernand Léger"],
            "prompt": (
                "synthetic cubism collage artwork, bold flat geometric color shapes, "
                "papier collé mixed media aesthetic, decorative patterns textures, "
                "bright vibrant color accents against earth tones, "
                "playful reconstructed forms, newspaper text elements, "
                "simplified bold shapes, modern art masterpiece, intricate details"
            ),
            "negative": "realistic, photographic, 3d render, smooth gradients, low resolution"
        },
        "cubism_orphism": {
            "name": "奥菲斯主义",
            "artists": ["Robert Delaunay", "Sonia Delaunay", "František Kupka"],
            "prompt": (
                "orphism abstract painting, simultaneous contrast of pure colors, "
                "concentric circles discs wheels, prismatic color harmonies, "
                "rhythmic circular forms, rainbow spectrum palette, "
                "lyrical abstraction, dynamic color movement, "
                "Eiffel Tower fragmentation, modern urban energy, ultra detailed"
            ),
            "negative": "monochrome, dull colors, realistic, figurative, simple"
        },

        # ===== 未来主义系列 =====
        "futurism_italian": {
            "name": "意大利未来主义",
            "artists": ["Umberto Boccioni", "Giacomo Balla", "Gino Severini"],
            "prompt": (
                "Italian Futurism dynamic painting, speed motion energy, "
                "force lines radiating movement, mechanical dynamism, "
                "fragmented simultaneous views of moving figure, "
                "vibrant colors electric blue orange red, "
                "urban modernity machines automobiles, "
                "aggressive angular composition, revolutionary art manifesto, highly detailed"
            ),
            "negative": "static, peaceful, traditional, classical, slow, simple"
        },
        "futurism_russian": {
            "name": "俄罗斯立体未来主义",
            "artists": ["Kazimir Malevich", "Natalia Goncharova", "Mikhail Larionov"],
            "prompt": (
                "Russian Cubo-Futurism avant-garde, bold primary colors, "
                "dynamic angular geometric forms, peasant folk art influence, "
                "rayonist light rays, suprematist elements, "
                "revolutionary energy, constructivist aesthetics, "
                "1910s Moscow St Petersburg modernism, intricate patterns"
            ),
            "negative": "western, classical, soft, pastel, traditional, minimal"
        },

        # ===== 表现主义系列 =====
        "expressionism_german": {
            "name": "德国表现主义",
            "artists": ["Ernst Ludwig Kirchner", "Emil Nolde", "Max Beckmann"],
            "prompt": (
                "German Expressionism Die Brücke painting, raw emotional intensity, "
                "distorted angular figures, bold jarring color contrasts, "
                "thick expressive brushstrokes, psychological tension, "
                "urban alienation anxiety, primitive simplified forms, "
                "woodcut aesthetic harsh lines, inner turmoil visualization, detailed textures"
            ),
            "negative": "calm, peaceful, realistic, smooth, pretty, simple"
        },
        "expressionism_abstract": {
            "name": "抽象表现主义",
            "artists": ["Willem de Kooning", "Franz Kline", "Robert Motherwell"],
            "prompt": (
                "Abstract Expressionism gestural painting, violent energetic brushwork, "
                "spontaneous automatic creation, existential angst raw emotion, "
                "large scale monumental canvas, black white dramatic contrast, "
                "action painting traces of artistic process, "
                "New York School 1950s avant-garde, complex layered textures"
            ),
            "negative": "controlled, precise, figurative, decorative, small, simple"
        },
        "expressionism_scream": {
            "name": "蒙克表现主义",
            "artists": ["Edvard Munch"],
            "prompt": (
                "Edvard Munch Scream style painting, swirling undulating lines, "
                "blood red orange sunset sky, psychological horror anxiety, "
                "distorted screaming figure, wavy flowing landscape, "
                "existential dread isolation, Nordic melancholy, "
                "symbolic emotional landscape, proto-expressionist masterpiece, rich details"
            ),
            "negative": "calm, happy, realistic, stable, peaceful, minimal"
        },

        # ===== 抽象几何系列 =====
        "suprematism": {
            "name": "至上主义",
            "artists": ["Kazimir Malevich"],
            "prompt": (
                "Suprematism geometric abstraction, pure geometric forms floating, "
                "black square red square white background, "
                "spiritual transcendence through geometry, "
                "dynamic diagonal compositions, weightless cosmic space, "
                "primary colors black white, non-objective art, "
                "revolutionary Russian avant-garde 1915, precise edges"
            ),
            "negative": "figurative, realistic, decorative, complex, busy, blurry"
        },
        "neoplasticism": {
            "name": "新造型主义",
            "artists": ["Piet Mondrian", "Theo van Doesburg"],
            "prompt": (
                "De Stijl Neoplasticism painting, strict grid composition, "
                "primary colors red yellow blue with black white, "
                "horizontal vertical lines only, rectangular planes, "
                "pure abstraction universal harmony, "
                "asymmetrical balance, spiritual geometric purity, "
                "Mondrian Broadway Boogie Woogie style, crisp clean lines"
            ),
            "negative": "curves, diagonals, figurative, natural, organic, blurry"
        },
        "constructivism": {
            "name": "构成主义",
            "artists": ["El Lissitzky", "Alexander Rodchenko", "Vladimir Tatlin"],
            "prompt": (
                "Russian Constructivism design, bold geometric propaganda poster, "
                "dynamic diagonal compositions, red black white color scheme, "
                "industrial materials aesthetic, photomontage elements, "
                "revolutionary Soviet art, utilitarian design, "
                "typography integration, social purpose art, sharp details"
            ),
            "negative": "decorative, bourgeois, traditional, ornamental, soft"
        },

        # ===== 超现实主义系列 =====
        "surrealism_dali": {
            "name": "达利超现实主义",
            "artists": ["Salvador Dalí"],
            "prompt": (
                "Salvador Dalí surrealist painting, melting clocks persistence of memory, "
                "dreamscape desert landscape, hyper-realistic technique, "
                "impossible juxtapositions, paranoid critical method, "
                "elephants on stilts, distorted faces figures, "
                "subconscious imagery, Spanish surrealism masterpiece, ultra detailed"
            ),
            "negative": "abstract, geometric, normal, logical, simple, low quality"
        },
        "surrealism_miro": {
            "name": "米罗超现实主义",
            "artists": ["Joan Miró"],
            "prompt": (
                "Joan Miró biomorphic abstraction, playful organic shapes, "
                "primary colors red yellow blue black on white, "
                "childlike symbolic figures stars moons eyes, "
                "automatic drawing spontaneous creation, "
                "constellation series, poetic dream imagery, "
                "Catalan surrealist master, joyful cosmic fantasy, intricate forms"
            ),
            "negative": "realistic, serious, geometric, rigid, heavy, simple"
        },
        "surrealism_magritte": {
            "name": "马格利特超现实主义",
            "artists": ["René Magritte"],
            "prompt": (
                "René Magritte philosophical surrealism, impossible realistic scenes, "
                "bowler hat man floating objects, Belgian surrealist, "
                "visual paradoxes word image relationships, "
                "clear precise painting technique, mysterious atmosphere, "
                "clouds sky motifs, conceptual art precursor, highly detailed"
            ),
            "negative": "abstract, expressive, loose brushwork, emotional, simple"
        },

        # ===== 野兽派与色彩系列 =====
        "fauvism": {
            "name": "野兽派",
            "artists": ["Henri Matisse", "André Derain", "Maurice de Vlaminck"],
            "prompt": (
                "Fauvism wild color explosion, non-naturalistic vivid hues, "
                "bold flat color areas, spontaneous brushwork, "
                "pure tube colors unmixed, emotional color expression, "
                "simplified forms outlines, joyful exuberance, "
                "Henri Matisse Dance joy of life, 1905 Salon d'Automne revolution, rich textures"
            ),
            "negative": "muted colors, realistic, detailed, academic, dull, simple"
        },
        "color_field": {
            "name": "色域绘画",
            "artists": ["Mark Rothko", "Barnett Newman", "Clyfford Still"],
            "prompt": (
                "Color Field painting, large expanses of flat color, "
                "soft-edged rectangular forms floating, "
                "spiritual transcendence through color, "
                "meditative contemplative atmosphere, "
                "luminous color relationships, sublime emotional depth, "
                "Mark Rothko chapel paintings, monumental scale feeling, subtle gradients"
            ),
            "negative": "busy, detailed, figurative, hard edges, small, simple"
        },

        # ===== 波普艺术系列 =====
        "pop_art_warhol": {
            "name": "沃霍尔波普",
            "artists": ["Andy Warhol"],
            "prompt": (
                "Andy Warhol pop art silkscreen, celebrity portrait Marilyn Monroe style, "
                "flat bold commercial colors, repetition serial imagery, "
                "high contrast posterized, mass media consumer culture, "
                "Campbell soup aesthetic, Factory production art, "
                "1960s New York pop culture icon, vibrant saturated colors"
            ),
            "negative": "painterly, traditional, unique, handmade, subtle, dull"
        },
        "pop_art_lichtenstein": {
            "name": "利希滕斯坦波普",
            "artists": ["Roy Lichtenstein"],
            "prompt": (
                "Roy Lichtenstein comic book pop art, Ben-Day dots halftone pattern, "
                "bold black outlines, primary colors red yellow blue, "
                "speech bubbles comic panels, melodramatic expression, "
                "enlarged comic strip aesthetic, ironic commentary, "
                "Whaam explosion style, American pop art master, detailed dots pattern"
            ),
            "negative": "realistic, painterly, subtle, serious, traditional, simple"
        },

        # ===== 当代艺术系列 =====
        "action_painting": {
            "name": "行动绘画",
            "artists": ["Jackson Pollock", "Lee Krasner"],
            "prompt": (
                "Jackson Pollock drip painting, all-over composition no focal point, "
                "rhythmic tangled lines splatters drips, "
                "automatism unconscious gesture, layered paint webs, "
                "energetic physical painting process, "
                "black silver aluminum paint, Number series abstract, "
                "Cedar Tavern New York School, complex intricate patterns"
            ),
            "negative": "controlled, figurative, centered, clean, precise, simple"
        },
        "op_art": {
            "name": "欧普艺术",
            "artists": ["Victor Vasarely", "Bridget Riley"],
            "prompt": (
                "Op Art optical illusion painting, geometric patterns create motion, "
                "black white contrasting shapes, vibrating visual effects, "
                "precise mathematical composition, perceptual tricks, "
                "concentric circles waves, dizzying depth illusion, "
                "1960s kinetic visual art movement, intricate geometric patterns"
            ),
            "negative": "static, soft, blurry, organic, random, simple"
        },
        "neo_expressionism": {
            "name": "新表现主义",
            "artists": ["Jean-Michel Basquiat", "Anselm Kiefer", "Georg Baselitz"],
            "prompt": (
                "Neo-Expressionism raw primitive painting, graffiti street art influence, "
                "crude figures symbols text, aggressive mark making, "
                "cultural social commentary, crown skull motifs, "
                "Basquiat SAMO style, 1980s East Village New York, "
                "intense emotional visceral imagery, layered complex textures"
            ),
            "negative": "refined, pretty, academic, polished, commercial, simple"
        },
        "pointillism": {
            "name": "点彩派",
            "artists": ["Georges Seurat", "Paul Signac"],
            "prompt": (
                "Pointillism Neo-Impressionism, tiny dots of pure color, "
                "optical color mixing divisionism, scientific color theory, "
                "luminous shimmering effect, Sunday afternoon Grande Jatte, "
                "precise methodical technique, complementary colors adjacent, "
                "Georges Seurat masterpiece style, millions of color dots"
            ),
            "negative": "blended colors, smooth, expressive, loose, dark, simple"
        },
        "art_nouveau": {
            "name": "新艺术运动",
            "artists": ["Alphonse Mucha", "Gustav Klimt", "Aubrey Beardsley"],
            "prompt": (
                "Art Nouveau decorative painting, flowing organic curves, "
                "elaborate ornamental patterns, feminine beauty figures, "
                "gold leaf Byzantine influence Klimt, floral botanical motifs, "
                "sinuous whiplash lines, Japanese ukiyo-e influence, "
                "turn of century Vienna Secession elegance, intricate decorative details"
            ),
            "negative": "geometric, minimal, industrial, harsh, angular, simple"
        },
        "bauhaus": {
            "name": "包豪斯",
            "artists": ["Wassily Kandinsky", "Paul Klee", "László Moholy-Nagy"],
            "prompt": (
                "Bauhaus design aesthetic, geometric abstraction primary colors, "
                "form follows function, clean minimal composition, "
                "circle triangle square basic shapes, "
                "Kandinsky Composition series, industrial modern design, "
                "Weimar Dessau school influence, rational artistic order, precise geometry"
            ),
            "negative": "ornamental, decorative, traditional, organic, messy, blurry"
        },

        # ===== 新增当代艺术风格 =====
        "minimalism": {
            "name": "极简主义",
            "artists": ["Donald Judd", "Dan Flavin", "Agnes Martin"],
            "prompt": (
                "Minimalism art movement, extreme simplicity geometric forms, "
                "industrial materials steel aluminum, repetitive modular units, "
                "pure color monochrome surfaces, rejection of expression, "
                "what you see is what you see, 1960s New York galleries, "
                "Agnes Martin subtle grids, serene contemplative, precise clean edges"
            ),
            "negative": "expressive, emotional, complex, decorative, figurative, busy"
        },
        "conceptual_art": {
            "name": "概念艺术",
            "artists": ["Sol LeWitt", "Joseph Kosuth", "Lawrence Weiner"],
            "prompt": (
                "Conceptual art idea over form, text-based artwork, "
                "Sol LeWitt wall drawings geometric instructions, "
                "language as art medium, dematerialization of art object, "
                "philosophical questioning, institutional critique, "
                "1960s avant-garde movement, clean typography, systematic approach"
            ),
            "negative": "traditional painting, emotional, decorative, realistic, ornate"
        },
        "digital_glitch": {
            "name": "数字故障艺术",
            "artists": ["Rosa Menkman", "Cory Arcangel", "JODI"],
            "prompt": (
                "Glitch art digital aesthetic, corrupted data visualization, "
                "pixel sorting datamoshing, RGB color channel separation, "
                "compression artifacts, broken digital imagery, "
                "cyberpunk vaporwave influence, neon colors on dark, "
                "technological malfunction beauty, fragmented distorted pixels"
            ),
            "negative": "clean, perfect, traditional, analog, smooth, simple"
        },
        "street_art": {
            "name": "街头艺术",
            "artists": ["Banksy", "Shepard Fairey", "KAWS"],
            "prompt": (
                "Street art urban graffiti, stencil spray paint aesthetic, "
                "political social commentary, bold graphic design, "
                "Banksy satirical style, Obey propaganda poster influence, "
                "wheat paste murals, underground subversive art, "
                "vibrant colors concrete walls, raw urban energy, detailed stencil work"
            ),
            "negative": "refined gallery art, traditional, academic, subtle, minimal"
        },
        "photorealism": {
            "name": "照相写实主义",
            "artists": ["Chuck Close", "Richard Estes", "Audrey Flack"],
            "prompt": (
                "Photorealism hyperrealistic painting, extreme detail precision, "
                "Chuck Close grid portrait technique, urban reflections, "
                "glossy surfaces chrome reflections, consumer objects, "
                "mechanical reproduction aesthetic, 1970s American art, "
                "photograph-like accuracy, meticulous brushwork invisible"
            ),
            "negative": "abstract, expressive, loose, impressionistic, stylized, simple"
        },
        "land_art": {
            "name": "大地艺术",
            "artists": ["Robert Smithson", "Michael Heizer", "Andy Goldsworthy"],
            "prompt": (
                "Land Art earthworks, Spiral Jetty monumental scale, "
                "natural materials rocks earth, site-specific installation, "
                "geological time scale, environmental intervention, "
                "Andy Goldsworthy ephemeral nature arrangements, "
                "desert landscapes, aerial view geometric patterns, organic forms"
            ),
            "negative": "indoor gallery, small scale, artificial materials, urban, simple"
        },
        "yba_art": {
            "name": "英国青年艺术家",
            "artists": ["Damien Hirst", "Tracey Emin", "Chris Ofili"],
            "prompt": (
                "YBA Young British Artists shock art, Damien Hirst spot paintings, "
                "provocative controversial imagery, Turner Prize aesthetic, "
                "mixed media installations, confessional autobiographical, "
                "1990s Saatchi collection, bold confrontational, "
                "pharmaceutical colors, geometric dot patterns, contemporary British art"
            ),
            "negative": "traditional, conservative, subtle, classical, academic, simple"
        },
        "kinetic_art": {
            "name": "动态艺术",
            "artists": ["Alexander Calder", "Jean Tinguely", "Jesús Rafael Soto"],
            "prompt": (
                "Kinetic art movement sculpture, Alexander Calder mobile forms, "
                "suspended geometric shapes, primary colors red blue yellow black, "
                "balance motion wind-driven, playful organic biomorphic, "
                "wire construction, negative space, "
                "mid-century modern aesthetic, dynamic floating elements"
            ),
            "negative": "static, heavy, grounded, realistic, figurative, simple"
        },
        "arte_povera": {
            "name": "贫穷艺术",
            "artists": ["Michelangelo Pistoletto", "Jannis Kounellis", "Mario Merz"],
            "prompt": (
                "Arte Povera Italian movement, humble everyday materials, "
                "industrial and natural elements combined, mirror reflections, "
                "neon igloo structures Mario Merz, raw unprocessed materials, "
                "anti-commercial aesthetic, 1960s radical Italian art, "
                "earth fire water elements, organic textures, conceptual depth"
            ),
            "negative": "precious materials, polished, commercial, decorative, simple"
        },
        "fluxus": {
            "name": "激浪派",
            "artists": ["Yoko Ono", "Nam June Paik", "George Maciunas"],
            "prompt": (
                "Fluxus intermedia art, experimental avant-garde, "
                "anti-art anti-commercial, event scores instructions, "
                "Nam June Paik video art aesthetic, playful anarchic, "
                "mail art correspondence, rubber stamps collage, "
                "1960s international movement, DIY aesthetic, mixed media chaos"
            ),
            "negative": "traditional, serious, commercial, polished, academic, simple"
        },
    }

    def __init__(self):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.pipe = None
        # 用于保存所有种子的完整日志
        self.seed_log = []
        self.guidance_scale = 10.0
        self.prompt_prefix = "stylized, painterly, abstract"
        self.negative_prefix = "photorealistic, realistic, photo, 3d render"

    def load_sd(self):
        print("📦 加载SDXL高质量模型...")
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=dtype,
        ).to(self.device)
        print("✅ 模型就绪")

    def clear_memory(self):
        """清理GPU内存"""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def generate_seed(self):
        """生成随机种子并记录"""
        return random.randint(1, 2147483647)

    def stylize(self, image, prompt, negative, strength, seed, steps=20):
        """高质量风格化 - 优化速度"""
        w, h = image.size
        # 高分辨率 - 1280px（80MB目标需要更高分辨率）
        new_w = min(1280, (w // 64) * 64)
        scale = new_w / w
        new_h = int(h * scale // 64) * 64
        new_h = min(new_h, 1280)
        image = image.resize((new_w, new_h), Image.LANCZOS)

        try:
            # 在MPS上使用CPU Generator更稳定（也便于跨设备复现）
            generator = torch.Generator(device="cpu").manual_seed(seed)
            result = self.pipe(
                prompt=f"{self.prompt_prefix}, {prompt}",
                negative_prompt=f"{self.negative_prefix}, {negative}",
                image=image,
                strength=strength,
                guidance_scale=self.guidance_scale,  # 稍微降低，加快速度
                num_inference_steps=steps,  # 20步足够了
                generator=generator,
            ).images[0]
        finally:
            self.clear_memory()

        return result

    def to_svg_ultra_quality(self, image, num_colors=256, simplify=0.0003):
        """艺术风格SVG转换 - 大色块 + 简洁多边形"""
        cv2.setNumThreads(mp.cpu_count())
        
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        # K-means颜色量化
        pixels = img_array.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1)

        tried = []
        labels = None
        centers = None
        for k in [num_colors, min(num_colors, 256), min(num_colors, 128), 64]:
            if k in tried:
                continue
            tried.append(k)
            try:
                _, labels, centers = cv2.kmeans(
                    pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
                )
                num_colors = k
                break
            except cv2.error:
                labels = None
                centers = None
                continue

        if labels is None or centers is None:
            raise RuntimeError("OpenCV kmeans failed for all attempted num_colors")
        centers = centers.astype(np.uint8)
        quantized = centers[labels.flatten()].reshape(img_array.shape)

        dwg = svgwrite.Drawing(size=(w, h))
        
        # 背景
        bg_color = centers[np.argmax(np.bincount(labels.flatten()))]
        dwg.add(dwg.rect(insert=(0, 0), size=(w, h), 
                        fill=f'rgb({bg_color[0]},{bg_color[1]},{bg_color[2]})'))

        # 按面积排序颜色（从大到小）
        color_areas = []
        for i, color in enumerate(centers):
            mask = np.all(quantized == color, axis=2)
            area = np.sum(mask)
            # 只保留较大的色块区域（面积 > 图像的0.1%）
            min_area = w * h * 0.001
            if area > min_area:
                color_areas.append((area, color, mask))
        color_areas.sort(reverse=True)

        for _, color, mask in color_areas:
            mask_uint8 = mask.astype(np.uint8) * 255
            
            # 形态学操作：先膨胀再腐蚀，填补小孔洞，平滑边缘
            kernel = np.ones((5, 5), np.uint8)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
            mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
            
            # 高斯模糊后重新二值化，让边缘更平滑
            mask_uint8 = cv2.GaussianBlur(mask_uint8, (7, 7), 0)
            _, mask_uint8 = cv2.threshold(mask_uint8, 127, 255, cv2.THRESH_BINARY)
            
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if len(contour) < 3:
                    continue
                # 计算轮廓面积，过滤太小的碎片
                contour_area = cv2.contourArea(contour)
                if contour_area < min_area:
                    continue
                    
                # 使用更激进的简化
                epsilon = simplify * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3:
                    continue
                    
                points = [(int(p[0][0]), int(p[0][1])) for p in approx]
                fill = f'rgb({color[0]},{color[1]},{color[2]})'
                dwg.add(dwg.polygon(points=points, fill=fill, stroke='none'))

        return dwg.tostring()


    def save_seed_log(self, output_dir, log_data):
        """保存种子日志 - 每次生成后立即保存"""
        log_path = Path(output_dir) / "seed_log.json"
        with open(log_path, 'w', encoding='utf-8') as f:
            json.dump({
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "total_count": len(log_data),
                "styles_count": len(self.ART_STYLES),
                "settings": {
                    "target_resolution": "<=1280px",
                    "target_colors": "24-64",
                    "target_file_size": "80MB+",
                    "simplify": "0.00001-0.00005",
                    "inference_steps": 30,
                    "guidance_scale": self.guidance_scale
                },
                "files": log_data
            }, f, ensure_ascii=False, indent=2)
        return log_path

    def generate_massive(self, image_path, output_dir, total_count=200, start_from=1):
        """生成200张超高质量多风格SVG"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image = Image.open(image_path).convert("RGB")
        
        styles = [k for k in self.ART_STYLES.keys() if k != "photorealism"]
        num_styles = len(styles)
        
        # 每种风格生成的数量
        per_style = total_count // num_styles
        extra = total_count % num_styles

        # 艺术风格参数 - 追求简洁有力的色块而非像素级细节
        strength_range = [0.80, 0.85, 0.90, 0.95]  # 更强的风格化
        color_range = [24, 32, 48, 64]  # 少量颜色 = 更明确的色块分区
        simplify_range = [0.008, 0.006, 0.004, 0.003]  # 高简化 = 更简洁的多边形边缘

        print("=" * 70)
        print("🎨 超高质量艺术风格SVG生成器")
        print("=" * 70)
        print(f"📊 总计生成: {total_count} 张")
        print(f"🎭 艺术流派: {num_styles} 种")
        print(f"📁 输出目录: {output_dir}")
        print(f"🎯 目标文件大小: 80MB+")
        print(f"🎨 颜色数量: {min(color_range)}-{max(color_range)}")
        print("=" * 70)
        print("\n包含的艺术流派：")
        for i, (key, style) in enumerate(self.ART_STYLES.items(), 1):
            artists = ", ".join(style["artists"])
            print(f"  {i:2d}. {style['name']} ({artists})")
        print("=" * 70)

        # 尝试加载已有的日志
        log_path = output_dir / "seed_log.json"
        if log_path.exists() and start_from > 1:
            with open(log_path, 'r', encoding='utf-8') as f:
                existing_log = json.load(f)
                log_data = existing_log.get("files", [])
                print(f"📋 已加载现有日志，包含 {len(log_data)} 条记录")
        else:
            log_data = []

        generated = 0
        skipped = 0

        for style_idx, style_key in enumerate(styles):
            style = self.ART_STYLES[style_key]
            count_for_style = per_style + (1 if style_idx < extra else 0)

            print(f"\n{'='*60}")
            print(f"🎨 [{style_idx+1}/{num_styles}] {style['name']}")
            print(f"   艺术家: {', '.join(style['artists'])}")
            print(f"   生成数量: {count_for_style}")
            print(f"{'='*60}")

            for var_idx in range(count_for_style):
                version = skipped + generated + 1
                
                # 跳过已生成的
                if version < start_from:
                    skipped += 1
                    continue
                
                # 参数组合
                strength = strength_range[var_idx % len(strength_range)]
                num_colors = color_range[var_idx % len(color_range)]
                simplify = simplify_range[var_idx % len(simplify_range)]
                
                # 生成随机种子
                seed = self.generate_seed()

                print(f"\n  [{var_idx+1}/{count_for_style}] art_v{version:03d}")
                print(f"     风格: {style['name']}")
                print(f"     strength: {strength}, colors: {num_colors}, simplify: {simplify}")
                print(f"     seed: {seed}")

                try:
                    # 风格化
                    inference_steps = 30
                    styled = self.stylize(
                        image, 
                        style["prompt"], 
                        style["negative"],
                        strength, 
                        seed,
                        steps=inference_steps  # 20步足够，速度快2倍
                    )

                    # 超高质量SVG转换
                    svg_content = self.to_svg_ultra_quality(styled, num_colors, simplify)

                    # 保存SVG
                    svg_path = output_dir / f"art_v{version:03d}_{style_key}.svg"
                    with open(svg_path, 'w') as f:
                        f.write(svg_content)

                    size_mb = svg_path.stat().st_size / (1024 * 1024)
                    print(f"     ✅ {svg_path.name} ({size_mb:.2f} MB)")

                    # 记录完整日志
                    log_entry = {
                        "version": version,
                        "style_key": style_key,
                        "style_name": style["name"],
                        "artists": style["artists"],
                        "seed": seed,
                        "strength": strength,
                        "num_colors": num_colors,
                        "simplify": simplify,
                        "inference_steps": inference_steps,
                        "guidance_scale": self.guidance_scale,
                        "file_size_mb": round(size_mb, 2),
                        "svg_file": svg_path.name,
                        "generated_at": datetime.now().isoformat(),
                        "prompt": style["prompt"],
                        "negative_prompt": style["negative"]
                    }
                    log_data.append(log_entry)

                    # 每次生成后立即保存日志
                    self.save_seed_log(output_dir, log_data)

                except Exception as e:
                    print(f"     ❌ 错误: {e}")
                    import traceback
                    traceback.print_exc()
                    self.clear_memory()

                generated += 1
                
                # 每3张清理一次内存
                if generated % 3 == 0:
                    self.clear_memory()
                    print(f"     🧹 内存已清理 (已生成{generated}张，总进度{version}/{total_count})")

        print("\n" + "=" * 70)
        print(f"✅ 完成！生成了 {generated} 个超高质量艺术SVG")
        print(f"📁 位置: {output_dir}")
        print(f"📋 种子日志: {log_path}")
        print("=" * 70)

        return generated

    def regenerate_single(self, image_path, output_dir, version, seed, style_key, 
                          strength=0.7, num_colors=256, simplify=0.0002):
        """根据种子重新生成单张图片 - 用于细化处理"""
        output_dir = Path(output_dir)
        image = Image.open(image_path).convert("RGB")
        
        if style_key not in self.ART_STYLES:
            print(f"❌ 未知风格: {style_key}")
            return None
            
        style = self.ART_STYLES[style_key]
        
        print(f"🔄 重新生成 v{version:03d} - {style['name']}")
        print(f"   seed: {seed}, strength: {strength}, colors: {num_colors}")
        
        styled = self.stylize(
            image, 
            style["prompt"], 
            style["negative"],
            strength, 
            seed,
            steps=40
        )
        
        svg_content = self.to_svg_ultra_quality(styled, num_colors, simplify)
        
        svg_path = output_dir / f"art_v{version:03d}_{style_key}_refined.svg"
        with open(svg_path, 'w') as f:
            f.write(svg_content)
            
        size_mb = svg_path.stat().st_size / (1024 * 1024)
        print(f"✅ 保存: {svg_path.name} ({size_mb:.2f} MB)")
        
        return svg_path


def main():
    input_image = "/Volumes/Seagate/SAM3/01_input/Ladygaga_2.jpg"
    output_dir = "/Volumes/Seagate/SAM3/06_style_art/output/massive_art_hq"
    
    # 从第1张开始（全新生成）
    START_FROM = 1
    # 先生成50张
    TOTAL_COUNT = 1

    if not Path(input_image).exists():
        print(f"❌ 找不到: {input_image}")
        return

    generator = MassiveArtGenerator()
    generator.load_sd()
    generator.generate_massive(input_image, output_dir, total_count=TOTAL_COUNT, start_from=START_FROM)

    # 打开文件夹
    import subprocess
    subprocess.run(["open", output_dir])


if __name__ == "__main__":
    main()
