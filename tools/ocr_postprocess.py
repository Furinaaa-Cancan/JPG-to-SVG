#!/usr/bin/env python3
"""
OCR后处理：修复常见拼写错误
针对科研图的专业术语纠正
"""

import re
from difflib import SequenceMatcher


# 科研图常见术语词典
SCIENTIFIC_TERMS = {
    # 电子/电路术语
    "excitation": ["excitaticn", "exc tation", "exc taiion", "excitat on"],
    "resistance": ["resistancc", "res stance", "resistace"],
    "compression": ["compresslon", "compress on"],
    "tension": ["tcnsion", "tens on"],
    "gauges": ["gaugcs", "gauqes", "qauges"],
    "output": ["ojtput", "outpjt", "0utput"],
    "circuit": ["circu t", "circjit"],
    "physical": ["physlcal", "phys cal"],
    "increases": ["incrcases", "lncreases", "increaces"],
    "decreases": ["decrcases", "decreaces"],
    "connect": ["conncc:", "connec:", "conncct"],
    "world": ["wor1d", "worId"],
    
    # 常见符号和数字
    "R1": ["r1", "R 1", "Rl"],
    "R2": ["r2", "R 2", "R?"],
    "R3": ["r3", "R 3"],
    "R4": ["r4", "R 4"],
    "T1": ["t1", "T 1", "Tl"],
    "T2": ["t2", "T 2"],
    "C1": ["c1", "C 1", "Cl"],
    "C2": ["c2", "C 2"],
    "V1": ["v1", "V 1", "Vl"],
    "V2": ["v2", "V 2"],
    "(+)": ["(+;", "(+ )", "( +)"],
    "(-)": ["(-;", "(- )", "( -)"],
    "&": ["=", "8"],
}

# 完整短语纠正
PHRASE_CORRECTIONS = {
    "Correct Tcnsion": "Connect Tension",
    "Gaugcs": "Gauges",
    "(T1, T2)0 R1": "(T1, T2) to R1 & R4",
    "Excitaticn": "Excitation",
    "Resistancc Incrcases": "Resistance Increases",
    "Exc tation": "Excitation",
    "OJtput": "Output",
    "Compression (-;": "Compression (-)",
    "Resistance Decreases": "Resistance Decreases",  # 正确
    "Exc taiion": "Excitation",
    "Conncc: Compression": "Connect Compression",
    "Physical World": "Physical World",  # 正确
    "Gauges (C1, C2) to R? =": "Gauges (C1, C2) to R2 & R3",
    "Circuit World": "Circuit World",  # 正确
    "Tension (+)": "Tension (+)",  # 正确
}


def similar(a: str, b: str) -> float:
    """计算两个字符串的相似度"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def correct_word(word: str) -> str:
    """纠正单个词"""
    word_lower = word.lower()
    
    for correct, variants in SCIENTIFIC_TERMS.items():
        if word_lower == correct.lower():
            return correct  # 已正确
        
        for variant in variants:
            if word_lower == variant.lower():
                # 保持原始大小写模式
                if word.isupper():
                    return correct.upper()
                elif word[0].isupper():
                    return correct.capitalize()
                return correct
    
    return word


def correct_phrase(text: str) -> str:
    """纠正完整短语"""
    # 先尝试精确匹配
    if text in PHRASE_CORRECTIONS:
        return PHRASE_CORRECTIONS[text]
    
    # 尝试模糊匹配
    best_match = None
    best_score = 0.7  # 最低相似度阈值
    
    for wrong, correct in PHRASE_CORRECTIONS.items():
        score = similar(text, wrong)
        if score > best_score:
            best_score = score
            best_match = correct
    
    if best_match:
        return best_match
    
    # 逐词纠正
    words = text.split()
    corrected_words = [correct_word(w) for w in words]
    return ' '.join(corrected_words)


def escape_xml(text: str) -> str:
    """转义XML特殊字符"""
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def process_svg_text(svg_content: str) -> str:
    """处理SVG中的所有文字"""
    
    def replace_text(match):
        full_tag = match.group(0)
        text_content = match.group(1)
        corrected = correct_phrase(text_content)
        # 转义XML特殊字符
        corrected = escape_xml(corrected)
        return full_tag.replace(f">{text_content}</text>", f">{corrected}</text>")
    
    # 匹配 <text ...>内容</text>
    pattern = r'<text[^>]*>([^<]+)</text>'
    corrected_svg = re.sub(pattern, replace_text, svg_content)
    
    return corrected_svg


def correct_svg_file(input_path: str, output_path: str = None) -> dict:
    """
    纠正SVG文件中的OCR错误
    """
    print("\n" + "="*60)
    print("🔤 OCR后处理：修复拼写错误")
    print("="*60)
    
    # 读取SVG
    with open(input_path, 'r', encoding='utf-8') as f:
        svg_content = f.read()
    
    # 提取所有文字
    pattern = r'<text[^>]*>([^<]+)</text>'
    texts = re.findall(pattern, svg_content)
    
    print(f"\n   找到 {len(texts)} 个文字元素")
    
    # 纠正
    corrections = []
    for text in texts:
        corrected = correct_phrase(text)
        if corrected != text:
            corrections.append({
                "original": text,
                "corrected": corrected
            })
            print(f"   ✓ '{text}' → '{corrected}'")
    
    if not corrections:
        print("   没有需要纠正的错误")
        return {"corrections": 0}
    
    # 应用纠正
    corrected_svg = process_svg_text(svg_content)
    
    # 保存
    if output_path is None:
        output_path = input_path.replace('.svg', '_corrected.svg')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(corrected_svg)
    
    print(f"\n   ✅ 已纠正 {len(corrections)} 处错误")
    print(f"   📄 保存到: {output_path}")
    
    return {
        "corrections": len(corrections),
        "details": corrections,
        "output_path": output_path
    }


def main():
    import glob
    
    # 找到最新的SVG文件
    svg_dir = "/Volumes/Seagate/SAM3/02_output/scientific_svg"
    svg_files = glob.glob(f"{svg_dir}/scientific_figure_*.svg")
    
    if not svg_files:
        print("未找到SVG文件")
        return
    
    latest_svg = sorted(svg_files)[-1]
    print(f"处理文件: {latest_svg}")
    
    result = correct_svg_file(latest_svg)
    
    print("\n" + "="*60)
    print("✅ OCR后处理完成")
    print("="*60)


if __name__ == "__main__":
    main()
