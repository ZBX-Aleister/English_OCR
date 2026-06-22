# -*- coding: utf-8 -*-
"""
=============================================================================
文本清洗模块 - OCR 输出后处理
=============================================================================
功能: 清洗 EasyOCR 输出文本中的乱码、拼写错误、格式问题
     针对手写英文 OCR 常见错误模式进行专项修复

常见手写 OCR 错误类型:
  1. 字符混淆: rn→m, cl→d, vv→w
  2. 数字字母混淆: 0↔o, 1↔l, 5↔s
  3. 多余空格或缺失空格
  4. 非英文字符混入
  5. 重复字符 / 随机乱码
=============================================================================
"""

import re
import string
from typing import List, Tuple

from config import OCR_CONFUSION_PAIRS


def normalize_whitespace(text: str) -> str:
    """
    规范化空白字符
    - 多个空格合并为单个
    - 去除首尾空格
    - 统一换行符
    """
    text = text.strip()
    text = re.sub(r"[ \t]+", " ", text)       # 多空格合并
    text = re.sub(r"\n{3,}", "\n\n", text)     # 多余空行合并
    return text


def remove_gibberish(text: str, min_word_len: int = 2, max_word_len: int = 30) -> str:
    """
    移除疑似乱码/非英文内容

    策略:
    - 过滤过长/过短的"单词"（手写OCR乱码常表现为超长字符串）
    - 过滤含过多非字母字符的词
    - 过滤连续辅音过多（不太可能是英文单词）
    """
    words = text.split()
    cleaned_words = []

    for word in words:
        # 去除首尾标点
        stripped = word.strip(string.punctuation + ".,;:!?\"'()[]{}")

        # 空字符串跳过
        if not stripped:
            cleaned_words.append(word)
            continue

        # 长度过滤（超长大概率是乱码）
        if len(stripped) > max_word_len or len(stripped) < min_word_len:
            # 但保留常见短词
            if stripped.lower() not in {"a", "i", "am", "is", "be", "to", "in", "on", "at", "an", "as", "we", "he", "she", "it", "of"}:
                continue

        # 非字母比例过高（>50%）视为乱码
        alpha_count = sum(1 for c in stripped if c.isalpha())
        if len(stripped) > 0 and alpha_count / len(stripped) < 0.5:
            continue

        # 连续相同字符过多（>4）
        if re.search(r"(.)\1{4,}", stripped):
            continue

        # 全是辅音（无元音且长度>3），不可能是英文单词
        vowels = set("aeiouAEIOU")
        if len(stripped) > 3 and not any(c in vowels for c in stripped):
            continue

        cleaned_words.append(word)

    return " ".join(cleaned_words)


def correct_ocr_confusions(text: str) -> str:
    """
    修正常见 OCR 字符混淆模式

    针对手写识别中容易混淆的字符对进行替换。
    这些混淆对来源于 EasyOCR 在手写体上的常见错误统计。
    """
    corrected = text
    for wrong, right in OCR_CONFUSION_PAIRS.items():
        # 只替换单词内部的混淆模式
        corrected = re.sub(
            r"(?<=\w)" + re.escape(wrong) + r"(?=\w)",
            right,
            corrected
        )
    return corrected


def clean_ocr_text(raw_text: str, confidence: float = 0.0) -> str:
    """
    主清洗函数 - 串联所有清洗步骤

    Args:
        raw_text: EasyOCR 输出的原始文本
        confidence: 平均置信度（用于判断是否需要更激进的清洗）

    Returns:
        清洗后的干净文本

    清洗流程:
        1. 去除不可打印字符
        2. 规范化空白
        3. 去除乱码
        4. 修正 OCR 混淆
        5. 最终英文验证
    """
    if not raw_text or not raw_text.strip():
        return ""

    # 1. 去除控制字符（保留换行）
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", raw_text)

    # 2. 规范化空白
    text = normalize_whitespace(text)

    # 3. 低置信度时使用更严格的清洗
    if confidence < 0.5:
        text = remove_gibberish(text, min_word_len=1, max_word_len=25)
    else:
        text = remove_gibberish(text)

    # 4. 修正常见 OCR 混淆
    text = correct_ocr_confusions(text)

    # 5. 最终空白规范化
    text = normalize_whitespace(text)

    return text


def validate_english_text(text: str) -> Tuple[bool, str]:
    """
    验证文本是否为有效的英文内容

    Returns:
        (is_valid, reason) 元组
    """
    if not text or not text.strip():
        return False, "文本为空"

    # 统计可打印 ASCII 字符比例
    printable = sum(1 for c in text if c in string.printable)
    if len(text) > 0 and printable / len(text) < 0.8:
        return False, "含过多非英文字符"

    # 至少包含一些字母
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 3:
        return False, "英文内容过少"

    return True, "有效英文文本"


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    从清洗后的文本中提取关键词（用于知识库检索）

    策略: 基于规则的关键词提取
    - 提取长度 >= 4 的实词（非停用词）
    - 按出现顺序返回
    """
    # 常见英文停用词
    stop_words = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "has",
        "have", "been", "some", "them", "who", "will", "more",
        "when", "your", "about", "which", "their", "this", "that",
        "with", "from", "they", "what", "were", "there", "would",
        "could", "should", "than", "then", "into", "also", "very",
        "just", "over", "such", "only", "other", "after", "before",
    }

    words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
    keywords = []
    seen = set()

    for word in words:
        if word not in stop_words and word not in seen:
            keywords.append(word)
            seen.add(word)
            if len(keywords) >= top_n:
                break

    return keywords
