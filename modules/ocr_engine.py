# -*- coding: utf-8 -*-
"""
=============================================================================
OCR 引擎模块 - EasyOCR 手写英文识别
=============================================================================
功能: 封装 EasyOCR，集成预处理流水线，提供统一的识别接口

核心类:
  OCREngine: 初始化一次，重复调用 recognize()

识别流程:
  加载图片 → 预处理 → EasyOCR 识别 → 置信度过滤 → 文本清洗 → 返回结果

调试方法:
  - 低准确率时先检查预处理效果（用 visualize_preprocessing 导出中间图片）
  - 调整 OCR_CONFIDENCE_THRESHOLD 过滤低质量结果
  - 检查清洗后文本，判断问题在 OCR 端还是清洗端
=============================================================================
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import traceback

# MUST set SSL bypass BEFORE importing easyocr (urllib caches SSL context on import)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[WARNING] EasyOCR not installed. Run: pip install easyocr")

from config import (
    OCR_CONFIDENCE_THRESHOLD,
    OCR_LANGUAGES,
    OCR_GPU_ENABLED,
    EASYOCR_MODEL_DIR,
)
from modules.preprocessing import preprocess_image
from utils.text_cleaner import clean_ocr_text, validate_english_text, extract_keywords


class OCREngine:
    """
    EasyOCR 手写英文识别引擎

    用法:
        engine = OCREngine()
        result = engine.recognize("handwritten.jpg")
        print(result["text"])        # 清洗后的文本
        print(result["confidence"])  # 平均置信度
        print(result["raw_words"])   # 逐词识别详情
    """

    def __init__(self):
        """初始化 EasyOCR Reader（单例模式，避免重复加载模型）"""
        self._reader = None
        self._initialized = False

    def _lazy_init(self):
        """延迟初始化 - 仅在首次调用时加载模型"""
        if self._initialized:
            return

        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR 未安装。请运行: pip install easyocr\n"
                "注意: 首次运行时 EasyOCR 会自动下载英文识别模型 (~68MB)"
            )

        try:
            gpu = OCR_GPU_ENABLED
            self._reader = easyocr.Reader(
                OCR_LANGUAGES,
                gpu=gpu,
                model_storage_directory=str(EASYOCR_MODEL_DIR),
                download_enabled=True,
                verbose=True,
            )
            self._initialized = True
            print(f"[OCR] EasyOCR initialized (GPU={gpu}, model_dir={EASYOCR_MODEL_DIR})")
        except Exception as e:
            if OCR_GPU_ENABLED:
                print(f"[OCR] GPU init failed ({e}), falling back to CPU")
                self._reader = easyocr.Reader(
                    OCR_LANGUAGES,
                    gpu=False,
                    model_storage_directory=str(EASYOCR_MODEL_DIR),
                    download_enabled=True,
                    verbose=True,
                )
                self._initialized = True
            else:
                raise

    def recognize(
        self,
        image: np.ndarray,
        preprocess: bool = True,
        enable_deskew: bool = True,
        enable_border_removal: bool = True,
    ) -> Dict:
        """
        识别图片中的手写英文

        Args:
            image: 输入图片 (numpy array, BGR/Gray)
            preprocess: 是否启用预处理（建议始终开启）
            enable_deskew: 预处理中是否启用倾斜矫正
            enable_border_removal: 预处理中是否裁剪白边

        Returns:
            {
                "text": str,           # 清洗后的完整文本
                "confidence": float,   # 平均置信度 (0-1)
                "raw_text": str,       # EasyOCR 原始输出文本
                "raw_words": [         # 逐词详细结果
                    {
                        "text": str,
                        "confidence": float,
                        "bbox": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
                    },
                    ...
                ],
                "keywords": [str, ...], # 提取的关键词
                "is_valid": bool,       # 是否为有效英文
                "error": str or None,   # 错误信息
            }
        """
        result = {
            "text": "",
            "confidence": 0.0,
            "raw_text": "",
            "raw_words": [],
            "keywords": [],
            "is_valid": False,
            "error": None,
        }

        try:
            # ---- 1. 延迟初始化 EasyOCR ----
            self._lazy_init()

            # ---- 2. 图像预处理 ----
            if preprocess:
                processed = preprocess_image(
                    image,
                    enable_deskew=enable_deskew,
                    enable_border_removal=enable_border_removal
                )
            else:
                # 不预处理，但确保是 numpy array
                processed = np.array(image)

            # ---- 3. EasyOCR 识别 ----
            # detail=1 返回每个词的置信度和边界框
            raw_results = self._reader.readtext(processed, detail=1)

            if not raw_results:
                result["error"] = "OCR 未识别到任何文字，请检查图片质量"
                return result

            # ---- 4. 置信度过滤 ----
            filtered_words = []
            total_conf = 0.0

            for (bbox, word_text, confidence) in raw_results:
                if confidence >= OCR_CONFIDENCE_THRESHOLD:
                    filtered_words.append({
                        "text": word_text,
                        "confidence": round(confidence, 4),
                        "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                    })
                    total_conf += confidence

            if not filtered_words:
                # 所有词都低于阈值，取置信度最高的几个
                raw_results.sort(key=lambda x: x[2], reverse=True)
                for (bbox, word_text, confidence) in raw_results[:3]:
                    filtered_words.append({
                        "text": word_text,
                        "confidence": round(confidence, 4),
                        "bbox": [[int(p[0]), int(p[1])] for p in bbox],
                        "low_confidence": True,
                    })
                    total_conf += confidence

            result["raw_words"] = filtered_words

            # ---- 5. 拼接原始文本 ----
            raw_text = " ".join([w["text"] for w in filtered_words])
            result["raw_text"] = raw_text

            # 计算平均置信度
            if filtered_words:
                result["confidence"] = round(
                    total_conf / len(filtered_words), 4
                )

            # ---- 6. 文本清洗 ----
            cleaned_text = clean_ocr_text(raw_text, result["confidence"])
            result["text"] = cleaned_text

            # ---- 7. 提取关键词 ----
            result["keywords"] = extract_keywords(cleaned_text)

            # ---- 8. 验证有效性 ----
            is_valid, _ = validate_english_text(cleaned_text)
            result["is_valid"] = is_valid

        except Exception as e:
            result["error"] = f"OCR 识别异常: {str(e)}\n{traceback.format_exc()}"

        return result

    def recognize_with_debug(
        self,
        image: np.ndarray,
    ) -> Dict:
        """
        调试模式识别 - 返回预处理中间图片供检查

        Returns:
            recognize() 的全部字段 + "preprocessed_image" (预处理后的图片数组)
        """
        from modules.preprocessing import preprocess_image

        processed, intermediates = preprocess_image(
            image, return_intermediates=True
        )

        result = self.recognize(image, preprocess=True)
        result["intermediates"] = intermediates  # 各预处理步骤图片
        result["preprocessed_image"] = processed   # 最终预处理结果

        return result

    @property
    def is_ready(self) -> bool:
        """检查引擎是否可用"""
        return EASYOCR_AVAILABLE
