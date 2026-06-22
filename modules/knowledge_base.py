# -*- coding: utf-8 -*-
"""
=============================================================================
轻量化本地知识库模块 - JSON + FAISS 向量检索
=============================================================================
功能: 基于 FAISS 向量检索的本地英语知识库，支持语义搜索
      当网络不可用（无法下载嵌入模型）时，自动回退到关键词匹配模式

数据来源:
  - vocabulary.json   : 考研/四六级高频词汇（~200 条）
  - grammar_errors.json: 常见语法错误模式（~50 条）
  - examples.json     : 标准写作例句（~100 条）

检索流程:
  FAISS 模式: 用户查询 → sentence-transformers 编码 → FAISS 搜索 → Top-K
  回退模式:   用户查询 → 关键词提取 → 字符串匹配打分 → Top-K

技术栈:
  - sentence-transformers (all-MiniLM-L6-v2) : 文本嵌入 (~80MB, 可选)
  - FAISS IndexFlatL2                        : L2 距离向量搜索 (可选)
  - 内置关键词匹配                            : 零依赖回退方案
=============================================================================
"""

import json
import os
import re
from typing import List, Dict, Optional, Tuple
from difflib import SequenceMatcher

from config import (
    KB_DIR,
    KB_EMBEDDING_MODEL,
    KB_RETRIEVAL_TOP_K,
    KB_INDEX_PATH,
    KB_META_PATH,
)


class KnowledgeBase:
    """
    本地英语知识库（JSON + FAISS，带回退策略）

    用法:
        kb = KnowledgeBase()
        kb.load_data()
        kb.build_index()        # 尝试构建 FAISS 索引（失败则回退）
        results = kb.search("sustainable development")
    """

    def __init__(self):
        """初始化知识库 - 延迟加载模型以节省启动时间"""
        self._embedding_model = None
        self._index = None           # FAISS index
        self._entries = []           # 所有知识库条目
        self._id_to_entry = {}       # ID 到条目的映射
        self._loaded = False
        self._use_fallback = False   # 是否使用回退检索模式
        self._fallback_reason = ""   # 回退原因

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def search_mode(self) -> str:
        """返回当前检索模式: 'faiss' 或 'fallback'"""
        return "fallback" if self._use_fallback else "faiss"

    def _try_load_embedding_model(self) -> bool:
        """
        Try to load the sentence-transformers embedding model.
        First attempts offline (fail fast, ~1s), then tries network download.
        Returns True if successfully loaded.
        """
        if self._embedding_model is not None:
            return True

        try:
            import os

            # Step 1: Quick offline check
            # Setting these env vars BEFORE importing makes it fail instantly if no cache
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"

            try:
                from sentence_transformers import SentenceTransformer
                print(f"[KB] Checking cached model: {KB_EMBEDDING_MODEL}...")
                self._embedding_model = SentenceTransformer(KB_EMBEDDING_MODEL)
                print("[KB] Embedding model loaded from cache")
                # Clear offline flags so subsequent calls work normally
                del os.environ["TRANSFORMERS_OFFLINE"]
                del os.environ["HF_HUB_OFFLINE"]
                return True
            except Exception as e:
                # Clean up offline flags
                if "TRANSFORMERS_OFFLINE" in os.environ:
                    del os.environ["TRANSFORMERS_OFFLINE"]
                if "HF_HUB_OFFLINE" in os.environ:
                    del os.environ["HF_HUB_OFFLINE"]

                # Check if sentence_transformers is even installed
                if "No module named" in str(e):
                    raise ImportError(str(e))

                print(f"[KB] Model not cached ({type(e).__name__}), will try download...")

            # Step 2: Try network download
            from sentence_transformers import SentenceTransformer
            mirror = os.environ.get("HF_ENDPOINT", "")
            if mirror:
                print(f"[KB] Using HF mirror: {mirror}")
            else:
                print("[KB] Tip: set HF_ENDPOINT=https://hf-mirror.com for China mirror")

            self._embedding_model = SentenceTransformer(KB_EMBEDDING_MODEL)
            print("[KB] Embedding model loaded successfully")
            return True

        except ImportError:
            self._fallback_reason = "sentence-transformers not installed"
            print("[KB] sentence-transformers not installed, using fallback mode")
            return False
        except Exception as e:
            self._fallback_reason = str(e)[:100]
            print(f"[KB] Failed to load embedding model: {e}")
            print(f"[KB] Automatically switching to keyword fallback mode")
            print(f"[KB] To download the model, ensure network access to huggingface.co")
            print(f"[KB]   or set mirror: set HF_ENDPOINT=https://hf-mirror.com")
            return False

    # ==================== 数据加载 ====================

    def load_data(self, vocab_path: str = None, grammar_path: str = None,
                   examples_path: str = None) -> int:
        """
        加载 JSON 数据文件并建立条目列表
        """
        vocab_path = vocab_path or str(KB_DIR / "vocabulary.json")
        grammar_path = grammar_path or str(KB_DIR / "grammar_errors.json")
        examples_path = examples_path or str(KB_DIR / "examples.json")

        self._entries = []
        entry_id = 0

        # 1. 加载词汇数据
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_data = json.load(f)
            for item in vocab_data.get("entries", []):
                text = (f"Word: {item['word']} | Phonetic: {item.get('phonetic', '')} | "
                        f"Meaning: {item.get('meaning', '')} | Synonyms: {item.get('synonyms', '')} | "
                        f"Collocations: {item.get('collocations', '')}")
                entry = {
                    "id": entry_id,
                    "type": "vocab",
                    "text": text,
                    "search_text": f"{item['word']} {item.get('meaning', '')} {item.get('synonyms', '')}",
                    "metadata": {
                        "word": item.get("word", ""),
                        "phonetic": item.get("phonetic", ""),
                        "meaning": item.get("meaning", ""),
                        "level": item.get("level", ""),
                        "synonyms": item.get("synonyms", ""),
                        "collocations": item.get("collocations", ""),
                        "example": item.get("example", ""),
                    },
                }
                self._entries.append(entry)
                self._id_to_entry[entry_id] = entry
                entry_id += 1
            print(f"[KB] Loaded vocabulary: {entry_id} entries")
        except FileNotFoundError:
            print(f"[KB] Vocabulary file not found: {vocab_path}")

        vocab_count = entry_id

        # 2. 加载语法错误数据
        try:
            with open(grammar_path, "r", encoding="utf-8") as f:
                grammar_data = json.load(f)
            for item in grammar_data.get("entries", []):
                text = (f"Grammar: {item['category']} | Pattern: {item['pattern']} | "
                        f"Error Example: {item.get('error_example', '')} | "
                        f"Correction: {item.get('correction', '')} | "
                        f"Explanation: {item.get('explanation', '')}")
                entry = {
                    "id": entry_id,
                    "type": "grammar",
                    "text": text,
                    "search_text": f"{item['category']} {item['pattern']} {' '.join(item.get('keywords', []))}",
                    "metadata": {
                        "category": item.get("category", ""),
                        "pattern": item.get("pattern", ""),
                        "error_example": item.get("error_example", ""),
                        "correction": item.get("correction", ""),
                        "explanation": item.get("explanation", ""),
                    },
                }
                self._entries.append(entry)
                self._id_to_entry[entry_id] = entry
                entry_id += 1
            print(f"[KB] Loaded grammar errors: {entry_id - vocab_count} entries")
        except FileNotFoundError:
            print(f"[KB] Grammar file not found: {grammar_path}")

        grammar_count = entry_id - vocab_count

        # 3. 加载例句数据
        try:
            with open(examples_path, "r", encoding="utf-8") as f:
                examples_data = json.load(f)
            for item in examples_data.get("entries", []):
                text = (f"Example: {item['sentence']} | Source: {item.get('source', '')} | "
                        f"Topic: {item.get('topic', '')} | Usage: {item.get('usage', '')} | "
                        f"Notes: {item.get('notes', '')}")
                entry = {
                    "id": entry_id,
                    "type": "example",
                    "text": text,
                    "search_text": f"{item['sentence']} {item.get('topic', '')} {item.get('usage', '')}",
                    "metadata": {
                        "sentence": item.get("sentence", ""),
                        "source": item.get("source", ""),
                        "topic": item.get("topic", ""),
                        "usage": item.get("usage", ""),
                        "level": item.get("level", ""),
                        "notes": item.get("notes", ""),
                    },
                }
                self._entries.append(entry)
                self._id_to_entry[entry_id] = entry
                entry_id += 1
            print(f"[KB] Loaded examples: {entry_id - vocab_count - grammar_count} entries")
        except FileNotFoundError:
            print(f"[KB] Examples file not found: {examples_path}")

        self._loaded = True
        print(f"[KB] Data loading complete: {len(self._entries)} total entries")
        return len(self._entries)

    # ==================== FAISS 索引构建 ====================

    def build_index(self, force_rebuild: bool = False) -> bool:
        """
        构建 FAISS 向量索引（或加载已有索引）
        如果嵌入模型不可用，自动回退到关键词匹配模式

        Returns:
            是否成功构建（回退模式也返回 True）
        """
        if not self._loaded:
            print("[KB] Data not loaded, please call load_data() first")
            return False

        # ---- 尝试加载已有 FAISS 索引 ----
        if not force_rebuild and os.path.exists(KB_INDEX_PATH) and os.path.exists(KB_META_PATH):
            try:
                import faiss
                self._index = faiss.read_index(str(KB_INDEX_PATH))
                with open(KB_META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                print(f"[KB] Loaded existing FAISS index ({meta.get('num_entries', '?')} entries)")
                self._use_fallback = False
                return True
            except Exception as e:
                print(f"[KB] Failed to load existing index: {e}, rebuilding...")

        # ---- 尝试加载嵌入模型 ----
        if not self._try_load_embedding_model():
            # 嵌入模型不可用，启用回退模式
            self._use_fallback = True
            print(f"[KB] Using keyword fallback search mode (reason: {self._fallback_reason})")
            print(f"[KB] The system is fully functional without the embedding model.")
            return True

        # ---- 构建 FAISS 索引 ----
        try:
            import faiss
            import numpy as np
        except ImportError:
            self._use_fallback = True
            self._fallback_reason = "faiss-cpu not installed"
            print("[KB] faiss-cpu not installed, using fallback mode")
            return True

        try:
            model = self._embedding_model
            texts = [entry["search_text"] for entry in self._entries]

            print(f"[KB] Encoding {len(texts)} entries...")
            embeddings = model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            dim = embeddings.shape[1]
            self._index = faiss.IndexFlatL2(dim)
            self._index.add(embeddings.astype(np.float32))

            print(f"[KB] FAISS index built (dim={dim}, entries={len(self._entries)})")
            self._use_fallback = False

            # 保存
            self._save_index()
            return True

        except Exception as e:
            self._use_fallback = True
            self._fallback_reason = str(e)[:100]
            print(f"[KB] FAISS index build failed: {e}")
            print(f"[KB] Falling back to keyword search mode")
            return True

    def _save_index(self):
        """保存 FAISS 索引和元数据"""
        if self._index is None:
            return
        try:
            import faiss
            faiss.write_index(self._index, str(KB_INDEX_PATH))
            meta = {
                "num_entries": len(self._entries),
                "embedding_model": KB_EMBEDDING_MODEL,
                "dim": self._index.d,
            }
            with open(KB_META_PATH, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"[KB] Index saved: {KB_INDEX_PATH}")
        except Exception as e:
            print(f"[KB] Index save failed: {e}")

    # ==================== 关键词回退检索 ====================

    def _keyword_search(self, query: str, k: int) -> List[Dict]:
        """
        回退检索策略 - 基于关键词和字符串相似度

        打分策略（加权组合）:
        1. 精确词匹配: +3 分/词
        2. 部分词匹配: +1 分/词
        3. 序列相似度: SequenceMatcher 比值

        返回得分最高的 k 个条目
        """
        if not self._entries:
            return []

        query_lower = query.lower()
        query_words = set(re.findall(r"[a-zA-Z]+", query_lower))

        if not query_words:
            return []

        scored_entries = []

        for entry in self._entries:
            search_text = entry["search_text"].lower()
            search_words = set(re.findall(r"[a-zA-Z]+", search_text))

            score = 0.0

            # 1. 精确词匹配加分
            exact_matches = query_words & search_words
            score += len(exact_matches) * 3.0

            # 2. 部分词匹配（子串包含）
            for qw in query_words:
                if len(qw) >= 3:  # 只对长度 >= 3 的词做子串匹配
                    for sw in search_words:
                        if qw in sw or sw in qw:
                            score += 0.5

            # 3. 序列相似度
            seq_ratio = SequenceMatcher(None, query_lower, search_text[:500]).ratio()
            score += seq_ratio * 2.0

            # 4. 词汇表条目：词本身匹配额外加分
            if entry["type"] == "vocab":
                word = entry.get("metadata", {}).get("word", "").lower()
                if word in query_words or any(qw in word or word in qw for qw in query_words if len(qw) >= 3):
                    score += 5.0

            if score > 0:
                scored_entries.append((score, entry))

        # 按分数降序排序
        scored_entries.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, entry in scored_entries[:k]:
            entry_copy = entry.copy()
            # 将得分转换为类似 FAISS 的 score 格式（越小越好）
            entry_copy["score"] = round(1.0 / (1.0 + score), 4)
            results.append(entry_copy)

        return results

    # ==================== 检索接口 ====================

    def search(self, query: str, k: int = None) -> List[Dict]:
        """
        统一检索接口 - 自动选择 FAISS 或回退模式

        Args:
            query: 查询文本
            k: 返回数量，默认 KB_RETRIEVAL_TOP_K

        Returns:
            [{id, type, text, metadata, score}, ...]
        """
        if k is None:
            k = KB_RETRIEVAL_TOP_K

        if not self._entries:
            return []

        # ---- 回退模式 ----
        if self._use_fallback:
            return self._keyword_search(query, k)

        # ---- FAISS 模式 ----
        if self._index is None:
            if not self.build_index():
                # build_index 又失败了，走回退
                self._use_fallback = True
                return self._keyword_search(query, k)

        if self._use_fallback:
            return self._keyword_search(query, k)

        try:
            import numpy as np
            model = self._embedding_model
            query_vec = model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            distances, indices = self._index.search(query_vec, min(k, len(self._entries)))

            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx >= 0 and idx < len(self._entries):
                    entry = self._entries[idx].copy()
                    entry["score"] = round(float(dist), 4)
                    results.append(entry)

            return results

        except Exception as e:
            print(f"[KB] FAISS search failed: {e}, using fallback")
            self._use_fallback = True
            return self._keyword_search(query, k)

    def search_by_keywords(self, keywords: List[str], k: int = None) -> List[Dict]:
        """
        关键词 + 语义混合检索
        """
        if k is None:
            k = KB_RETRIEVAL_TOP_K

        if not keywords:
            return []

        query = " ".join(keywords)
        return self.search(query, k)

    # ==================== 增量更新 ====================

    def add_entry(self, entry_type: str, data: dict) -> int:
        """添加单条知识库条目"""
        entry_id = len(self._entries)

        if entry_type == "vocab":
            text = f"Word: {data.get('word', '')} | Meaning: {data.get('meaning', '')}"
            search_text = f"{data.get('word', '')} {data.get('meaning', '')}"
        elif entry_type == "grammar":
            text = f"Grammar: {data.get('category', '')} - {data.get('pattern', '')}"
            search_text = f"{data.get('category', '')} {data.get('pattern', '')}"
        elif entry_type == "example":
            text = f"Example: {data.get('sentence', '')}"
            search_text = f"{data.get('sentence', '')} {data.get('topic', '')}"
        else:
            text = str(data)
            search_text = str(data)

        entry = {
            "id": entry_id,
            "type": entry_type,
            "text": text,
            "search_text": search_text,
            "metadata": data,
        }

        self._entries.append(entry)
        self._id_to_entry[entry_id] = entry

        # 每 10 条重建索引（仅在 FAISS 模式下）
        if self._index is not None and not self._use_fallback and len(self._entries) % 10 == 0:
            self.build_index(force_rebuild=True)

        return entry_id

    def batch_import(self, json_path: str) -> int:
        """批量导入 JSON 文件到知识库"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            count = 0
            for item in data.get("entries", []):
                entry_type = item.get("type", "unknown")
                self.add_entry(entry_type, item)
                count += 1

            self.build_index(force_rebuild=True)
            print(f"[KB] Batch import complete: {count} entries")
            return count
        except Exception as e:
            print(f"[KB] Batch import failed: {e}")
            return 0

    # ==================== 工具方法 ====================

    def get_entry_by_id(self, entry_id: int) -> Optional[Dict]:
        """根据 ID 获取条目"""
        return self._id_to_entry.get(entry_id)

    def format_results(self, results: List[Dict]) -> str:
        """
        将检索结果格式化为可读文本（用于嵌入 LLM Prompt）
        """
        if not results:
            return "(No relevant knowledge base entries found)"

        lines = ["[Knowledge Base Results]"]
        for i, item in enumerate(results, 1):
            item_type = item.get("type", "")
            type_label = {"vocab": "[VOCAB]", "grammar": "[GRAMMAR]", "example": "[EXAMPLE]"}.get(item_type, "[OTHER]")
            score = item.get("score", 0)
            lines.append(f"\n{type_label} #{i} (relevance: {1/(1+score):.2f})")
            lines.append(item.get("text", ""))

            meta = item.get("metadata", {})
            if meta and item_type == "vocab":
                lines.append(f"   Example: {meta.get('example', 'N/A')}")

        return "\n".join(lines)
