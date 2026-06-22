# -*- coding: utf-8 -*-
"""
=============================================================================
知识库构建脚本 - 一次性运行
=============================================================================
功能: 加载 JSON 数据文件 → 构建 FAISS 向量索引 → 保存到磁盘

运行方法:
    cd "English OCR"
    python knowledge/build_kb.py

说明:
  - 首次运行会自动下载 sentence-transformers 模型 (~80MB)
  - 生成的索引文件保存在 knowledge/ 目录下
  - 如果 JSON 文件有更新，重新运行此脚本即可
=============================================================================
"""

import sys
import os
import io

# 强制 UTF-8 输出（解决 Windows GBK 编码问题）
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 确保项目根目录在路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.knowledge_base import KnowledgeBase


def main():
    print("=" * 60)
    print("📚 English OCR 知识库构建工具")
    print("=" * 60)

    # 1. 初始化
    kb = KnowledgeBase()

    # 2. 加载数据
    print("\n[1/3] 加载 JSON 数据文件...")
    count = kb.load_data()
    if count == 0:
        print("❌ 未找到任何数据文件，请检查 knowledge/ 目录")
        return 1

    # 3. 构建索引
    print("\n[2/3] 构建 FAISS 向量索引...")
    success = kb.build_index(force_rebuild=True)
    if not success:
        print("❌ 索引构建失败")
        return 1

    # 4. 验证
    print("\n[3/3] 验证检索功能...")
    test_queries = [
        "sustainable development",
        "artificial intelligence",
        "grammar error subject verb agreement",
        "The quick brown fox",
    ]

    for query in test_queries:
        results = kb.search(query, k=3)
        print(f"\n  查询: '{query}'")
        if results:
            for r in results:
                print(f"    → [{r['type']}] (score={r['score']:.4f}) {r['text'][:80]}...")
        else:
            print("    → (无结果)")

    print("\n" + "=" * 60)
    print(f"Knowledge base built: {count} entries")
    print(f"   Search mode: {kb.search_mode}")
    if kb.search_mode == "faiss":
        print(f"   Index file: knowledge/faiss_index.bin")
        print(f"   Metadata: knowledge/index_metadata.json")
    else:
        print(f"   Using keyword matching (no network required)")
        print(f"   To enable FAISS mode, download the model:")
        print(f"     set HF_ENDPOINT=https://hf-mirror.com")
        print(f"     Then re-run this script")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
