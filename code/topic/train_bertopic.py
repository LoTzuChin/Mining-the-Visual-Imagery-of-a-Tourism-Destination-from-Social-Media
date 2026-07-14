# 若在本機執行遇到 PowerShell 執行政策限制，可改在 Colab 執行或調整權限

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import numpy as np
import re

def _auto_language_guess(sample: str) -> str:
    # 依據字元是否多為 ASCII 以及是否包含英文字母，進行極簡語言判斷
    if not sample:
        return "multilingual"
    ascii_ratio = sum(1 for ch in sample if ord(ch) < 128) / max(1, len(sample))
    if ascii_ratio > 0.95 and re.search(r"[A-Za-z]", sample):
        return "english"
    return "multilingual"


def train(df):
    # 從資料框抓出清理後的文本欄位，濾除短句並移除重複內容
    descriptions = df['cleaned_text'].astype(str).str.strip()
    descriptions = descriptions[descriptions.str.len() >= 5].drop_duplicates()
    descriptions = descriptions.tolist()

    # 取前 50 筆樣本拼成字串，推測主要語言以決定 BERTopic 的語系設定
    sample_text = " ".join(descriptions[:50])
    lang = _auto_language_guess(sample_text)
    print(f"[INFO] Detected language: {lang} | docs: {len(descriptions)}")

    # 建立句向量模型供 BERTopic 使用，採用多語版 MiniLM 以支援多種語言
    print("[INFO] Loading embedding model...")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # 以語言設定與嵌入模型初始化 BERTopic，並訓練取得主題與對應文件索引
    print("[INFO] Fitting BERTopic...")
    topic_model = BERTopic(embedding_model=embedding_model, language=lang, calculate_probabilities=False, verbose=True)
    topics, probs = topic_model.fit_transform(descriptions)
    return topic_model, np.array(topics)


def writeCsv(topic_model, topics, filename="topic.csv"):
    # 組合每個主題的關鍵字與文件數量，輸出為報表
    topic_keywords = []
    for topic_num in topic_model.get_topics().keys():
        if topic_num == -1:
            # 主題 -1 代表雜訊群組，通常不納入報告
            continue
        keywords_with_weights = topic_model.get_topic(topic_num) or []
        keywords_str = "; ".join([f"{word}:{weight:.4f}" for word, weight in keywords_with_weights[:10]])
        doc_count = int((topics == topic_num).sum())
        topic_keywords.append({
            "Topic": topic_num,
            "Document_Count": doc_count,
            "Keywords_Weights": keywords_str
        })
    df_topic = pd.DataFrame(topic_keywords).sort_values("Document_Count", ascending=False)
    df_topic.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved {filename}")


if __name__ == "__main__":
    print("[INFO] Reading CSV...")
    # 若原始資料為 Big5/CP950，可調整為 encoding="utf-8-sig" 並加入 errors="ignore"
    df = pd.read_csv('data\init_data\prepocessing_data\cleaned_texts_final_ig_twitter.csv')
    topic_model, topics = train(df)
    writeCsv(topic_model, topics, "topic.csv")
    topic_model.save("bertopic_finetuned_model")
    print("[DONE] topic.csv 已儲存，模型存放於 bertopic_finetuned_model/")
