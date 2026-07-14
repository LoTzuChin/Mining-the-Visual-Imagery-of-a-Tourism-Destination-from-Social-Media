import argparse
import os
import sys
import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel, CoherenceModel
from collections import Counter, defaultdict

def read_csv_flex(path: str) -> pd.DataFrame:
    # 嘗試多種常見編碼，避免 Windows/CJK 編碼問題
    encodings = ["utf-8-sig", "utf-8", "cp950", "big5"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

def simple_tokenize(text: str):
    # 假設 cleaned_text 已完成前處理並以空白分詞
    if isinstance(text, str):
        return [t for t in text.strip().split() if t]
    return []

def build_corpus(texts_tokens, no_below=5, no_above=0.5, keep_n=100000):
    dictionary = Dictionary(texts_tokens)
    # 過濾極端詞：少見詞/過於常見詞
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    dictionary.compactify()
    corpus = [dictionary.doc2bow(doc) for doc in texts_tokens]
    return dictionary, corpus

def train_lda(dictionary, corpus, num_topics, passes=10, iterations=400, random_state=42):
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=random_state,
        chunksize=2000,
        passes=passes,
        iterations=iterations,
        alpha='symmetric',
        eta='auto',
        eval_every=None
    )
    return lda

def compute_coherence(lda_model, texts_tokens, dictionary, measure="c_v"):
    cm = CoherenceModel(model=lda_model, texts=texts_tokens, dictionary=dictionary, coherence=measure)
    return cm.get_coherence()

def choose_best_k(dictionary, corpus, texts_tokens, k_values, passes, iterations, random_state, coherence_measure="c_v"):
    results = []
    best = {"k": None, "coh": -np.inf, "model": None}
    for k in k_values:
        print(f"[INFO] Training LDA for K={k} ...", flush=True)
        lda = train_lda(dictionary, corpus, k, passes=passes, iterations=iterations, random_state=random_state)
        coh = compute_coherence(lda, texts_tokens, dictionary, measure=coherence_measure)
        print(f"[INFO] K={k}, {coherence_measure}={coh:.5f}", flush=True)
        results.append({"K": k, "coherence": coh})
        if coh > best["coh"]:
            best = {"k": k, "coh": coh, "model": lda}
    return results, best

def topic_doc_coverage(lda_model, corpus):
    """回傳每個主題覆蓋的文件數（以每篇文件的最大機率主題作為歸屬）"""
    topic_counts = Counter()
    for bow in corpus:
        # 取得該文件的主題分佈（僅回傳非零主題）
        dist = lda_model.get_document_topics(bow, minimum_probability=0.0)
        # dist 是 (topic_id, prob) 列表
        if dist:
            top_topic = max(dist, key=lambda x: x[1])[0]
            topic_counts[top_topic] += 1
    return topic_counts

def topic_top_terms(lda_model, topn=10):
    """回傳 {topic_id: [(term, weight), ...]}"""
    id2term = lda_model.id2word
    topic_terms = {}
    for t in range(lda_model.num_topics):
        topic_terms[t] = [(id2term[id_], float(w)) for id_, w in lda_model.get_topic_terms(t, topn=topn)]
    return topic_terms

def save_coherence_table(results, out_dir):
    df = pd.DataFrame(results).sort_values("K")
    path = os.path.join(out_dir, "coherence_results.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved coherence table -> {path}")

def save_topic_summary_csv(lda_model, corpus, out_dir, topn=10):
    cover = topic_doc_coverage(lda_model, corpus)
    top_terms = topic_top_terms(lda_model, topn=topn)

    rows = []
    for t in range(lda_model.num_topics):
        doc_count = int(cover.get(t, 0))
        pairs = top_terms.get(t, [])
        # 以「詞(權重)」串接，權重保留 4 位小數
        terms_joined = ", ".join([f"{w}({wt:.4f})" for w, wt in pairs])
        rows.append({
            "topic_id": t,
            "doc_count": doc_count,
            "top10_terms_with_weights": terms_joined
        })

    df = pd.DataFrame(rows).sort_values("doc_count", ascending=False)
    path = os.path.join(out_dir, "topic_summary.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved topic summary -> {path}")

def save_best_model(lda_model, out_dir, k, coh):
    path = os.path.join(out_dir, f"best_lda_K{k}_coh{coh:.4f}.gensim")
    lda_model.save(path)
    print(f"[INFO] Saved best model -> {path}")



# ========== 主程式入口 ==========
if __name__ == "__main__":
    # 這裡直接寫死路徑與參數
    input_csv = "D:\Social_Media_Analysis\prepocessing_data\cleaned_texts.csv"
    text_col = "cleaned_text"
    output_dir = "D:\Social_Media_Analysis\lda_out"
    os.makedirs(output_dir, exist_ok=True)

    k_min, k_max, k_step = 4, 20, 2
    passes, iterations, random_state = 10, 400, 42
    coherence_measure = "c_v"

    # 讀取 CSV
    print("[INFO] Reading CSV ...")
    df = read_csv_flex(input_csv)
    texts_raw = df[text_col].astype(str).fillna("").tolist()
    print(f"[INFO] Documents: {len(texts_raw)}")

    # Tokenize
    texts_tokens = [simple_tokenize(t) for t in texts_raw]

    # Corpus
    dictionary, corpus = build_corpus(texts_tokens, no_below=5, no_above=0.5, keep_n=100000)

    # 嘗試不同 K
    results = []
    best_k, best_coh, best_model = None, -np.inf, None
    for k in range(k_min, k_max+1, k_step):
        print(f"[INFO] Training K={k} ...")
        lda = train_lda(dictionary, corpus, k, passes=passes, iterations=iterations, random_state=random_state)
        coh = compute_coherence(lda, texts_tokens, dictionary, measure=coherence_measure)
        results.append({"K": k, "coherence": coh})
        print(f"[INFO] K={k}, coherence={coh:.4f}")
        if coh > best_coh:
            best_k, best_coh, best_model = k, coh, lda

    # 輸出 coherence 結果
    pd.DataFrame(results).to_csv(os.path.join(output_dir, "coherence_results.csv"), index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved coherence_results.csv")

    # 儲存最佳模型
    best_model.save(os.path.join(output_dir, f"best_lda_K{best_k}_coh{best_coh:.4f}.gensim"))
    print(f"[INFO] Best model K={best_k}, coh={best_coh:.4f}")

    # 輸出主題摘要
    cover = topic_doc_coverage(best_model, corpus)
    top_terms = topic_top_terms(best_model, topn=10)
    rows = []
    for t in range(best_model.num_topics):
        doc_count = int(cover.get(t, 0))
        pairs = top_terms[t]
        terms_joined = ", ".join([f"{w}({wt:.4f})" for w, wt in pairs])
        rows.append({"topic_id": t, "doc_count": doc_count, "top10_terms_with_weights": terms_joined})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, "topic_summary.csv"), index=False, encoding="utf-8-sig")
    print("[INFO] Saved topic_summary.csv")
