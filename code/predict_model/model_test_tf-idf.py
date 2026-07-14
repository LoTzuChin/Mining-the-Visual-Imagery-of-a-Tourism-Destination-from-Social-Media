import json
import pandas as pd
import numpy as np
from pathlib import Path
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import hstack
import joblib

# ====== 檔案設定 (使用與 train_svm_linearsvc.py 相同的設定) ======
TEXTS_CSV = "data\init_data\prepocessing_data\cleaned_texts_final_ig_twitter.csv"
LABELS_CSV = "data\\topic_output\\output_bertopic\\bertopic_labels.csv"
SPLIT_CSV = "else_file\split.csv"

# ====== 搜索結果輸出目錄 ======
SEARCH_DIR = Path("data\model_output\only")
SEARCH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = SEARCH_DIR / "tfidf_search_results.csv"

def tlog(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ====== 讀取資料 (加上編碼容錯) ======
def load_texts(path: str) -> pd.DataFrame:
    df = None
    for enc in ("utf-8", "utf-8-sig", "cp950"):
        try:
            df = pd.read_csv(path, encoding=enc)
            break
        except Exception:
            pass
    if df is None:
        raise RuntimeError(f"Failed to read {path} using utf-8/utf-8-sig/cp950 encodings.")

    for cand in ["cleaned_text", "text", "desc", "description"]:
        if cand in df.columns:
            text_col = cand
            break
    else:
        raise ValueError(f"{path} must contain one of columns cleaned_text/text/desc/description")

    if "image_name" not in df.columns:
        if "image_path" in df.columns:
            df["image_name"] = df["image_path"].apply(lambda p: Path(str(p)).name)
        else:
            raise ValueError(f"{path} must contain image_name or image_path column.")

    return df[["image_name", text_col]].rename(columns={text_col: "text"})

def main():
    tlog("--- Start TF-IDF Parameter Search (Baseline: LinearSVC) ---")

    # ====== 載入與合併數據 ======
    texts = load_texts(TEXTS_CSV)
    labels = pd.read_csv(LABELS_CSV)[["image_name", "new_topic_num"]]
    split = pd.read_csv(SPLIT_CSV)[["image_name", "split"]]

    df = (
        texts.merge(labels, on="image_name", how="inner")
        .merge(split, on="image_name", how="inner")
    ).dropna(subset=["text", "new_topic_num", "split"])

    df["text"] = df["text"].astype(str)
    df["label"] = df["new_topic_num"].astype(str)

    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()

    X_train_text = df_train["text"].values
    y_train = df_train["label"].values
    X_test_text = df_test["text"].values
    y_test = df_test["label"].values
    
    # 保持 LinearSVC C 值固定，專注搜索特徵
    BASE_C = 3.0
    
    # ====== 定義搜索空間 ======
    search_space = {
        "word_ngram_range": [(1, 2), (1, 3)],  # Word 1-gram + 2-gram 或 1-gram + 2-gram + 3-gram
        "char_ngram_range": [(3, 5), (3, 6)],  # Char 3-gram 到 5-gram 或 6-gram
        "min_df": [2, 5],                       # 最小文檔頻率 (2: 低度過濾, 5: 中度過濾)
    }

    results = []
    total_runs = (len(search_space["word_ngram_range"]) * len(search_space["char_ngram_range"]) * len(search_space["min_df"]))
    run_count = 0

    # ====== 循環搜索 ======
    tlog(f"Total combinations to test: {total_runs}")

    for w_ngram in search_space["word_ngram_range"]:
        for c_ngram in search_space["char_ngram_range"]:
            for min_df_val in search_space["min_df"]:
                run_count += 1
                
                tlog(f"--- Run {run_count}/{total_runs}: W:{w_ngram}, C:{c_ngram}, MinDF:{min_df_val} ---")
                
                # 1. 配置 TF-IDF
                word_vect = TfidfVectorizer(
                    analyzer="word", ngram_range=w_ngram, min_df=min_df_val, 
                    max_df=0.9, strip_accents="unicode", dtype=np.float32
                )
                char_vect = TfidfVectorizer(
                    analyzer="char", ngram_range=c_ngram, min_df=min_df_val, 
                    max_df=0.95, dtype=np.float32
                )
                
                # 2. 訓練/轉換特徵
                Xw_tr = word_vect.fit_transform(X_train_text)
                Xc_tr = char_vect.fit_transform(X_train_text)
                X_train = hstack([Xw_tr, Xc_tr])
                
                Xw_te = word_vect.transform(X_test_text)
                Xc_te = char_vect.transform(X_test_text)
                X_test = hstack([Xw_te, Xc_te])

                n_features = X_train.shape[1]
                tlog(f"Feature Dim: {n_features}")

                # 3. 訓練 LinearSVC (固定 C=3.0)
                clf = LinearSVC(C=BASE_C, class_weight="balanced", random_state=42)
                clf.fit(X_train, y_train)

                # 4. 評估
                y_pred = clf.predict(X_test)
                p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="macro", zero_division=0
                )
                p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="weighted", zero_division=0
                )

                # 5. 儲存結果
                results.append({
                    "word_ngram": str(w_ngram),
                    "char_ngram": str(c_ngram),
                    "min_df": min_df_val,
                    "C_val": BASE_C,
                    "num_features": n_features,
                    "Macro_F1": f1_macro,
                    "Weighted_F1": f1_weight,
                })
                tlog(f"Result: Macro F1={f1_macro:.4f}, Weighted F1={f1_weight:.4f}")

    # ====== 整理並輸出結果 ======
    results_df = pd.DataFrame(results)
    results_df.sort_values(by="Weighted_F1", ascending=False, inplace=True)
    results_df.to_csv(RESULTS_CSV, index=False)

    best_run = results_df.iloc[0]
    tlog("\n=======================================================")
    tlog("[SEARCH COMPLETE]")
    tlog(f"Best Configuration (Weighted F1: {best_run['Weighted_F1']:.4f}):")
    tlog(f"Word Ngram: {best_run['word_ngram']}, Char Ngram: {best_run['char_ngram']}, Min DF: {best_run['min_df']}")
    tlog(f"Full results saved to: {RESULTS_CSV}")
    tlog("=======================================================")

if __name__ == "__main__":
    main()