# search_tfidf_multilabel.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import precision_recall_fscore_support
from scipy.sparse import hstack

# ====== 檔案設定 (請確認檔名是否為多標籤版本) ======
TEXTS_CSV   = "data\init_data\prepocessing_data\cleaned_texts_final_ig_twitter.csv"  # 需確認路徑
LABELS_CSV  = "data\\topic_output\output_bertopic\\bertopic_labels_multilabel.csv"       # [新] 多標籤結果
SPLIT_CSV   = "else_file\split_multilabel.csv"                 # [新] 多標籤 split

# ====== 搜索結果輸出目錄 ======
SEARCH_DIR = Path("data\\model_output\\new_multi_output")
SEARCH_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = SEARCH_DIR / "tfidf_multilabel_search_results.csv"

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

    # 找文字欄位
    text_col = next((c for c in ["cleaned_text", "text", "desc", "description"] if c in df.columns), None)
    if not text_col:
        raise ValueError(f"{path} must contain one of columns cleaned_text/text/desc/description")

    # 找 image_name
    if "image_name" not in df.columns:
        if "image_path" in df.columns:
            df["image_name"] = df["image_path"].apply(lambda p: Path(str(p)).name)
        else:
            raise ValueError(f"{path} must contain image_name or image_path column.")

    return df[["image_name", text_col]].rename(columns={text_col: "text"})

def main():
    tlog("--- Start TF-IDF Multi-Label Parameter Search ---")

    # ====== 1. 載入與合併數據 ======
    texts = load_texts(TEXTS_CSV)
    # 讀取多標籤欄位 (topic_ids)
    if not Path(LABELS_CSV).exists():
        raise FileNotFoundError(f"找不到 {LABELS_CSV}，請確認是否已執行 topic_data_collation_multilabel.py")
        
    labels = pd.read_csv(LABELS_CSV)[["image_name", "topic_ids"]]
    split = pd.read_csv(SPLIT_CSV)[["image_name", "split"]]

    df = (
        texts.merge(labels, on="image_name", how="inner")
        .merge(split, on="image_name", how="inner")
    ).dropna(subset=["text", "topic_ids", "split"])

    df["text"] = df["text"].astype(str)
    
    # 解析 "1,5" -> ['1', '5']
    df["label_list"] = df["topic_ids"].astype(str).apply(
        lambda x: [s.strip() for s in x.split(',') if s.strip()]
    )

    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()

    X_train_text = df_train["text"].values
    X_test_text  = df_test["text"].values
    
    # ====== 2. 多標籤編碼 (MultiLabelBinarizer) ======
    tlog("Encoding Labels...")
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(df_train["label_list"]) # 0/1 Matrix
    y_test  = mlb.transform(df_test["label_list"])
    
    tlog(f"Classes: {len(mlb.classes_)}")

    # 保持 LinearSVC C 值固定 (根據之前的分析，5.0 效果不錯)
    BASE_C = 5.0
    
    # ====== 3. 定義搜索空間 ======
    search_space = {
        "word_ngram_range": [(1, 2), (1, 3)],  
        "char_ngram_range": [(3, 5), (3, 6)],  
        "min_df": [2, 5],                       
    }

    results = []
    total_runs = (len(search_space["word_ngram_range"]) * len(search_space["char_ngram_range"]) * len(search_space["min_df"]))
    run_count = 0

    # ====== 4. 循環搜索 ======
    tlog(f"Total combinations to test: {total_runs}")

    for w_ngram in search_space["word_ngram_range"]:
        for c_ngram in search_space["char_ngram_range"]:
            for min_df_val in search_space["min_df"]:
                run_count += 1
                
                tlog(f"--- Run {run_count}/{total_runs}: W:{w_ngram}, C:{c_ngram}, MinDF:{min_df_val} ---")
                
                # A. 配置 TF-IDF
                word_vect = TfidfVectorizer(
                    analyzer="word", ngram_range=w_ngram, min_df=min_df_val, 
                    max_df=0.9, strip_accents="unicode", dtype=np.float32
                )
                char_vect = TfidfVectorizer(
                    analyzer="char", ngram_range=c_ngram, min_df=min_df_val, 
                    max_df=0.95, dtype=np.float32
                )
                
                # B. 訓練/轉換特徵
                Xw_tr = word_vect.fit_transform(X_train_text)
                Xc_tr = char_vect.fit_transform(X_train_text)
                X_train = hstack([Xw_tr, Xc_tr])
                
                Xw_te = word_vect.transform(X_test_text)
                Xc_te = char_vect.transform(X_test_text)
                X_test = hstack([Xw_te, Xc_te])

                n_features = X_train.shape[1]
                tlog(f"Feature Dim: {n_features}")

                # C. 訓練 OneVsRest + LinearSVC
                # n_jobs=-1 可加速多標籤訓練
                base_clf = LinearSVC(C=BASE_C, class_weight="balanced", random_state=42, dual="auto")
                clf = OneVsRestClassifier(base_clf, n_jobs=-1) 
                
                clf.fit(X_train, y_train)

                # D. 評估
                y_pred = clf.predict(X_test)
                
                # Samples Average (最重要的多標籤指標)
                _, _, f1_samples, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="samples", zero_division=0
                )
                
                # Micro Average (全域指標)
                _, _, f1_micro, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="micro", zero_division=0
                )

                # Macro Average (類別平均)
                _, _, f1_macro, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="macro", zero_division=0
                )

                # E. 儲存結果
                results.append({
                    "word_ngram": str(w_ngram),
                    "char_ngram": str(c_ngram),
                    "min_df": min_df_val,
                    "C_val": BASE_C,
                    "num_features": n_features,
                    "Samples_F1": f1_samples, # 推薦排序依據
                    "Micro_F1": f1_micro,
                    "Macro_F1": f1_macro,
                })
                tlog(f"Result: Samples F1={f1_samples:.4f} | Micro F1={f1_micro:.4f}")

    # ====== 5. 整理並輸出結果 ======
    results_df = pd.DataFrame(results)
    # 根據 Samples F1 排序 (因為是多標籤任務)
    results_df.sort_values(by="Samples_F1", ascending=False, inplace=True)
    results_df.to_csv(RESULTS_CSV, index=False)

    best_run = results_df.iloc[0]
    tlog("\n=======================================================")
    tlog("[SEARCH COMPLETE]")
    tlog(f"Best Configuration (Samples F1: {best_run['Samples_F1']:.4f}):")
    tlog(f"Word Ngram: {best_run['word_ngram']}, Char Ngram: {best_run['char_ngram']}, Min DF: {best_run['min_df']}")
    tlog(f"Full results saved to: {RESULTS_CSV}")
    tlog("=======================================================")

if __name__ == "__main__":
    main()