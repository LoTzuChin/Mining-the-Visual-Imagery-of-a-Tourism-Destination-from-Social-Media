# train_rf_multilabel.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# Sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, hamming_loss
)
from scipy.sparse import hstack

# ====== 檔案路徑設定 ======
# 假設你已經跑完 topic_data_collation_multilabel.py 並產生了以下檔案
TEXTS_CSV   = "data\init_data\prepocessing_data\cleaned_texts_final_ig_twitter.csv"  # 需確認路徑
LABELS_CSV  = "data\\topic_output\\output_bertopic\\bertopic_labels_multilabel.csv"       # [新] 多標籤結果
SPLIT_CSV   = "else_file\split_multilabel.csv"     

# ** 外部控制變數 **
CURRENT_EXP_ID = "RF"

# ====== 輸出目錄設定 ======
RF_DIR = Path("data\\model_output\\new_multi_output") / CURRENT_EXP_ID
RF_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV      = RF_DIR / "rf_predictions.csv"
REPORT_TXT    = RF_DIR / "rf_report.txt"
MODEL_JOBLIB  = RF_DIR / "rf_multilabel.joblib"
MLB_JOBLIB    = RF_DIR / "mlb.joblib"            # [重要] 儲存標籤編碼器
VECT_WORD     = RF_DIR / "tfidf_word.joblib"
VECT_CHAR     = RF_DIR / "tfidf_char.joblib"


def load_texts(path: str) -> pd.DataFrame:
    """讀取文字檔並統一欄位名稱"""
    df = pd.read_csv(path)
    # 嘗試自動找文字欄位
    text_col_candidates = ["cleaned_text", "text", "desc", "description"]
    text_col = next((c for c in text_col_candidates if c in df.columns), None)
    
    if text_col is None:
        raise ValueError(f"{path} 必須包含其中一個文字欄位: {text_col_candidates}")
    
    if "image_name" not in df.columns:
        if "image_path" in df.columns:
            df["image_name"] = df["image_path"].apply(lambda p: Path(str(p)).name)
        else:
            raise ValueError(f"{path} 需包含 image_name 欄位")
            
    return df[["image_name", text_col]].rename(columns={text_col: "text"})

def to_serializable(obj):
    """JSON 序列化輔助函數"""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    return str(obj)

def main():
    print(f"[INFO] Loading data...")
    texts  = load_texts(TEXTS_CSV)
    labels = pd.read_csv(LABELS_CSV)
    split  = pd.read_csv(SPLIT_CSV)

    # 1. 資料合併
    # labels 必須包含 "topic_ids" 欄位 (例如字串 "1,5")
    if "topic_ids" not in labels.columns:
        raise ValueError(f"{LABELS_CSV} 必須包含 'topic_ids' 欄位。請先執行 topic_data_collation_multilabel.py")

    df = (
        texts.merge(labels[["image_name", "topic_ids"]], on="image_name", how="inner")
             .merge(split[["image_name", "split"]], on="image_name", how="inner")
    )
    
    # 過濾缺值
    df = df.dropna(subset=["text", "topic_ids", "split"])
    df["text"] = df["text"].astype(str)

    # 2. 解析多標籤字串 "1,5" -> ['1', '5'] (List of strings)
    df["label_list"] = df["topic_ids"].astype(str).apply(
        lambda x: [s.strip() for s in x.split(',') if s.strip()]
    )

    # 切分 Train / Test
    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()
    
    print(f"[INFO] Train size: {len(df_train)}, Test size: {len(df_test)}")

    X_train_text = df_train["text"].values
    X_test_text  = df_test["text"].values

    # 3. 標籤編碼 (MultiLabelBinarizer)
    # 隨機森林接收 0/1 矩陣作為 y
    print(f"[INFO] Encoding labels...")
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(df_train["label_list"])
    y_test  = mlb.transform(df_test["label_list"])
    
    print(f"[INFO] Classes detected: {mlb.classes_}")

    # 4. TF-IDF 特徵提取
    print(f"[INFO] Extracting TF-IDF features...")
    word_vect = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 3), min_df=2, max_df=0.9, dtype=np.float32
    )
    char_vect = TfidfVectorizer(
        analyzer="char", ngram_range=(3, 6), min_df=2, max_df=0.95, dtype=np.float32
    )

    Xw_tr = word_vect.fit_transform(X_train_text)
    Xc_tr = char_vect.fit_transform(X_train_text)
    X_train = hstack([Xw_tr, Xc_tr])

    Xw_te = word_vect.transform(X_test_text)
    Xc_te = char_vect.transform(X_test_text)
    X_test = hstack([Xw_te, Xc_te])

    # 5. 建模：Random Forest (原生支援多標籤)
    print(f"[INFO] Training Random Forest...")
    
    # 這裡不需要 OneVsRestClassifier，RF 本身就可以吃二維的 y
    clf = RandomForestClassifier(
        n_estimators=200,    # 樹的數量
        max_depth=None,      # 深度不限
        min_samples_split=5,
        class_weight="balanced", # 針對不平衡樣本做加權
        n_jobs=-1,           # 使用所有 CPU 核心
        random_state=42,
        verbose=1
    )
    
    clf.fit(X_train, y_train)

    # 6. 預測與評估
    print(f"[INFO] Evaluating...")
    y_pred = clf.predict(X_test)  # 回傳 0/1 矩陣

    # --- 計算多標籤指標 ---
    
    # A. Exact Match Accuracy (全對才算對)
    acc = accuracy_score(y_test, y_pred)
    
    # B. Hamming Loss (錯誤標籤的比例，越低越好)
    h_loss = hamming_loss(y_test, y_pred)

    # C. Samples Average (推薦：針對每個樣本算 F1 再平均)
    p_samp, r_samp, f1_samp, _ = precision_recall_fscore_support(
        y_test, y_pred, average="samples", zero_division=0
    )

    # D. Micro Average (全局混淆矩陣)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="micro", zero_division=0
    )
    
    # 詳細報告
    report = classification_report(
        y_test, y_pred, target_names=mlb.classes_, zero_division=0
    )

    # 儲存報告
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"[Random Forest Multi-Label Evaluation]\n")
        f.write(f"Subset Accuracy (Exact Match) : {acc:.4f}\n")
        f.write(f"Hamming Loss (lower is better): {h_loss:.4f}\n\n")
        
        f.write(f"[Samples Average] (Per-instance performance) ★Recommended\n")
        f.write(f"Precision : {p_samp:.4f}\n")
        f.write(f"Recall    : {r_samp:.4f}\n")
        f.write(f"F1 Score  : {f1_samp:.4f}\n\n")

        f.write(f"[Micro Average] (Global performance)\n")
        f.write(f"Precision : {p_micro:.4f}\n")
        f.write(f"Recall    : {r_micro:.4f}\n")
        f.write(f"F1 Score  : {f1_micro:.4f}\n\n")
        
        f.write("[Per-class Report]\n")
        f.write(report)
        
        f.write("\n[Parameters]\n")
        param_snapshot = {
            "rf_params": clf.get_params(),
            "tfidf_word": word_vect.get_params(),
            "tfidf_char": char_vect.get_params()
        }
        f.write(json.dumps(to_serializable(param_snapshot), ensure_ascii=False, indent=2))

    # 7. 儲存預測結果 (轉回人類可讀標籤)
    # y_pred 是 0/1 矩陣，轉回 [('1', '5'), ('2',)]
    pred_labels_tuples = mlb.inverse_transform(y_pred)
    true_labels_tuples = mlb.inverse_transform(y_test)
    
    out_pred = pd.DataFrame({
        "image_name": df_test["image_name"],
        "true_labels": [",".join(x) for x in true_labels_tuples],
        "pred_labels": [",".join(x) for x in pred_labels_tuples]
    })
    out_pred.to_csv(PRED_CSV, index=False)

    # 8. 儲存模型
    print(f"[INFO] Saving artifacts to {RF_DIR}...")
    joblib.dump(clf, MODEL_JOBLIB)
    joblib.dump(mlb, MLB_JOBLIB)   # 重要：預測新資料需要它
    joblib.dump(word_vect, VECT_WORD)
    joblib.dump(char_vect, VECT_CHAR)

    print("[OK] Random Forest Multi-Label Training Completed.")
    print(f"Samples-F1={f1_samp:.4f} | Micro-F1={f1_micro:.4f}")
    print(f"- Report: {REPORT_TXT}")
    print(f"- Predictions: {PRED_CSV}")

if __name__ == "__main__":
    main()