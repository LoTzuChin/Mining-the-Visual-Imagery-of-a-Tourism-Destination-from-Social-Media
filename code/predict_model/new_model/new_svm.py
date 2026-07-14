# train_svm_multilabel.py
import json
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    hamming_loss
)
from scipy.sparse import hstack

# ====== 檔案設定 ======
# 假設你已經跑完 topic_data_collation_multilabel.py 並產生了以下檔案
TEXTS_CSV   = "data\init_data\prepocessing_data\cleaned_texts_final_ig_twitter.csv"  # 需確認路徑
LABELS_CSV  = "data\\topic_output\\output_bertopic\\bertopic_labels_multilabel.csv"       # [新] 多標籤結果
SPLIT_CSV   = "else_file\split_multilabel.csv"     

# ** 外部控制變數 **
CURRENT_EXP_ID = "SVM" 

# ====== Output artifacts ======
SVM_DIR = Path("data\\model_output\\new_multi_output") / CURRENT_EXP_ID
SVM_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV      = SVM_DIR / "svm_predictions.csv"
REPORT_TXT    = SVM_DIR / "svm_report.txt"
MODEL_JOBLIB  = SVM_DIR / "svm_multilabel.joblib"
MLB_JOBLIB    = SVM_DIR / "mlb.joblib"            # [新] 儲存標籤編碼器
VECT_WORD     = SVM_DIR / "tfidf_word.joblib"
VECT_CHAR     = SVM_DIR / "tfidf_char.joblib"


def load_texts(path: str) -> pd.DataFrame:
    """讀取文字檔並統一欄位名稱"""
    df = pd.read_csv(path)
    text_col_candidates = ["cleaned_text", "text", "desc", "description"]
    text_col = next((c for c in text_col_candidates if c in df.columns), None)
    
    if text_col is None:
        raise ValueError(f"{path} 必須包含其中一個文字欄位: {text_col_candidates}")
    
    if "image_name" not in df.columns:
        if "image_path" in df.columns:
            df["image_name"] = df["image_path"].apply(lambda p: Path(str(p)).name)
        else:
            raise ValueError(f"{path} 需包含 image_name")
            
    return df[["image_name", text_col]].rename(columns={text_col: "text"})

def to_serializable(obj):
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
    # labels 中應包含 "image_name" 和 "topic_ids" (e.g., "1,5")
    df = (
        texts.merge(labels[["image_name", "topic_ids"]], on="image_name", how="inner")
             .merge(split[["image_name", "split"]], on="image_name", how="inner")
    )
    
    # 過濾缺值
    df = df.dropna(subset=["text", "topic_ids", "split"])
    df["text"] = df["text"].astype(str)

    # 2. 解析多標籤字串 "1,5" -> ['1', '5']
    # 注意：這裡轉成字串列表，因為 MultiLabelBinarizer 需要
    df["label_list"] = df["topic_ids"].astype(str).apply(
        lambda x: [s.strip() for s in x.split(',') if s.strip()]
    )

    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()
    
    print(f"[INFO] Train size: {len(df_train)}, Test size: {len(df_test)}")

    X_train_text = df_train["text"].values
    X_test_text  = df_test["text"].values

    # 3. 標籤編碼 (MultiLabelBinarizer)
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(df_train["label_list"])
    y_test  = mlb.transform(df_test["label_list"])
    
    print(f"[INFO] Classes: {mlb.classes_}")

    # 4. TF-IDF 特徵
    print(f"[INFO] Extracting features...")
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

    # 5. 建模：OneVsRest + LinearSVC
    # class_weight="balanced" 在 OVR 中會對每個二元分類器個別平衡
    print(f"[INFO] Training Model...")
    base_clf = LinearSVC(C=5.0, class_weight="balanced", random_state=42, max_iter=2000)
    clf = OneVsRestClassifier(base_clf, n_jobs=-1)
    
    clf.fit(X_train, y_train)

    # 6. 預測與評估
    print(f"[INFO] Evaluating...")
    y_pred = clf.predict(X_test)

    # --- 各種多標籤指標 ---
    # A. Exact Match Accuracy (全對才算對)
    acc = accuracy_score(y_test, y_pred)
    
    # B. Hamming Loss (錯標籤的比例，越低越好)
    h_loss = hamming_loss(y_test, y_pred)

    # C. Samples Average (針對每個樣本看它對了幾成，再平均) -> 最直覺
    p_samp, r_samp, f1_samp, _ = precision_recall_fscore_support(
        y_test, y_pred, average="samples", zero_division=0
    )

    # D. Micro Average (全局混淆矩陣)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="micro", zero_division=0
    )
    
    # E. Macro Average (各類別獨立算再平均)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    # 詳細報告
    report = classification_report(
        y_test, y_pred, target_names=mlb.classes_, zero_division=0
    )

    # 儲存報告
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"[Multi-Label Evaluation]\n")
        f.write(f"Subset Accuracy (Exact Match) : {acc:.4f}\n")
        f.write(f"Hamming Loss (lower is better): {h_loss:.4f}\n\n")
        
        f.write(f"[Samples Average] (Per-instance performance)\n")
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
            "linear_svc": base_clf.get_params(),
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
    joblib.dump(clf, MODEL_JOBLIB)
    joblib.dump(mlb, MLB_JOBLIB)   # 重要：之後預測新資料需要這個來解碼
    joblib.dump(word_vect, VECT_WORD)
    joblib.dump(char_vect, VECT_CHAR)

    print("[OK] Multi-Label Training Completed.")
    print(f"Exact Acc={acc:.4f} | Samples-F1={f1_samp:.4f} | Micro-F1={f1_micro:.4f}")
    print(f"- Report: {REPORT_TXT}")
    print(f"- Predictions: {PRED_CSV}")

if __name__ == "__main__":
    main()