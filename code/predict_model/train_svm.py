# train_svm_linearsvc.py
import json
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from scipy.sparse import hstack
import joblib

# ====== 檔案設定 ======
TEXTS_CSV = "data\init_data\prepocessing_data\cleaned_texts_final_ig_twitter.csv"
LABELS_CSV = "data\\topic_output\\output_bertopic\\bertopic_labels.csv"
SPLIT_CSV = "else_file\split.csv"

# ** 外部控制變數 **
CURRENT_EXP_ID = "SVM" # <-- 每次運行時修改或從命令行傳入

# ====== Output artifacts ======
# 根據 Exp ID 創建新的輸出目錄
SVM_DIR = Path("data\model_output\only_output") / CURRENT_EXP_ID
SVM_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV      = SVM_DIR / "svm_predictions.csv"
REPORT_TXT    = SVM_DIR / "svm_report.txt"
CM_CSV        = SVM_DIR / "svm_confusion_matrix.csv"
MODEL_JOBLIB  = SVM_DIR / "svm_linearsvc.joblib"
VECT_WORD     = SVM_DIR / "tfidf_word.joblib"
VECT_CHAR     = SVM_DIR / "tfidf_char.joblib"


# ====== 讀取資料 ======
def load_texts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 嘗試自動找文字欄位
    text_col_candidates = ["cleaned_text", "text", "desc", "description"]
    text_col = None
    for c in text_col_candidates:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(
            f"{path} 必須包含其中一個文字欄位: {text_col_candidates}"
        )
    if "image_name" not in df.columns:
        # 常見替代：image_path -> 取檔名
        if "image_path" in df.columns:
            df["image_name"] = df["image_path"].apply(lambda p: Path(str(p)).name)
        else:
            raise ValueError(f"{path} 需包含 image_name 欄位（或提供 image_path 以轉成檔名）")
    return df[["image_name", text_col]].rename(columns={text_col: "text"})


def to_serializable(obj):
    """Recursively convert objects into JSON-serializable forms."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    return str(obj)

def main():
    texts  = load_texts(TEXTS_CSV)
    labels = pd.read_csv(LABELS_CSV)
    split  = pd.read_csv(SPLIT_CSV)

    # 檢查標籤欄位
    for col in ["image_name", "new_topic_num"]:
        if col not in labels.columns:
            raise ValueError(f"{LABELS_CSV} 缺少欄位: {col}")

    # 合併
    df = (
        texts.merge(labels[["image_name", "new_topic_num"]], on="image_name", how="inner")
             .merge(split[["image_name", "split"]], on="image_name", how="inner")
    ).dropna(subset=["text", "new_topic_num", "split"])

    # 文字轉字串
    df["text"] = df["text"].astype(str)
    # 標籤轉字串可避免某些整數編碼陷阱（部署推論時也穩定）
    df["label"] = df["new_topic_num"].astype(str)

    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()

    X_train_text = df_train["text"].values
    y_train = df_train["label"].values
    X_test_text  = df_test["text"].values
    y_test = df_test["label"].values

    # ====== TF-IDF (word + char 3–5) ======
    word_vect = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2), # (1, 2) (1, 3) (1, 4)
        min_df=5, #5, 10
        max_df=0.9,
        strip_accents="unicode",
        dtype=np.float32
    )
    char_vect = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5), # (3, 5) (3, 6)
        min_df=5, #5, 10
        max_df=0.95,
        dtype=np.float32
    )

    Xw_tr = word_vect.fit_transform(X_train_text)
    Xc_tr = char_vect.fit_transform(X_train_text)
    X_train = hstack([Xw_tr, Xc_tr])

    Xw_te = word_vect.transform(X_test_text)
    Xc_te = char_vect.transform(X_test_text)
    X_test = hstack([Xw_te, Xc_te])

    # ====== 建模：LinearSVC ======
    clf = LinearSVC(C=3.0, class_weight="balanced", random_state=42) #0.5, 1, 3, 5, 10

    clf.fit(X_train, y_train)

    # ====== 預測與評估 ======
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=sorted(np.unique(y_test)), columns=sorted(np.unique(y_test)))
    cm_df.to_csv(CM_CSV, index=True)

    # 儲存文字報告
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("[Overall]\n")
        f.write(f"accuracy  : {acc:.4f}\n")
        f.write(f"precision(macro): {p_macro:.4f}\n")
        f.write(f"recall(macro)   : {r_macro:.4f}\n")
        f.write(f"f1(macro)       : {f1_macro:.4f}\n\n")
        f.write(f"precision(weighted): {p_weight:.4f}\n")
        f.write(f"recall(weighted)   : {r_weight:.4f}\n")
        f.write(f"f1(weighted)       : {f1_weight:.4f}\n\n")
        f.write("[Per-class report]\n")
        f.write(report)
        f.write("\n[Parameters]\n")
        param_snapshot = {
            "linear_svc": clf.get_params(),
            "tfidf_word": word_vect.get_params(),
            "tfidf_char": char_vect.get_params()
        }
        f.write(json.dumps(to_serializable(param_snapshot), ensure_ascii=False, indent=2))
        f.write("\n")

    # 儲存預測明細
    out_pred = df_test[["image_name", "label"]].copy()
    out_pred["pred"] = y_pred
    out_pred.rename(columns={"label": "true"}, inplace=True)
    out_pred.to_csv(PRED_CSV, index=False)

    # 儲存模型＆向量器（部署用）
    joblib.dump(clf, MODEL_JOBLIB)
    joblib.dump(word_vect, VECT_WORD)
    joblib.dump(char_vect, VECT_CHAR)

    print("[OK] LinearSVC 訓練完成")
    print(f"Accuracy={acc:.4f}  Macro-F1={f1_macro:.4f}  Weighted-F1={f1_weight:.4f}")
    print(f"- 詳細報告: {REPORT_TXT}")
    print(f"- 混淆矩陣: {CM_CSV}")
    print(f"- 預測檔案: {PRED_CSV}")
    print(f"- 模型保存: {MODEL_JOBLIB}, {VECT_WORD}, {VECT_CHAR}")

if __name__ == "__main__":
    main()
