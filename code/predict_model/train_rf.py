# train_rf_tfidf_chi2.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier

# ====== Input files ======
TEXTS_CSV = "data\init_data\prepocessing_data\cleaned_texts_final_ig_twitter.csv"
LABELS_CSV = "data\\topic_output\\output_bertopic\\bertopic_labels.csv"
SPLIT_CSV = "else_file\split.csv"


# ** 外部控制變數 **
CURRENT_EXP_ID = "RF" # <-- 每次運行時修改或從命令行傳入

# ====== Output artifacts ======
# 根據 Exp ID 創建新的輸出目錄
RF_DIR = Path("data\model_output\only_output") / CURRENT_EXP_ID
RF_DIR.mkdir(parents=True, exist_ok=True)

# ====== Output artifacts ======
PRED_CSV       = RF_DIR / "rf_predictions.csv"
REPORT_TXT     = RF_DIR / "rf_report.txt"
CM_CSV         = RF_DIR / "rf_confusion_matrix.csv"
MODEL_JOBLIB   = RF_DIR / "rf_model.joblib"
VECT_WORD      = RF_DIR / "tfidf_word.joblib"
VECT_CHAR      = RF_DIR / "tfidf_char.joblib"
SEL_JOBLIB     = RF_DIR / "chi2_selector.joblib"
LABEL_MAP_JSON = RF_DIR / "label_mapping.json"
PARAMS_JSON    = RF_DIR / "rf_params.json"
META_JSON      = RF_DIR / "meta.json"


def safe_params(obj):
    """Convert nested parameter structures into JSON-serializable types."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): safe_params(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_params(v) for v in obj]
    return str(obj)


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
    print("[INFO] Loading and merging data...")
    texts = load_texts(TEXTS_CSV)
    labels = pd.read_csv(LABELS_CSV)[["image_name", "new_topic_num"]]
    split = pd.read_csv(SPLIT_CSV)[["image_name", "split"]]

    df = (
        texts.merge(labels, on="image_name", how="inner")
            .merge(split, on="image_name", how="inner")
    ).dropna(subset=["text", "new_topic_num", "split"])

    if df.empty:
        raise ValueError("Merged dataframe is empty. Check that image_name values match across files.")

    df["text"] = df["text"].astype(str)
    df["label_str"] = df["new_topic_num"].astype(str)

    print(f"[INFO] Samples: {len(df)} | Classes: {df['label_str'].nunique()}")
    print(df["split"].value_counts())

    df_train = df[df["split"] == "train"].copy()
    df_test = df[df["split"] == "test"].copy()
    if df_train.empty or df_test.empty:
        raise ValueError("train/test splits contain no samples. Verify split.csv.")

    X_train_text = df_train["text"].values
    X_test_text = df_test["text"].values

    print("[INFO] Building TF-IDF vectorizers...")
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
    X_train_sparse = hstack([Xw_tr, Xc_tr]).tocsr()

    Xw_te = word_vect.transform(X_test_text)
    Xc_te = char_vect.transform(X_test_text)
    X_test_sparse = hstack([Xw_te, Xc_te]).tocsr()

    n_samples, n_features = X_train_sparse.shape
    print(f"[INFO] TF-IDF shape -> samples: {n_samples}, features: {n_features}")

    K = min(2000, n_features) #1000, 5000, 10000
    print(f"[INFO] Applying chi2 feature selection with k={K}")
    selector = SelectKBest(score_func=chi2, k=K)
    X_train_sel = selector.fit_transform(X_train_sparse, df_train["label_str"].values)
    X_test_sel = selector.transform(X_test_sparse)

    print("[INFO] Converting selected features to dense matrices for RandomForest...")
    X_train = X_train_sel.toarray()
    X_test = X_test_sel.toarray()

    le = LabelEncoder()
    y_train = le.fit_transform(df_train["label_str"].values)
    y_test = le.transform(df_test["label_str"].values)

    with open(LABEL_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "classes": le.classes_.tolist(),
            "mapping": {int(i): cls for i, cls in enumerate(le.classes_)}
        }, f, ensure_ascii=False, indent=2)

    params = {
        "n_estimators": 600, #400, 800, 1200
        "max_depth": None, #20, 30, None
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "bootstrap": True,
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced_subsample",
        "verbose": 0
    }
    with open(PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    print("[INFO] Training RandomForest...")
    clf = RandomForestClassifier(**params)
    clf.fit(X_train, y_train)

    print("[INFO] Evaluating...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    report = classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        labels=le.classes_
    )
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(CM_CSV, index=True)

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
            "random_forest": safe_params(params),
            "tfidf_word": safe_params(word_vect.get_params()),
            "tfidf_char": safe_params(char_vect.get_params()),
            "chi2": {"k": int(K)}
        }
        f.write(json.dumps(param_snapshot, ensure_ascii=False, indent=2))
        f.write("\n[Confusion Matrix saved to] " + str(CM_CSV) + "\n")

    out_pred = df_test[["image_name"]].copy()
    out_pred["true"] = le.inverse_transform(y_test)
    out_pred["pred"] = le.inverse_transform(y_pred)
    out_pred.to_csv(PRED_CSV, index=False)

    joblib.dump(clf, MODEL_JOBLIB)
    joblib.dump(word_vect, VECT_WORD)
    joblib.dump(char_vect, VECT_CHAR)
    joblib.dump(selector, SEL_JOBLIB)

    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "vectorizer_word_params": safe_params(word_vect.get_params()),
            "vectorizer_char_params": safe_params(char_vect.get_params()),
            "chi2_k": int(K),
            "train_samples": int(X_train.shape[0]),
            "train_dim": int(X_train.shape[1])
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] RF (chi2) training finished @ {RF_DIR.resolve()}")
    print(f"Accuracy={acc:.4f}  Macro-F1={f1_macro:.4f}  Weighted-F1={f1_weight:.4f}")
    print(f"- Report: {REPORT_TXT}")
    print(f"- Confusion matrix: {CM_CSV}")
    print(f"- Predictions: {PRED_CSV}")
    print(f"- Model: {MODEL_JOBLIB}")
    print(f"- Vectorizers: {VECT_WORD}, {VECT_CHAR}")
    print(f"- Feature selector: {SEL_JOBLIB}")
    print(f"- Label map: {LABEL_MAP_JSON}")


if __name__ == "__main__":
    main()
