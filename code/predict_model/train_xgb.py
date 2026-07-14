# train_xgb_gpu.py
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from scipy.sparse import hstack, csr_matrix
import json
import joblib
import xgboost as xgb

# ====== 輸入檔 ======
TEXTS_CSV = "data\init_data\prepocessing_data\cleaned_texts_final_ig_twitter.csv"
LABELS_CSV = "data\\topic_output\\output_bertopic\\bertopic_labels.csv"
SPLIT_CSV = "else_file\split.csv"

# ** 外部控制變數 **
CURRENT_EXP_ID = "XGB" # <-- 每次運行時修改或從命令行傳入

# ====== Output artifacts ======
# 根據 Exp ID 創建新的輸出目錄
XGB_DIR = Path("data\model_output\only_output") / CURRENT_EXP_ID
XGB_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV      = XGB_DIR / "xgb_predictions.csv"
REPORT_TXT    = XGB_DIR / "xgb_report.txt"
CM_CSV        = XGB_DIR / "xgb_confusion_matrix.csv"
MODEL_JSON    = XGB_DIR / "xgb_model.json"      # 可移植部署
VECT_WORD     = XGB_DIR / "tfidf_word.joblib"
VECT_CHAR     = XGB_DIR / "tfidf_char.joblib"
LABEL_MAP_JSON= XGB_DIR / "label_mapping.json"  # 儲存 label↔id 對應
PARAMS_JSON   = XGB_DIR / "xgb_params.json"

# ====== 讀取文字欄位 ======
def load_texts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for cand in ["cleaned_text", "text", "desc", "description"]:
        if cand in df.columns:
            text_col = cand
            break
    else:
        raise ValueError(f"{path} 必須包含文字欄位 cleaned_text/text/desc/description")

    if "image_name" not in df.columns:
        if "image_path" in df.columns:
            df["image_name"] = df["image_path"].apply(lambda p: Path(str(p)).name)
        else:
            raise ValueError(f"{path} 必須有 image_name 欄位或 image_path 欄位")

    return df[["image_name", text_col]].rename(columns={text_col: "text"})

def build_sample_weight(labels: np.ndarray) -> np.ndarray:
    """依類別頻率建立 sample_weight：freq 越低權重越高"""
    vals, counts = np.unique(labels, return_counts=True)
    freq = dict(zip(vals, counts))
    total = len(labels)
    weights = np.array([total / (len(freq) * freq[y]) for y in labels], dtype=np.float32)
    return weights

def to_serializable(obj):
    """將參數物件轉成可序列化格式，避免 dtype 等型別寫檔失敗。"""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    return str(obj)

def main():
    # ====== 載入與合併 ======
    texts  = load_texts(TEXTS_CSV)
    labels = pd.read_csv(LABELS_CSV)[["image_name", "new_topic_num"]]
    split  = pd.read_csv(SPLIT_CSV)[["image_name", "split"]]

    df = (
        texts.merge(labels, on="image_name", how="inner")
             .merge(split, on="image_name", how="inner")
    ).dropna(subset=["text", "new_topic_num", "split"])

    df["text"] = df["text"].astype(str)
    # 以字串保存原始類別，對 XGB 再做整數編碼
    df["label_str"] = df["new_topic_num"].astype(str)

    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()

    X_train_text = df_train["text"].values
    X_test_text  = df_test["text"].values

    # ====== TF-IDF（word + char 3–5）======
    word_vect = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2), # (1, 2) (1, 3) (1, 4)
        min_df=20, #5, 10
        max_df=0.9,
        strip_accents="unicode",
        max_features=40000,     # ⭐️ 限制特徵數量
        dtype=np.float32
    )
    char_vect = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5), # (3, 5) (3, 6)
        min_df=20, #5, 10
        max_df=0.95,
        max_features=30000,    # ⭐️ 限制特徵數量
        dtype=np.float32
    )

    Xw_tr = word_vect.fit_transform(X_train_text)
    Xc_tr = char_vect.fit_transform(X_train_text)
    X_train = hstack([Xw_tr, Xc_tr]).tocsr()

    Xw_te = word_vect.transform(X_test_text)
    Xc_te = char_vect.transform(X_test_text)
    X_test = hstack([Xw_te, Xc_te]).tocsr()

    # ====== 標籤整數化（XGBoost 需要 0..K-1）======
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["label_str"].values)
    y_test  = le.transform(df_test["label_str"].values)
    num_classes = len(le.classes_)

    # 儲存 label 對應，部署或還原報表會用到
    with open(LABEL_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "classes": le.classes_.tolist(),
            "mapping": {int(i): cls for i, cls in enumerate(le.classes_)}
        }, f, ensure_ascii=False, indent=2)

    # ====== 類別不平衡處理（sample_weight）======
    sample_weight = build_sample_weight(y_train)

    # ====== XGBoost GPU 參數 ======
    # 1) 參數：移除 n_estimators，避免多餘警告
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "device": "cuda",
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 1.0,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": 42,
    }

    num_boost_round = 2000  # 原本放在 params["n_estimators"] 的數字，獨立拿出來


    with open(PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    # ====== DMatrix 與 eval_set（可得更快的 GPU 管線）======
    # 也可直接用 scikit API：xgb.XGBClassifier(**params).fit(...)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weight)
    dvalid = xgb.DMatrix(X_test,  label=y_test)

    # 早停
    # 2) 訓練：把 verbose_eval 調小，方便觀察
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=50,
        verbose_eval=10,  # <- 每 10 輪列印一次
    )

    # ====== 預測與評估 ======
    y_prob = bst.predict(dvalid)  # shape: [N, num_classes]
    y_pred = y_prob.argmax(axis=1)

    # 報表（同時提供 macro 與 weighted 指標）
    acc = accuracy_score(y_test, y_pred)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # 逐類別報告（用原始類別字串顯示）
    report = classification_report(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        digits=4,
        zero_division=0
    )

    # 混淆矩陣（以原始類別順序顯示）
    cm = confusion_matrix(
        le.inverse_transform(y_test),
        le.inverse_transform(y_pred),
        labels=le.classes_
    )
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
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
        f.write("\n[Confusion Matrix saved to] " + str(CM_CSV) + "\n\n")
        f.write("[Parameters]\n")
        param_snapshot = {
            "xgboost": params,
            "tfidf_word": word_vect.get_params(),
            "tfidf_char": char_vect.get_params()
        }
        f.write(json.dumps(to_serializable(param_snapshot), ensure_ascii=False, indent=2))
        f.write("\n")

    # 預測明細
    out_pred = df_test[["image_name"]].copy()
    out_pred["true"] = le.inverse_transform(y_test)
    out_pred["pred"] = le.inverse_transform(y_pred)
    PRED_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_pred.to_csv(PRED_CSV, index=False)

    # ====== 儲存模型與向量器 ======
    bst.save_model(str(MODEL_JSON))          # JSON/UBJ 皆可，JSON較通用
    joblib.dump(word_vect, VECT_WORD)
    joblib.dump(char_vect, VECT_CHAR)

    print(f"[OK] XGBoost(GPU) 訓練完成 @ {XGB_DIR.resolve()}")
    print(f"Accuracy={acc:.4f}  Macro-F1={f1_macro:.4f}  Weighted-F1={f1_weight:.4f}")
    print(f"- 報告: {REPORT_TXT}")
    print(f"- 混淆矩陣: {CM_CSV}")
    print(f"- 預測: {PRED_CSV}")
    print(f"- 模型: {MODEL_JSON}")
    print(f"- 向量器: {VECT_WORD}, {VECT_CHAR}")
    print(f"- 標籤對應: {LABEL_MAP_JSON}")

if __name__ == "__main__":
    main()
