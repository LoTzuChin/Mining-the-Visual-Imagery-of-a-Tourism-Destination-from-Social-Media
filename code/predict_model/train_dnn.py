# train_dnn.py  — DNN + TF-IDF + SparseRandomProjection（SRP）
import os, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# ===(可選) 關閉 oneDNN，有時可避免 Windows 上卡頓 ===
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# 限制 BLAS/Omp 執行緒，避免多工互卡（可選）
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 讓 GPU 顯存按需增長（即使目前 GPUs=[] 也安全）
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ====== 檔案設定 ======
TEXTS_CSV = "data\init_data\prepocessing_data\cleaned_texts_final_ig_twitter.csv"
LABELS_CSV = "data\\topic_output\\output_bertopic\\bertopic_labels.csv"
SPLIT_CSV = "else_file\split.csv"

# ** 外部控制變數 **
CURRENT_EXP_ID = "DNN" # <-- 每次運行時修改或從命令行傳入

# ====== Output artifacts ======
# 根據 Exp ID 創建新的輸出目錄
DNN_DIR = Path("data\model_output\only_output") / CURRENT_EXP_ID
DNN_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV       = DNN_DIR / "dnn_predictions.csv"
REPORT_TXT     = DNN_DIR / "dnn_report.txt"
CM_CSV         = DNN_DIR / "dnn_confusion_matrix.csv"
HISTORY_JSON   = DNN_DIR / "dnn_history.json"
MODEL_DIR      = DNN_DIR / "tf_model"         # SavedModel 目錄
MODEL_H5       = DNN_DIR / "tf_model.h5"      # ModelCheckpoint 最佳權重
VECT_WORD      = DNN_DIR / "tfidf_word.joblib"
VECT_CHAR      = DNN_DIR / "tfidf_char.joblib"
SRP_JOBLIB     = DNN_DIR / "srp_projector.joblib"
LABEL_MAP_JSON = DNN_DIR / "label_mapping.json"
PARAMS_JSON    = DNN_DIR / "dnn_params.json"

def tlog(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_texts(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for cand in ["cleaned_text", "text", "desc", "description"]:
        if cand in df.columns:
            text_col = cand
            break
    else:
        raise ValueError(f"{path} 必須包含 cleaned_text/text/desc/description 任一欄位")
    if "image_name" not in df.columns:
        if "image_path" in df.columns:
            df["image_name"] = df["image_path"].apply(lambda p: Path(str(p)).name)
        else:
            raise ValueError(f"{path} 必須包含 image_name 或 image_path 欄位")
    return df[["image_name", text_col]].rename(columns={text_col: "text"})


def to_serializable(obj):
    """Convert objects into JSON-serializable structures."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    return str(obj)

def main():
    t0 = time.time()
    tlog(f"TensorFlow {tf.__version__}, GPUs={tf.config.list_physical_devices('GPU')}")

    # ====== 超參數（含 SRP 維度）======
    params = {
        "proj_components": 256,   # (128/ 256/ 384/ 512）
        "batch_size": 32,
        "epochs": 100,
        "patience": 6,
        "lr": 1e-3, # 5e-4, 2e-3, 1e-3
        "dropout": 0.6, # 0.4, 0.5, 0.6, 0.7
        "hidden1": 512, # 384, 768, 512
        "hidden2": 256, #192, 384, 256
        "seed": 42,
    }
    with open(PARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    # ====== 讀取 & 合併 ======
    tlog("Loading CSVs ...")
    texts  = load_texts(TEXTS_CSV)
    labels = pd.read_csv(LABELS_CSV)[["image_name", "new_topic_num"]]
    split  = pd.read_csv(SPLIT_CSV)[["image_name", "split"]]
    df = texts.merge(labels, on="image_name").merge(split, on="image_name")
    df = df.dropna(subset=["text", "new_topic_num", "split"])
    df["text"] = df["text"].astype(str)
    df["label_str"] = df["new_topic_num"].astype(str)

    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()
    X_train_text = df_train["text"].values
    X_test_text  = df_test["text"].values
    tlog(f"Total samples: {len(df)} | Train: {len(df_train)} | Test: {len(df_test)}")

    # ====== TF-IDF（word + char；限制特徵數與 df）======
    tlog("Building TF-IDF vectorizers (with caps) ...")
    # word_vect = TfidfVectorizer(
    #     analyzer="word",
    #     ngram_range=(1, 2),
    #     min_df=params["word_min_df"],
    #     max_df=0.9,
    #     max_features=params["word_max_features"],
    #     strip_accents="unicode"
    # )
    # char_vect = TfidfVectorizer(
    #     analyzer="char",
    #     ngram_range=(params["char_ngram_lo"], params["char_ngram_hi"]),
    #     min_df=params["char_min_df"],
    #     max_df=0.95,
    #     max_features=params["char_max_features"]
    # )

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

    tlog("Fitting TF-IDF on train ...")
    Xw_tr = word_vect.fit_transform(X_train_text).astype(np.float32)
    Xc_tr = char_vect.fit_transform(X_train_text).astype(np.float32)
    X_train_sparse = hstack([Xw_tr, Xc_tr], format="csr").astype(np.float32)

    Xw_te = word_vect.transform(X_test_text).astype(np.float32)
    Xc_te = char_vect.transform(X_test_text).astype(np.float32)
    X_test_sparse = hstack([Xw_te, Xc_te], format="csr").astype(np.float32)

    tlog(f"TF-IDF shapes -> train: {X_train_sparse.shape}, test: {X_test_sparse.shape}, nnz(train)={X_train_sparse.nnz}")

    # ====== 稀疏 -> 低維 dense：SparseRandomProjection ======
    tlog(f"Fitting SparseRandomProjection to {params['proj_components']} dims ...")
    srp = SparseRandomProjection(
        n_components=params["proj_components"],
        dense_output=True,
        random_state=params["seed"]
    )
    X_train = srp.fit_transform(X_train_sparse)   # -> dense (n_train, proj_components)
    X_test  = srp.transform(X_test_sparse)
    joblib.dump(srp, SRP_JOBLIB)
    tlog(f"SRP outputs -> train: {X_train.shape}, test: {X_test.shape}")

    # ====== 標籤 & 類別權重 ======
    le = LabelEncoder()
    y_train = le.fit_transform(df_train["label_str"].values)
    y_test  = le.transform(df_test["label_str"].values)
    num_classes = len(le.classes_)
    class_weight_vals = compute_class_weight(
        class_weight="balanced", classes=np.arange(num_classes), y=y_train
    )
    class_weight = {i: float(w) for i, w in enumerate(class_weight_vals)}
    with open(LABEL_MAP_JSON, "w", encoding="utf-8") as f:
        json.dump({"classes": le.classes_.tolist(),
                   "mapping": {int(i): cls for i, cls in enumerate(le.classes_)}},
                  f, ensure_ascii=False, indent=2)
    tlog(f"Num classes: {num_classes}")

    # ====== 建立 DNN ======
    tf.keras.utils.set_random_seed(params["seed"])
    model = keras.Sequential([
        layers.Input(shape=(params["proj_components"],)),
        layers.Dense(params["hidden1"], activation="relu"),
        layers.Dropout(params["dropout"]),
        layers.Dense(params["hidden2"], activation="relu"),
        layers.Dropout(params["dropout"]),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["lr"]),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=params["patience"], restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_H5), monitor="val_accuracy", save_best_only=True
        ),
    ]

    tlog("Start training ...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2
    )
    tlog("Training finished. Saving artifacts ...")

    # ====== 保存模型與前處理器 ======
    # 儲存模型 (Keras 格式)
    MODEL_KE = DNN_DIR / "tf_model.keras"
    model.save(MODEL_KE)
    # model.save(MODEL_DIR)                    # SavedModel 目錄
    joblib.dump(word_vect, VECT_WORD)
    joblib.dump(char_vect, VECT_CHAR)

    # ====== 評估 ======
    y_pred = model.predict(X_test, batch_size=512, verbose=0).argmax(axis=1)

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
    pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(CM_CSV, index=True)

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
            "dnn_hyperparams": params,
            "class_weight": class_weight,
            "tfidf_word": word_vect.get_params(),
            "tfidf_char": char_vect.get_params(),
            "sparse_random_projection": srp.get_params()
        }
        f.write(json.dumps(to_serializable(param_snapshot), ensure_ascii=False, indent=2))
        f.write("\n[Confusion Matrix saved to] " + str(CM_CSV) + "\n")

    out_pred = df_test[["image_name"]].copy()
    out_pred["true"] = le.inverse_transform(y_test)
    out_pred["pred"] = le.inverse_transform(y_pred)
    out_pred.to_csv(PRED_CSV, index=False)

    with open(HISTORY_JSON, "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()},
                  f, ensure_ascii=False, indent=2)

    tlog(f"All done in {round(time.time()-t0, 2)}s @ {DNN_DIR.resolve()}")
    tlog(f"Accuracy={acc:.4f}  Macro-F1={f1_macro:.4f}  Weighted-F1={f1_weight:.4f}")

if __name__ == "__main__":
    main()
