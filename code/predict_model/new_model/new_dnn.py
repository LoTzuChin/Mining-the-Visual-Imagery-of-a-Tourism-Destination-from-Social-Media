# train_dnn_multilabel.py
import os
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import joblib

# Sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, hamming_loss
)

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# === 環境設定 ===
# 讓 GPU 顯存按需增長
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass

# ====== 檔案設定 ======
# 假設你已經跑完 topic_data_collation_multilabel.py 並產生了以下檔案
TEXTS_CSV   = "data\init_data\prepocessing_data\cleaned_texts_final_ig_twitter.csv"  # 需確認路徑
LABELS_CSV  = "data\\topic_output\\output_bertopic\\bertopic_labels_multilabel.csv"       # [新] 多標籤結果
SPLIT_CSV   = "else_file\split_multilabel.csv"                 # [新] 多標籤 split

# ** 外部控制變數 **
CURRENT_EXP_ID = "DNN"

# ====== 輸出目錄設定 ======
DNN_DIR = Path("data\\model_output\\new_multi_output") / CURRENT_EXP_ID
DNN_DIR.mkdir(parents=True, exist_ok=True)

PRED_CSV      = DNN_DIR / "dnn_predictions.csv"
REPORT_TXT    = DNN_DIR / "dnn_report.txt"
MODEL_KERAS   = DNN_DIR / "dnn_model.keras"       # Keras 模型檔
MLB_JOBLIB    = DNN_DIR / "mlb.joblib"            # [重要] 儲存標籤編碼器
VECT_WORD     = DNN_DIR / "tfidf_word.joblib"
VECT_CHAR     = DNN_DIR / "tfidf_char.joblib"
SRP_JOBLIB    = DNN_DIR / "srp.joblib"            # 降維投影矩陣


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
            raise ValueError(f"{path} 需包含 image_name 欄位")
            
    return df[["image_name", text_col]].rename(columns={text_col: "text"})

def to_serializable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    return str(obj)

def build_multilabel_model(input_dim, output_dim, learning_rate=1e-3):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # 增加神經元數量，並加入 L2 正規化
        layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5), # 提高 Dropout 比例
        
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(output_dim, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        # 除了 accuracy，增加 AUC 指標能更全面觀察多標籤效能
        metrics=['binary_accuracy', keras.metrics.AUC(multi_label=True)] 
    )
    return model

def main():
    print(f"[INFO] Loading data...")
    texts  = load_texts(TEXTS_CSV)
    labels = pd.read_csv(LABELS_CSV)
    split  = pd.read_csv(SPLIT_CSV)

    # 1. 資料合併
    if "topic_ids" not in labels.columns:
        raise ValueError(f"{LABELS_CSV} 必須包含 'topic_ids' 欄位。")

    df = (
        texts.merge(labels[["image_name", "topic_ids"]], on="image_name", how="inner")
             .merge(split[["image_name", "split"]], on="image_name", how="inner")
    )
    
    df = df.dropna(subset=["text", "topic_ids", "split"])
    df["text"] = df["text"].astype(str)

    # 2. 解析多標籤字串 "1,5" -> ['1', '5']
    df["label_list"] = df["topic_ids"].astype(str).apply(
        lambda x: [s.strip() for s in x.split(',') if s.strip()]
    )

    df_train = df[df["split"] == "train"].copy()
    df_test  = df[df["split"] == "test"].copy()
    
    print(f"[INFO] Train size: {len(df_train)}, Test size: {len(df_test)}")

    X_train_text = df_train["text"].values
    X_test_text  = df_test["text"].values

    # 3. 標籤編碼 (MultiLabelBinarizer)
    print(f"[INFO] Encoding labels...")
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(df_train["label_list"])
    y_test  = mlb.transform(df_test["label_list"])
    
    num_classes = len(mlb.classes_)
    print(f"[INFO] Classes detected: {num_classes}")
    print(f"Classes: {mlb.classes_}")

    # 4. TF-IDF + SRP (保持原 DNN 邏輯，使用 SRP 降維以加速)
    print(f"[INFO] Extracting TF-IDF features...")
    word_vect = TfidfVectorizer(analyzer="word", ngram_range=(1, 3), min_df=2, max_df=0.9, dtype=np.float32)
    char_vect = TfidfVectorizer(analyzer="char", ngram_range=(3, 6), min_df=2, max_df=0.95, dtype=np.float32)

    Xw_tr = word_vect.fit_transform(X_train_text)
    Xc_tr = char_vect.fit_transform(X_train_text)
    X_train_sparse = hstack([Xw_tr, Xc_tr])

    Xw_te = word_vect.transform(X_test_text)
    Xc_te = char_vect.transform(X_test_text)
    X_test_sparse = hstack([Xw_te, Xc_te])
    
    # 執行 SRP 降維 (例如降到 3000 維，依硬體可調整)
    target_dim = 3000
    if X_train_sparse.shape[1] > target_dim:
        print(f"[INFO] Reducing dimensions with SRP to {target_dim}...")
        srp = SparseRandomProjection(n_components=target_dim, dense_output=True, random_state=42)
        X_train_dense = srp.fit_transform(X_train_sparse)
        X_test_dense  = srp.transform(X_test_sparse)
    else:
        print("[INFO] Dimension is small enough, skipping SRP.")
        srp = None
        X_train_dense = X_train_sparse.toarray()
        X_test_dense  = X_test_sparse.toarray()

    # 5. 建立與訓練模型
    print(f"[INFO] Building DNN Model...")
    model = build_multilabel_model(input_dim=X_train_dense.shape[1], output_dim=num_classes)
    
    # Early Stopping
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    print(f"[INFO] Training...")
    history = model.fit(
        X_train_dense, y_train,
        validation_split=0.1,  # 從訓練集切 10% 做驗證
        epochs=30,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    # 6. 預測與評估
    print(f"[INFO] Evaluating...")
    # model.predict 輸出的是 0~1 的機率矩陣
    y_prob = model.predict(X_test_dense)
    
    # 設定門檻轉為 0/1 (通常用 0.5)
    threshold = 0.5
    y_pred = (y_prob >= threshold).astype(int)

    # --- 計算多標籤指標 ---
    acc = accuracy_score(y_test, y_pred)
    h_loss = hamming_loss(y_test, y_pred)

    p_samp, r_samp, f1_samp, _ = precision_recall_fscore_support(
        y_test, y_pred, average="samples", zero_division=0
    )
    
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="micro", zero_division=0
    )

    report = classification_report(
        y_test, y_pred, target_names=mlb.classes_, zero_division=0
    )

    # 儲存報告
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(f"[DNN Multi-Label Evaluation]\n")
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
        
        f.write("\n[Training History]\n")
        # 紀錄 Loss 變化
        hist_df = pd.DataFrame(history.history)
        f.write(hist_df.to_string())

        f.write("\n\n[Parameters]\n")
        param_snapshot = {
            "dnn_layers": ["256-relu", "128-relu", "sigmoid"],
            "threshold": threshold,
            "srp_dim": target_dim,
            "tfidf_word": word_vect.get_params(),
            "tfidf_char": char_vect.get_params()
        }
        f.write(json.dumps(to_serializable(param_snapshot), ensure_ascii=False, indent=2))

    # 7. 儲存預測結果
    pred_labels_tuples = mlb.inverse_transform(y_pred)
    true_labels_tuples = mlb.inverse_transform(y_test)
    
    out_pred = pd.DataFrame({
        "image_name": df_test["image_name"],
        "true_labels": [",".join(x) for x in true_labels_tuples],
        "pred_labels": [",".join(x) for x in pred_labels_tuples]
    })
    out_pred.to_csv(PRED_CSV, index=False)

    # 8. 儲存模型物件
    print(f"[INFO] Saving artifacts to {DNN_DIR}...")
    model.save(MODEL_KERAS)  # 儲存 Keras 模型
    joblib.dump(mlb, MLB_JOBLIB)
    joblib.dump(word_vect, VECT_WORD)
    joblib.dump(char_vect, VECT_CHAR)
    if srp:
        joblib.dump(srp, SRP_JOBLIB)

    print("[OK] DNN Multi-Label Training Completed.")
    print(f"Samples-F1={f1_samp:.4f} | Micro-F1={f1_micro:.4f}")
    print(f"- Report: {REPORT_TXT}")
    print(f"- Predictions: {PRED_CSV}")

if __name__ == "__main__":
    main()