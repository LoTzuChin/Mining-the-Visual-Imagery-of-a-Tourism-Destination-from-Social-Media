# 注意：若在本機執行遇到 PowerShell 執行原則限制，可自行調整政策或改用雲端環境。

import os
import pandas as pd
import numpy as np
from bertopic import BERTopic
from typing import Optional, List, Tuple, Union


def load_model(model_path: str) -> BERTopic:
    """載入指定路徑的 BERTopic 模型，若路徑不存在則拋出錯誤。"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    print(f"[INFO] Loading BERTopic model from: {model_path}")
    return BERTopic.load(model_path)


def load_texts(csv_path: str, text_col: str) -> pd.DataFrame:
    """讀取輸入 CSV 並將目標欄位轉為字串，缺漏值以空字串補足。"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    print(f"[INFO] Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not in CSV. Existing: {df.columns.tolist()}")
    df[text_col] = df[text_col].fillna("").astype(str)
    return df


def get_topic_ids(topic_model: BERTopic) -> List[int]:
    """取得所有主題編號，排除 BERTopic 以 -1 標示的離群文件。"""
    info = topic_model.get_topic_info()
    return [int(t) for t in info["Topic"].tolist() if int(t) != -1]


def coerce_to_matrix(probabilities: Union[np.ndarray, List]) -> Optional[np.ndarray]:
    """將 BERTopic 傳回的 probabilities 整理成二維矩陣，無法判定時回傳 None。"""
    if probabilities is None:
        return None
    if isinstance(probabilities, list):
        try:
            arrs = [np.asarray(p).ravel() for p in probabilities]
            if len(arrs) == 0:
                return None
            if arrs[0].size == 1:
                # 每筆僅有單一分數代表隸屬強度，無法形成完整主題分布
                return None
            return np.vstack(arrs)
        except Exception:
            return None
    if isinstance(probabilities, np.ndarray):
        if probabilities.ndim == 2:
            return probabilities
        if probabilities.ndim == 1:
            # 一維向量通常只有單一強度分數，直接視為無法使用的結果
            if np.issubdtype(probabilities.dtype, np.floating) or np.issubdtype(probabilities.dtype, np.integer):
                return None
            try:
                arrs = [np.asarray(p).ravel() for p in probabilities]
                if len(arrs) and arrs[0].size > 1:
                    return np.vstack(arrs)
            except Exception:
                return None
    return None


def get_topic_distribution(topic_model: BERTopic, docs: List[str]) -> Tuple[np.ndarray, List[int], str]:
    """回傳主題分布矩陣、主題編號與所使用的推論模式。"""
    topics, probs = topic_model.transform(docs)
    mat = coerce_to_matrix(probs)
    topic_ids = get_topic_ids(topic_model)

    if mat is not None:
        if mat.shape[1] != len(topic_ids):
            # 若欄數不一致，提醒使用者主題被重新編號的可能性
            print(f"[WARN] transform matrix topics({mat.shape[1]}) != topic_ids({len(topic_ids)}).")
        return mat, topic_ids, "transform"

    # transform 未提供完整矩陣時改採 approximate_distribution 取得估計值
    print("[INFO] Falling back to approximate_distribution(...) to get topic proportions.")
    res = topic_model.approximate_distribution(docs, window=8, stride=4)
    topic_distr = res[0] if isinstance(res, tuple) else res
    if not isinstance(topic_distr, np.ndarray) or topic_distr.ndim != 2:
        raise RuntimeError("approximate_distribution did not return a 2D matrix; cannot build topic proportions.")
    if topic_distr.shape[1] != len(topic_ids):
        print(f"[WARN] approx matrix topics({topic_distr.shape[1]}) != topic_ids({len(topic_ids)}). Using sequential IDs.")
        topic_ids = list(range(topic_distr.shape[1]))
    return topic_distr, topic_ids, "approx"


def build_prob_df(prob_matrix: np.ndarray, topic_ids: List[int]) -> pd.DataFrame:
    """根據主題編號建立對應的主題機率欄位。"""
    cols = [f"topic_{tid}_prob" for tid in topic_ids]
    return pd.DataFrame(prob_matrix, columns=cols)


def add_dominant(prob_df: pd.DataFrame) -> pd.DataFrame:
    """找出每篇文件機率最高的主題與對應機率。"""
    probs = prob_df.values
    argmax_idx = np.argmax(probs, axis=1)
    max_probs = probs[np.arange(probs.shape[0]), argmax_idx]
    colnames = np.array(prob_df.columns)
    dom_topic_cols = colnames[argmax_idx]
    dom_topic_ids = [c.replace("topic_", "").replace("_prob", "") for c in dom_topic_cols]
    return pd.DataFrame({"dominant_topic": dom_topic_ids, "dominant_prob": max_probs})


def main(model_path: str, input_csv: str, text_col: str, output_csv: str):
    """整合流程：載入模型、推論主題分布並輸出結果檔。"""
    topic_model = load_model(model_path)
    df = load_texts(input_csv, text_col)

    docs = df[text_col].tolist()
    print(f"[INFO] Transforming {len(docs)} documents ...")
    topic_matrix, topic_ids, mode = get_topic_distribution(topic_model, docs)
    print(f"[INFO] Using topic proportions from: {mode}")

    prob_df = build_prob_df(topic_matrix, topic_ids)
    dom_df = add_dominant(prob_df)

    # 將原始資料與主題分布、主題標籤合併輸出
    out_df = pd.concat([df.reset_index(drop=True),
                        prob_df.reset_index(drop=True),
                        dom_df.reset_index(drop=True)], axis=1)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved topic proportions to: {os.path.abspath(output_csv)}")


if __name__ == "__main__":
    # ==== 依據 Colab 預設路徑寫死，可依需要自行修改 ====
    MODEL_PATH  = "bertopic_finetuned_model"
    INPUT_CSV   = "data\init_data\prepocessing_data\cleaned_texts_final_ig_twitter.csv"
    TEXT_COLUMN = "cleaned_text"
    OUTPUT_CSV  = "topic_proportions.csv"

    main(MODEL_PATH, INPUT_CSV, TEXT_COLUMN, OUTPUT_CSV)
