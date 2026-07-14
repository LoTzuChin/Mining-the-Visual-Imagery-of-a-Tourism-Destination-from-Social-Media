import os
import json
from pathlib import Path
from collections import Counter
from itertools import combinations

import pandas as pd

# =========================
# 基本參數設定（請自行修改）
# =========================

# 放 Vision JSON 檔的資料夾
topic = "topic_12"
BASE_DIR = Path(r"data\vision_data")
INPUT_DIR = BASE_DIR / topic / "vision_description"


# 輸出結果的資料夾
OUTPUT_BASE_DIR = Path(r"data\\vision_method_output\\an_la_coocuurrence")
OUTPUT_DIR = OUTPUT_BASE_DIR / topic
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 只保留 score >= 這個門檻的 label（如果不想過濾就設 0）
SCORE_THRESHOLD = 0.0

# 至少在多少張圖片中出現過的標籤才納入統計（避免極冷門 label）
MIN_LABEL_IMAGE_COUNT = 1

# 顯示前幾名共現最頻繁的 pair 在 console
TOP_K_PRINT = 30


# =========================
# 讀取 JSON 並蒐集標籤
# =========================

def load_labels_from_json_file(json_path: Path, score_threshold: float):
    """
    從單一 Vision JSON 檔中取出 label description list（依 score 過濾）。
    回傳：set[str]，同張圖片中同一 label 只算一次。

    ✅ 同時支援兩種常見格式：
        1) "labels": [ {"description": "...", "score": 0.8}, ... ]
        2) "labels": [ "Beach", "Summer", "Sea" ]
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] 讀取失敗，跳過：{json_path}，錯誤：{e}")
        return set()

    labels_raw = data.get("labels", [])
    labels = set()

    # 若整個就是 list[str] 形式，直接處理
    # 例：["Beach", "Summer", "Sea"]
    if all(isinstance(x, str) for x in labels_raw):
        for desc in labels_raw:
            desc = str(desc).strip()
            if not desc:
                continue
            # 沒有 score 的情況，當作 1.0（一定通過門檻）
            if 1.0 >= score_threshold:
                labels.add(desc)
        return labels

    # 否則逐一處理 list 裡的元素，可能是 dict 或 str 混合
    for item in labels_raw:
        desc = None
        score = 1.0  # 沒有 score 時當作 1.0

        if isinstance(item, dict):
            # 正常 Vision 格式：{"description": "...", "score": 0.8, ...}
            desc = item.get("description") or item.get("label") or item.get("name")
            score = float(item.get("score", 1.0))
        elif isinstance(item, str):
            # 混合: 例如 ["Beach", {"description": "Sea", "score": 0.9}]
            desc = item.strip()
            score = 1.0
        else:
            # 其他奇怪型態就跳過
            continue

        if not desc:
            continue
        if score < score_threshold:
            continue

        labels.add(str(desc).strip())

    return labels



def collect_all_image_labels(input_dir: str, score_threshold: float):
    """
    走訪資料夾，把每張圖片的 label set 收集起來。
    回傳：
    - image_labels_list: List[Set[str]] 每一項代表一張圖片的標籤集合
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"輸入資料夾不存在：{input_dir}")

    image_labels_list = []

    json_files = sorted(list(input_path.glob("*.json")))
    print(f"[INFO] 在 {input_dir} 找到 {len(json_files)} 個 JSON 檔。")

    for idx, json_path in enumerate(json_files, start=1):
        labels = load_labels_from_json_file(json_path, score_threshold)
        if labels:
            image_labels_list.append(labels)
        # 若你想看進度，可開啟以下註解
        # if idx % 1000 == 0:
        #     print(f"[INFO] 已處理 {idx} / {len(json_files)} 個 JSON 檔")

    print(f"[INFO] 有 {len(image_labels_list)} 張圖片至少有一個符合條件的 label。")
    return image_labels_list


# =========================
# 建立共現統計
# =========================

def compute_cooccurrence(image_labels_list, min_label_image_count: int):
    """
    給定每張圖的 label set 列表，計算：
    - label_counts: 每個 label 出現於多少張圖片
    - pair_counts: 每個 label pair 共現於多少張圖片（無向、A-B 與 B-A 視為同一對）
    """
    label_counts = Counter()
    pair_counts = Counter()

    for labels in image_labels_list:
        # 先只計算這張圖片的唯一 label
        unique_labels = set(labels)
        for lbl in unique_labels:
            label_counts[lbl] += 1

        # 計算這張圖片中的所有兩兩組合（無序組合）
        # 例如 {A,B,C} -> (A,B), (A,C), (B,C)
        for a, b in combinations(sorted(unique_labels), 2):
            pair_counts[(a, b)] += 1

    # 依照最低出現次數過濾掉冷門 label
    if min_label_image_count > 1:
        valid_labels = {lbl for lbl, cnt in label_counts.items()
                        if cnt >= min_label_image_count}
        print(f"[INFO] label 依出現次數 >= {min_label_image_count} 過濾後，剩餘 {len(valid_labels)} 個。")

        # 過濾 label_counts
        label_counts = Counter({lbl: cnt for lbl, cnt in label_counts.items()
                                if lbl in valid_labels})

        # 過濾 pair_counts（兩邊都要在 valid_labels 中）
        pair_counts = Counter({
            (a, b): cnt
            for (a, b), cnt in pair_counts.items()
            if a in valid_labels and b in valid_labels
        })

    else:
        print(f"[INFO] 未過濾冷門 label，總共有 {len(label_counts)} 個 label。")

    return label_counts, pair_counts


# =========================
# 計算條件機率 + 輸出 CSV
# =========================

def export_pair_stats(label_counts: Counter,
                      pair_counts: Counter,
                      output_dir: str,
                      top_k_print: int):
    """
    輸出長表格形式的 pair 統計：
    - label_A
    - label_B
    - co_occurrence
    - P_B_given_A (P(B|A))
    - P_A_given_B (P(A|B))
    並在 console 顯示共現最多的前 K 組。
    """
    rows = []

    for (a, b), co_cnt in pair_counts.items():
        count_a = label_counts.get(a, 0)
        count_b = label_counts.get(b, 0)

        if count_a > 0:
            p_b_given_a = co_cnt / count_a
        else:
            p_b_given_a = 0.0

        if count_b > 0:
            p_a_given_b = co_cnt / count_b
        else:
            p_a_given_b = 0.0

        rows.append({
            "label_A": a,
            "label_B": b,
            "co_occurrence": co_cnt,
            "P_B_given_A": p_b_given_a,
            "P_A_given_B": p_a_given_b,
        })

    df_pairs = pd.DataFrame(rows)
    df_pairs.sort_values(by="co_occurrence", ascending=False, inplace=True)
    out_path = Path(output_dir) / "label_pair_stats.csv"
    df_pairs.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已輸出 pair 統計到：{out_path}")

    # 顯示前 TOP_K_PRINT 筆共現最多的 pair
    if top_k_print > 0 and not df_pairs.empty:
        print(f"\n=== 共現次數最高的前 {top_k_print} 組標籤 ===")
        print(df_pairs.head(top_k_print).to_string(index=False))


def export_cooccurrence_matrix(label_counts: Counter,
                               pair_counts: Counter,
                               output_dir: str):
    """
    輸出共現矩陣（DataFrame），index/columns 都是 label，值為共現次數。
    對角線 (A,A) 保留單一 label 在多少張圖片出現，用 label_counts。
    """
    labels = sorted(label_counts.keys())
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    # 初始化矩陣
    import numpy as np
    n = len(labels)
    mat = np.zeros((n, n), dtype=int)

    # 對角線為個別 label 出現次數
    for lbl, cnt in label_counts.items():
        i = label_to_idx[lbl]
        mat[i, i] = cnt

    # pair 共現（對稱）
    for (a, b), co_cnt in pair_counts.items():
        i = label_to_idx[a]
        j = label_to_idx[b]
        mat[i, j] = co_cnt
        mat[j, i] = co_cnt

    df_mat = pd.DataFrame(mat, index=labels, columns=labels)
    out_path = Path(output_dir) / "label_cooccurrence_matrix.csv"
    df_mat.to_csv(out_path, encoding="utf-8-sig")
    print(f"[INFO] 已輸出共現矩陣到：{out_path}")


# =========================
# 主程式
# =========================

def main():
    # 1) 收集所有圖片的 label set
    image_labels_list = collect_all_image_labels(INPUT_DIR, SCORE_THRESHOLD)

    if not image_labels_list:
        print("[WARN] 沒有任何圖片有符合條件的 label，程式結束。")
        return

    # 2) 計算 label 與 pair 的統計
    label_counts, pair_counts = compute_cooccurrence(
        image_labels_list,
        min_label_image_count=MIN_LABEL_IMAGE_COUNT,
    )

    # 3) 輸出 pair 長表（含 P(B|A)、P(A|B)）
    export_pair_stats(label_counts, pair_counts, OUTPUT_DIR, TOP_K_PRINT)

    # 4) 輸出共現矩陣
    export_cooccurrence_matrix(label_counts, pair_counts, OUTPUT_DIR)

    print("\n[INFO] 分析完成。你可以用 Excel / pandas 進一步做探索。")


if __name__ == "__main__":
    main()
