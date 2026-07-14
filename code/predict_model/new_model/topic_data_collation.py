# topic_data_collation_multilabel.py
# -*- coding: utf-8 -*-

import re
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

try:
    from docx import Document  # python-docx
except Exception:
    Document = None

# ====== 設定 ======
TOPIC_PROP_CSV = "topic_proportions.csv"
DOCX_PATH      = "小主題聚合成大主題_0930.docx"
OUT_LABELS_CSV = "bertopic_labels_multilabel.csv"  # 改名以示區別
OUT_SPLIT_CSV  = "split_multilabel.csv"

SEED = 42
PROB_THRESHOLD = 0.05  # [新增] 機率門檻：機率 > 5% 的主題才算數

# ====== 後備對應表 (保持不變) ======
FALLBACK_MAJOR_TO_SMALL = {
    1:  [1, 23, 24, 60, 63, 89],
    2:  [12, 13, 29, 32, 48, 55, 83],
    3:  [25, 34, 72, 77, 80, 81, 82, 84, 87],   # 注意：100 同時出現在 1 與 3，程式會警告並以先出現者為主
    4:  [2, 21, 27, 41, 65],
    5:  [35, 36, 38, 45, 52, 53, 54],
    6:  [15, 19, 20, 57, 58, 61, 69, 76],
    7:  [7, 30, 50, 59, 64, 70, 73, 75, 79],
    8:  [0, 4, 5, 18, 31, 42, 44, 49, 68, 71],
    9:  [3, 14, 22, 51],
    10: [6, 9, 10, 26, 33, 39, 40, 47, 62, 66, 74, 88],
    11: [17, 28, 43, 56, 78],
    12: [8, 11, 16, 37, 46, 67, 85, 86],
}

def build_mapping_from_dict(major_to_small: dict) -> dict:
    small_to_major = {}
    for major, small_list in major_to_small.items():
        for s in small_list:
            if s not in small_to_major: # 簡單處理重複，先到先得
                small_to_major[s] = major
    return small_to_major

# ... (保留 try_parse_docx_mapping 函數，此處省略以節省篇幅，請照舊複製即可) ...
def try_parse_docx_mapping(docx_path: str) -> dict | None:
    # 這裡請貼上您原本的 try_parse_docx_mapping 函數內容
    # 如果不想麻煩，可以直接回傳 None 讓它用 FALLBACK
    if Document is None: return None
    if not Path(docx_path).exists(): return None
    # (為簡化顯示，此處假設會使用 FALLBACK 或您原本的邏輯)
    return None 

def extract_multi_labels(row, topic_cols, col_name_to_id, small_to_major, threshold):
    """
    對每一列資料：
    1. 掃描所有 topic columns
    2. 若機率 > threshold，則取出該 topic id
    3. 查表轉成大主題 id (major_id)
    4. 收集成 set (去重，例如小主題 1 和 2 都對應大主題 A，則只算一次 A)
    """
    major_labels = set()
    
    for col in topic_cols:
        prob = row[col]
        # 確保機率有效且大於門檻
        if pd.notnull(prob) and prob >= threshold:
            raw_id = col_name_to_id.get(col)
            
            # 查找對應的大主題
            if raw_id is not None and raw_id in small_to_major:
                major_id = small_to_major[raw_id]
                major_labels.add(major_id)
    
    # 轉成排序後的列表，方便存成字串
    return sorted(list(major_labels))

def main():
    # 1. 建立映射表
    small_to_major = try_parse_docx_mapping(DOCX_PATH)
    if small_to_major is None:
        small_to_major = build_mapping_from_dict(FALLBACK_MAJOR_TO_SMALL)
        print("[INFO] 使用預設 FALLBACK 映射表")

    # 2. 讀取原始機率檔
    print(f"[INFO] 讀取 {TOPIC_PROP_CSV} ...")
    df = pd.read_csv(TOPIC_PROP_CSV)
    
    # 3. 辨識 topic 欄位 (e.g., topic_0, topic_1...)
    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    
    # 建立 { "topic_0": 0, "topic_1": 1 ... } 的快速查找表
    col_name_to_id = {}
    for c in topic_cols:
        # 抓出數字部分
        m = re.search(r"(-?\d+)", c)
        if m:
            col_name_to_id[c] = int(m.group(1))

    # 4. [核心修改] 針對每一列，提取多個標籤
    print(f"[INFO] 正在提取多標籤 (Threshold={PROB_THRESHOLD})...")
    
    # Apply 逐行處理
    # 結果會是一個 list，例如 [1, 5]
    df["label_list"] = df.apply(
        lambda row: extract_multi_labels(row, topic_cols, col_name_to_id, small_to_major, PROB_THRESHOLD),
        axis=1
    )

    # 5. [功能] 刪除不屬於任何一類 (list 為空) 的項目
    initial_count = len(df)
    df_filtered = df[df["label_list"].map(len) > 0].copy()
    dropped_count = initial_count - len(df_filtered)
    
    if dropped_count > 0:
        print(f"[WARN] 已刪除 {dropped_count} 筆完全不屬於任何大主題的資料 (空標籤)")

    # 6. 將 list 轉為字串 (例如 "1,5") 方便存 CSV
    # 這裡我們存一個字串欄位 topic_ids
    df_filtered["topic_ids"] = df_filtered["label_list"].apply(lambda x: ",".join(map(str, x)))
    
    # 整理輸出欄位
    # 我們只需要 image_name 和 topic_ids
    out_df = df_filtered[["image_name", "topic_ids"]].copy()
    
    out_df.to_csv(OUT_LABELS_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] 已輸出多標籤結果：{OUT_LABELS_CSV} (共 {len(out_df)} 筆)")

    # 7. 產生 Split (Train/Val/Test)
    # 注意：這裡要在過濾後的資料上做切分
    unique_imgs = out_df["image_name"].drop_duplicates()
    
    train_imgs, temp_imgs = train_test_split(unique_imgs, test_size=0.20, random_state=SEED, shuffle=True)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(20/36), random_state=SEED, shuffle=True)

    split_df = pd.concat([
        pd.DataFrame({"image_name": train_imgs, "split": "train"}),
        pd.DataFrame({"image_name": val_imgs,   "split": "val"}),
        pd.DataFrame({"image_name": test_imgs,  "split": "test"}),
    ], ignore_index=True)

    split_df.to_csv(OUT_SPLIT_CSV, index=False, encoding="utf-8-sig")
    print(f"[OK] 已輸出 Split：{OUT_SPLIT_CSV}")

if __name__ == "__main__":
    print(123)
    main()