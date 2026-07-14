# split_dataset.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# === 參數 ===
LABELS_CSV = "bertopic_labels_except_no_topic.csv"   # 必須包含: image_name, new_topic_num
OUTPUT_SPLIT_CSV = "split.csv"       # 產出: image_name, split(train/test)
TEST_SIZE = 0.2
SEED = 42

def main():
    df = pd.read_csv(LABELS_CSV)
    required_cols = {"image_name", "new_topic_num"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{LABELS_CSV} 缺少欄位: {missing}")

    # 去除重複 image_name（若有），保留第一筆並提示
    dup_count = df["image_name"].duplicated().sum()
    if dup_count > 0:
        print(f"[WARN] 偵測到 {dup_count} 個重複的 image_name，將保留第一筆並移除後續重複。")
        df = df.drop_duplicates(subset=["image_name"], keep="first").reset_index(drop=True)

    # 檢查每一類別的樣本數，若所有類別皆 >= 2，嘗試做分層抽樣
    label_counts = df["new_topic_num"].value_counts()
    can_stratify = (label_counts.min() >= 2) and (label_counts.size > 1)

    if can_stratify:
        try:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
            y = df["new_topic_num"].astype(str)  # stratify 需要 1D array
            idx_train, idx_test = next(sss.split(df, y))
            train_df = df.iloc[idx_train].copy()
            test_df  = df.iloc[idx_test].copy()
            stratified = True
        except ValueError as e:
            print(f"[WARN] Stratified split 失敗（{e}），改用隨機切分。")
            stratified = False
    else:
        stratified = False

    if not stratified:
        train_df, test_df = train_test_split(
            df,
            test_size=TEST_SIZE,
            random_state=SEED,
            shuffle=True,
            stratify=None  # 無法分層時退回隨機
        )

    # 產出 split.csv
    out = pd.concat([
        train_df.assign(split="train")[["image_name", "new_topic_num", "split"]],
        test_df.assign(split="test")[["image_name", "new_topic_num", "split"]],
    ], ignore_index=True)

    # 為了簡潔，split.csv 只保留需求欄位（image_name, split）
    out_min = out[["image_name", "split"]].sort_values("image_name").reset_index(drop=True)
    out_min.to_csv(OUTPUT_SPLIT_CSV, index=False)
    print(f"[OK] 已輸出 {OUTPUT_SPLIT_CSV}（共 {len(out_min)} 筆；train={sum(out_min['split']=='train')}, test={sum(out_min['split']=='test')}）")

    # 額外資訊（可忽略）：查看分佈是否大致一致
    print("\n[Info] 標籤分佈（原始資料）：")
    print(label_counts.sort_index())
    print("\n[Info] 標籤分佈（train）：")
    print(train_df["new_topic_num"].value_counts().sort_index())
    print("\n[Info] 標籤分佈（test）：")
    print(test_df["new_topic_num"].value_counts().sort_index())

if __name__ == "__main__":
    main()
