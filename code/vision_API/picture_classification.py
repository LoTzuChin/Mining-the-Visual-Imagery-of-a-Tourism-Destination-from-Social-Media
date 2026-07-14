# group_images_by_topic_windows.py
import os
import shutil
from pathlib import Path
import pandas as pd

# ==== 參數（已寫死路徑）====
CSV_PATH   = r"data\\topic_output\\output_bertopic\bertopic_labels.csv"
IMAGE_DIR  = r"data\\init_data\\original_data\\img_ig_with_twitter"          # 放全部圖片的資料夾
OUTPUT_DIR = r"data\\vision_data"    # 依主題輸出到這裡
MOVE_FILES = False  # False=複製, True=移動

# ==== 主程式 ====
def main():
    csv_path = Path(CSV_PATH)
    img_dir  = Path(IMAGE_DIR)
    out_dir  = Path(OUTPUT_DIR)

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 檔：{csv_path}")
    if not img_dir.exists():
        raise FileNotFoundError(f"找不到圖片資料夾：{img_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required_cols = {"image_name", "new_topic_num"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV 需包含欄位：{required_cols}，實際為：{list(df.columns)}")

    # 正規化檔名與 topic
    df["image_name"] = df["image_name"].astype(str).str.strip()
    df["new_topic_num"] = df["new_topic_num"].astype(str).str.strip()

    total = len(df)
    found = 0
    missing = 0

    for idx, row in df.iterrows():
        img_name = row["image_name"]
        topic    = row["new_topic_num"]

        src = img_dir / img_name
        topic_folder = out_dir / f"topic_{topic}"
        topic_folder.mkdir(parents=True, exist_ok=True)

        if src.exists():
            dst = topic_folder / img_name
            if MOVE_FILES:
                shutil.move(str(src), str(dst))
            else:
                # 若已存在則略過（避免重複複製）
                if not dst.exists():
                    shutil.copy2(str(src), str(dst))
            found += 1
        else:
            print(f"[WARN] 找不到圖片：{src}")
            missing += 1

        if (idx + 1) % 200 == 0:
            print(f"[INFO] 進度 {idx+1}/{total}，已處理：{found}，缺失：{missing}")

    print("====== 完成 ======")
    print(f"總列數：{total}，成功處理：{found}，找不到：{missing}")
    print(f"輸出資料夾：{out_dir.resolve()}")
    print(f"模式：{'移動' if MOVE_FILES else '複製'}")

if __name__ == "__main__":
    main()
