# -*- coding: utf-8 -*-
"""
Google Vision Label Frequency Analysis (固定輸入 / 輸出路徑版本)
"""

import os
import json
from pathlib import Path
from collections import defaultdict, Counter

import pandas as pd
import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

from pathlib import Path

topic = "topic_12"
BASE_DIR = Path(r"data\vision_data")
INPUT_DIR = BASE_DIR / topic / "vision_description"

# ====== 你只需要改這兩行 ======
OUTPUT_DIR = Path(r"data\\vision_method_output\\la_fre_analysis\\label_stats")
TOP_K = 20
# =================================


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    """讀取單一 JSON 檔並回傳 dict"""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] 無法讀取 {path}: {e}")
        return None


def extract_labels(data: dict):
    """支援 labels / labelAnnotations 兩種格式"""
    if data is None:
        return []

    labels = []

    if isinstance(data.get("labels"), list):
        for item in data["labels"]:
            desc = item.get("description")
            score = item.get("score", None)
            if desc:
                labels.append((desc, score))

    if isinstance(data.get("labelAnnotations"), list):
        for item in data["labelAnnotations"]:
            desc = item.get("description")
            score = item.get("score", None)
            if desc:
                labels.append((desc, score))

    return labels


def analyze_labels(input_dir: Path):
    """統計所有 JSON 檔案中的 label 出現頻率"""
    label_counter = Counter()
    label_score_sum = defaultdict(float)
    label_score_max = defaultdict(float)
    label_image_set = defaultdict(set)

    json_files = list(input_dir.rglob("*.json"))

    if not json_files:
        print(f"[INFO] 在 {input_dir} 沒有找到任何 JSON")
        return pd.DataFrame()

    print(f"[INFO] 找到 {len(json_files)} 個 JSON，開始分析...")

    for idx, json_path in enumerate(json_files, start=1):
        if idx % 100 == 0 or idx == len(json_files):
            print(f"  進度：{idx}/{len(json_files)}")

        data = load_json(json_path)
        labels = extract_labels(data)
        image_name = data.get("image_name") if isinstance(data, dict) else json_path.stem

        for desc, score in labels:
            label_counter[desc] += 1

            if score is not None:
                score = float(score)
                label_score_sum[desc] += score
                label_score_max[desc] = max(label_score_max.get(desc, 0), score)

            label_image_set[desc].add(image_name)

    records = []
    for label, cnt in label_counter.most_common():
        image_count = len(label_image_set[label])
        avg_score = label_score_sum[label] / cnt if cnt > 0 else None
        max_score = label_score_max[label] if label in label_score_max else None

        records.append(
            {
                "label": label,
                "count": cnt,
                "image_count": image_count,
                "avg_score": avg_score,
                "max_score": max_score,
            }
        )

    df = pd.DataFrame(records).sort_values(by="count", ascending=False).reset_index(drop=True)
    return df


# =============== 主程式開始 ===============
def main():
    print("[INFO] 開始執行 Vision Label 分析...")

    df = analyze_labels(INPUT_DIR)
    if df.empty:
        print("[INFO] 無資料可分析")
        return

    csv_path = OUTPUT_DIR / f"{topic}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] CSV 輸出 → {csv_path}")


    print("[INFO] 任務完成！")


if __name__ == "__main__":
    main()
