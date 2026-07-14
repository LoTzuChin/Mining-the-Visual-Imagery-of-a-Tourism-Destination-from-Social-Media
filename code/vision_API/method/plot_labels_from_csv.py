# -*- coding: utf-8 -*-
"""
從 label_stats.csv 產生：
1. Top-K label 長條圖
2. 文字雲 (Word Cloud)
"""

from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # 使用非 GUI 後端，避免在伺服器或無顯示環境出錯
import matplotlib.pyplot as plt

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception as e:
    WORDCLOUD_AVAILABLE = False
    print("[INFO] wordcloud 未安裝：", e)


# ====== 你可以依照實際情況改這兩行 ======
# CSV_PATH = Path(r"D:\Social_Media_Analysis\data\vision_method_output\la_fre_analysis\label_stats.csv")

topic = "topic_12"
BASE_DIR = Path(r"data\vision_method_output\la_fre_analysis")
CSV_PATH = BASE_DIR / "label_stats"/ f"{topic}.csv"

OUTPUT_DIR = BASE_DIR / "png_output" / topic
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = 20
# ==========================================


def plot_topk_bar(df, output_path: Path, top_k: int = 20):
    if df.empty:
        print("[WARN] DataFrame 為空，跳過長條圖")
        return

    try:
        top_df = df.head(top_k)
        labels = top_df["label"].tolist()
        counts = top_df["count"].tolist()

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.xlabel("Label")
        plt.ylabel("Frequency")
        plt.title(f"Top {top_k} Google Vision Labels")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"[INFO] 長條圖輸出成功 → {output_path}")
    except Exception as e:
        print(f"[ERROR] 無法輸出長條圖: {output_path}")
        print("原因：", e)


def generate_wordcloud(df, output_path: Path, max_words: int = 100):
    if not WORDCLOUD_AVAILABLE:
        print("[INFO] 未安裝 wordcloud，略過文字雲產生")
        return

    if df.empty:
        print("[WARN] DataFrame 為空，跳過文字雲")
        return

    try:
        top_df = df.head(max_words)
        freq_dict = {row["label"]: row["count"] for _, row in top_df.iterrows()}

        wc = WordCloud(
            width=1600,
            height=800,
            background_color="white"
        ).generate_from_frequencies(freq_dict)

        plt.figure(figsize=(16, 8))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        print(f"[INFO] 文字雲輸出成功 → {output_path}")
    except Exception as e:
        print(f"[ERROR] 無法輸出文字雲: {output_path}")
        print("原因：", e)


def main():
    if not CSV_PATH.exists():
        print(f"[ERROR] 找不到 CSV 檔：{CSV_PATH}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 讀取 CSV：{CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    # 確認必要欄位存在
    if "label" not in df.columns or "count" not in df.columns:
        print("[ERROR] CSV 少了必要欄位 'label' 或 'count'")
        print(f"目前欄位：{list(df.columns)}")
        return

    bar_path = OUTPUT_DIR / f"label_top{TOP_K}_bar.png"
    wc_path = OUTPUT_DIR / "label_wordcloud.png"

    plot_topk_bar(df, bar_path, top_k=TOP_K)
    generate_wordcloud(df, wc_path, max_words=100)

    print("[INFO] 完成所有圖像輸出")


if __name__ == "__main__":
    main()
