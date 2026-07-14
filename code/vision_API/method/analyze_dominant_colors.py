# analyze_dominant_color_tones.py
"""
從 Google Vision JSON 中抽取每張圖片的 Top-1 主色，
再依照色調分類（紅、橙、黃、綠、青綠、藍、紫、粉、黑、白、灰），
最後輸出 CSV + 圓餅圖。
"""

import json
import os
from pathlib import Path
from collections import Counter
from typing import Optional, Tuple, Dict, List
import colorsys
import matplotlib.pyplot as plt
import csv

# ======== 路徑設定（寫死）========
# 放 Vision JSON 檔的資料夾
topic = "topic_12"
BASE_DIR = Path(r"data\\vision_data")
JSON_DIR = BASE_DIR / topic / "vision_description"


# 輸出結果的資料夾
OUTPUT_BASE_DIR = Path(r"data\\vision_method_output\\an_do_colors")
OUTPUT_DIR = OUTPUT_BASE_DIR / topic
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 輸出 CSV 路徑
OUTPUT_CSV = OUTPUT_DIR / "dominant_colors_summary.csv"

# 輸出圓餅圖路徑
OUTPUT_PLOT = OUTPUT_DIR / "dominant_color_pie.png"




def ensure_output_dir(path: Path) -> None:
    """確保輸出資料夾存在。"""
    path.parent.mkdir(parents=True, exist_ok=True)


# ===================== 抽主色 =====================

def extract_top1_color_from_processed(data: Dict) -> Optional[Tuple[int, int, int]]:
    """
    從「你目前的 preprocessed JSON 格式」中抽出第一名主色：
    image_properties -> dominant_colors[0] -> color.{red, green, blue}
    有些檔案的 image_properties 可能是字串或 None，要先檢查型別。
    """
    image_props = data.get("image_properties")

    # 避免 image_props 是 str / list / None
    if not image_props or not isinstance(image_props, dict):
        return None

    dominant_colors = image_props.get("dominant_colors") or []
    if not dominant_colors:
        return None

    top = dominant_colors[0]
    if not isinstance(top, dict):
        return None

    color = top.get("color") or {}
    if not isinstance(color, dict):
        return None

    r = int(color.get("red", 0))
    g = int(color.get("green", 0))
    b = int(color.get("blue", 0))
    return (r, g, b)


def extract_top1_color_from_raw(data: Dict) -> Optional[Tuple[int, int, int]]:
    """
    若是直接存 Google Vision 原始回傳格式：
    imagePropertiesAnnotation -> dominantColors -> colors[0].color.{red, green, blue}
    """
    img_props = data.get("imagePropertiesAnnotation")
    if not img_props or not isinstance(img_props, dict):
        return None

    dominant = img_props.get("dominantColors") or {}
    if not isinstance(dominant, dict):
        return None

    colors_list = dominant.get("colors") or []
    if not colors_list:
        return None

    top = colors_list[0]
    if not isinstance(top, dict):
        return None

    color = top.get("color") or {}
    if not isinstance(color, dict):
        return None

    r = int(color.get("red", 0))
    g = int(color.get("green", 0))
    b = int(color.get("blue", 0))
    return (r, g, b)


def extract_top1_color(data: Dict) -> Optional[Tuple[int, int, int]]:
    """
    先試你的 preprocessed 格式，若沒有再試 Vision 原始格式。
    """
    color = extract_top1_color_from_processed(data)
    if color is not None:
        return color

    color = extract_top1_color_from_raw(data)
    return color


# ===================== RGB → 色調分類 =====================

def classify_color_tone(rgb: Tuple[int, int, int]) -> str:
    """
    把 RGB 分到幾個大色系：
    Black, White, Gray, Red, Orange, Yellow, Green, Cyan, Blue, Purple, Pink
    規則：先用 HSV 判斷黑白灰，再用 Hue 決定色系。
    """
    r, g, b = rgb
    rf, gf, bf = r / 255.0, g / 255.0, b / 255.0
    h, s, v = colorsys.rgb_to_hsv(rf, gf, bf)  # h in [0,1), s/v in [0,1]
    h_deg = h * 360.0

    # 先抓黑 / 白 / 灰（saturation 低）
    if v < 0.15:
        return "Black"
    if s < 0.12:
        if v > 0.85:
            return "White"
        else:
            return "Gray"

    # 再依照 Hue 分類
    if (h_deg >= 345) or (h_deg < 15):
        return "Red"
    elif h_deg < 45:
        return "Orange"
    elif h_deg < 65:
        return "Yellow"
    elif h_deg < 150:
        return "Green"
    elif h_deg < 210:
        return "Cyan"
    elif h_deg < 270:
        return "Blue"
    elif h_deg < 300:
        return "Purple"
    else:  # 300–345
        return "Pink"


# 給每個色調一個代表色，用來畫圓餅圖
TONE_COLOR_MAP = {
    "Black": (0.05, 0.05, 0.05),
    "White": (0.98, 0.98, 0.98),
    "Gray":  (0.6, 0.6, 0.6),
    "Red":   (0.9, 0.2, 0.2),
    "Orange": (0.95, 0.5, 0.1),
    "Yellow": (0.97, 0.9, 0.2),
    "Green": (0.2, 0.7, 0.3),
    "Cyan":  (0.1, 0.7, 0.8),
    "Blue":  (0.2, 0.3, 0.9),
    "Purple": (0.6, 0.3, 0.8),
    "Pink":  (0.95, 0.6, 0.8),
}

# 中文名稱（要寫報告時可以用）
TONE_ZH_MAP = {
    "Black": "Black",
    "White": "White",
    "Gray": "Gray",
    "Red": "Red",
    "Orange": "Orange",
    "Yellow": "Yellow",
    "Green": "Green",
    "Cyan": "Cyan",
    "Blue": "Blue",
    "Purple": "Purple",
    "Pink": "Pink",
}


# ===================== 掃描檔案並統計 =====================

def scan_json_files(json_dir: Path) -> Counter:
    """
    走訪資料夾中所有 .json，抽出每張圖的 Top-1 主色，轉成色調後統計。
    回傳 Counter: { tone: count }
    """
    counter: Counter = Counter()

    json_files: List[Path] = sorted(json_dir.glob("*.json"))
    print(f"[INFO] 在 {json_dir} 找到 {len(json_files)} 個 JSON 檔。")

    for idx, json_path in enumerate(json_files, start=1):
        try:
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] 無法讀取 JSON: {json_path}，錯誤：{e}")
            continue

        rgb = extract_top1_color(data)
        if rgb is None:
            # 如果想 debug 哪些檔案沒有主色，可以打開這行
            # print(f"[DEBUG] 無主色資訊：{json_path.name}")
            continue

        tone = classify_color_tone(rgb)
        counter[tone] += 1

        if idx % 500 == 0:
            print(f"[INFO] 已處理 {idx}/{len(json_files)} 個檔案...")

    return counter


# ===================== 輸出 CSV / 繪圖 =====================

def save_csv(counter: Counter, output_csv: Path) -> None:
    """
    把每個色調的統計結果存成 CSV。
    欄位：tone_en, tone_zh, count, fraction
    """
    ensure_output_dir(output_csv)

    total = sum(counter.values())
    rows = []

    for tone, cnt in counter.most_common():
        fraction = cnt / total if total > 0 else 0.0
        rows.append({
            "tone_en": tone,
            "tone_zh": TONE_ZH_MAP.get(tone, tone),
            "count": cnt,
            "fraction": fraction,
        })

    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["tone_en", "tone_zh", "count", "fraction"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] 已輸出 CSV：{output_csv}")


def plot_pie(counter: Counter, output_plot: Path) -> None:
    """
    把色調統計結果畫成圓餅圖。
    每一塊是「一種色調」，不會再細分到一堆 RGB。
    """
    ensure_output_dir(output_plot)

    if not counter:
        print("[WARN] 沒有任何色調資料，無法繪圖。")
        return

    tones = list(counter.keys())
    sizes = [counter[t] for t in tones]

    colors = []
    labels = []

    for tone in tones:
        colors.append(TONE_COLOR_MAP.get(tone, (0.8, 0.8, 0.8)))
        labels.append(TONE_ZH_MAP.get(tone, tone))

    plt.figure(figsize=(8, 8))
    plt.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
    )
    plt.title("Dominant Color Tone Distribution")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.close()

    print(f"[INFO] 已輸出圓餅圖：{output_plot}")


# ===================== main =====================

def main():
    if not JSON_DIR.exists():
        print(f"[ERROR] JSON 資料夾不存在：{JSON_DIR}")
        return

    counter = scan_json_files(JSON_DIR)

    total_images = sum(counter.values())
    print(f"[INFO] 共成功取得 {total_images} 張圖片的主色（色調）。")
    print(f"[INFO] 不同色調數量：{len(counter)}")
    print("[INFO] 色調分佈：")
    for tone, cnt in counter.most_common():
        frac = cnt / total_images if total_images > 0 else 0
        print(f"  - {tone} ({TONE_ZH_MAP.get(tone, tone)}): {cnt} ({frac:.1%})")

    if not counter:
        print("[WARN] 沒有資料，中止。")
        return

    save_csv(counter, OUTPUT_CSV)
    plot_pie(counter, OUTPUT_PLOT)


if __name__ == "__main__":
    main()
