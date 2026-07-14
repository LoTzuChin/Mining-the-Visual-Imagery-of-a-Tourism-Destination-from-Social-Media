#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 批次呼叫 Google Gemini 產生圖片描述 (升級全新 google-genai SDK 版本)

import json
import mimetypes
import os
from pathlib import Path
import time
import csv
from typing import List, Dict, Set

# === 改用全新的 SDK import 方式 ===
from google import genai
from google.genai import types
from google.genai.errors import APIError

def load_api_key_from_json(json_path: Path) -> str:
    if not json_path.exists():
        raise FileNotFoundError(f"Secrets file not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = data.get("GOOGLE_API_KEY")
    if not key:
        raise ValueError("\"GOOGLE_API_KEY\" not found in secret.json")
    return key

def detect_mime_type(file_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(file_path))
    return mime or "application/octet-stream"

def list_image_files(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tiff"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    files.sort(key=lambda p: p.name.lower())
    return files

def get_processed_images(csv_path: Path) -> Set[str]:
    """讀取現有的 CSV，回傳已經處理過的圖片名稱集合。"""
    processed = set()
    if not csv_path.exists():
        return processed
        
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Image Name") and row.get("Description", "").strip():
                processed.add(row["Image Name"])
    return processed

def describe_images_and_save(client: genai.Client, model_name: str, images: List[Path], prompt: str, out_csv: Path) -> None:
    """使用指定模型產生描述，並即時寫入 CSV 檔案。"""
    file_exists = out_csv.exists()
    
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Image Name", "Description"])
        if not file_exists:
            writer.writeheader()

        for idx, img_path in enumerate(images, start=1):
            mime_type = detect_mime_type(img_path)
            with img_path.open("rb") as img_file:
                image_bytes = img_file.read()

            # === 新 SDK 封裝圖片資料的方式 ===
            image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # === 新 SDK 呼叫 API 的方式 ===
                    resp = client.models.generate_content(
                        model=model_name,
                        contents=[prompt, image_part]
                    )
                    text = (resp.text or "").strip()
                    
                    # 即時寫入 CSV
                    writer.writerow({"Image Name": img_path.name, "Description": text})
                    f.flush() 
                    
                    print(f"[ok] ({idx}/{len(images)}) {img_path.name}")
                    time.sleep(1) # 避免打太快
                    break 
                    
                except APIError as e:
                    # 新 SDK 有專屬的 APIError 可以捕捉
                    error_msg = str(e)
                    if e.code == 429 or "Quota" in error_msg:
                        wait_time = 60
                        print(f"[warning] 觸發 API 限制，暫停 {wait_time} 秒後重試 (第 {attempt+1}/{max_retries} 次重試)...")
                        time.sleep(wait_time)
                    else:
                        print(f"[fail] {img_path.name}: {e}")
                        writer.writerow({"Image Name": img_path.name, "Description": f"ERROR: {e}"})
                        f.flush()
                        break 
                except Exception as e:
                    print(f"[fail] {img_path.name} (未知的系統錯誤): {e}")
                    writer.writerow({"Image Name": img_path.name, "Description": f"ERROR: {e}"})
                    f.flush()
                    break 

if __name__ == "__main__":
    # === 參數設定 ===
    secrets_path = Path("secrets.json")
    input_folder = Path("data\init_data\Twitter_0423")
    output_csv = Path("data/init_data/prepocessing_data/image_descriptions/image_descriptions_twitter.csv")
    
    # 移除舊的 "models/" 前綴，直接寫模型名稱 (推薦使用最新版 2.5)
    model_name = "gemini-2.5-flash" 
    prompt = (
        "Describe the image in detail, focusing on the main objects and their "
        "context. Provide a fully descriptive response without using bullet points "
        "and in English."
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1) 初始化新版 Client
    api_key = load_api_key_from_json(secrets_path)
    client = genai.Client(api_key=api_key)

    # 2) 掃描並過濾圖片
    all_image_paths = list_image_files(input_folder)
    processed_set = get_processed_images(output_csv)
    pending_images = [p for p in all_image_paths if p.name not in processed_set]
    
    print(f"[info] 資料夾總共: {len(all_image_paths)} 張圖片")
    print(f"[info] 已完成略過: {len(processed_set)} 張圖片")
    print(f"[info] 本次待處理: {len(pending_images)} 張圖片")
    print("-" * 30)

    if not pending_images:
        print("[done] 所有圖片都已處理完畢！")
    else:
        # 3) 執行推論
        describe_images_and_save(client, model_name, pending_images, prompt, output_csv)
        print(f"[done] 本次任務執行結束。")