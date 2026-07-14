import shutil
from pathlib import Path

def comprehensive_vision_sync(base_path_str: str):
    base_dir = Path(base_path_str)
    history_dir = base_dir / "history"
    valid_img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff'}
    
    # 1. 建立全局 JSON 索引 (解決檔案錯置問題)
    # 格式: { 'json_filename': Path_to_source_json }
    print("正在掃描 history 資料夾建立全局索引...")
    global_json_map = {}
    for json_path in history_dir.rglob("*.json"):
        if "vision_description" in json_path.parts:
            global_json_map[json_path.name] = json_path

    # 用於統計與追蹤
    matched_count = 0
    missing_images = []
    used_jsons = set()
    
    # 找出所有要處理的目標資料夾 (vision_x)
    vision_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("topic_") and d.name != "history"]

    print(f"開始處理 {len(vision_dirs)} 個分類資料夾...")

    for v_dir in vision_dirs:
        target_json_dir = v_dir / "vision_description"
        target_json_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in v_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in valid_img_exts:
                # 根據規則推算預期的 JSON 名稱
                expected_json_name = f"{img_path.stem}_vision_results.json"
                
                # 從全局索引中尋找
                if expected_json_name in global_json_map:
                    source_path = global_json_map[expected_json_name]
                    target_path = target_json_dir / expected_json_name
                    
                    # 複製檔案
                    shutil.copy2(source_path, target_path)
                    
                    matched_count += 1
                    used_jsons.add(expected_json_name)
                else:
                    missing_images.append(f"{v_dir.name}/{img_path.name}")

    # 2. 找出「孤兒 JSON」：存在於 history 但沒有對應到任何現有圖片的 JSON
    all_history_jsons = set(global_json_map.keys())
    orphaned_jsons = all_history_jsons - used_jsons

    # --- 最終稽核報告 ---
    print("\n" + "="*50)
    print("📊 執行結果與數據稽核報告")
    print("="*50)
    print(f"✅ 成功配對並複製: {matched_count} 件")
    print(f"❓ 圖片無 JSON (正常現象): {len(missing_images)} 件")
    print(f"⚠️ 孤兒 JSON (有 JSON 但找不到圖片): {len(orphaned_jsons)} 件")
    print("-"*50)

    if orphaned_jsons:
        print("\n🔎 以下 JSON 在 history 中存在，但在 vision_x 資料夾中找不到對應圖片：")
        print("   (這可能是因為圖片被刪除、改名，或尚未被分類到 vision_x 中)")
        for o_json in sorted(list(orphaned_jsons))[:10]: # 只列前10筆
            print(f"   - {o_json} (來源: {global_json_map[o_json].relative_to(base_dir)})")
        if len(orphaned_jsons) > 10:
            print(f"   ... 以及其他 {len(orphaned_jsons)-10} 個檔案")

    if missing_images:
        print("\n📝 圖片無對應 JSON 的範例 (前 5 筆):")
        for m_img in missing_images[:5]:
            print(f"   - {m_img}")
    
    print("\n" + "="*50)
    print("處理完成。")

if __name__ == "__main__":
    # 設定根目錄
    DATA_ROOT = "data/vision_data" 
    comprehensive_vision_sync(DATA_ROOT)