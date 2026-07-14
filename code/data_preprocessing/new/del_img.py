import pandas as pd
import os

# ================= 參數設定 =================
folder_path = 'data\\init_data\\tourism_images_filtered_0423'  # 替換成你的圖片資料夾路徑
csv_path = 'data\init_data\prepocessing_data\image_descriptions\image_descriptions_all.csv'         # 替換成你的原始 CSV 檔案路徑
output_csv_path = 'data\\init_data\\images_filtereded_final.csv' # 處理後的 CSV 輸出路徑

# 替換成你 CSV 中用來記錄圖片名稱的「欄位名稱」
target_column = 'Image Name'  
# ============================================

def clean_csv_by_images():
    # 1. 獲取資料夾內的所有圖片名稱清單
    # 這裡預設抓取包含副檔名的完整檔名 (例如: 'cat_01.jpg')
    if not os.path.exists(folder_path):
        print(f"找不到資料夾: {folder_path}")
        return
        
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    print(f"已從資料夾讀取 {len(image_files)} 個圖片名稱。")

    # 2. 讀取 CSV 檔案
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"找不到 CSV 檔案: {csv_path}")
        return

    if target_column not in df.columns:
        print(f"錯誤: CSV 中找不到 '{target_column}' 欄位。")
        return

    original_count = len(df)

    # 3. 核心邏輯：進行比對與刪除
    # df[target_column].isin(image_files) 會找出有對應到的列
    # 前面的波浪號 (~) 代表反轉條件 (Not in)，也就是保留「不在」圖片清單中的資料
    df_cleaned = df[df[target_column].isin(image_files)]

    # 4. 輸出處理後的資料
    df_cleaned.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    deleted_count = original_count - len(df_cleaned)
    print("="*30)
    print(f"處理完成！")
    print(f"原始資料: {original_count} 筆")
    print(f"已刪除對應的資料: {deleted_count} 筆")
    print(f"剩餘資料: {len(df_cleaned)} 筆")
    print(f"檔案已儲存至: {output_csv_path}")

if __name__ == '__main__':
    clean_csv_by_images()