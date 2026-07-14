import pandas as pd

# 讀取兩個 CSV 檔案
df1 = pd.read_csv('data\init_data\prepocessing_data\image_descriptions\images_filtereded_final.csv')
df2 = pd.read_csv('data\init_data\prepocessing_data\image_descriptions\image_descriptions_twitter.csv')

# 合併資料
df_all = pd.concat([df1, df2], ignore_index=True)

# 儲存合併後的 CSV
df_all.to_csv('data\init_data\prepocessing_data\image_descriptions\image_descriptions_ig_twitter.csv', index=False)