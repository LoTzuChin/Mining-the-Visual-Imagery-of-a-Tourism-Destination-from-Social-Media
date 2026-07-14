import os
import pandas as pd
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# === 路徑設定（請改成你自己的） ===
input_csv = "D:\Social_Media_Analysis\prepocessing_data\cleaned_texts.csv"
text_col = "cleaned_text"
id_col     = "image_name"   # 要輸出的欄位
model_path = "D:\\Social_Media_Analysis\\lda_out\\best_lda_K10_coh0.5948.gensim"
output_dir = "D:\Social_Media_Analysis\lda_out"
os.makedirs(output_dir, exist_ok=True)

def simple_tokenize(text: str):
    return [t for t in str(text).strip().split() if t]

# 1) 載入資料
df = pd.read_csv(input_csv, encoding="utf-8-sig")
texts_tokens = [simple_tokenize(t) for t in df[text_col].astype(str).fillna("")]

# 2) 載入模型（含訓練時的字典 id2word）
lda = LdaModel.load(model_path)
dictionary = lda.id2word  # 這很關鍵：用訓練時的字典！

# 3) 用訓練字典把當前文本轉成 bow（未出現在訓練字典的詞會被忽略）
corpus = [dictionary.doc2bow(doc) for doc in texts_tokens]

# 4) 推論主題分布並輸出 CSV
rows = []
for i, bow in enumerate(corpus):
    dist = lda.get_document_topics(bow, minimum_probability=0.0)
    dominant_topic, top_prob = max(dist, key=lambda x: x[1])
    row = {
        id_col: df.iloc[i][id_col],     # 用 image_path 取代 doc_index
        "dominant_topic": int(dominant_topic),
        "topic_prob": float(top_prob),
    }
    # 展開完整分布
    for topic_id, prob in dist:
        row[f"topic_{topic_id}"] = float(prob)
    rows.append(row)

df_out = pd.DataFrame(rows)
out_path = os.path.join(output_dir, "doc_topic.csv")
df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"[INFO] Saved document-topic distribution -> {out_path}")