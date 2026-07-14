# Mining the Visual Imagery of a Tourism Destination from Social Media
Title
An Automated Computer Vision and Machine Learning Framework with Evidence from Taiwan

## Description

This repository contains the complete source code for reproducing the experiments presented in the study: "Mining the Visual Imagery of a Tourism Destination from Social Media: An Automated Computer Vision and Machine Learning Framework with Evidence from Taiwan."

The project develops an end-to-end visual content-mining framework applied to 5,031 unique public images tagged #ILoveTaiwan on Instagram and Twitter. The Gemini API translates each image into a descriptive narrative; the Google Cloud Vision API extracts semantic labels and color data. BERTopic is then applied to the image descriptions to derive 12 macro-level visual themes through topic modeling and expert content analysis. Four supervised classification algorithms — Support Vector Machines (SVM), Random Forests (RF), Extreme Gradient Boosting (XGBoost), and Deep Neural Networks (DNN) — are compared for automated multi-label theme classification. The repository also contains scripts for label co-occurrence analysis and color distribution analysis across visual themes.

---

## Dataset Information

The dataset consists of 5,031 unique public images tagged #ILoveTaiwan, collected from Instagram and Twitter via Phantombuster, covering posts published between January 2019 and April 2025. Due to platform terms of service and privacy considerations, the raw image data is not included in this repository.

The dataset should be organized as follows:

```
data/
├── init_data/
│   ├── original_data/
│   │   └── img_ig_with_twitter/          # Raw social media images
│   └── prepocessing_data/
│       └── image_descriptions/
│           └── image_descriptions_ig_twitter.csv   # Raw Gemini / Vision API outputs
├── topic_output/
│   └── output_bertopic/
│       ├── bertopic_labels.csv                # Single-label topic assignments
│       └── bertopic_labels_multilabel.csv     # Multi-label assignments (used for classification)
└── model_output/
    └── new_multi_output/
        ├── DNN/
        ├── RF/
        ├── SVM/
        └── XGB/

---

## Code Information

```
code/
├── data_preprocessing/
│   ├── describe_images.py       # Batch image description via Google Gemini API
│   ├── cleaned_text.py          # NLP preprocessing pipeline
│   ├── merge_img_describe.py    # Merge image descriptions with metadata
│   └── new/
│       └── del_img.py           # Image filtering utilities
│
├── vision_API/
│   ├── describe_vision.py       # Google Cloud Vision API: label and color extraction
│   ├── picture_classification.py
│   ├── mergy_old_json.py
│   └── method/
│       ├── analyze_dominant_colors.py
│       ├── analyze_label_cooccurrence.py
│       ├── label_frequency_analysis.py
│       └── plot_labels_from_csv.py
│
├── topic/
│   ├── train_bertopic.py        # Train BERTopic model
│   ├── train_LDA.py             # Train LDA model with coherence evaluation
│   ├── docTopic_BERTopic.py     # Assign BERTopic-derived theme labels to documents
│   ├── docTopic_LDA.py          # Assign LDA topic labels to documents
│   └── topic_data_collation.py
│
└── predict_model/
    ├── split_dataset.py         # Stratified 80/20 train/test split
    ├── train_svm.py             # LinearSVC classifier
    ├── train_rf.py              # Random Forest classifier
    ├── train_xgb.py             # XGBoost classifier
    ├── train_dnn.py             # DNN classifier (Keras)
    ├── model_test_tf-idf.py     # Unified model evaluation
    └── new_model/               # Multi-label classifiers used in the paper
        ├── new_svm.py
        ├── new_rf.py
        ├── new_xgboost.py
        ├── new_dnn.py
        ├── new_model_test_tf-idf.py
        └── topic_data_collation.py  # Multi-label assignment + train/val/test split generation
```

The multi-label scripts output to `data/model_output/new_multi_output/<MODEL_NAME>/`:

```
<MODEL_NAME>/
  *_predictions.csv
  *_report.txt
  *_confusion_matrix.csv
  *_history.json        (DNN only)
  *.joblib / *.h5 / *.keras
```

---

## Usage Instructions

1. Place image data in the folder structure described above.
2. Place your Google Cloud Vision service account key JSON and `secrets.json` (containing `GOOGLE_API_KEY` for Gemini) in the project root. Both files are excluded from version control.
3. Install dependencies.
4. Run the pipeline in order.

**Install dependencies:**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Step 1 — Extract visual features:**
```bash
python code/vision_API/describe_vision.py
python code/data_preprocessing/describe_images.py
```

**Step 2 — Preprocess text descriptions:**
```bash
python code/data_preprocessing/cleaned_text.py
python code/data_preprocessing/merge_img_describe.py
```

**Step 3 — Topic modeling:**
```bash
python code/topic/train_bertopic.py
python code/topic/docTopic_BERTopic.py
```

**Step 4 — Generate multi-label assignments and split:**
```bash
python code/predict_model/new_model/topic_data_collation.py
```

**Step 5 — Train multi-label classifiers:**
```bash
python code/predict_model/new_model/new_svm.py
python code/predict_model/new_model/new_rf.py
python code/predict_model/new_model/new_xgboost.py
python code/predict_model/new_model/new_dnn.py
```

> For Windows users, set `OMP_NUM_THREADS=1` as an environment variable if multiprocessing issues arise during training.

---

## Methodology

Images are described in natural language using the Gemini API. The Google Cloud Vision API independently extracts semantic labels (confidence threshold ≥ 0.5) and dominant color data (RGB values and pixel fractions) for each image.

Text descriptions are cleaned through an NLP pipeline: tokenization, stopword removal (NLTK), bigram/trigram phrase detection (Gensim), lemmatization and POS filtering — retaining only nouns, adjectives, and adverbs (spaCy `en_core_web_sm`) — and document frequency filtering.

BERTopic (`paraphrase-multilingual-MiniLM-L12-v2` embeddings) is applied to the cleaned descriptions, initially yielding 90 micro-topics. These are consolidated into 12 macro-level tourism visual themes through coherence score evaluation and expert content analysis. Multi-label ground truth is derived from BERTopic's topic–document probability distributions (threshold = 0.05), yielding 4,902 labeled images for the classification task.

All classifiers use TF-IDF features (word n-grams and character n-grams) fitted on the training partition only. The dataset is split 80/20 using a fixed random seed to ensure reproducibility across all models.

---

## Requirements

Recommended environment:

- Python >= 3.10
- TensorFlow >= 2.10.0

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
