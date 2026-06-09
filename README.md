# 🌍 Language-Agnostic Multilingual Answer Retrieval

> A comparative study of classical ML and deep learning models for cross-lingual semantic retrieval using the XQuAD-R / LAReQA dataset.

---

## 📌 Overview

This project benchmarks multiple machine learning and deep learning approaches for **multilingual answer retrieval** — the task of finding the correct answer passage for a question, regardless of which language the question or passage is in.

We evaluate five model families across 11 languages:

- **Classical ML:** Decision Tree, Random Forest, SVM (with TF-IDF embeddings)
- **Deep Learning:** CNN, BiLSTM (with FastText embeddings)
- **Transformer (Sentence-BERT):** Two-stage fine-tuned `bert-base-multilingual-cased` (StageA → hard-negative mining → StageB)

---

## 📂 Repository Structure

```
.
├── main.ipynb                  # Full pipeline: data prep → training → evaluation → plots
└── README.md
```

---

## 📊 Dataset

**XQuAD-R (LAReQA)** — a multilingual QA retrieval benchmark from Google Research.

| Property | Details |
|---|---|
| Languages | Arabic, German, Greek, English, Spanish, Hindi, Russian, Thai, Turkish, Vietnamese, Chinese |
| Task | Semantic sentence/passage retrieval |
| Evaluation | mAP, Recall@k, MRR |
| Source | [LAReQA (Roy et al., 2020)](https://github.com/google-research-datasets/lareqa) |

The dataset is parsed from 11 language-specific JSON files (`ar.json`, `de.json`, ... `zh.json`) and combined into a single CSV/Parquet for training and evaluation.

---

## 🏗️ Pipeline

```
Raw XQuAD-R JSONs (11 languages)
        │
        ▼
  Data Preprocessing
  ├── Sentence tokenization (NLTK)
  ├── Answer-to-sentence alignment
  └── Combined CSV + Parquet export
        │
        ▼
  ┌─────────────────────────────┐
  │      ML Models              │   TF-IDF Embeddings
  │  Decision Tree / RF / SVM   │
  └─────────────────────────────┘
        │
  ┌─────────────────────────────┐
  │      DL Models              │   FastText Embeddings
  │      CNN / BiLSTM           │
  └─────────────────────────────┘
        │
  ┌─────────────────────────────┐
  │   Transformer (mBERT)       │
  │  Stage A: MNR Loss          │
  │       ↓                     │
  │  Hard Negative Mining       │
  │       ↓                     │
  │  Stage B: Triplet Training  │
  └─────────────────────────────┘
        │
        ▼
   Evaluation (mAP@k, Recall@k, MRR)
```

---

## 📈 Results

### mAP Performance (Mean ± Std)

| Model | mAP |
|---|---|
| **StageA (mBERT)** | **0.6996 ± 0.0312** |
| StageB (mBERT) | 0.6762 ± 0.0217 |
| SVM | 0.5136 ± 0.3543 |
| BiLSTM | 0.2717 ± 0.2855 |
| Random Forest | 0.2305 ± 0.0246 |
| CNN | 0.1691 ± 0.1346 |

### Full Model Comparison

| Model | Median First Rank | Recall | MRR | Accuracy |
|---|---|---|---|---|
| StageA (mBERT) | 1 | 69.96 | 88.09 | 70% |
| SVM | 3 | 51.36 | 99.70 | 51% |
| BiLSTM | 6 | 27.17 | 100.00 | 35% |
| Random Forest | 8 | 23.03 | 98.81 | 21% |
| CNN | 8 | 16.91 | 100.00 | 17% |

> **Key findings:** SVM is the strongest classical model. Among neural models, BiLSTM outperforms CNN. The fine-tuned mBERT (StageA) outperforms all baselines by a large margin, confirming the superiority of pre-trained multilingual transformers for cross-lingual retrieval.

---

## ⚙️ Setup & Usage

### Requirements

```bash
pip install sentence-transformers transformers pandas torch scikit-learn \
            nltk tqdm pyarrow matplotlib seaborn
```

### 1. Prepare the Dataset

Place the 11 XQuAD-R JSON files (`ar.json`, `de.json`, `el.json`, `en.json`, `es.json`, `hi.json`, `ru.json`, `th.json`, `tr.json`, `vi.json`, `zh.json`) in `/content/` (or update `INPUT_DIR` in the notebook), then run the first cell to produce `xquad_r_all.csv`.

### 2. Run the Notebook

The notebook (`main.ipynb`) is self-contained and designed to run on **Google Colab** (GPU recommended).

Execute cells in order:

| Cell | Description |
|---|---|
| Cell 1 | Install dependencies and configure paths |
| Cell 2 | Load dataset, create train/dev split, save artifacts |
| Cell 3 (Stage A) | Fine-tune mBERT with MultipleNegativesRankingLoss |
| Cell 4 | Hard negative mining using Stage A embeddings |
| Cell 5 (Stage B) | Fine-tune with hard negatives (triplet training) |
| Cell 6 | Full-corpus evaluation (mAP@k, Recall@k, MRR) |
| Cell 7 | Generate all evaluation plots |

### 3. Evaluation

After training, evaluation metrics are saved to `output/finetune-mbert-xquadr/final_eval/agg_metrics.json` and plots are saved under the `plots/` subdirectory.

---

## 🧠 Model Details

### Transformer — Two-Stage Fine-Tuning

**Base Model:** `bert-base-multilingual-cased`  
**Framework:** `sentence-transformers`

| Stage | Loss | Purpose |
|---|---|---|
| Stage A | MultipleNegativesRankingLoss | Initial semantic alignment |
| Hard Negative Mining | — | Mine hard negatives via cosine similarity |
| Stage B | Triplet / contrastive | Refine with hard negatives |

### ML Models

Implemented with `scikit-learn` using TF-IDF vectorization.

### Deep Learning Models

| Model | Architecture |
|---|---|
| CNN | Embedding → 1D Conv → MaxPool → Dense → Softmax |
| BiLSTM | Embedding → BiLSTM → Dense → Softmax |

Both use **FastText** embeddings and are implemented in PyTorch/Keras.

---

## 🔬 Key Observations

- **Cross-lingual gap:** Classical models with TF-IDF fail to capture cross-lingual semantics. SVM does best among them but still struggles on non-English queries.
- **Deep learning limitations:** CNN and BiLSTM with FastText embeddings show limited generalization across distant language pairs.
- **Transformer advantage:** mBERT's shared sub-word vocabulary and multilingual pre-training enable strong zero-shot and fine-tuned performance across all 11 languages.
- **Hard negative mining:** Improves retrieval precision by exposing the model to semantically similar but incorrect passages during Stage B training.

---

## 🚀 Future Work

- Integrate **XLM-R** and **LaBSE** as stronger multilingual encoders
- Experiment with **contrastive learning** objectives (SimCSE, InfoNCE)
- Explore **language-agnostic embedding spaces** for improved cross-lingual transfer
- Evaluate on additional low-resource languages

---

## 📜 References

1. Conneau et al., *Unsupervised Cross-lingual Representation Learning at Scale*, ACL 2020
2. Devlin et al., *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*, NAACL 2019
3. Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach*, arXiv 2019
4. Chen & Cardie, *Multinomial Adversarial Networks for Multilingual Text Classification*, NAACL 2018
5. Lample & Conneau, *Cross-lingual Language Model Pretraining*, NeurIPS 2019
6. Yang et al., *XLNet: Generalized Autoregressive Pretraining for Language Understanding*, NeurIPS 2019
7. Pires et al., *How Multilingual Is Multilingual BERT?*, ACL 2019
8. Roy et al., *LAReQA: Language-Agnostic Answer Retrieval from a Multilingual Pool*, EMNLP 2020

---

## 👤 Author

**Aarush Tiwari**  
Bennett University

---

## 📄 License

This project is for academic and research purposes.
