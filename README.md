# 🧠 NLP Course Final Project

Final project for the **Natural Language Processing (NLP)** course at *Guilan University*.

---

## 📘 Project Overview

This project focuses on building a **Question Answering (QA)** system in Persian using **Transformer-based models**.  
It involves fine-tuning large language models, performing **Retrieval-Augmented Generation (RAG)**, and evaluating the system using multiple metrics.

---

## ⚙️ Key Components

### 1. Fine-tuning
- Base model: **Llama-3.2-1B-bnb-4bit**
- Fine-tuned on datasets:
  - [PQuAD](https://huggingface.co/datasets/Gholamreza/pquad)
  - [PersianQA](https://huggingface.co/datasets/SajjadAyoubi/persian_qa)
- Used **LoRA / QLoRA** to reduce GPU memory usage.

### 2. Embedding Models
- [`paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
- [`distiluse-base-multilingual-cased-v2`](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2)
- [`multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base)

### 3. Retrieval Methods
- **BM25**, **TF-IDF**, and **Semantic Search**
- Vector databases: `FAISS`, `Chroma`, `LanceDB`

---

## 📊 Evaluation Metrics
| Metric | Description |
|---------|--------------|
| EM | Exact Match |
| F1 | Token-level overlap |
| MRR | Mean Reciprocal Rank |
| Hit@k | Correct answer in top-k |
| Cosine Similarity | Semantic similarity between model and ground truth |

---

## 💻 Tools and Libraries
- Python 3.10+
- `transformers`, `sentence-transformers`
- `faiss-cpu`, `chromadb`, `lancedb`
- `scikit-learn`, `numpy`, `pandas`, `matplotlib`
- Optional UI: **Streamlit** or **Gradio**

---

## 📂 Project Files
| File | Description |
|------|--------------|
| `A.ipynb` | Fine-tuning experiments |
| `BC.ipynb` | Evaluation and embedding comparisons |
| `report.pdf` | Full project documentation |
| `README.md` | This file |

---

## 📚 References
- [Fine-Tuning Embeddings for RAG](https://medium.com/llamaindex-blog/fine-tuning-embeddings-for-rag-with-synthetic-data-e534409a3971)
- [Llama Model on Hugging Face](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)

---

## ✨ Author
**Faryafth**  
**AmirHossein Mortezaei**  
Final Project, NLP Course — Guilan University, 1403–1404

