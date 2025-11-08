\# Persian QA - NLP Project



This repository contains only the \*\*bonus part\*\* of the \*Speech and Language Processing\* course project (1403–1404).



\## 🌟 Goals



\- Fine-tune embedding model: `distiluse-base-multilingual-cased-v2`

\- Evaluate embedding quality using:

&nbsp; - `Cosine Similarity`

&nbsp; - `Mean Reciprocal Rank (MRR)`

\- Compare retrieval performance with:

&nbsp; - \*\*multilingual-e5-base\*\*

\- Test vector databases:

&nbsp; - `FAISS`, `Chroma`, `LanceDB`



\## 📊 Evaluation Metrics

| Metric | Description |

|---------|--------------|

| MRR | Measures ranking quality |

| Cosine Similarity | Semantic similarity |

| Precision / Recall / Hit@k | Retrieval effectiveness |



\## 🧠 Libraries

\- `sentence-transformers`

\- `faiss-cpu`

\- `chromadb`

\- `lancedb`

\- `scikit-learn`

\- `pandas`, `numpy`, `matplotlib`



\## 🚀 How to Run

```bash

pip install -r requirements.txt

python src/fine\_tune\_distiluse.py

python src/retrieval\_analysis.py



