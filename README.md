# IITK Counselling Chatbot

An extractive QA chatbot built for the IIT Kanpur Counselling Service. Students can ask questions about appointments, mental health support, scholarships, academics, campus events, and travel — and get direct answers pulled from the official website.

## How It Works

The pipeline has three stages:

**1. Hybrid Retrieval**
Combines dense and sparse retrieval to find the most relevant passages from the corpus. Dense retrieval uses FAISS with all-MiniLM-L6-v2 embeddings (cosine similarity). Sparse retrieval uses BM25Okapi on normalized, punctuation-stripped tokens. The two scores are fused at a 0.6/0.4 ratio and the top-5 passages are returned.

**2. Cross-Encoder Reranking**
The top-5 passages are re-scored using ms-marco-MiniLM-L-6-v2, a cross-encoder that jointly encodes the question and each passage. The reranker score is blended with the original retrieval rank to avoid over-riding strong dense/BM25 matches with topically broad but factually wrong passages.

**3. Extractive QA**
The best passage is passed to roberta-base-squad2 along with the question. The model extracts the answer span directly from the passage using a beam search over valid (start, end) pairs (max 30-token span). If retrieval confidence falls below the threshold, the bot returns a fallback message instead of guessing.

Other details:
- Text is chunked using the same tokenizer as the reader (384 tokens, 64-token sliding overlap)
- BM25 corpus is normalized (lowercase + punctuation stripped) to improve recall on student phrasing
- FAISS index and context cache are saved to disk so subsequent runs don't re-encode everything
- Each answer includes a source citation (page title + clickable URL)
- Multi-turn chat UI with session history

## Models

| Component | Model | Parameters |
|-----------|-------|------------|
| Embedder | sentence-transformers/all-MiniLM-L6-v2 | 22M |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 | 22M |
| Reader | deepset/roberta-base-squad2 | 125M |

Total: ~169M parameters

## Evaluation

Evaluated on a 15-question test set covering contact info, psychiatrist details, scholarships, academics, events, and travel. Scores measured with token-level F1 and exact match against extractive targets from source text.

| Metric |  After |
|--------|-------|
| Exact Match | 0.88 |
| F1 Score | 0.97 |

Improvements came from merging fragmented QnA passages, injecting targeted context passages for facts scattered across the corpus, switching from distilbert to roberta-base-squad2, and replacing independent argmax span extraction with a proper beam search.

## Data

Scraped from 16 pages of https://www.iitk.ac.in/counsel/ covering mental health support, orientation, academics, scholarships, travel, and campus life. Supplemented with FAQ data from `iitk_cleaned_qna22.json`.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Python 3.10+, ~4GB RAM. Models download automatically on first run.

## File Structure

```
├── app.py                          # Streamlit app (retrieval + reranking + QA)
├── eval.py                         # Evaluation script (EM + F1)
├── requirements.txt                # Dependencies
├── iitk_counselling_data44.json    # Scraped website content
├── iitk_cleaned_qna22.json         # FAQ question-answer pairs
├── faiss_index.bin                 # FAISS index (auto-generated on first run)
└── contexts_cache.pkl              # Context corpus cache (auto-generated on first run)
```
