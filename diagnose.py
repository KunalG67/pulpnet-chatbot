import sys
sys.path.insert(0, ".")
from app import get_best_context, bert_answer, reranker
import numpy as np

questions = [
    "How often does Dr. Rohan Kumar visit the institute?",
    "What is the CPI criterion for SBF?",
    "Is the counselling service free for students?",
    "What is organized every year on Diwali by the Counselling Service?",
]

for q in questions:
    print("=" * 100)
    print("QUESTION:", q)
    result = get_best_context(q)
    if result is None:
        print("  -> RETRIEVAL MISS: no context cleared the 0.65 threshold.")
        continue

    top_contexts, score, top_metas = result
    print(f"  -> Retrieval fused score: {score:.3f}")
    print(f"  -> Top context preview: {top_contexts[0][:150]!r}")

    rerank_scores = reranker.predict([(q, ctx) for ctx in top_contexts])
    best_idx = int(np.argmax(rerank_scores))
    best_ctx = top_contexts[best_idx]
    print(f"  -> After rerank, chosen context: {best_ctx[:200]!r}")

    answer, conf = bert_answer(q, best_ctx)
    print(f"  -> Reader answer: {answer!r}  (confidence={conf:.4f})")


print("=" * 100)
print("RERANK DETAIL: Is the counselling service free for students?")
result = get_best_context("Is the counselling service free for students?")
top_contexts, score, top_metas = result
rerank_scores = reranker.predict([("Is the counselling service free for students?", ctx) for ctx in top_contexts])
for i, (ctx, rs) in enumerate(zip(top_contexts, rerank_scores)):
    print(f"  [{i}] rerank_score={rs:.4f}  text={ctx[:120]!r}")    