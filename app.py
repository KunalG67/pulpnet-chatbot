import streamlit as st
import json
import pickle
import re
import string
import torch
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util, CrossEncoder
 
 
try:
    with open("iitk_cleaned_qna22.json", "r", encoding="utf-8") as f2:
        qna = json.load(f2)
 
        seen = set()
        qna_passages = []
        buffer = ""
        for pair in qna:
            a = pair.get("answer", "")
            try:
                a = a.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            a = a.strip()
            if not a:
                continue
            if buffer and not buffer[-1] in '.!?"':
                buffer = buffer + " " + a
            else:
                if len(buffer) >= 30 and buffer not in seen:
                    seen.add(buffer)
                    qna_passages.append(buffer)
                buffer = a
        if len(buffer) >= 30 and buffer not in seen:
            qna_passages.append(buffer)
        qna = qna_passages
 
    with open("iitk_counselling_data44.json", "r", encoding="utf-8") as f1:
        scraped = json.load(f1)
except FileNotFoundError as e:
    st.error(f"Data file missing: {e}")
    st.stop()
 
 
READER_MODEL_NAME = "deepset/roberta-base-squad2"
 
 
def chunk_text(text, max_tokens=384, overlap_tokens=64):
    if not hasattr(chunk_text, "_tokenizer"):
        chunk_text._tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    tokenizer = chunk_text._tokenizer
 
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks = []
    current_ids = []
 
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sent_ids = tokenizer.encode(sentence, add_special_tokens=False)
 
        if len(current_ids) + len(sent_ids) > max_tokens and current_ids:
            chunks.append(
                tokenizer.decode(current_ids, skip_special_tokens=True).strip()
            )
            if len(current_ids) >= overlap_tokens:
                current_ids = current_ids[-overlap_tokens:]
 
        current_ids.extend(sent_ids)
 
    if current_ids:
        chunks.append(
            tokenizer.decode(current_ids, skip_special_tokens=True).strip()
        )
 
    return chunks
 
 
def normalize_for_bm25(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()
 
 
index = None
contexts = []
context_meta = []
 
try:
    with open("contexts_cache.pkl", "rb") as f:
        contexts, context_meta = pickle.load(f)
    index = faiss.read_index("faiss_index.bin")
    if len(contexts) != len(context_meta):
        raise ValueError("cache corpus size mismatch, forcing rebuild")
except (FileNotFoundError, pickle.PickleError, Exception):
 
    contexts = []
    context_meta = []
    for item in scraped:
        if "content" in item:
            chunks = chunk_text(item["content"])
            contexts.extend(chunks)
            context_meta.extend([{"title": item["title"], "url": item["url"]}] * len(chunks))
 
    contexts.extend(qna)
    context_meta.extend(
        [{"title": "Counselling Service FAQ", "url": "https://www.iitk.ac.in/counsel/faq.php"}] * len(qna)
    )
 
    # FIX 2: inject hand-crafted passages for facts that get lost in fragmented text
    contact_passage = (
        "The Institute Counselling Service office timings are 11:00 AM - 7:00 PM (Mon to Fri). "
        "Phone: +91 512 2597784. "
        "Email for appointments: counselor@iitk.ac.in. "
        "Email for head: head_cs@iitk.ac.in. "
        "All registered undergraduate and postgraduate students can avail the service free of cost."
    )
    contexts.append(contact_passage)
    context_meta.append({"title": "Counselling Service Contact", "url": "https://www.iitk.ac.in/counsel/"})
 
    distance_passage = (
        "IIT Kanpur is about 17 km from Kanpur Central Railway Station. "
        "The journey takes about 45 minutes by auto-rickshaw or cab."
    )
    contexts.append(distance_passage)
    context_meta.append({"title": "Reaching IITK", "url": "https://www.iitk.ac.in/counsel/new-ug-information.php"})

    bajpai_passage = (
        "Dr. Alok Bajpai is a psychiatrist trained at NIMHANS, Bangalore. "
        "Dr. Bajpai visits the institute on a weekly basis."
    )
    contexts.append(bajpai_passage)
    context_meta.append({"title": "Counselling Service Team", "url": "https://www.iitk.ac.in/counsel/team.php"})

    free_passage = (
        "The counselling service is free of cost for all students. "
        "All registered undergraduate and postgraduate students can avail the service free of cost."
    )
    contexts.append(free_passage)
    context_meta.append({"title": "Counselling Service FAQ", "url": "https://www.iitk.ac.in/counsel/faq.php"})

    cpi_passage = (
        "There is no CPI criterion for applying for SBF scholarship."
    )
    contexts.append(cpi_passage)
    context_meta.append({"title": "SBF Scholarship", "url": "https://www.iitk.ac.in/counsel/SBF.php"})

    semester_passage = (
        "A semester load is defined as 36 credits."
    )
    contexts.append(semester_passage)
    context_meta.append({"title": "PG Information", "url": "https://www.iitk.ac.in/counsel/pg-information.php"})
 
 
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    model = AutoModelForQuestionAnswering.from_pretrained(READER_MODEL_NAME)
 
    if index is None:
        context_embeddings = embedder.encode(contexts, convert_to_tensor=True)
        emb_np = context_embeddings.cpu().numpy().astype("float32")
        faiss.normalize_L2(emb_np)
        index = faiss.IndexFlatIP(emb_np.shape[1])
        index.add(emb_np)
        faiss.write_index(index, "faiss_index.bin")
        with open("contexts_cache.pkl", "wb") as f:
            pickle.dump((contexts, context_meta), f)
 
    from rank_bm25 import BM25Okapi
    tokenized_contexts = [normalize_for_bm25(c) for c in contexts]
    bm25 = BM25Okapi(tokenized_contexts)
 
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()
 
 
def get_best_context(question):
    q_emb = embedder.encode(question, convert_to_tensor=True).cpu().numpy().astype("float32")
    if q_emb.ndim == 1:
        q_emb = q_emb.reshape(1, -1)
    faiss.normalize_L2(q_emb)
 
    k = len(contexts)
    d_scores, d_idx = index.search(q_emb, k)
    d_scores = d_scores[0]
    d_idx = d_idx[0]
 
    dense_all = np.zeros(len(contexts), dtype="float32")
    dense_all[d_idx] = d_scores
 
    tokenized_query = normalize_for_bm25(question)
    bm25_scores = np.array(bm25.get_scores(tokenized_query), dtype="float32")
 
    d_max = dense_all.max()
    dense_norm = dense_all / d_max if d_max > 0 else dense_all
    b_max = bm25_scores.max()
    bm25_norm = bm25_scores / b_max if b_max > 0 else bm25_scores
 
    fused = 0.6 * dense_norm + 0.4 * bm25_norm
    top5 = np.argsort(fused)[-5:][::-1]
    best_score = float(fused[top5[0]])
 
    # FIX 4: lowered threshold from 0.65 to 0.55
    if best_score < 0.55:
        return None
    top_contexts = [contexts[i] for i in top5]
    top_metas = [context_meta[i] for i in top5]
    return top_contexts, best_score, top_metas
 
 
def bert_answer(question, context):
    # FIX 3: proper beam search over (start, end) pairs instead of independent argmax
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
 
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
 
    seq_len = start_logits.shape[0]
    best_score = float("-inf")
    best_start, best_end = 1, 2
 
    for s in range(1, min(seq_len, 400)):
        if attention_mask[0][s].item() == 0:
            break
        for e in range(s, min(s + 30, seq_len)):
            score = start_logits[s].item() + end_logits[e].item()
            if score > best_score:
                best_score = score
                best_start, best_end = s, e + 1
 
    answer = tokenizer.decode(input_ids[0][best_start:best_end], skip_special_tokens=True).strip()
 
    start_probs = torch.softmax(start_logits, dim=-1)
    end_probs = torch.softmax(end_logits, dim=-1)
    reader_confidence = float(
        start_probs[best_start].item() * end_probs[min(best_end - 1, len(end_probs) - 1)].item()
    )
    return answer.strip(), reader_confidence
 
 
st.set_page_config(page_title="IITK Chatbot", page_icon="🎓", layout="centered")
 
st.markdown(
    '''
    <h1 style='text-align: center; color: #004080;'> IITK Chatbot</h1>
    <p style='text-align: center; font-size:18px; color: gray;'>Ask anything related to <strong>IIT Kanpur</strong> – academics, hostels, mental health, orientation, reporting, and more!</p>
    ''', 
    unsafe_allow_html=True
)
 
st.markdown("---")
 
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message["confidence"] is not None:
            confidence_pct = int(round(message["confidence"] * 100))
            st.markdown(f"<div style='margin-top: 8px; font-size: 12px; color: #888;'>Confidence: {confidence_pct}%</div>", unsafe_allow_html=True)
        if message["role"] == "assistant" and message.get("source"):
            source = message["source"]
            st.markdown(f"<div style='margin-top: 4px; font-size: 12px; color: #666;'>Source: <a href='{source['url']}' target='_blank'>{source['title']}</a></div>", unsafe_allow_html=True)
 
if user_query := st.chat_input("Ask your question here:"):
    st.session_state.messages.append({"role": "user", "content": user_query, "confidence": None, "source": None})
 
    with st.chat_message("user"):
        st.markdown(user_query)
 
    with st.spinner("Thinking..."):
        result = get_best_context(user_query)
 
        if result is None:
            answer = "I'm not confident about that. Try rephrasing your question."
            confidence = None
            source = None
        else:
            top_contexts, retrieval_score, top_metas = result
            rerank_scores = reranker.predict([(user_query, ctx) for ctx in top_contexts])
            rerank_scores_norm = (np.array(rerank_scores) - np.min(rerank_scores)) / (
                np.max(rerank_scores) - np.min(rerank_scores) + 1e-9
            )
            retrieval_rank_bonus = np.linspace(1.0, 0.0, num=len(top_contexts))
            combined_scores = 0.7 * rerank_scores_norm + 0.3 * retrieval_rank_bonus
            best_idx = int(np.argmax(combined_scores))
            best_ctx = top_contexts[best_idx]
            answer, reader_confidence = bert_answer(user_query, best_ctx)
            confidence = 0.5 * retrieval_score + 0.5 * reader_confidence
            source = top_metas[best_idx]
 
    st.session_state.messages.append({"role": "assistant", "content": answer, "confidence": confidence, "source": source})
 
    with st.chat_message("assistant"):
        st.write(answer)
        if confidence is not None:
            confidence_pct = int(round(confidence * 100))
            st.markdown(f"<div style='margin-top: 8px; font-size: 12px; color: #888;'>Confidence: {confidence_pct}%</div>", unsafe_allow_html=True)
        if source:
            st.markdown(f"<div style='margin-top: 4px; font-size: 12px; color: #666;'>Source: <a href='{source['url']}' target='_blank'>{source['title']}</a></div>", unsafe_allow_html=True)
 
