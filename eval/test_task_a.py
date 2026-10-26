import os
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
import numpy as np


NONFOOD_BLACKLIST = {
"plastic", "air", "water", "moon", "laptop", "phone", "car", "metal", "rock", "table"
}
GENERIC = {
    "is","are","does","do","contain","contains","free","allergy","safe","can","i","eat",
    "with","without","for","a","an","the","to","of","in","on","my","your"
}


# =========================
# CONFIG (match your project)
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
KB_PATH = os.path.join(PROJECT_ROOT, "kb", "allergy_qa_kb.csv")
EVAL_PATH = os.path.join(PROJECT_ROOT, "eval", "task_a_eval_set.csv")

NLTK_DIR = os.path.join(PROJECT_ROOT, "nltk_data")
if NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DIR)

SIM_THRESHOLD = 0.35  # same as in mybot-basic.py




# =========================
# SAFETY CHECKS
# =========================
def assert_nltk_resources():
    """
    Keep this clean: do NOT download here (demo-friendly).
    If resources are missing, show a clear message.
    """
    required = ["tokenizers/punkt", "corpora/stopwords", "corpora/wordnet"]
    missing = []
    for r in required:
        try:
            nltk.data.find(r)
        except LookupError:
            missing.append(r)

    if missing:
        raise RuntimeError(
            "Missing NLTK resources: "
            + ", ".join(missing)
            + "\nRun scripts/setup_nltk.py first."
        )


# =========================
# PREPROCESSING (match your chatbot)
# =========================
_stopwords = None
_lemmatizer = None

def init_nlp():
    global _stopwords, _lemmatizer
    _stopwords = set(stopwords.words("english"))
    # keep negations (important for allergy statements)
    _stopwords.discard("no")
    _stopwords.discard("not")
    _lemmatizer = WordNetLemmatizer()

def preprocess(text: str) -> str:
    text = (text or "").lower().strip()
    tokens = word_tokenize(text)

    cleaned = []
    for t in tokens:
        if t.isalpha() and t not in _stopwords:
            cleaned.append(_lemmatizer.lemmatize(t))
    return " ".join(cleaned)


# =========================
# DATA LOADING
# =========================
def load_kb(kb_path: str) -> Tuple[List[str], List[str]]:
    questions, answers = [], []
    with open(kb_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("KB CSV has no header row.")
        # require these headers
        for col in ["question", "answer"]:
            if col not in reader.fieldnames:
                raise RuntimeError(f"KB CSV missing required column: {col}")

        for row in reader:
            q = (row.get("question") or "").strip()
            a = (row.get("answer") or "").strip()
            if q and a:
                questions.append(q)
                answers.append(a)

    if not questions:
        raise RuntimeError("KB loaded 0 questions. Check your CSV content.")
    return questions, answers


def load_eval(eval_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """
    Returns rows and detected column names.
    Your eval CSV can be minimal (just 'query') or richer.
    """
    with open(eval_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise RuntimeError("Eval CSV has no header row.")
        rows = [dict(r) for r in reader]
    if not rows:
        raise RuntimeError("Eval set is empty.")
    return rows, reader.fieldnames


# =========================
# MATCHING
# =========================
@dataclass
class MatchResult:
    query: str
    best_idx: int
    best_score: float
    matched_question: str
    matched_answer: str
    decision: str  # "ANSWER" or "REFUSE"


def find_best_match(
    query: str,
    vectorizer: TfidfVectorizer,
    kb_tfidf,
    kb_questions: List[str],
    kb_answers: List[str],
    tfidf_threshold: float,
    mode: str="TFIDF", 
    sbert_model=None,
    kb_sbert=None,
    sbert_threshold=0.55
) -> MatchResult:

    q_proc = preprocess(query)
    q_vec = vectorizer.transform([q_proc])

    sims = cosine_similarity(q_vec, kb_tfidf).flatten()
    best_idx = int(sims.argmax())
    best_score = float(sims[best_idx])
    
    
    # --- Optional SBERT scoring (extension) ---
    if mode.upper() == "SBERT":
        q_emb = sbert_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
            )[0]
        sbert_sims = np.dot(kb_sbert, q_emb)
        best_idx = int(np.argmax(sbert_sims))
        best_score = float(sbert_sims[best_idx])
        threshold = sbert_threshold
    else:
            threshold = tfidf_threshold

    
    




    query_tokens = set(preprocess(query).split())
    match_tokens = set(preprocess(kb_questions[best_idx]).split())

    meaningful_query = query_tokens - GENERIC
    meaningful_match = match_tokens - GENERIC
    meaningful_overlap = meaningful_query & meaningful_match

    # if query contains obvious non-food terms, refuse
    contains_nonfood = len(meaningful_query & NONFOOD_BLACKLIST) >= 1

    if best_score >= threshold and len(meaningful_overlap) >= 1 and not contains_nonfood:
        decision = "ANSWER"
    else:
        decision = "REFUSE"

    return MatchResult(
        query=query,
        best_idx=best_idx,
        best_score=best_score,
        matched_question=kb_questions[best_idx],
        matched_answer=kb_answers[best_idx],
        decision=decision
        )

# =========================
# EVALUATION (auto-detect)
# =========================
def main():
    assert_nltk_resources()
    init_nlp()

    kb_questions, kb_answers = load_kb(KB_PATH)
    
    

    # TF-IDF fit once
    kb_questions_proc = [preprocess(q) for q in kb_questions]
    vectorizer = TfidfVectorizer()
    kb_tfidf = vectorizer.fit_transform(kb_questions_proc)

    # SBERT setup (extension evaluation)
    SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
    SBERT_THRESHOLD = 0.55

    sbert_model = SentenceTransformer(SBERT_MODEL_NAME)
    kb_sbert = sbert_model.encode(
        kb_questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
        )




    eval_rows, cols = load_eval(EVAL_PATH)

    print("=== Task A Evaluation ===")
    print("KB questions:", len(kb_questions))
    print("Eval rows:", len(eval_rows))
    print("Eval columns:", cols)
    print("SIM_THRESHOLD:", SIM_THRESHOLD)
    print()
    def run_eval(mode_name: str):
        total = 0
        correct_behavior = 0
        mismatches = []
        
        answered = []
        refused = []


        for row in eval_rows:
            query = (row.get("query") or row.get("Query") or "").strip()
            if not query:
                continue

            res = find_best_match(
                query=query,
                vectorizer=vectorizer,
                kb_tfidf=kb_tfidf,
                kb_questions=kb_questions,
                kb_answers=kb_answers,
                tfidf_threshold=SIM_THRESHOLD,
                mode=mode_name,
                sbert_model=sbert_model,
                kb_sbert=kb_sbert,
                sbert_threshold=SBERT_THRESHOLD
                )

            total += 1
            
            if res.decision == "ANSWER":
                answered.append(res)
            else:
                refused.append(res)

            exp = (row.get("expected_behavior") or "").strip().upper()
            if exp in ["ANSWER", "REFUSE"]:
                if res.decision == exp:
                    correct_behavior += 1
                else:
                    mismatches.append((query, f"behavior expected {exp} got {res.decision}", res))
                    
                    


        print(f"\n=== {mode_name} Summary ===")
        print("Total evaluated:", total)
        print(f"Behavior correctness: {correct_behavior}/{total} = {correct_behavior/total:.2%}")

        if mismatches:
            print(f"\n=== {mode_name} Mismatches (first 5) ===")
            for q, why, res in mismatches[:5]:
                print(f"\nQuery: {q}")
                print("Issue:", why)
                print(f"Decision: {res.decision} (score={res.best_score:.3f})")
                print("Top KB match:", res.matched_question)
                print("Answer:", res.matched_answer)

        
    
        print(f"\n=== {mode_name} Evidence samples (use in write-up) ===")

        # Show up to 3 ANSWER examples
        for r in answered[:3]:
            print(f"\nQuery: {r.query}")
            print(f"Decision: {r.decision} (score={r.best_score:.3f})")
            print("Top KB match:", r.matched_question)
            print("Answer:", r.matched_answer)

        # Show up to 3 REFUSE examples
        for r in refused[:3]:
            print(f"\nQuery: {r.query}")
            print(f"Decision: {r.decision} (score={r.best_score:.3f})")
            print("Top KB match:", r.matched_question)
            print("Answer:", r.matched_answer)

        return correct_behavior, total

    run_eval("TFIDF")
    run_eval("SBERT")


if __name__ == "__main__":
    main()


