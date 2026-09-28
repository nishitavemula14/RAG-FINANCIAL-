import re
from typing import List, Any, Tuple

STOPWORDS = {
    "how", "what", "which", "where", "when", "why", "who", "whom", "does", "do", "did",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "a", "an", "the", "and", "or", "but", "if", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through", "during",
    "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here", "there", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "can", "will", "just",
    "should", "now", "its", "it", "their", "they", "we", "our", "my", "your"
}

def stem_word(w: str) -> str:
    """Lightweight morphological stemmer for English inflections."""
    w = w.lower()
    if len(w) <= 3:
        return w
    if w.endswith("al"):
        w = w[:-2]
    if w.endswith("tions") or w.endswith("tion"):
        w = w[:-len("tions")] + "t" if w.endswith("tions") else w[:-len("tion")] + "t"
    for s in ["ing", "edly", "ment", "ments", "ability", "ibility", "ness"]:
        if len(w) > len(s) + 2 and w.endswith(s):
            w = w[:-len(s)]
            break
    for s in ["ed", "es", "ly", "er", "or", "ive"]:
        if len(w) > len(s) + 2 and w.endswith(s):
            w = w[:-len(s)]
            break
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    return w

class LexicalReranker:
    """Lightweight local reranker using stemmed token overlap, phrase density, and segment alignment."""
    def __init__(self):
        pass

    def rerank(self, query: str, chunks: List[Any], top_k: int = 5) -> List[Any]:
        """Reranks chunks based on stemmed query term matches, segment awareness, and phrase overlap."""
        if not chunks:
            return []

        q_lower = query.lower()
        raw_words = re.findall(r"\w+", q_lower)
        q_keywords = [w for w in raw_words if w not in STOPWORDS]
        if not q_keywords:
            q_keywords = raw_words

        q_stems = [stem_word(w) for w in q_keywords]

        scored: List[Tuple[float, Any]] = []
        for c in chunks:
            text_lower = c.text.lower()
            doc_words = re.findall(r"\w+", text_lower)
            doc_stems = [stem_word(w) for w in doc_words]
            doc_stem_set = set(doc_stems)

            # 1. Stemmed keyword overlap with term-frequency bonus
            score = 0.0
            for qs in q_stems:
                if qs in doc_stem_set:
                    cnt = doc_stems.count(qs)
                    score += 1.5 + min(cnt * 0.25, 1.5)

            # 2. Key phrase bonus
            if q_lower in text_lower:
                score += 4.0

            # 3. Segment alignment bonus
            has_us_query = any(tok in q_lower for tok in ["u.s.", "u.s", "us segment", "u.s. segment", "united states"])
            if has_us_query:
                if "walmart u.s." in text_lower or "walmart u.s" in text_lower or "in the u.s." in text_lower:
                    score += 3.0
                if "sam's club" in text_lower and "sam" not in q_lower and "club" not in q_lower:
                    score -= 1.5
                if "walmart international" in text_lower and "walmart u.s" not in text_lower:
                    score -= 2.0

            has_intl_query = "international" in q_lower
            if has_intl_query and "walmart international" in text_lower:
                score += 3.0

            has_sams_query = "sam" in q_lower or "club" in q_lower
            if has_sams_query and "sam's club" in text_lower:
                score += 3.0

            # 4. Action/Section match bonus (e.g. distribute/distribution, risk, operate/operation)
            for target_stem in ["distribut", "operat", "risk", "competit"]:
                if any(stem_word(kw) == target_stem for kw in q_keywords):
                    if target_stem in doc_stem_set:
                        score += 2.5

            scored.append((score, c))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]

