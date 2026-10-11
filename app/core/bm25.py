"""
BM25 (Best Matching 25) Sparse Keyword Search Implementation.
Provides exact Okapi BM25 scoring with term frequency saturation and length normalization.
"""
import math
import re
from typing import List, Dict, Tuple, Set


STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than",
    "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've",
    "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what",
    "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's",
    "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd",
    "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"
}


def tokenize(text: str) -> List[str]:
    """Tokenize and clean text into lowercase alphanumeric tokens."""
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: List[int] = []
        self.doc_ids: List[str] = []
        self.doc_freqs: Dict[str, int] = {}  # term -> number of docs containing it
        self.term_freqs_list: List[Dict[str, int]] = []  # doc_idx -> {term: count}

    def index(self, documents: List[Tuple[str, str]]):
        """
        Index list of tuples: [(doc_id, text), ...]
        """
        self.doc_ids = []
        self.doc_lens = []
        self.doc_freqs = {}
        self.term_freqs_list = []
        self.corpus_size = len(documents)

        total_len = 0
        for doc_id, text in documents:
            tokens = tokenize(text)
            doc_len = len(tokens)
            self.doc_ids.append(doc_id)
            self.doc_lens.append(doc_len)
            total_len += doc_len

            term_counts: Dict[str, int] = {}
            for t in tokens:
                term_counts[t] = term_counts.get(t, 0) + 1
            self.term_freqs_list.append(term_counts)

            for t in term_counts:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0

    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Calculates BM25 score for query across all indexed documents.
        Returns: [(doc_id, score), ...] sorted descending.
        """
        if self.corpus_size == 0:
            return []

        query_tokens = tokenize(query_text)
        if not query_tokens:
            return []

        scores: List[float] = [0.0] * self.corpus_size

        for token in query_tokens:
            if token not in self.doc_freqs:
                continue

            # IDF computation with smoothing
            df = self.doc_freqs[token]
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

            for idx in range(self.corpus_size):
                tf = self.term_freqs_list[idx].get(token, 0)
                if tf == 0:
                    continue

                doc_len = self.doc_lens[idx]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len if self.avg_doc_len > 0 else 1.0))
                scores[idx] += idf * (numerator / denominator)

        # Pair with doc_ids and sort
        ranked = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
