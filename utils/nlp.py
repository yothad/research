from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


def compute_tfidf_cosine_similarity(free_text_drugs, atc_drug_names):
    # Character-level n-grams help with misspellings
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5))

    # Fit on ATC names (the reference list)
    tfidf_atc = vectorizer.fit_transform(atc_drug_names)

    # Transform the free-text entries
    tfidf_free = vectorizer.transform(free_text_drugs)

    # -----------------------------
    # Compute cosine similarity
    # -----------------------------
    cosine_sim = cosine_similarity(tfidf_free, tfidf_atc)

    rows = []
    for i, drug in enumerate(free_text_drugs):
        # Get top 3 matches
        top_indices = np.argsort(cosine_sim[i])[::-1][:3]
        for idx in top_indices:
            rows.append({
                "text_drug_name": drug,
                "atc_drug_name": atc_drug_names[idx],
                "match_rate": round(cosine_sim[i][idx], 3)
            })

    return pd.DataFrame(rows)

