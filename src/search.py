import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ----------------------------
# PATHS
# ----------------------------
DATA_PATH = r"D:\Research_paper_recommend\data\processed\cleaned_data.csv"
FAISS_INDEX_PATH = r"D:\Research_paper_recommend\data\processed\faiss.index"

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH).reset_index(drop=True)

# ----------------------------
# LOAD FAISS
# ----------------------------
index = faiss.read_index(FAISS_INDEX_PATH)

# ----------------------------
# MODEL
# ----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# SEARCH
# ----------------------------
def search(query, top_k=10):

    # 🔥 STEP 1: encode query
    query_vec = model.encode(query)

    query_vec = np.array([query_vec]).astype("float32")

    # 🔥 STEP 2: ensure safe FAISS search
    scores, indices = index.search(query_vec, top_k * 2)

    indices = indices[0]
    scores = scores[0]

    # 🚨 remove invalid results (-1 issue fix)
    valid_mask = indices != -1
    indices = indices[valid_mask]
    scores = scores[valid_mask]

    if len(indices) == 0:
        return pd.DataFrame()

    # 🔥 STEP 3: map results
    results = df.iloc[indices].copy()
    results["score"] = scores

    # ----------------------------
    # HYBRID RANKING
    # ----------------------------
    if "n_citation" in results.columns:
        results["citation_norm"] = results["n_citation"] / (results["n_citation"].max() + 1)

        results["final_score"] = (
            0.7 * results["score"] +
            0.3 * results["citation_norm"]
        )

        results = results.sort_values("final_score", ascending=False)

    return results.head(top_k)

# ----------------------------
# TEST
# ----------------------------
if __name__ == "__main__":
    res = search("graph neural network", 10)

    if res.empty:
        print("❌ No results found - FAISS issue")
    else:
        print(res[["title", "score"]])