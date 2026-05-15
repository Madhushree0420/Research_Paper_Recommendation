import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# ----------------------------
# PATHS
# ----------------------------
DATA_PATH = r"D:\Research_paper_recommend\data\processed\cleaned_data.csv"
FAISS_INDEX_PATH = r"D:\Research_paper_recommend\data\processed\faiss.index"

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df = df.reset_index(drop=True)

    # SAFE numeric conversion
    df["year"] = pd.to_numeric(df.get("year", 0), errors="coerce").fillna(0).astype(int)
    df["n_citation"] = pd.to_numeric(df.get("n_citation", 0), errors="coerce").fillna(0).astype(int)

    # SAFE text fallback
    df["text"] = (df["title"].astype(str) + " " + df["abstract"].astype(str)).fillna("")

    return df

df = load_data()

# ----------------------------
# LOAD FAISS
# ----------------------------
@st.cache_resource
def load_faiss():
    return faiss.read_index(FAISS_INDEX_PATH)

index = load_faiss()

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------
# LABEL FUNCTION
# ----------------------------
def get_relevance(score):
    if score >= 0.75:
        return "🟢 Highly Relevant", "green"
    elif score >= 0.55:
        return "🟡 Relevant", "orange"
    elif score >= 0.35:
        return "🟠 Somewhat Relevant", "darkorange"
    else:
        return "🔴 Related Match", "red"

# ----------------------------
# FAISS SEARCH (FIXED)
# ----------------------------
def search_papers(query, top_k=10):

    query_vec = model.encode(query, normalize_embeddings=True)
    query_vec = np.array([query_vec]).astype("float32")

    # 🔥 overfetch to avoid filtering losing results
    scores, indices = index.search(query_vec, top_k * 5)

    indices = indices[0]
    scores = scores[0]

    # remove invalid
    valid = indices != -1
    indices = indices[valid]
    scores = scores[valid]

    results = df.iloc[indices].copy()
    results["score"] = scores

    # ----------------------------
    # SIMPLE HYBRID (safe)
    # ----------------------------
    if "n_citation" in results.columns:
        results["citation_norm"] = results["n_citation"] / (results["n_citation"].max() + 1)
        results["final_score"] = 0.8 * results["score"] + 0.2 * results["citation_norm"]
    else:
        results["final_score"] = results["score"]

    return results.sort_values("final_score", ascending=False).head(top_k)

# ----------------------------
# UI
# ----------------------------
st.title("📚 AI Research Paper Search System")
st.write("FAISS + Sentence-BERT + Hybrid Ranking")

query = st.text_input("Enter your research topic")
top_k = st.slider("Number of results", 5, 20, 10)

# ----------------------------
# SEARCH
# ----------------------------
if query:

    with st.spinner("Searching papers..."):
        results = search_papers(query, top_k)

    st.subheader("🔍 Top Recommended Papers")

    if len(results) == 0:
        st.warning("No results found. Try another keyword.")
    else:
        for i, (_, row) in enumerate(results.iterrows()):

            score = row["score"]
            label, color = get_relevance(score)

            st.markdown(f"### 📄 {row['title']}")

            if i == 0:
                st.success("⭐ Best Match")

            st.markdown(
                f"<span style='color:{color}; font-weight:bold;'>📊 {label}</span>"
                f"<br><sub>Score: {score:.4f}</sub>",
                unsafe_allow_html=True
            )

            st.progress(float(score))

            st.markdown(f"""
            **👨‍💻 Authors:** {row.get('authors','Unknown')}  
            **🏛 Venue:** {row.get('venue','Unknown')}  
            **📅 Year:** {row.get('year','N/A')}  
            **📈 Citations:** {row.get('n_citation',0)}
            """)

            st.write("📝 Abstract:", row["abstract"][:300] + "...")

            st.write("---")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("---")
st.markdown("🚀 FAISS + Sentence-BERT + Working Recommendation System")