import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

# ----------------------------
# PATHS
# ----------------------------
DATA_PATH = r"D:\Research_paper_recommend\data\dblp-v10.csv"
SAVE_DIR = r"D:\Research_paper_recommend\data\processed"

os.makedirs(SAVE_DIR, exist_ok=True)

EMBEDDINGS_PATH = os.path.join(SAVE_DIR, "embeddings.npy")
DATA_CLEAN_PATH = os.path.join(SAVE_DIR, "cleaned_data.csv")
FAISS_INDEX_PATH = os.path.join(SAVE_DIR, "faiss.index")

MAX_ROWS = 5000

# ----------------------------
# LOAD DATA
# ----------------------------
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# ----------------------------
# CLEANING (SAFE + CONSISTENT)
# ----------------------------
df = df.dropna(subset=["title", "abstract"])
df = df.drop_duplicates()

df = df.head(MAX_ROWS).reset_index(drop=True)

# Ensure required columns exist
for col in ["authors", "venue", "year", "n_citation"]:
    if col not in df.columns:
        df[col] = "Unknown" if col in ["authors", "venue"] else 0

# Type fixing
df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
df["n_citation"] = pd.to_numeric(df["n_citation"], errors="coerce").fillna(0).astype(int)

# Create embedding text
df["text"] = (df["title"].astype(str) + " " + df["abstract"].astype(str)).str.strip()

print(f"✅ Cleaned dataset: {len(df)} papers")

# ----------------------------
# LOAD MODEL
# ----------------------------
print("🧠 Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ----------------------------
# EMBEDDINGS
# ----------------------------
print("⚡ Generating embeddings...")

embeddings = model.encode(
    df["text"].tolist(),
    batch_size=64,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print(f"✅ Embeddings shape: {embeddings.shape}")

# ----------------------------
# SAVE DATA
# ----------------------------
print("💾 Saving files...")

np.save(EMBEDDINGS_PATH, embeddings)
df.to_csv(DATA_CLEAN_PATH, index=False)

# ----------------------------
# FAISS INDEX (IMPORTANT FIXED VERSION)
# ----------------------------
print("⚡ Building FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)  # cosine similarity (because normalized)
index.add(embeddings.astype("float32"))

faiss.write_index(index, FAISS_INDEX_PATH)

# ----------------------------
# FINAL DEBUG CHECK
# ----------------------------
print("\n🔍 DEBUG CHECK")
print("DF size:", len(df))
print("FAISS vectors:", index.ntotal)

print("\n✅ Precompute DONE successfully!")
print(f"📁 Embeddings: {EMBEDDINGS_PATH}")
print(f"📁 Clean data: {DATA_CLEAN_PATH}")
print(f"📁 FAISS index: {FAISS_INDEX_PATH}")