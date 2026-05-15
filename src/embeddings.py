import pandas as pd
from sentence_transformers import SentenceTransformer

# Load dataset
df = pd.read_csv(r"D:\Research_paper_recommend\data\dblp-v10.csv")

df = df.dropna(subset=["title", "abstract"])
df["text"] = df["title"] + " " + df["abstract"]

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# IMPORTANT: start small first (VERY IMPORTANT for your laptop)
df_small = df.head(5000)

# Generate embeddings
df_small["embedding"] = df_small["text"].apply(lambda x: model.encode(x))

print("Embeddings created for:", len(df_small))