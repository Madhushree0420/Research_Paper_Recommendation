import pandas as pd

df = pd.read_csv(r"D:\Research_paper_recommend\data\dblp-v10.csv")

# Step 1: remove missing abstracts
df = df.dropna(subset=["title", "abstract"])

# Step 2: remove duplicates
df = df.drop_duplicates()

# Step 3: create text field for NLP
df["text"] = df["title"] + " " + df["abstract"]

print(df.head())
print("Clean dataset size:", len(df))