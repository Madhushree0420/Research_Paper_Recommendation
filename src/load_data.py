import pandas as pd

df = pd.read_csv(r"D:\Research_paper_recommend\data\dblp-v10.csv")

print("Shape:", df.shape)
print("Columns:", df.columns)
print(df.head())