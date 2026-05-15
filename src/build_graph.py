import pandas as pd
from neo4j import GraphDatabase
import ast

# ----------------------------
# LOAD CLEANED DATA (IMPORTANT)
# ----------------------------
df = pd.read_csv(r"D:\Research_paper_recommend\data\processed\cleaned_data.csv")

# ----------------------------
# NEO4J CONNECTION
# ----------------------------
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "12345678")
)

# ----------------------------
# CREATE PAPER NODE
# ----------------------------
def create_paper(tx, paper_id, title):
    tx.run("""
    MERGE (p:Paper {id: $id})
    SET p.title = $title
    """, id=paper_id, title=title)

# ----------------------------
# CREATE CITATION EDGE
# ----------------------------
def create_citation(tx, src, tgt):
    tx.run("""
    MATCH (a:Paper {id: $src})
    MATCH (b:Paper {id: $tgt})
    MERGE (a)-[:CITES]->(b)
    """, src=src, tgt=tgt)

# ----------------------------
# BUILD GRAPH
# ----------------------------
with driver.session() as session:
    print("📥 Creating nodes and edges...")

    for _, row in df.head(5000).iterrows():

        # Create paper node
        session.execute_write(create_paper, row["id"], row["title"])

        # Handle references safely
        refs = row.get("references")

        if pd.isna(refs) or refs == "":
            continue

        try:
            # Convert string → list safely
            ref_list = ast.literal_eval(refs)

            if not isinstance(ref_list, list):
                continue

            for ref in ref_list:
                session.execute_write(create_citation, row["id"], ref)

        except:
            continue

print("✅ Graph created successfully!")