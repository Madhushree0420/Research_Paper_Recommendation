# AI Research Paper Search & Recommendation System

An intelligent research paper recommendation system built using NLP, Sentence-BERT, FAISS, and Hybrid Ranking.  
The system enables users to search research papers semantically using natural language queries and retrieve the most relevant papers instantly.


## Displays:
1. Title  
2. Authors  
3. Abstract  
4. Year  
5. Venue  
6. Citation Count  


## Technologies Used
- Python  
- Streamlit  
- Sentence-Transformers  
- FAISS  
- Pandas  
- NumPy  
- Scikit-learn  
- Neo4j  

```bash
## Installation

## 1️⃣ Clone Repository

git clone <your-github-repo-link>
cd Research_paper_recommend


##2️⃣ Install Dependencies
pip install -r requirements.txt
Or install manually:
pip install streamlit pandas numpy sentence-transformers faiss-cpu scikit-learn neo4j
Dataset

Place your dataset file:

data/dblp-v10.csv
Dataset Columns Used
title
abstract
authors
venue
year
n_citation
references
id
⚡ Generate Embeddings & FAISS Index

Run:

python precompute.py

This will generate:

cleaned_data.csv
embeddings.npy
faiss.index

inside:

data/processed/
▶️ Run Streamlit App
streamlit run app.py
🔍 Example Queries

##Try searching:

-deep learning
-recommendation system
-graph neural network
-machine learning
-NLP
-data mining
##🧠 How It Works
Step 1: Data Preprocessing
Cleans dataset
Removes null values
Combines title + abstract
Step 2: Embedding Generation
Uses Sentence-BERT (all-MiniLM-L6-v2)
Converts papers into vector embeddings
Step 3: FAISS Indexing
Creates fast similarity search index
Enables real-time retrieval
Step 4: Hybrid Ranking

Final ranking combines:

Semantic similarity score
Citation count score
-📊 Future Enhancements
-🔗 Citation graph visualization
-💬 Chat with research papers
-📄 PDF paper links
-🧠 RAG-based AI assistant
-☁️ Cloud deployment
-📈 Trending paper analytics
-🖼 Sample UI Features
-Semantic search bar
-Relevance badges
-Citation-aware ranking
-Research metadata display
-Interactive recommendations


**👨‍💻 Author**
Developed by Madhushree .S

**📜 License**

This project is for educational and research purposes.
