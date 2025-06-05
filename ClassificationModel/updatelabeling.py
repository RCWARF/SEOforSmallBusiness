#used to add NLP feature into training csv I already updated with Is_Good_Keyword Field
print("starting script....")
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Load your labeled file
df = pd.read_csv("outputs/labeling_sampleupdated.csv", encoding='windows-1252')

# Lazy-load embedding model
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-mpnet-base-v2')
    return _embedding_model

# Business keyword reference list
BUSINESS_KEYWORDS = [
    "digital marketing for course creators",
    "content strategy",
    "email marketing tips",
    "simplify your business",
    "marketing funnel strategy",
    "marketing strategy for course creators",
    "email marketing for digital products",
    "marketing for online educators",
    "simplified marketing for small business owners",
    "slow business growth strategies"
]

# Precompute the average embedding
model = get_embedding_model()
BUSINESS_EMBEDDING = np.mean(model.encode(BUSINESS_KEYWORDS), axis=0)

# Similarity scoring function
def compute_similarity_to_business_focus(keyword: str) -> float:
    try:
        keyword_embedding = model.encode(keyword)
        dot = np.dot(keyword_embedding, BUSINESS_EMBEDDING)
        norm = np.linalg.norm(keyword_embedding) * np.linalg.norm(BUSINESS_EMBEDDING)
        return dot / norm
    except:
        return 0.0

# Apply the feature
df['Business_Keyword_Similarity'] = df['Keyword'].apply(compute_similarity_to_business_focus)

# Save the updated file
df.to_csv("outputs/labeling_sample_enriched2.csv", index=False)
print("Updated file saved to: outputs/labeling_sample_enriched2.csv")