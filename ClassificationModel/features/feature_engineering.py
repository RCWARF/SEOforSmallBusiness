from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd

_embedding_model = None  # module-level cache

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-mpnet-base-v2')
    return _embedding_model

# Define your business "ideal" keyword topics
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

BUSINESS_EMBEDDING = np.mean(get_embedding_model().encode(BUSINESS_KEYWORDS), axis=0)


# Cosine Similarity of Data keywords and Business Keywordscosine_similarity= 

def compute_similarity_to_business_focus(keyword: str) -> float:
    try:
        model = get_embedding_model()
        keyword_embedding = model.encode(keyword)
        dot = np.dot(keyword_embedding, BUSINESS_EMBEDDING)
        norm = np.linalg.norm(keyword_embedding) * np.linalg.norm(BUSINESS_EMBEDDING)
        return dot / norm
    except Exception as e:
        return 0.0

def apply_feature_engineering(df):


    # KEYWORD LENGTH FEATURE

    df['Keyword_Word_Count'] = df['Keyword'].apply(lambda x: len(str(x).split()))
    df['Keyword_Length_Score'] = df['Keyword_Word_Count'].apply(lambda count:
        0 if count <= 2 else 1 if count == 3 else 2 if count == 4 else 3)
    
    # KEYWORD SIMILARITY
    df['Business_Keyword_Similarity'] = df['Keyword'].apply(compute_similarity_to_business_focus)


    # SEARCH VOLUME FEATURE

    sv_threshold = 1400  #set threshold to cap outliers
    df['Search Volume'] = df['Search Volume'].fillna(0)
    df['Search_Volume_Capped'] = df['Search Volume'].apply(lambda x: min(x, sv_threshold))
    df['Search_Volume_Log'] = np.log1p(df['Search_Volume_Capped'])  #using log function to compress data field(fixing distribution)
    df['Search_Volume_Scaled'] = df['Search_Volume_Log'] / df['Search_Volume_Log'].max()

    # RANKING DIFFICULTY FEATURE

    df['Ranking Difficulty'] = df['Ranking Difficulty'].fillna(0)
    df['Ranking Difficulty Score'] = 100 - df['Ranking Difficulty']
    df['Ranking Difficulty Scaled'] = df['Ranking Difficulty Score'] / 100

    # TOTAL MONTHLY CLICKS FEATURE

    mc_threshold = 1000 #threshhold to cap outlier
    nonzero_clicks = df['Total Monthly Clicks'][(df['Total Monthly Clicks'] > 0) & (df['Total Monthly Clicks'] <= mc_threshold)]
    low_neutral_fill = np.percentile(nonzero_clicks, 10) #creates fill value so that outlier doesnt skew and zero values dont get advantage over low values(data cant be right that there are zero clicks if they showed up on list)
    df['Total Monthly Clicks Filled'] = df['Total Monthly Clicks'].apply(
        lambda x: low_neutral_fill if pd.isna(x) or x < low_neutral_fill else x) #adds low percentile value to all zero values and any values less that low percentile value
    df['Total Monthly Clicks Capped'] = df['Total Monthly Clicks Filled'].apply(lambda x: min(x, mc_threshold))
    df['Total Monthly Clicks Log'] = np.log1p(df['Total Monthly Clicks Capped'])  #log function to compress data(fixing distribution)
    df['Total Monthly Clicks Scaled'] = df['Total Monthly Clicks Log'] / df['Total Monthly Clicks Log'].max()

    # COST PER CLICK FEATURE
    df['Broad CPC Score'] = df['Broad Cost Per Click'].apply(lambda x: 1 if pd.isna(x) or x == 0 else 0) #dont want to compete for pay to play

    # IS QUESTION FEATURE
    df['Is Question Score'] = df['Is Question?'].astype(int) #useful feature do to search habits

    # NSFW
    df['NSFW Score'] = (~df['Is Not Safe For Work?']).astype(int) #dont want nsfw keyword content

    return df