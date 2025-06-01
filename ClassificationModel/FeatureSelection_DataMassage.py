import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import os
print("Current working directory:", os.getcwd())
# Load your CSV
df = pd.read_csv("SalesFunnelSEO.csv")




# KEYWORD LENGTH FEATURE



# Count the number of words in the keyword
df['Keyword_Word_Count'] = df['Keyword'].apply(lambda x: len(str(x).split()))

# Assign score based on how many words are in the keyword
def keyword_length_score(count):
    if count <= 2:
        return 0
    elif count == 3:
        return 1
    elif count == 4:
        return 2
    else:
        return 3

df['Keyword_Length_Score'] = df['Keyword_Word_Count'].apply(keyword_length_score)





# SEARCH VOLUME FEATURE




# Fill missing values with 0
df['Search Volume'] = df['Search Volume'].fillna(0)

# Cap outliers at a threshold
sv_threshold = 1400

df['Search_Volume_Capped'] = df['Search Volume'].apply(lambda x: min(x, sv_threshold))

# Logarithmic Scaling to a 0–1 range
df['Search_Volume_Log'] = np.log1p(df['Search_Volume_Capped'])
df['Search_Volume_Scaled'] = df['Search_Volume_Log'] / df['Search_Volume_Log'].max()

# PLOT
sns.histplot(df['Search_Volume_Scaled'], bins=50, kde=True)
plt.title("Log-Scaled_Search_Volume")
plt.show()



# RANKING DIFFICULTY FEATURE




# Fill missing with a high score (e.g., 100 = easy to rank)
df['Ranking Difficulty'] = df['Ranking Difficulty'].fillna(0)

# Invert the values so higher = better
df['Ranking Difficulty Score'] = 100 - df['Ranking Difficulty']

# Optional: Scale to 0–1
df['Ranking Difficulty Scaled'] = df['Ranking Difficulty Score'] / 100




# TOTAL MONTHLY CLICKS FEATURE




mc_threshold = 1000 #threshold to cap outlier

# Get 10th percentile of non-zero click data to apply to zero(omitted) click data
# Looking at the data and the amount of missing data for the respected keywords I dont believe it is wise to unfairly punish
# the missing data so I want a lowest 10th percentile to fill the anything less that the 10th percentile
nonzero_clicks = df['Total Monthly Clicks'][(df['Total Monthly Clicks'] > 0) & (df['Total Monthly Clicks'] <= mc_threshold)]
low_neutral_fill = np.percentile(nonzero_clicks, 10)

# Fill zero/missing/lessthan10% values with 10th percentile
df['Total Monthly Clicks Filled'] = df['Total Monthly Clicks'].apply(
    lambda x: low_neutral_fill if pd.isna(x) or x < low_neutral_fill else x)

# Cap at 1000 to handle outlier
df['Total Monthly Clicks Capped'] = df['Total Monthly Clicks Filled'].apply(lambda x: min(x, 1000))

# Log + scale
# Decided to compress magnitude of the total monthly clicks because the Post-capped distribution is still heavily scewed
# With many Values being low. 
df['Total Monthly Clicks Log'] = np.log1p(df['Total Monthly Clicks Capped'])
df['Total Monthly Clicks Scaled'] = df['Total Monthly Clicks Log'] / df['Total Monthly Clicks Log'].max()

fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Filled version (after neutral fill, before log)
sns.histplot(df['Total Monthly Clicks Filled'], bins=50, ax=axs[0], kde=True)
axs[0].set_title("Total Monthly Clicks (Filled)")

# Log-transformed version
sns.histplot(df['Total Monthly Clicks Log'], bins=50, ax=axs[1], kde=True)
axs[1].set_title("Log-Transformed")

# Scaled version
sns.histplot(df['Total Monthly Clicks Scaled'], bins=50, ax=axs[2], kde=True)
axs[2].set_title("Scaled (0–1)")

plt.tight_layout()
plt.show()




# COST PER CLICK FEATURE



# straight forward for this, do not want to compete for SEO KEYWORDS THAT COST ADVERTISING DOLLARS
df['Broad CPC Score'] = df['Broad Cost Per Click'].apply(
    lambda x: 1 if pd.isna(x) or x == 0 else 0
)



# IS QUESTION? FEATURE


# Boolean value to int.  This is an interesting feature because being a question is a great SEO target
# due to how many people utilize search engine's and ask questions when looking for things.
# not necessarily the end all be all but it should be important.
df['Is Question Flag'] = df['Is Question?'].astype(int)


# IS NOT SAFE FOR WORK FEATURE



# This is an immediate fail for the model. We want to immediately toss out not safe for work keywords
df['NSFW Score'] = (~df['Is Not Safe For Work?']).astype(int)




#PREVIEW
def preview_feature_engineering(df, columns=None, n=10, export_csv=False, filename="feature_preview.csv"):
    """
    Display the first n rows of selected columns (or all engineered columns if none specified).
    Optionally export the result to a CSV file.
    """
    if columns is None:
        # Auto-select only newly created/engineered columns
        engineered_cols = [col for col in df.columns if any(sub in col.lower() for sub in [
            'score', 'scaled', 'capped', 'filled', 'log', 'flag', 'keyword_length'])]
        display_cols = ['Keyword'] + engineered_cols  # Keep the keyword for context
    else:
        display_cols = columns

    preview_df = df[display_cols].head(n)
    
    # Print to terminal
    print(preview_df)

    # Optionally export to CSV
    if export_csv:
        preview_df.to_csv(filename, index=False)
        print(f"\nPreview exported to: {filename}")

preview_feature_engineering(df, None, 100, True, filename="feature_preview.csv")