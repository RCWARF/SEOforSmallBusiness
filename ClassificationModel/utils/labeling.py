


def create_labeling_sample(df, sample_size=300, bin_column='Search_Volume_Scaled', bins=5, filename='labeling_sample.csv'):
    import numpy as np
    import pandas as pd
    
    df = df.copy()

    #stratefy bins for labeling accross multiple segements of data
    df['bin'] = pd.qcut(df[bin_column], q=bins, duplicates='drop')

    sample_df = df.groupby('bin').apply(
        lambda x: x.sample(n=min(len(x), sample_size // bins), random_state=42)
    ).reset_index(drop=True)

    sample_df['Is_Good_Keyword'] = ''
    sample_df['Label_Notes'] = ''

    cols_to_include = ['Keyword'] + [col for col in df.columns if 'score' in col.lower() or 'scaled' in col.lower()]
    sample_df = sample_df[cols_to_include + ['Is_Good_Keyword', 'Label_Notes']]

    sample_df.to_csv(f"outputs/{filename}", index=False)
    print(f"Sample exported to: outputs/{filename}")