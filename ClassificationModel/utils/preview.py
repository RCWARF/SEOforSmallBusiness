def preview_feature_engineering(df, columns=None, n=10, export_csv=False, filename="feature_preview.csv"):
    import numpy as np
    import pandas as pd

    if columns is None:
        engineered_cols = [col for col in df.columns if any(sub in col.lower() for sub in [
            'score', 'scaled', 'capped', 'filled', 'log', 'flag', 'keyword_length'])]
        display_cols = ['Keyword'] + engineered_cols
    else:
        display_cols = columns

    preview_df = df[display_cols].head(n)
    print(preview_df)

    if export_csv:
        preview_df.to_csv(f"outputs/{filename}", index=False)
        print(f"Preview exported to: outputs/{filename}")