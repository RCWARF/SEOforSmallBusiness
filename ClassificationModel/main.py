import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.preview import preview_feature_engineering
from utils.labeling import create_labeling_sample
from features.feature_engineering import apply_feature_engineering

df = pd.read_csv("data/SalesFunnelSEO.csv")

df = apply_feature_engineering(df)

preview_feature_engineering(df, n=100, export_csv=True)
create_labeling_sample(df, sample_size=400,bin_column='Search_Volume_Scaled')