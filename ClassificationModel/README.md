Feature Selection & Preprocessing Strategy

This repository contains a Random Forest Classification model designed to predict SEO keyword viability for small businesses that rely on organic reach rather than paid advertising.

Objective

The model classifies keywords as either "viable" or "not viable" for organic SEO targeting, using structured features extracted and transformed from keyword analysis data. The goal is to help small businesses identify long-tail, low-competition, high-intent keywords worth targeting — without relying on pay-per-click strategies.

---

Selected Features and Data-Engineering Rationale

| Feature                       | Description                                                                |
|-------------------------------|-----------------------------------------------------------------------------|
| `Keyword_Length_Score`        | Ordinal score based on the number of words in the keyword (more = better).  |
| `Search Volume Scaled`        | Log-scaled and capped search volume to reduce the influence of outliers.    |
| `Ranking Difficulty Scaled`   | Inverted and scaled difficulty; lower difficulty = higher score.            |
| `Total Monthly Clicks Scaled` | Log-scaled click data; smoothed using the 10th percentile for missing/low.  |
| `Broad CPC Score`             | Binary: 1 = zero or missing cost; 0 = any CPC value.                        |
| `Is Question Flag`            | Binary: 1 if the keyword is a question (`True`), 0 otherwise.               |
| `NSFW Score`                  | Binary: 1 = safe content, 0 = not safe for work (disqualifying).            |

Unstructured columns like `Ads`, `SERP Features CSV`, and `Organic Clicks Percent` were excluded due to:
- Ambiguous interpretation without additional NLP processing
- Strong correlation to paid advertising which this model is designed to avoid

---

Preprocessing Steps

1. Missing Values
   - `Search Volume` and `Total Monthly Clicks` filled using zeros or low percentiles to avoid penalizing uncertain data
   - `Ranking Difficulty` filled with `0` and inverted (100 - x) to reflect ease of ranking, (not having a difficulty ranking means the keywords arent difficult to compete for)

2. Outlier Handling
   - Capped `Search Volume` at 1400 and `Total Monthly Clicks` at 1000 based on dataset inspection
   - Capped values were log-transformed for compression. (The Bulk of the data for `Search Volume`and `Total Monthly Clicks` were in 0-300 type range but a few top heavy values caused any kind of linear scaling to have dominated data at the top and almost all of the data points were squeezed too close together for differentation.)

3. Feature Transformation
   - `np.log1p()` used on skewed continuous data
   - All continuous features scaled to a 0–1 range post-log for comparability (ranking Difficulty did not require log scaling)

4. Binary and Ordinal Encoding
   - Boolean values converted to `int`
   - Manual scoring systems applied to `Keyword Length` and CPC relevance

---

Random Forest

- Supports both continuous and categorical features natively
- Automatically identifies important features
- Robust to noise, overfitting, and missing values
- Provides interpretable feature importances for model review

---

Next Steps

- Manually label a portion of the dataset (e.g., `Is_Good_Keyword = 1/0`)
- Split labeled data for training and validation
- Use the trained model to predict unlabeled keyword viability
- Export and review top-performing keywords for use in SEO campaigns

