# Flight Ticket Price Prediction — Final Project Report

> **Machine Learning Course Project — I**  
> **Evaluation Metric**: Root Mean Squared Error (RMSE)  
> **Course Scope**: Supervised Learning (Part 1 Only — Strictly No Unsupervised Learning)

---

## 1. Problem and Dataset

### 1.1 Problem Description
Predicting commercial airline ticket prices is a core regression problem in revenue management and aviation economics. Airline ticket pricing is highly dynamic and non-linear, influenced by carrier market share, booking lead time, geographical routing, cabin class distinctions, layovers, and airport landing fees.

The goal of this project is to develop an end-to-end, leak-free supervised machine learning regression pipeline to predict the continuous flight price (`price` in USD) for domestic commercial flights across California. The primary evaluation metric for the competition is **Root Mean Squared Error (RMSE)**:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

To provide a thorough evaluation, secondary metrics including **Mean Absolute Error (MAE)**, **$R^2$ Score (Coefficient of Determination)**, **Median Absolute Error (MedAE)**, **Root Mean Squared Logarithmic Error (RMSLE)**, and **Mean Absolute Percentage Error (MAPE)** are also tracked across all models.

### 1.2 Dataset Overview
The dataset contains commercial domestic flight records connecting five major California cities: Los Angeles, San Diego, San Francisco, San Jose, and Sacramento.

| Split | Number of Rows | Number of Columns | Target Variable | Missing Values |
|:---|:---:|:---:|:---|:---:|
| **Training Set** (`train.csv`) | 18,076 | 12 | `price` ($0.00 to $8,996.00) | 0 (after deterministic prefix imputer) |
| **Test Set** (`test.csv`) | 4,520 | 11 | `price` (Required Prediction) | 0 (after deterministic prefix imputer) |
| **Submission** (`submission.csv`) | 4,520 | 2 | `id`, `price` | Strict 4,521 lines including header |

### 1.3 Feature Descriptions

| Feature Name | Data Type | Description | Values / Range |
|---|---|---|---|
| `id` | Identifier | Unique index for each flight record | Integer index |
| `airline` | Categorical | Operating commercial air carrier | Southwest, United, American Airlines, Alaska Airlines, Delta, Frontier, Southern Airways Express |
| `flight` | String | Flight code with 2-letter IATA carrier prefix and number | e.g., `WN 3596`, `UA 1763`, `AA 6274`, `AS 3359` |
| `source_city` | Categorical | Departure origin airport city | Los Angeles, San Diego, San Francisco, San Jose, Sacramento |
| `departure_time` | Categorical | Time slot of flight departure | Early_Morning, Morning, Afternoon, Evening, Night, Late_Night |
| `stops` | Categorical | Number of intermediate transit stops | `zero` (non-stop), `one`, `two_or_more` |
| `arrival_time` | Categorical | Time slot of flight destination arrival | Early_Morning, Morning, Afternoon, Evening, Night, Late_Night |
| `destination_city` | Categorical | Arrival destination airport city | Same 5 California cities |
| `flight_class` | Categorical | Cabin travel class | `Economy`, `Business` |
| `duration` | Continuous | Total travel duration in hours | 0.87 to 28.65 hours |
| `days_left` | Discrete | Booking lead time (days remaining until departure) | 1 to 44 days |
| **`price`** | **Continuous** | **Target variable: Ticket price in USD** | **$0.00 to $8,996.00** |

---

## 2. EDA

Exploratory Data Analysis was performed to uncover key structural patterns, pricing distributions, and domain dynamics. Visual plots were generated and saved directly to the [`eda_plots/`](file:///c:/Users/pradh/Downloads/kaggleReg/EXP12/EXP12/eda_plots) directory:

### 2.1 Key Observations from EDA

1. **Target Variable (`price`) Distribution & Skewness**:
   - **Summary Statistics**: Minimum = **$0.00**, 25th percentile ($Q_1$) = **$157.00**, Median ($Q_2$) = **$234.00**, Mean = **$280.26**, 75th percentile ($Q_3$) = **$336.00**, 95th percentile = **$656.25**, 99th percentile = **$1,016.00**, Maximum = **$8,996.00**.
   - Standard Deviation = **$221.97**, Skewness = **+7.07**, Kurtosis = **170.25**.
   - *Observation*: The target exhibits a massive positive right tail. Exactly 191 records (1.06% of the data) exceed $1,000, reaching up to $8,996. Meanwhile, 98.94% of flights are densely clustered below $1,000 with a median of $234.
   - *Zero-Price Records*: 25 promotional/award-mile flights in the training data have `price == 0.00`. Retaining them preserves model calibration on commercial award routes.
   - *Plot Reference*: `eda_plots/01_price_distribution.png`

2. **Cabin Class Disparities**:
   - **Business Class**: Mean fare = **$393.96** (Median: $323.00, Std: $278.75).
   - **Economy Class**: Mean fare = **$202.09** (Median: $194.00, Std: $121.34).
   - Business class commands an average **+94.94% price premium** over economy seats.
   - *Plot Reference*: `eda_plots/02_price_by_airline_and_class.png`

3. **The Southwest Airlines Fleet Anomaly**:
   - Southwest Airlines operates **100% Economy seats** (3,895 rows in the training set, exactly 0 Business rows). This accurately reflects real-world single-cabin airline fleet configurations.
   - *Carrier Mean Price Hierarchy*:
     1. American Airlines: **$349.44** (Business: $448.40, Economy: $240.90)
     2. United Airlines: **$319.03** (Business: $425.89, Economy: $213.64)
     3. Alaska Airlines: **$302.62** (Business: $393.13, Economy: $192.17)
     4. Delta Air Lines: **$286.53** (Business: $356.64, Economy: $209.17)
     5. Southwest Airlines: **$197.84** (Economy only: $197.84)
     6. Frontier Airlines: **$178.58** (Business: $234.24, Economy: $135.85)

4. **Booking Lead Time Curve (`days_left` vs `price`)**:
   - Ticket prices rise exponentially as departure day approaches:
     - **1–2 days left**: Mean price = **$382.86** (urgency peak; max spike of $8,996 occurs here).
     - **3–7 days left**: Mean price = **$355.65**.
     - **8–15 days left**: Mean price = **$320.47**.
     - **16–30 days left**: Mean price = **$257.28**.
     - **31+ days left**: Mean price = **$220.64** (advance purchase window).
   - *Plot Reference*: `eda_plots/03_days_left_vs_price.png`

5. **Flight Duration & Layover Stops**:
   - **Direct flights (`zero` stops)**: 75.0% of data (13,559 flights), mean price = **$229.95**.
   - **1 stop (`one`)**: 23.9% of data (4,324 flights), mean price = **$431.53**.
   - **2+ stops (`two_or_more`)**: 1.1% of data (193 flights), mean price = **$425.74**.
   - Flights with 1 or more stops cost **87.7% more** than direct flights due to multi-airport landing fees and connection costs.
   - *Plot Reference*: `eda_plots/04_duration_vs_price.png`

6. **Route Geography**:
   - **Most expensive route**: `Sacramento -> San Jose` (Mean: **$583.65**, Median: $535.50).
   - **Most frequent route**: `San Francisco <-> Los Angeles` (> 4,000 flights, Mean: ~$233 – $235).
   - **Lowest average route**: `San Francisco <-> San Diego` (Mean: ~$189 – $190).
   - *Plot Reference*: `eda_plots/05_route_price_heatmap.png` & `eda_plots/06_departure_arrival_heatmap.png`

---

## 3. Data Preprocessing

All preprocessing is strictly supervised and domain-driven, implemented inside [`preprocessor.py`](file:///c:/Users/pradh/Downloads/kaggleReg/EXP12/EXP12/preprocessor.py). **No unsupervised methods (no PCA, no K-Means clustering, no autoencoders, no unsupervised KNN)** were utilized, ensuring strict alignment with Course Part 1 syllabus requirements.

### 3.1 Preprocessing Steps & Justification

1. **Deterministic Tier-1 Domain Imputation for Airline Carriers**:
   - **Implementation**: The flight code prefix maps deterministically to the operating airline:
     - `WN` $\rightarrow$ Southwest
     - `UA` $\rightarrow$ United
     - `AA` $\rightarrow$ American Airlines
     - `AS` $\rightarrow$ Alaska Airlines
     - `DL` $\rightarrow$ Delta
     - `F9` $\rightarrow$ Frontier
     - `9X` $\rightarrow$ Southern Airways Express
   - **Justification**: Solves missing or corrupted airline labels with 100.0% deterministic domain ground-truth without heuristic imputation.

2. **Categorical Mappings & Encodings**:
   - `stops_num`: Ordinal mapping `{'zero': 0, 'one': 1, 'two_or_more': 2}`.
   - `flight_class_num`: Binary mapping `{'Economy': 0, 'Business': 1}` (`is_business`).
   - `is_direct`: Binary flag indicating non-stop travel (`stops == 'zero'`).
   - `route`: Composite origin-destination feature (`source_city + '_' + destination_city`).
   - `airline_route`: Interaction between operating carrier and route corridor.
   - `dep_arr_time`: Interaction between departure and arrival time blocks.

3. **Geodesic Haversine Distance & Flight Physics**:
   - **Airport Coordinates**: Exact coordinates assigned for Los Angeles `(34.0522, -118.2437)`, San Diego `(32.7157, -117.1611)`, San Francisco `(37.7749, -122.4194)`, San Jose `(37.3382, -121.8863)`, and Sacramento `(38.5816, -121.4944)`.
   - `dist_km`: Haversine great-circle aerial distance in kilometers.
   - `effective_speed`: Ratio of aerial distance to duration ($\frac{\text{dist\_km}}{\text{duration}}$ in km/h), distinguishing cruising direct flights from layover-delayed flights.
   - `est_layover_hours`: Estimated ground delay $\max(0, \text{duration} - \frac{\text{dist\_km}}{600})$.
   - `duration_per_stop`: Average flying time per transit segment ($\frac{\text{duration}}{\text{stops\_num} + 1}$).

4. **Zero-Leakage Frequency Encodings**:
   - Computed for `flight_freq`, `route_freq`, and `airline_route_freq`.
   - **Justification**: Captures flight frequency and carrier schedule density. Frequencies are learned **strictly on the training split** of each cross-validation fold to prevent data leakage. Unseen test frequencies default safely to 1.

5. **Non-Linear Lead Time Transformations**:
   - `days_left_log`: $\ln(1 + \text{days\_left})$
   - `days_left_inv`: $\frac{1}{\text{days\_left} + 1}$
   - Binary urgency flags: `is_last_minute` ($\le 2$ days), `is_urgent` ($\le 7$ days), `is_advance` ($\ge 30$ days).
   - **Justification**: Models the sharp exponential inflection in airline yield curves during the final 48 hours.

6. **Domain Interaction Features**:
   - `business_x_duration`: Premium pricing scaling with travel duration.
   - `business_x_stops`: Extra connection surcharges for business travelers.
   - `business_x_speed`: Isolates direct, non-stop business express flights.
   - `business_x_urgent`: Captures corporate emergency bookings.

7. **Dual-Representation Pipeline (Tree vs Linear)**:
   - **Tree-Based Models (Decision Tree, Random Forest, LightGBM, CatBoost, XGBoost)**:
     - 35 dense features using label encoding for categoricals and raw non-linear continuous features. Trees naturally split on monotonic thresholds.
   - **Linear Models (Linear Regression, Ridge, Lasso)**:
     - 76 features using `StandardScaler` on continuous features + `OneHotEncoder(handle_unknown='ignore')` on categorical features.
     - **Justification**: Linear and regularized models require unit-variance scaling and one-hot orthogonal representation so that regularization penalties ($L_1$ and $L_2$) treat all dimensions equitably.

---

## 4. Models Used

### 4.1 All Models Tried & Best Score Achieved (Primary Competition Table)

| Model | Best Score Achieved (5-Fold CV RMSE) |
|---|:---:|
| **Calibrated Ensemble (Nelder-Mead)** | **121.15** |
| **Top-3 Equal Ensemble** | **122.13** |
| **Hybrid CatBoost (2-Stage Dual-Regime)** | **122.54** |
| **Hybrid XGBoost (2-Stage Dual-Regime)** | **122.63** |
| **XGBoost (Global Log-Space)** | **122.79** |
| **Hybrid LightGBM (2-Stage Dual-Regime)** | **123.05** |
| **LightGBM (Global Log-Space)** | **123.16** |
| **CatBoost (Global Log-Space)** | **125.34** |
| **Random Forest (Regression Forest)** | **127.04** |
| **Decision Tree (Regression Tree)** | **134.83** |
| **Linear Regression (OLS Baseline)** | **150.14** |
| **Ridge Regression ($L_2$ Regularized)** | **150.59** |
| **Lasso Regression ($L_1$ Regularized)** | **151.07** |

---

### 4.2 Comprehensive Multi-Metric Regression Breakdown

| Model / Pipeline | RMSE ($) | MAE ($) | $R^2$ Score | MedAE ($) | RMSLE | MAPE (%) | Ensemble Weight | Category / Role |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **Calibrated Ensemble** | **121.15** | **41.99** | **0.7021** | **17.47** | **0.2944** | **43.95%** | **1.0000** | **Final Submission** |
| **Top-3 Equal Ensemble** | **122.13** | **42.74** | **0.6973** | **17.57** | **0.3064** | **56.28%** | **0.3333** | **Alternative Submission** |
| **Hybrid CatBoost (2-Stage)** | **122.54** | **44.90** | **0.6952** | **21.43** | **0.3150** | **59.58%** | **0.2087** | **Core Ensemble** |
| **Hybrid XGBoost (2-Stage)** | **122.63** | **43.09** | **0.6947** | **17.65** | **0.3089** | **55.65%** | **0.2282** | **Core Ensemble** |
| **XGBoost (Global Log)** | 122.79 | 41.89 | 0.6939 | 16.51 | 0.2759 | 30.01% | 0.0000 | Core Ensemble Candidate |
| **Hybrid LightGBM (2-Stage)** | 123.05 | 42.80 | 0.6927 | 16.48 | 0.3056 | 54.70% | 0.1144 | Core Ensemble |
| **LightGBM (Global Log)** | 123.16 | 41.96 | 0.6921 | 16.01 | 0.2780 | 28.76% | 0.2571 | Core Ensemble |
| **CatBoost (Global Log)** | 125.34 | 46.15 | 0.6811 | 21.91 | 0.2930 | 30.18% | 0.1916 | Core Ensemble |
| **Random Forest (Regression Forest)** | 127.04 | 44.68 | 0.6724 | 14.54 | 0.3180 | 55.61% | 0.0000 | Core Tree Candidate |
| **Decision Tree (Regression Tree)** | 134.83 | 49.29 | 0.6310 | 15.25 | 0.3389 | 57.43% | 0.0000 | Standalone Tree Baseline |
| **Linear Regression (OLS)** | 150.14 | 75.83 | 0.5425 | 49.84 | 0.5140 | 88.12% | 0.0000 | Standalone Linear Baseline |
| **Ridge Regression ($L_2$)** | 150.59 | 75.75 | 0.5397 | 49.90 | 0.5103 | 87.15% | 0.0000 | Regularized Linear Baseline |
| **Lasso Regression ($L_1$)** | 151.07 | 75.64 | 0.5368 | 49.65 | 0.4950 | 87.59% | 0.0000 | Regularized Linear Baseline |

---

## 5. Hyperparameter Tuning

All tuning was performed under strict 5-fold cross-validation (`random_state=42`) with zero test leakage:

### 5.1 Baseline Linear Regressors
1. **Linear Regression (OLS)**:
   - Trained on StandardScaled numerical features + OneHot encoded categorical dummies.
   - Non-negative projection: `np.clip(pred, 0, None)`.
   - Settings: `fit_intercept=True`, `copy_X=True`.
2. **Ridge Regression ($L_2$ Regularization)**:
   - Evaluated $\alpha \in \{0.01, 0.1, 1.0, 10.0, 100.0\}$. Optimal setting: $\alpha = 10.0$.
   - Settings: `alpha=10.0`, `solver='auto'`, `random_state=42`.
   - Effect: Effectively shrinks multicollinear weights from one-hot route indicators.
3. **Lasso Regression ($L_1$ Regularization)**:
   - Evaluated $\alpha \in \{0.001, 0.01, 0.1, 1.0\}$. Optimal setting: $\alpha = 0.1$.
   - Settings: `alpha=0.1`, `max_iter=2000`, `tol=1e-4`, `random_state=42`.
   - Effect: Sparse feature selection by setting minor dummy coefficients to exactly zero.

### 5.2 Tree Models (Decision Tree & Random Forest)
4. **Decision Tree Regressor (Regression Tree)**:
   - Explored `max_depth` $\in \{6, 8, 10, 12, 16, \text{None}\}$. Depths beyond 12 caused severe training overfitting ($R^2_{\text{train}} > 0.85$ vs $R^2_{\text{val}} \approx 0.59$).
   - Optimal setting: `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=5`, `criterion='squared_error'`.
5. **Random Forest Regressor (Regression Forest)**:
   - Explored `n_estimators` $\in \{50, 100, 200\}$, `max_depth` $\in \{12, 16, 20\}$, `max_features` $\in \{'sqrt', 0.5, 'all'\}$.
   - Optimal setting: `n_estimators=100`, `max_depth=16`, `min_samples_split=6`, `min_samples_leaf=3`, `max_features='sqrt'`, `n_jobs=-1`.
   - Effect: Ensemble bagging dramatically reduced variance over the single decision tree, lowering RMSE from 134.83 to 127.04.

### 5.3 Global Log-Space Gradient Boosters
6. **LightGBM Regressor**:
   - Target transformed via $\ln(1 + \text{price})$.
   - Optimal setting: `n_estimators=650`, `learning_rate=0.035`, `num_leaves=63`, `subsample=0.85`, `colsample_bytree=0.85`, `random_state=42`.
7. **CatBoost Regressor**:
   - Target transformed via $\ln(1 + \text{price})$.
   - Optimal setting: `iterations=650`, `learning_rate=0.04`, `depth=6`, `l2_leaf_reg=3.0`, `loss_function='RMSE'`.
8. **XGBoost Regressor**:
   - Target transformed via $\ln(1 + \text{price})$.
   - Optimal setting: `n_estimators=550`, `learning_rate=0.04`, `max_depth=6`, `min_child_weight=3`, `subsample=0.85`, `colsample_bytree=0.85`, `reg_lambda=2.0`, `tree_method='hist'`.

### 5.4 Hybrid 2-Stage Dual-Regime Models
To overcome the limitation where extreme outliers (> $1,000) distort MSE gradients on normal rows while global log-space models amplify small errors upon exponentiation:
- **Stage 1 (Tail Classifier)**: LightGBM Classifier (`n_estimators=300`, `learning_rate=0.05`, `num_leaves=31`, `max_depth=6`) predicting $P(\text{price} > \$1,000)$.
- **Stage 2 (Normal Regressor)**: Trained on records with $\text{price} \le \$1,000$ using **RAW price** $p$.
- **Stage 2 (Outlier Regressor)**: Trained on records with $\text{price} > \$1,000$ using **$\ln(1 + \text{price})$**.
- **Soft-Gated Blending**:
  $$\hat{y} = (1 - P) \cdot \hat{y}_{\text{normal}} + P \cdot \text{expm1}(\hat{y}_{\text{outlier}})$$
- **Tuned Hyperparameters**:
  - *Hybrid CatBoost*: Normal (`iterations=750, depth=7, lr=0.04`), Outlier (`iterations=250, depth=4, lr=0.05`).
  - *Hybrid XGBoost*: Normal (`n_estimators=700, max_depth=6`), Outlier (`n_estimators=200, max_depth=4`).
  - *Hybrid LightGBM*: Normal (`n_estimators=750, num_leaves=95`), Outlier (`n_estimators=250, num_leaves=31`).

### 5.5 Optimal Ensemble Optimization (Nelder-Mead)
- Multi-start bounded Scipy Nelder-Mead optimization across 5 candidate models to find non-negative weights summing to 1.0 that directly minimize cross-validation RMSE:
  - **LightGBM (Global Log)**: **0.2571** (25.71%)
  - **Hybrid XGBoost**: **0.2282** (22.82%)
  - **Hybrid CatBoost**: **0.2087** (20.87%)
  - **CatBoost (Global Log)**: **0.1916** (19.16%)
  - **Hybrid LightGBM**: **0.1144** (11.44%)
- **Resulting RMSE**: **121.15** (Best score across the entire project).

---

## 6. Results and Discussion

### 6.1 Algorithmic Performance Progression

```
Hierarchy of Evaluated Supervised Models:
1. Linear Regression (OLS)        : 150.14 RMSE (R² = 0.5425)  [Baseline]
2. Decision Tree (Regression Tree): 134.83 RMSE (R² = 0.6310)  [-15.31 error reduction]
3. Random Forest (100 Trees)      : 127.04 RMSE (R² = 0.6724)  [-7.79 error reduction]
4. Gradient Boosting (XGB/LGB)    : 122.79 RMSE (R² = 0.6939)  [-4.25 error reduction]
5. Hybrid Dual-Regime (CatBoost)  : 122.54 RMSE (R² = 0.6952)  [-0.25 error reduction]
6. Top-3 Equal Ensemble           : 122.13 RMSE (R² = 0.6973)  [-0.41 error reduction]
7. Calibrated Optimal Ensemble     : 121.15 RMSE (R² = 0.7021)  [-0.98 error reduction]
```

### 6.2 Discussion & Comparative Insights
1. **Limitations of Linear Models on Dynamic Fares**:
   - Linear Regression, Ridge, and Lasso achieve RMSE ~150.1–151.0 with $R^2 \approx 0.54$. Although one-hot encoding captures airline-specific mean price baselines, linear formulations cannot naturally handle the exponential price jump in the final 48 hours or route-specific connection surcharges without manual high-degree polynomial expansions.
2. **Decision Tree vs. Random Forest Ensemble**:
   - A single Decision Tree with depth 12 lowers RMSE by **15.31 points** to 134.83.
   - Ensembling 100 de-correlated trees in Random Forest averages out individual split variance, driving RMSE down to **127.04** (an 8.2% relative error reduction over the single tree).
3. **Power of Gradient Boosted Decision Trees**:
   - XGBoost, LightGBM, and CatBoost achieve RMSE scores of 122.79, 123.16, and 125.34 respectively. By iteratively fitting shallow trees to the negative gradient of the loss function, gradient boosters efficiently map non-linear relationships across effective speed, layovers, and booking urgency.
4. **Benefit of Hybrid Dual-Regime Routing**:
   - The dual-regime architecture decouples the dense normal fare structure ($\le \$1,000$) from sparse extreme business spikes ($> \$1,000$). Hybrid CatBoost and Hybrid XGBoost achieve **122.54** and **122.63 RMSE**, outperforming their global single-model counterparts.
5. **Ensemble Gain via Mathematical Diversity**:
   - Blending raw-space hybrid models with log-space global models provides complementary error profiles. Nelder-Mead optimization finds the optimal weighting that achieves the lowest overall error: **121.15 RMSE**, **41.99 MAE**, and **$R^2 = 0.7021$** (capturing over 70.2% of total price variance).

### 6.3 Critical Engineering Insight: Detection and Elimination of the `id` Feature Leakage Artifact
During initial exploratory prototyping, an earlier experimental run incorporated the `id` column into feature representations. This prototype achieved an artificially high score that topped the public leaderboard. 

However, rigorous diagnostic analysis immediately revealed that this was a classic manifestation of **target leakage via dataset indexing artifacts**:
1. **Origin of the Leak**: The row `id` values were not purely random identifiers; they reflected the sequential order of web scraping or database insertion. Because flight queries were scraped in batches by route and date, `id` acted as an accidental proxy for price clusters and temporal scrape windows.
2. **Artificial Leaderboard Inflation vs. Genuine Generalization**: While exploiting `id` allowed the model to memorize the test split's indexing pattern, it provided zero real-world economic utility and would fail catastrophically on genuinely new flight queries. Relying on an index artifact also defeats the academic integrity and foundational principles of machine learning.
3. **The Clean, Leak-Free Paradigm**: In accordance with course guidelines and sound engineering ethics, **all `id`-based features were permanently eliminated**. The `id` column is strictly restricted to joining the final output rows with the submission file. The current pipeline (`preprocessor.py` and `train_and_predict.py`) relies exclusively on domain physics, geodesic distance, layover hours, and booking lead-time dynamics. The resulting 5-fold CV score of **121.15 RMSE** ($R^2 = 0.7021$) reflects an honest, robust, and reproducible benchmark of true out-of-sample generalization.

---

## 7. Best Model

The best-performing model identified across all experiments and cross-validation folds is the **Calibrated Blended Ensemble**:

- **Model Identity**: Multi-Model Calibrated Optimal Ensemble combining LightGBM (Global Log, 25.71%), Hybrid XGBoost (22.82%), Hybrid CatBoost (20.87%), CatBoost (Global Log, 19.16%), and Hybrid LightGBM (11.44%).
- **Cross-Validation Performance**:
  - **RMSE**: **121.15**
  - **MAE**: **41.99**
  - **$R^2$ Score**: **0.7021**
  - **Median Absolute Error (MedAE)**: **$17.47**
  - **RMSLE**: **0.2944**
  - **MAPE**: **43.95%**
- **Public Submission File**: [`submission.csv`](file:///c:/Users/pradh/Downloads/kaggleReg/EXP12/EXP12/submission.csv)
- **Submission Verification Checklist**:
  - [x] Strict **4,520 rows** (4,521 lines with header `id,price`).
  - [x] Zero missing, null, or infinite values.
  - [x] All IDs match test set monotonically (min ID: 3, max ID: 22,594).
  - [x] Predicted ticket prices are strictly non-negative:
    - Minimum predicted price: **$27.94**
    - Maximum predicted price: **$2,029.21**
    - Mean predicted price: **$277.24** (closely matching train set mean of $280.26).

---

## 8. GitHub Link

**Source Code Repository**:  
[`https://github.com/pradh-ai/flight-price-prediction-course-project`](https://github.com/pradh-ai/flight-price-prediction-course-project) *(Replace with your team's GitHub repository link prior to report submission)*

### Repository Reproduction Instructions:
```bash
# Clone the repository
git clone https://github.com/pradh-ai/flight-price-prediction-course-project.git
cd flight-price-prediction-course-project

# Run the complete end-to-end multi-model pipeline
python train_and_predict.py
```
*Executing the script automatically performs 5-fold cross-validation, trains all 11 supervised models, performs Nelder-Mead ensemble optimization, computes all metrics, and generates all 13 individual submission CSVs in `predictions/` and summaries in `metrics/`.*
