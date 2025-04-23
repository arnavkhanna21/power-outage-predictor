# Power Outage Predictor

## Introduction
Power outages continue to be some of the most disruptive infrastructure failures in the United States, given our reliance on electricity for basic services, work, and other essential components of our daily lives. They affect millions of Americans yearly. 
The Outage dataset, with **1,534** rows and **56** columns, highlights details and metadata about outages in the Continental US from 2000 to 2016. Some of these features include outage causes, number of customers affected, economic and geographic indictors, grid and utility data, and more. 

The question we aim to answer is: 
**Can we predict the duration of a power outage using a combination of climate, economic, demographic, and grid-related features?**

This is a regression problem, with our target variable being the column 'OUTAGE.DURATION'. The relevant features we wanted to include are:

- `CUSTOMERS.AFFECTED`: Number of customers impacted
- `CAUSE.CATEGORY`: High-level cause of the outage (e.g., severe weather, equipment failure)
- `UTIL.CONTRI`: Utility sector’s contribution to the state GDP
- `POPPCT_URBAN`: Population percentage of the state that is urban
- `AREAPCT_URBAN`: Area percentage of the state that is Urban
- `TOTAL.PRICE`: Average electricity price in the state 
- `CLIMATE.CATEGORY`: Weather climate classification (e.g. hot, cold)
- `U.S._STATE`: State where the outage occurred

With this question, we aim to discover which features are most predictive of how long outages last — and identify which features tend to cause longer interruptions.


## Data Cleaning and Exploratory Data Analysis

### Cleaning Steps 
After loading the excel file to our notebook, we formatted appropriately, setting the header file to the right row, and dropping the placeholder column in the dataset. We also set the index to be a column that the dataset already had as an index, called OBS. We also kept only 13 columns that we were interested in exploring for our regression task, including economic data, urbanization metrics, and climate/geographical data. Furthermore, to reduce skew and modeling noise, we decided to eliminate outliers, or outages that lasted longer than three days (4320 minutes). This left us with 1230 rows, compared to our original 1534. Lastly, all the columns had dtype='OBJECT', so we converted the columns with numeric values to have numeric types.  

Below is the head of our cleaned dataset:

<div align='center'><iframe src="assets/head_table.html" width="100%" height="250" frameborder="0"></iframe></div>

### Univariate Analysis

<div align="center">
  <iframe
    src="assets/duration_hist.html"
    width="800"
    height="400"
    frameborder="0"
  ></iframe>
</div>

The histogram above shows the distribution `OUTAGE.DURATION` under 3 days in the US. Even after removing all the entries with an outage duration longer than 3 days, the data is still heavily right-skewed, with nearly 40% of our data having an outage duration of under 200 minutes.  

### Bivariate Analysis

<div align="center">
  <iframe
    src="assets/scatter_util.html"
    width="800"
    height="400"
    frameborder="0"
  ></iframe>
</div>

This scatter plot shows the relationship Utility Sector contribution to State GDP and Outage Duration. We theorized that if the contribution to the utility sector is a high percentage, then outage duration would be lower due to the larger investment in infrastructure. However, from the data, we saw that there wasn't a strong linear relationship between the two fields, and that it would not be a good predictor for outage duration alone. 

### Grouped Table

| CAUSE.CATEGORY                |   Avg Duration (min) |   Median Duration (min) |   Event Count |   Avg Customers Affected |   Median Customers Affected |
|:------------------------------|---------------------:|------------------------:|--------------:|-------------------------:|----------------------------:|
| fuel supply emergency         |               2283.5 |                  2283.5 |             2 |                      0.5 |                         0.5 |
| severe weather                |               1700.1 |                  1500   |           487 |                 153252   |                    100000   |
| public appeal                 |                920.3 |                   394   |            15 |                   9426.3 |                      7935   |
| system operability disruption |                543.8 |                   191   |            81 |                 210562   |                     69000   |
| equipment failure             |                389.7 |                   184   |            26 |                 109223   |                     49250   |
| intentional attack            |                308.2 |                    44.5 |           190 |                   1875.3 |                         0   |
| islanding                     |                237.1 |                   115.5 |            34 |                   6169.1 |                      2342.5 |

This grouped table shows the aggregated outage and customers affected statistics by the `CAUSE.CATEGORY` column, which has types like severe weather, equipment failure, intentional attack, and others listed in the table above. Severe weather accounts for most of the events and a high mean outage duration. Fuel Supply Emergency accounts for a the longest average duration, but only had 2 events, suggesting that those events are not frequent but extreme. Intentional attacks had the second highest frequency of occurrences, but a low median and mean duration, potentially indicating that these types of outages could be contained and fixed faster. 

### Imputation

For our task, we had to impute 12 values for the `TOTAL.PRICE` column, and 395 values for the `CUSTOMERS.AFFECTED` column. 

For customers affected, we saw that the distribution was heavily right-skewed, and knew that the median would be more robust against outliers, better reflecting the typical outage event. The figure below shows the distribution of the column before and after the imputation, with the taller spike caused by the 395 values imputed with the median. 

<div align='center'><iframe src="assets/customers_imputation_side_by_side.html" width="800" height="400" frameborder="0"></iframe></div>

For total price, we knew that the field was the average monthly electricity price in the corresponding U.S. state of the event. Since there were also only 12 missing values, we decided to impute these by going with the median of the other TOTAL.PRICE values of entries in the same state to preserve regional pricing differences. The figure below shows the difference in the distribution before and after the imputation, with there being little to no difference between the two. 

<div align='center'><iframe src="assets/price_imputation_side_by_side.html" width="800" height="400" frameborder="0"></iframe></div>

## Framing a Prediction Problem
Our goal is to predict the duration of a power outage in minutes, using only information known at the start of the outage This is a regression problem, as the response variable `OUTAGE.DURATION` is numerical and continuous.

We chose outage duration as the prediction target because it is a critical piece of information for both utility companies and the public. Estimating how long an outage might last helps with operational decisions, communication with customers, and improving future response strategies.


### Included Features:
All features used in the model are available at the time of prediction:
- `CUSTOMERS.AFFECTED`: Number of customers impacted
- `CAUSE.CATEGORY`: High-level cause of the outage (e.g., severe weather, equipment failure)
- `UTIL.CONTRI`: Utility sector’s contribution to the state GDP
- `POPPCT_URBAN`: Population percentage of the state that is urban
- `AREAPCT_URBAN`: Area percentage of the state that is Urban
- `TOTAL.PRICE`: Average electricity price in the state 
- `CLIMATE.CATEGORY`: Weather climate classification (e.g. hot, cold)
- `U.S._STATE`: State where the outage occurred

### Evaluation Metrics:
We use two metrics to evaluate performance:
- **Mean Absolute Error/MAE:** Reports average absolute prediction error in minutes. It is easy to interpret but less sensitive to outliers. We preferred MAE to MSE as our data has a lot of outliers and there is a skew in the distribution, so it is robust to these outliers.
- **Mean Squared Error/MSE:** Penalizes large errors heavily and is useful for model comparisons.

The features that we chose to include are known only at the start of the outage, which is crucial for real-world prediction.

## Baseline Model

For our baseline, we trained a Linear Regression model to predict outage duration. Our dataset contained both quantitative and categorical features, so we used a full sklearn Pipeline to preprocess and train the model in one step.

### Features Used
We selected 11 total features for our baseline model:

- **Quantitative Features** :  
  - `CUSTOMERS.AFFECTED_MED_IMPUTED`  
  - `UTIL.CONTRI`  
  - `PC.REALGSP.STATE`  
  - `PC.REALGSP.USA`  
  - `TOTAL.PRICE_IMPUTED`  
  - `POPPCT_URBAN`  
  - `AREAPCT_URBAN`  
  - `POPDEN_URBAN`

- **Categorical features**:  
  - `CAUSE.CATEGORY`  
  - `CLIMATE.CATEGORY`  
  - `U.S._STATE`  

These categorical features were encoded using `OneHotEncoding` with `drop='first'` to avoid multicollinearity.

### Model Evaluation

We trained our model using an 80/20 train-test split. The baseline results were:

- **Mean Absolute Error:** 720.22 minutes  
- **Mean Squared Error:** 985,250.22 sq. minutes

Although the baseline model's absolute error is high with an average nearly 12 hours, this is expected due to the high variance and skew in outage durations, given our histogram from earlier. Some outages last only minutes, while others extend over multiple days. 

## Final Model

For our final model, we aimed to improve on our baseline linear regression by introducing new engineered features, applying more appropriate scaling methods to certain features, and experimenting with more flexible models  that could capture non-linear relationships in the data.

### Engineered Features and Transformations

- `URBAN_DENSITY_RATIO` - Calculated as the percent of population living in urban areas divided by the percent of land that is urban, or the ratio between `POPPCT_URBAN` and `AREAPCT_URBAN`. This feature captures the population in cities relative to their land use, and could affect infrastructure resilience, and in turn outage duration.
We applied tailored preprocessing techniques to specific features:

#### StandardScaler

We used a `StandardScaler` on 

- `UTIL.CONTRI`: Utility sector’s contribution to the state GDP

This column is already on a percentage scale, but it has low variance with a lot of the utility sector's contribution to the GDP being between 1% and 3%. Standardization  ensures that it is on the same scale as other inputs, particularly helpful for regularized models like Lasso.

#### QuantileTransformer

We applied a `QuantileTransformer` with a normal output distributionto:

- `TOTAL.PRICE_IMPUTED`
- `URBAN_DENSITY_RATIO`

We chose these variables as they had highly skewed distributions. The quantile transformer helps smooth extreme values and reshapes the data into a standard normal distribution, which makes learning more stable.

#### Modeling approach

We trained and evaluated two advanced models: **Lasso Regression** and **Random Forest Regression**.


#### Lasso Regression

We used `GridSearchCV` to find the best alpha for lasso regression.

- Hyperparameter tuning: `GridSearchCV` over 10 values of alpha (log-spaced)
- Best alpha: 2.15
- **Performance**:
  - MAE: 708.42 mins
  - MSE: 973,790.75 sq. mins

**Top Weighted Features for Lasso**

| Feature                             | Coefficient |
|-------------------------------------|-------------|
| U.S._STATE_Michigan                 | 1107.60     |
| CAUSE.CATEGORY_severe weather       | 1074.04     |
| CAUSE.CATEGORY_fuel supply emergency| 814.37      |
| U.S._STATE_Minnesota                | 725.66     |
| U.S._STATE_Pennsylvania             | 650.61      |

Lasso helped us identify the most important predictors by assigning zero weight to less relevant features, however it struggled with non-linear relationships with features, and we felt that was more important for prediction as multiple factors combined for outage duration, rather than one feature having a strict importance over the other.

#### Random Forest Regressor

`GridSearchCV` was used to tune max_depth, min_samples_split, and n_estimators.

- Hyperparameter tuning: `GridSearchCV` with
  - `max_depth`: `[10, 20, None]`
  - `n_estimators`: `[100, 200]`
- Best parameters:
  - `max_depth=10`, `min_samples_split=2`, `n_estimators=200`
- **Performance**:
  - MAE: 652.68 mins
  - MSE: 922,291.54 sq. mins

**5 Most Important Features for Random Forest**:

| Feature                          | Importance |
|----------------------------------|------------|
| CAUSE.CATEGORY_severe weather    | 0.301      |
| CUSTOMERS.AFFECTED_MED_IMPUTED   | 0.154      |
| TOTAL.PRICE_IMPUTED              | 0.099      |
| UTIL.CONTRI                      | 0.096      |
| PC.REALGSP.STATE                 | 0.068      |

### Comparison with Baseline Model

| Model                  | MAE (mins)   | MSE (sq. mins) |
|-----------------------|---------------|----------------|
| Baseline (Linear)     | 720.22        | 985,250.22     |
| Lasso Regression      | 708.42        |  973,790.75    |
| Random Forest         | **652.68**    | **922,291.54** |

The Random Forest model did better than both Lasso and the Baseline model on MAE and MSE, which we prioritized as evaluation metrics since they one more intuitively measured the average prediction error in minutes and the other was robust to outliers. 

The Random Forest model also captures nonlinear relationships and interactions between features, which was not possible with the Baseline linear model or Lasso. With our multitude of inpt features, this was the ideal choice for the final model. This,combined with its strong MAE performance and better interpretability with feature importance scores, made Random Forest the best candidate for our final model.
