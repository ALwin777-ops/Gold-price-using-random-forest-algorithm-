# Gold Price Prediction using Random Forest Regressor

This project predicts the price of gold (GLD) using machine learning, specifically a Random Forest Regressor model. The dataset used is `gld_price_data.csv`, which contains historical gold price data along with several other features. The aim is to predict future gold prices based on historical data.

## Steps Involved

1. **Data Collection & Preprocessing**:
   - The dataset is loaded using pandas and basic exploratory data analysis (EDA) is performed, including:
     - Checking for missing values
     - Understanding the shape of the data
     - Viewing the first and last few rows

2. **Correlation Analysis**:
   - The correlation between different features and the target variable (GLD) is calculated, and the most relevant features are identified.

3. **Feature and Target Separation**:
   - The features (`x`) are separated from the target variable (`y`), and the data is split into training and testing sets (80% training, 20% testing).

4. **Model Building**:
   - A Random Forest Regressor model is trained using the training set. The model learns the patterns in the data to predict gold prices.

5. **Prediction & Evaluation**:
   - After training, the model predicts the gold prices for the test set. The model's performance is evaluated using the R-squared metric.
   - The predicted values are plotted against the actual values to visually assess the model's accuracy.

## Requirements

To run this project, you'll need the following libraries:

- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
