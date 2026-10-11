# 📈 S&P 500 Stock Analysis and Next-Day Prediction

Exploring 5 years of S&P 500 stock data to find out what can and cannot be predicted about stock price movements, using statistics and machine learning in Python.

🔗 **Original notebook on Kaggle:** [View on Kaggle](https://www.kaggle.com/code/mrshaidary/s-p-500-stock-analysis-and-next-day-prediction)

## 🎯 Key finding
**The direction of price moves is close to random, but the size of moves (volatility) is partly predictable.**

## 📊 Dataset
[S&P 500 stock data (Kaggle)](https://www.kaggle.com/datasets/camnugent/sandp500): daily prices and volumes for 505 companies, February 2013 – February 2018 (619,029 rows after cleaning).

## 🛠️ Tools
Python · pandas · NumPy · matplotlib · SciPy · scikit-learn

## 🔍 What I did
1. **Data cleaning and exploration:** cleaned 619,040 rows and explored Apple's price history
2. **Statistical analysis of returns:** tested normality (Jarque-Bera), fat tails and autocorrelation across all 505 stocks
3. **Direction prediction:** logistic regression and random forest models with engineered features (moving averages, volatility, volume, market-wide returns)
4. **Volatility prediction:** regression models predicting next-week volatility
5. **Honest evaluation:** time-based train/test splits, simple baselines, and a gap to prevent look-ahead bias

## 📈 Results

**Returns are not normally distributed.** Apple's daily returns have fat tails (excess kurtosis 3.77), with 19 extreme days compared with 3.4 expected under a normal distribution, so models assuming normal returns underestimate risk.

**Direction is close to random.** Day-to-day returns show almost no autocorrelation across all 505 stocks.

| Model (next-day direction) | Test accuracy |
|---|---|
| Always guess "up" (baseline) | 52.4% |
| Random forest | 51.6% |
| Logistic regression | 49.6% |

**Volatility is partly predictable.** Big moves tend to follow big moves (volatility clustering: lag-1 correlation of 0.127 for move size vs 0.029 for direction).

| Model (next-week volatility) | MAE |
|---|---|
| Baseline: average volatility | 0.00510 |
| Baseline: last 20 days | 0.00473 |
| **Linear regression** | **0.00435** |
| Random forest | 0.00493 |

Linear regression reduced average error by about 15% compared with the baseline.

## 💡 What I learned
- In financial data, the useful question is often not *which way* prices will move but *how much*
- A more complex model does not help when the signal is not in the data
- Careful baselines and time-based splits are essential for honest evaluation
- Market conditions change (2017 was unusually calm), which affects models trained on past data

## 🚀 Next steps
- Add earnings-announcement dates as a feature
- Use walk-forward validation to retrain on recent data
- Extend the analysis to all 505 stocks

## ▶️ How to run
Open the `.ipynb` notebook in Jupyter or on Kaggle, attach the dataset above, and run the cells in order.
