import pandas as pd
import numpy as np
import cvxpy as cp
import yfinance as yf

esg_data = pd.read_csv("data.csv")  # replaced with whatever the ESG data path, but #this is the kaggle data from link above
esg_data_og = esg_data[["ticker", "total_score"]].dropna(subset=["total_score"])
esg_data_og.rename(columns={"ticker": "Ticker", "total_score": "ESG_Score"}, inplace=True)
esg_data_og["Ticker"] = esg_data_og["Ticker"].str.upper()

# this is the historical prices data
start_date = "2018-01-01"
end_date = "2023-01-01"

tickers = [
   "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMGN", "AMT", "AMZN",
   "APD", "AVGO", "AXP", "BA", "BAC", "BDX", "BK", "BKNG", "BLK", "BMY",
   "BRK-B", "C", "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM",
   "CSCO", "CVS", "CVX", "D", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC",
   "F", "FDX", "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS",
   "HD", "HON", "IBM", "INTC", "INTU", "ISRG", "JNJ", "JPM", "KHC", "KMI",
   "LLY", "LMT", "LOW", "MA", "MAR", "MCD", "MDT", "MET", "MMM", "MO",
   "MRK", "MS", "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE",
   "PG", "PM", "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T",
   "TGT", "TMO", "TMUS", "TRV", "TXN", "UNH", "UNP", "UPS", "USB", "V",
   "VZ", "WBA", "WFC", "WMT", "XOM"
]
# pandas and yfinance data analysis and jargon.
data = yf.download(tickers, start=start_date, end=end_date)
prices = data.xs("Close", axis=1, level=0)
prices = prices.dropna(axis=1, how="all")

all_price_tickers = prices.columns
all_esg_tickers = esg_data_og["Ticker"].unique()


print("All Price Tickers:", all_price_tickers)
print("ESG Tickers:", all_esg_tickers)
valid_tickers = [t for t in all_esg_tickers if t in all_price_tickers]


esg_data_filtered = esg_data_og[esg_data_og["Ticker"].isin(valid_tickers)]
esg_data_filtered = esg_data_filtered.set_index("Ticker")

prices = prices[valid_tickers]

returns = prices.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()

esg_scores = esg_data_filtered["ESG_Score"]
esg_min, esg_max = esg_scores.min(), esg_scores.max()
normalized_esg = (esg_scores - esg_min) / (esg_max - esg_min)

tickers = valid_tickers 
N = len(tickers)
mu = mean_returns[tickers].values
Sigma = cov_matrix.loc[tickers, tickers].values
ESG = normalized_esg[tickers].values


# Building mathematical models used below.
def solve_min_variance(Sigma):
   """
   Traditional risk minimization:
   Minimizing w^T Sigma w subject to sum(w) = 1, w >= 0
   """
   w = cp.Variable(N)
   risk = cp.quad_form(w, Sigma)
   constraints = [cp.sum(w) == 1, w >= 0]
   prob = cp.Problem(cp.Minimize(risk), constraints)
   prob.solve()
   return w.value, prob.value

def solve_max_esg(ESG):
   """
   ESG maximization:
   Maximize sum(w_i * ESG_i) subject to sum(w) = 1, w >= 0
   """
   w = cp.Variable(N)
   esg_obj = ESG @ w
   constraints = [cp.sum(w) == 1, w >= 0]
   prob = cp.Problem(cp.Maximize(esg_obj), constraints)
   prob.solve()
   return w.value, prob.value

def solve_esg_risk_tradeoff(Sigma, ESG, lam):
   """
   Combined ESG-Risk optimization:
   Maximize lam * (sum(w_i * ESG_i)) - (1 - lam)*(w^T Sigma w)
   Constraints ofsum(w)=1, w>=0
   """
   w = cp.Variable(N)
   risk = cp.quad_form(w, Sigma)
   esg_obj = ESG @ w
   objective = cp.Maximize(lam * esg_obj - (1 - lam) * risk)
   constraints = [cp.sum(w) == 1, w >= 0]
   prob = cp.Problem(objective, constraints)
   prob.solve()
   return w.value, prob.value

#results showing
lambda_values = [0.0, 0.5, 1.0]
for lam in lambda_values:
   w_opt, obj_val = solve_esg_risk_tradeoff(Sigma, ESG, lam)
   portfolio_esg = ESG @ w_opt
   portfolio_risk = w_opt.T @ Sigma @ w_opt
   print(f"Lambda = {lam}:")
   print(f"  ESG Score: {portfolio_esg:.4f}")
   print(f"  Risk (Variance): {portfolio_risk:.6f}")
   print("  Weights:", dict(zip(tickers, w_opt)))
   print()

