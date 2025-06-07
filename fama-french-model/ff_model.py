import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm

style.use("default")

params = {
    "axes.labelsize" : 8, "font.size": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "text.usetex": False,
    "font.family": "sans-serif", "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": "grey", "axes.grid": True, "grid.alpha": 0.5, "grid.linestyle": ":",
}

plt.rcParams.update(params)

# using yahoo finance to download historical data for QQQ over the last 15 years
qqq_daily = yf.download("QQQ", start="2006-01-01", end="2023-12-31", auto_adjust=False)

# Adj Close is not available anymore, therefore using Close or Use auto_adjust to False when downloading data
qqq_daily["Adj Close"].plot(title="QQQ Daily Close", figsize=(5, 3))
plt.show()

#calculate monthly returns of QQQ
print(qqq_daily["Adj Close"])
qqq_monthly = qqq_daily["Adj Close"].resample("ME").ffill()
qqq_monthly.index = qqq_monthly.index.to_period("M")
qqq_monthly["Return"] = qqq_monthly["QQQ"].pct_change() * 100
qqq_monthly.dropna(inplace=True)

print(qqq_monthly)

#Step 2: Load the monthly three factors into a dataframe
# CSV columns: , Mkt-RF, SMB, HML, RF

ff_factors_monthly = pd.read_csv(
    "./data/F-F_Research_Data_Factors-monthly.CSV", index_col=0
)
ff_factors_monthly.index.names = ["Date"]
ff_factors_monthly.index = pd.to_datetime(ff_factors_monthly.index, format="%Y%m")
ff_factors_monthly.index = ff_factors_monthly.index.to_period("M")
# print(ff_factors_monthly)

# Filter factor dates to match the asset
ff_factors_subset = ff_factors_monthly[
    ff_factors_monthly.index.isin(qqq_monthly.index)
].copy()

# Step 3: Calculate excess returns for the asset
ff_factors_subset["Excess_Return"] = qqq_monthly["Return"] - ff_factors_subset["RF"]
print(ff_factors_subset)


# Step 4: Running regression model
X = sm.add_constant(ff_factors_subset[["Mkt-RF", "SMB", "HML"]])
y = ff_factors_subset["Excess_Return"]
model = sm.OLS(y, X).fit()

# Display the summary of the regression
print(model.summary())

# Plot the coefficients and their confidence intervals
factors = model.params.index[1:] # ['Mkt-RF', 'SMB', 'HML']
coefficients = model.params.values[1:]
confidence_intervals = model.conf_int().diff(axis=1).iloc[1]

# Create a Dataframe
ols_data = pd.DataFrame(
    {
        "Factor": factors,
        "Coefficient": coefficients,
        "Confidence_Lower": confidence_intervals[0],
        "Confidence_Upper": confidence_intervals[1]
    }
)

# Plotting
plt.figure(figsize=(3, 3))
sns.barplot(x="Factor", y="Coefficient", data=ols_data, capsize=0.2, palette="coolwarm")

# Add the p-value for each factor to the plot
for i, row in ols_data.iterrows():
    plt.text(
        i, 
        0.2, 
        f"p-value:{model.pvalues[row['Factor']]:.4f}",
        ha="center",
        va="bottom",
        fontsize=6,
    )
    
plt.title("Impact of Fama-French Factors on QQQ Monthly Returns (2006-2023)")
plt.xlabel("Factor"); plt.ylabel("Coefficient Value")
plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
plt.show()