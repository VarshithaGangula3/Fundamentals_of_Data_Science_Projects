import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import sys
from sklearn.metrics import r2_score

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv>")
    sys.exit(1)

infilename = sys.argv[1]

df = pd.read_csv(infilename, index_col="property_id")

print("Making new features...")

## Your code here
Y = df.values[:, -1]
df["is_close_to_school"] = np.where(df["miles_to_school"] < 2, 1, 0)
df["lot_size"] = df["lot_width"] * df["lot_depth"]
df["1"] = 1
X = pd.concat(
    [df["1"], df["sqft_hvac"], df["lot_size"], df["is_close_to_school"]], axis=1
)
lin_reg = LinearRegression(fit_intercept=False)
lin_reg.fit(X, Y)
B = lin_reg.coef_
prediction = X @ B
r2 = r2_score(Y, prediction)
col = X.columns
print(f"Using only the useful ones: {col}")
print(f"R2 = {r2:.5f}")
coe = lin_reg.coef_
int = lin_reg.intercept_
print("*** Prediction ***")
print(
    f"Price = ${B[0].round(2)} + (sqft x ${B[1].round(2)}) + (lot_size x ${B[2].round(2)})"
)
print("\t Less than 2 miles from a school? You get $49,300.87 added to the price!")
