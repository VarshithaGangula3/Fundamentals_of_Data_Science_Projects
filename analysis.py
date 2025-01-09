import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys

# Deal with command-line
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv>")
    sys.exit(1)
infilename = sys.argv[1]
# Read in the basic data frame
df = pd.read_csv(infilename, index_col="property_id")
X_basic = df.values[:, :-1]
labels_basic = df.columns[:-1]
Y = df.values[:, -1]
# Expand to a 2-degree polynomials
## Your code here
trans = PolynomialFeatures(degree=2)
data = trans.fit_transform(X_basic)
labels_basic_c = trans.get_feature_names_out(labels_basic)
# labels_basic_t
# Prepare for loop
residual = Y
X_basic_c = pd.DataFrame(data, columns=labels_basic_c)
in1 = X_basic_c.iloc[:, 0]
feature_indices = [0]
while len(feature_indices) < 3:
    a = []
    for i in range(1, 21):
        pe, p = pearsonr(data[:, i], residual)
        a.append({"feature:": labels_basic_c[i], " vs residual: pvalue": p})
    a.sort(key=lambda x: x[" vs residual: pvalue"])
    for i in range(len(a)):
        print(
            f'\t"{a[i]["feature:"]}"vs residual: p_value={a[i][" vs residual: pvalue"]}'
        )
    # We always need the column of zeros to
    # include the intercept
    h = a[0]["feature:"]
    in2 = X_basic_c[h]
    input = pd.concat([in1, in2], axis=1)
    in1 = input
    lin_reg = LinearRegression(fit_intercept=False)
    lin_reg.fit(in1, Y)
    B = lin_reg.coef_
    prediction = in1 @ B
    from sklearn.metrics import r2_score
    r2 = r2_score(Y, prediction)
    print(f"*** Fitting with {list(in1)} ***")
    print(f"R2 = ", r2)
    residual = Y - prediction
    feature_indices.append(3)
    print("Residual is updated")
print("Making scatter plot: age_of_roof vs final residual")
fig, ax = plt.subplots()
ax.scatter(X_basic[:, 3], residual, marker="+")
fig.savefig("ResidualRoof.png")

print("Making a scatter plot: miles_from_school vs final residual")
fig, ax = plt.subplots()
ax.scatter(X_basic[:, 4], residual, marker="+")
fig.savefig("ResidualMiles.png")