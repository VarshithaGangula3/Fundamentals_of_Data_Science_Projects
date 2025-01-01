from contextlib import AsyncExitStack
import pandas as pd
import numpy as np


# Read in the excel file
# Returns:
#   X: first column is 1s, the rest are from the spreadsheet
#   Y: The last column from the spreadsheet
#   labels: The list of headers for the columns of X from the spreadsheet
def read_excel_data(infilename):
    df = pd.read_excel(infilename)
    # replace property_id with 1s
    df["property_id"] = 1
    Y = df["price"].to_numpy()
    df.drop("price", axis=1, inplace=True)
    X = df.to_numpy()
    labels = list(df.columns)
    return X, Y, labels


# Make it pretty
def format_prediction(B, labels):
    # df=pd.read_excel('properties.xlsx',index_col='property_id')
    # pred_string = 32362.85 + (85.61 * df['sqft_hvac']) + (2.73 * df['sqft_yard']) + (59195.07 * df['bedrooms']) + (9599.24 * df['bathrooms']) +   ((-17421.84) * df['miles_to_school'])
    # pred_string= B[0] + (B[1] * X[1]) + (B[2] * X[2]) + (B[3] * X[3]) + (B[4] * X[4]) + (B[5] * X[5])

    pred_string = f"${B[0]:,.2f} + (${B[1]:,.2f} * {labels[1]}) + (${B[2]:,.2f} * {labels[2]}) + (${B[3]:,.2f} * {labels[3]}) + (${B[4]:,.2f} * {labels[4]}) + (${B[5]:,.2f} * {labels[5]})"
    return pred_string


# Return the R2 score for coefficients B
# Given inputs X and outputs Y
def score(B, X, Y):
    ## Your code here
    y = X @ B
    R2 = 1 - ((np.sum((Y - y) ** 2)) / ((len(Y) - 1) * np.var(Y, ddof=1)))
    return R2
