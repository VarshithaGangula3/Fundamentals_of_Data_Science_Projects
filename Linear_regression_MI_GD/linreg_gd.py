import pandas as pd
from array import array
import numpy as np
import sys
import util
import matplotlib.pyplot as plt

# Check the command line
if len(sys.argv) != 2:
    print(f"{sys.argv[0]} <xlsx>")
    exit(1)

# Learning rate
t = 0.001

# Limit interations
max_steps = 1000

# Get the arg and read in the spreadsheet
infilename = sys.argv[1]
X, Y, labels = util.read_excel_data(infilename)
n, d = X.shape
print(f"Read {n} rows, {d - 1} features from '{infilename}'.")


# Get the mean and standard deviation for each column
## Your code here
# for i in range(1, X.shape[1]):

Xmean = X.mean(axis=0)
Xstd = X.std(axis=0)

# Don't mess with the first column (the 1s)
## Your code here
Xmean[0] = 0.0
Xstd[0] = 1.0
# Standardize X to be X'
Xp = (X - Xmean) / Xstd  ## Your code here

# First guess for B is "all coefficents are zero"
# cB = np.zeros((6), float) ## Your code here
New_B = np.zeros(6, float)

# Create a numpy array to record avg error for each step
errors = np.array(max_steps)
# errors=np.sum(((y - Y)**2),axis = 0) / 2
er = np.zeros(max_steps)  ## Your code here


for i in range(max_steps):
    cB = New_B
    # Compute the gradient
    ## Your code here
    G = Xp.T @ (Xp @ cB - Y)
    # Compute a new B (use `t`)
    ## Your code here
    New_B = cB - t * G
    # Figure out the average squared error using the new B
    ## Your code here
    y = Xp @ New_B
    R = Y - Xp @ New_B
    # errors=np.sum(((y-Y)**2),axis = 0) / 2 #len(X)
    E = np.dot(R, R) / n
    er[i] = E
    # Store it in `errors``
    ## Your code here

    # Check to see if we have converged
    if np.sum((New_B - cB) ** 2) < 0.001:
        break

print(f"Took {i} iterations to converge")

# "Unstandardize" the coefficients
## Your code here
# X= Xp*Xstd +Xmean
Xmean[0] = -1.0
New_B[0] = New_B @ (-1 * Xmean / Xstd)

B = New_B / Xstd
# Show the result
print(util.format_prediction(B, labels))

# Get the R2 score
R2 = util.score(B, X, Y)
print(f"R2 = {R2:f}")

# Draw a graph
fig1 = plt.figure(1, (4.5, 4.5))
## Your code ehre
ax1 = fig1.add_axes([0.15, 0.15, 0.6, 0.7])
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.plot(np.arange(i), er[:i])
ax1.set_title("Convergence")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Mean squared error")
fig1.savefig("err.png")
