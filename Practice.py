#import required libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy
import statsmodels.api as sm

#create new data frame for accounts receivable
recs = pd.read_csv("accounts_receivable.csv")

#convert fields inv_date, due_date, and paid_date to type datetime
recs["inv_date"] = pd.to_datetime(recs["inv_date"])
recs["due_date"] = pd.to_datetime(recs["due_date"])
recs["paid_date"] = pd.to_datetime(recs["paid_date"])

#create new column titled age that reflects the age of the receivable
recs["age"] = recs["paid_date"] - recs["inv_date"]

#create new column titled late that represents whether the invoice was paid late
recs["late"] = recs["paid_date"] > recs["due_date"]

#generate and display descriptive statistics for score and age
print("** Descriptive statistics for credit score **")
print(recs["score"].describe())
print("\n** Descriptive statistics for receivable age **")
print(recs["age"].dt.days.describe())

#generate and display histogram for data frame with axis labels and title
plt.hist(x=recs["age"].dt.days, bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age of Receivables When Paid")
plt.show()

#generate and display scatterplot for data frame with axis labels and title
plt.scatter(recs["age"].dt.days, recs["score"])
plt.xlabel("Age")
plt.ylabel("Credit Score")
plt.title("Age of Receivables When Paid by Credit Score")
plt.show()

#generate and display correlation between credit score and age of receivable
print("\n** Correlation coefficient and p-value **")
print(pearsonr(recs["age"].dt.days, recs["score"]))

#create new variables that will store independent and dependent variable values 
y = recs["age"].dt.days
x = recs["score"]
x = sm.add_constant(x)

#create the model
mod = sm.OLS(y,x)

#estimate the model fit
results = mod.fit()

#display the summarized results
print(results.summary())