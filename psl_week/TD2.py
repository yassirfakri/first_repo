####################################################################
# This TD is using body data to explore :
#  - Linear regression
#  - Multiple linear regression
#  - Ridge / Lasso

################# ################# #################
# Import packages
################# ################# #################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from random import choices
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pdb as debugger
import scipy.stats as stats

pdb = debugger.set_trace

# Open the data
dat = pd.read_csv('./TD1/data_body/body.dat', sep='\s+',
                  header=None)  # Open the data file
# Open description.csv with description and names of each variables
Description = pd.read_csv('./TD1/data_body/description.csv', sep=';', header=0)
# Rename columns from dat with the name of the variable
dat = dat.rename(mapper=Description.loc[:, 'Name'], axis=1)

# Remove outliers
# Replace the two outliers with Nan
dat.loc[dat['Weight'] > 200, 'Weight'] = np.nan
# Replace the two outliers with Nan
dat.loc[dat['Height'] > 1000, 'Height'] = np.nan
# Replace the two outliers with Nan
dat.loc[dat['Height'] < 50, 'Height'] = np.nan


################# ################# #################
# Simple Linear regression
################# ################# #################
# Run a linear regression between two variables
linear_reg = LinearRegression(fit_intercept=False)
X = dat[['Weight']]
Y = dat['Height']
# linear_reg.fit(X, Y)
# -> Won't work because of nan values => Need to clean the dataframe from incomplete data

# Q1 Remove subjects with nan values
dat = dat.dropna(ignore_index=True)
X = dat[['Weight']]  # Extract Weights
Y = dat['Height']  # Extract Heights
reg = linear_reg.fit(X, Y)  # Fit the linear regression model

# Q2 print the coefficient of this regression
print(reg.coef_)
print(reg.intercept_)  # Ordonnée à l'origine

# Q3 make a scatter plot of Weight and Height and superimpose the linear regression
# plt.scatter(X, Y, color='green')  # Scatter plot
# plt.plot(X, reg.predict(X), 'k')  # Draw linear regression on top
# plt.show()

# Q4 -> What is the problem with this regression ?
# => This regression model shows that it the fit is not good

# Q5 -> Redo the same but correcting the problem of the previous regression
linear_reg = LinearRegression(fit_intercept=True)
X = dat[['Weight']]
Y = dat['Height']
reg = linear_reg.fit(X, Y)

# Q6 look at the intercept and the coef of this model
intercept = reg.intercept_
coef = reg.coef_

# Q7 make a scatter plot of Weight and Height and superimpose the linear regression
# plt.scatter(X, Y, color='green')  # Scatter plot
# plt.plot(X, reg.predict(X), 'k')  # Draw linear regression on top
# plt.show()


################# ################# #################
# Multiple Linear regression
################# ################# #################
# Q8 Before computing regression with multiple factors, we need to zscore all factors so that they are comparable.
dat = stats.zscore(dat)
dat = (dat - dat.mean()) / dat.std()  # Second way to do it

# Q9  Make a regression of the Height with two factors : Weight and Knee_girth
linear_reg = LinearRegression(fit_intercept=True)
X = dat[['Weight', 'Knee_girth']]
Y = dat['Height']
linear_reg.fit(X, Y)


# Q10 make a barplot showing the two coefficients of this regression
# plt.bar([1, 2], linear_reg.coef_)
# plt.show()

# Q11 How stable this is ? Compute confidence interval on the regression parameters using a boostrap approach. Create the loop to compute 1000 estimation of each coefs
coef_1 = []
coef_2 = []
for _ in range(1000):
    reg_i = LinearRegression(fit_intercept=True)
    indexes = choices(dat.index, k=dat.shape[0])
    reg_i.fit(X.iloc[indexes], Y.iloc[indexes])
    coef_1.append(reg_i.coef_[0])
    coef_2.append(reg_i.coef_[1])

# Q12 From these 1000 estimation, compute the confidence interval for both coefs
CI1 = np.percentile(coef_1, [2.5, 97.5])
CI2 = np.percentile(coef_2, [2.5, 97.5])

# Q13 redo the bar plot and add the confidence interval
a, b = linear_reg.coef_[0], linear_reg.coef_[1]
# plt.bar([1, 2], linear_reg.coef_)
# plt.errorbar([1, 2], linear_reg.coef_, yerr=[
#              (CI1[1]-CI1[0])/2, (CI2[1]-CI2[0])/2], fmt="o", color="r")
# plt.show()

# Q14 Know look only at the simple regression between Height and Knee_girth
reg = LinearRegression()
X = dat[['Knee_girth']]
Y = dat['Height']
reg.fit(X, Y)

# plt.scatter(X, Y)
# plt.plot(X, reg.predict(X), 'r')
# plt.show()


# Q15 -> What can you conclude ?
# => The linear regression model Height as a function of Knee_girth is a good fit => it's probably a latent variable

# Q16 Now make a multiple linear regression predicting the Height based on all other factors.
linear_reg = LinearRegression(fit_intercept=True)
X = dat.loc[:, dat.columns != 'Height']
Y = dat['Height']
linear_reg.fit(X, Y)

# Q17 And plot all coefs of this regression
coefs = linear_reg.coef_
# plt.bar(np.arange(len(coefs)), coefs)
# plt.show()


# Q18 Redo the same but with using a ridge regression with alpha = 100
ridge_reg = Ridge(alpha=100)
X = dat.loc[:, dat.columns != 'Height']
Y = dat['Height']
ridge_reg.fit(X, Y)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.bar(X.keys(), ridge_reg.coef_)
# ax.tick_params(axis='x', rotation=90)
# plt.show()

# Q19 what do you observe ?
# => We observe different correlation coefficients compared to the previous model

# Q20 Compute the values of the parameters for ridge regressions with alpha varying between 10^-5 and 10^5
X = dat.loc[:, dat.columns != 'Height']
Y = dat['Height']
alphas = np.logspace(-5, 5, num=1000)
allcoefs = []
for alpha in alphas:
    ridge_reg = Ridge(alpha=alpha)
    ridge_reg.fit(X, Y)
    allcoefs.append(ridge_reg.coef_)

# plt.xscale("log")
# plt.plot(alphas, allcoefs)
# plt.show()
# Rq: Tous les coefficients convergent vers 0 quand alpha (coef de pénalité) tend vers +inf

# Q21 Now compute a lasso regression with alpha = 0.1
lasso_reg = Lasso(alpha=0.1)
X = dat.loc[:, dat.columns != 'Height']
Y = dat['Height']
lasso_reg.fit(X, Y)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.bar(X.keys(), lasso_reg.coef_)
# ax.tick_params(axis='x', rotation=90)
# plt.show()

# Q22 Compute and plot the values of the parameters for lasso regressions with alpha varying between 10^-5 and 10^5
alphas = np.logspace(-5, 5, num=1000)
allcoefs = []
for alpha in alphas:
    lasso_reg = Ridge(alpha=alpha)
    lasso_reg.fit(X, Y)
    allcoefs.append(lasso_reg.coef_)

plt.xscale("log")
plt.plot(alphas, allcoefs)
plt.show()
