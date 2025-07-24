####################################################################
# This TD is using body data to explore :
#  - Panda dataframe, import .dat, header, separator, change key names, save csv, sorting dataframe, replacing values, drop nan
#  - Use of numpy : mean, median, max, min, var
#  - plotting histogram, density function, bimodal
#  - Confidence interval and boostrap computation, significance
#  - Create and call python functions
#  - Scatter plots, correlations, matrix of cross-correlation
#  - Simpson paradox, bar plot


################# ################# #################
# Import packages
################# ################# #################
# If you are not familiar with Python : this part allow to import packages functions.
# You can either import one function (from Package import Function) or the full package (import package as pkg) and call eah function of this package as pkg.function
# Pandas : see https://pandas.pydata.org/docs/user_guide/10min.html
import pandas as pd
import numpy as np               # Numpy : see https://numpy.org
# matplotlib for visualization : https://matplotlib.org
import matplotlib.pyplot as plt
# seaborn is another plotting library : https://seaborn.pydata.org
import seaborn as sns
from random import choices
import pdb as debugger

pdb = debugger.set_trace

################# ################# #################
# Open and format the data
################# ################# #################
# -> Using pandas, you can open .dat or .csv data files
# Open body.dat using read_csv function from pandas library.
dat = pd.read_csv('./TD1/data_body/body.dat', sep='\s+', header=None)

# Q1 ##  Using the same read_csv function from panda, open the description.csv file. Choose the correct 'sep' and 'header' parameters
Description = pd.read_csv(".\TD1\data_body\description.csv", sep=';')

# What is the role of this line ?
dat = dat.rename(mapper=Description.loc[:, 'Name'], axis=1)
# => This line renames the columns of the data dataframe in order using the first column of the Description dataframe

# Q2 ## save the dataframe as csv.
dat.to_csv('./TD1/data_body/body.csv')

################# ################# #################
# Look at mean and median, min, max...
################# ################# #################
dat['Weight']  # Print the Weight of all people

# Q3 ## Using a function from numpy, compute the mean Weight
mean_weight = dat['Weight'].mean()
# Q4 ## Using a function from numpy, compute the median Weight
median_weight = np.median(dat['Weight'])

# Q5 ## Redo the same for the height
mean_height = dat['Height'].mean()
median_height = np.median(dat['Height'])

# Q6 ## Now select the column corresponding to biological sex of subjects
dat['Sex']
# Q7 ## And look at its average value => What does it correspond to ? : 0.48717948717948717
Mean_sex = dat['Sex'].mean()

# -> With numpy, you can alo find the Maximum / Minimum
np.max(dat['Biacromial'])  # Max Biacromial
np.min(dat['Biacromial'])  # Min Biacromial


################# ################# #################
# Sorting the data
################# ################# #################
# -> Pandas dataframe can be sorted based of a given column. Find the correct method to apply on a dataframe to do so.
# Q7 ## First sort the based on Biacromial value, ascending and plot those values. Plot another column of the dataframe.
dat = dat.sort_values('Biacromial', ascending=True)
# Q8 ## And now base on the Biiliac, descending.
dat = dat.sort_values('Biiliac', ascending=False)


# ->  Note that all other columns, including the first one containing the index has been re-arranged to match the new order.


################# ################# #################
# Look at the variability of the data
################# ################# #################

# Q9 ## Look at the doc from matplotlib.pyplot to plot the histogram of Weight with 100 bins.
# plt.hist(dat['Weight'], bins=100)
# plt.show()
# -> What do you notice ?
# ==> The weight is mostly a normal distribution between 42 and 120 with the exception of very few outliers of 600kg, that are probably incorrect

# Q10 ## This command will find the outliers : 'dat['Weight']>200'. Use this to replace the outliers by nans in the dataframe
dat[dat['Weight'] > 200] = np.nan  # Replace the two outliers with Nan

# Q11 ## Now replot the hitogram and check that outliers are removed
# plt.hist(dat['Weight'], bins=100)
# plt.show()
# ==> The outliers are now removed, and the maximal weight is about 116kg
np.var(dat['Weight'])


# Q12 ## Redo the same process with Height to plot the histogram without outliers
plt.clf()
dat[dat['Height'] > 250] = np.nan
dat[dat['Height'] < 100] = np.nan
# plt.hist(dat['Height'], bins=100)

# Q13 ## barplot are interesting but if we want to see a continuous distribution, density plot are interesting. Find a use a function from seaborn package to compute density plot of the Weight/Height and age
plt.clf()
# sns.displot(dat[['Weight']], x='Weight', stat='density')
# plt.show()

plt.clf()
# sns.displot(dat[['Height']], x='Height', stat='density')
# plt.show()

plt.clf()
# sns.displot(dat[['Age']], x='Age', stat='density')
# plt.show()

# Q14 ## Play with the kernel size and observe how it changes the distribution plot.
plt.clf()
# sns.kdeplot(dat['Weight'], bw_adjust=0.5, color='blue')
# sns.kdeplot(dat['Weight'], bw_adjust=0.2, color='red')
# sns.kdeplot(dat['Weight'], bw_adjust=1, color='green')
# plt.show()

# Documentation :
# bw_adjust : number, optional
# Factor that multiplicatively scales the value chosen using bw_method. Increasing will make the curve smoother. See Notes.


# Q15 ## Plot the diameter of elbows with a kernel size of 1
plt.clf()
# sns.kdeplot(dat['Elbow_diam'], bw_adjust=1, color='green')
# plt.show()

# Q16 ## Redo the same with a kernel size of 0.2 : What do you notice ?
plt.clf()
# sns.kdeplot(dat['Elbow_diam'], bw_adjust=0.2, color='green')
# plt.show()
# ==> We notice quite variation compared to the previous density plot

# Q17 ## If you do not close the figure between two plots, they will overlapp. Use this to overlapp the density plt of elbow diameters of the whole population with the one with only males and the one with only females (alway with kernel width = 0.2). : What can you conclude ?
plt.clf()
filter = (dat['Sex'] == 1)
# sns.kdeplot(dat['Elbow_diam'][filter].squeeze(),
#             bw=0.2, color='green')
# sns.kdeplot(dat['Elbow_diam'][~filter].squeeze(),
#             bw=0.2, color='red')
# plt.show()
# ==> We can tell that each sex has its own normal distribution, one of them is centered around 12.5cm and the other around 14.5cm


# Q18 ## Now we will estimate the confidence intervall of the weight using a boostrap method.
Weight = np.array(dat['Weight'])
mean = Weight.mean()
n_suj = np.shape(Weight)[0]
n_boostrap = 1000  # We will do 1000 boostrap iterations to estimate the mean
allmeans = []  # In this list, we will put every estimation of the mean for each boostrap
for i in range(n_boostrap):
    sample = np.nanmean(choices(Weight, k=len(Weight)))
    allmeans.append(sample)
allmeans = np.array(allmeans)  # Contains 1000 values

# Q19 ## plot the density plot of all means
plt.clf()
# sns.kdeplot(allmeans, bw=0.3)  # PLot the distribution of estimated means
# plt.show()

# Q20 ## Find the 2.5% lowest and 2.5% highest values of allmean :
allmeans.sort()  # first : sort the list from low to high
# Slice the array to remove the 2.5% lowest and the 2.5% highest values

allmeans = allmeans[int(len(allmeans) * 0.025): int(len(allmeans) * 0.975)]
confidence_interval = [min(allmeans), max(allmeans)]  # Find the 95% CI

# ->  Note that this three last line can be replaced by the following function : np.percentile(allmeans, [2.5, 97.5])


# Q21 ## Now you will write a function that takes the name of the variable to consider, the index of subset of subject to consider, and the number of boostrap iteration and return the confidence interval of the mean
def Mean_boostrapCI(name, subjects, n_boot):
    variable = np.array(subjects[name])
    allmeans = [np.nanmean(choices(variable, k=len(variable)))
                for _ in range(n_boot)]
    confidence_interval = np.percentile(allmeans, [2.5, 97.5])  # find CI
    return (confidence_interval)


# Q22 ## Use this function to compute the Weight CI male and female
male = dat['Sex'] == 1
female = dat['Sex'] == 0
Male_CI = Mean_boostrapCI('Weight', dat.loc[male], 1000)
# [76.91439796, 79.53104082]
Female_CI = Mean_boostrapCI(
    'Weight', dat.loc[female], 1000)  # [59.48004845, 61.74900194]
# Q23 ## What can be concluded on the significance of mean Weight diff between male and female ? : XXXXXXXXXX
# ==> We can conclude that sex impacts the weight of the person

# Q24 ## re-use the same function to compute the CI of the mean Height between people < 40 yo and people > 40 yo
young = dat['Age'] < 40
old = dat['Age'] > 40
young_CI = Mean_boostrapCI('Height', dat.loc[young], 1000)
# [170.07474453, 171.9485219 ]
old_CI = Mean_boostrapCI('Height', dat.loc[old], 1000)
# [170.13700617, 174.22006173]
# Q25 ## What can you conclude ?
# We can notice that the age doesn't really impact the height in this dataset, especially because the youngest person is about 20 years old


################# ################# #################
# Explore correlations between variables.
################# ################# #################
# Q26 ##  Using the scatter function from matplotlib.pyplot, show Height and Weight for each subject
plt.clf()
dat = dat.dropna()
plt.scatter(np.arange(len(dat)), dat['Height'])
plt.scatter(np.arange(len(dat)), dat['Weight'])
plt.legend(['Height', 'Weight'])
plt.show()

# Q27 ##  Compute the correlation between the two variables Height and Weight
corr = np.corrcoef(dat['Height'], dat['Weight'])[0, 1]
pdb()

# Q28 ## Redo the same for male and female separately
plt.scatter(XXXXXXXXXX)
plt.scatter(XXXXXXXXXX)
corr_maleonly = XXXXXXXXXX
corr_femaleonly = XXXXXXXXXX


# Q29 ## Write loops to look at the correlation of each pair of variable
variables = dat.keys()  # Find all keys in dataframe
n_var = np.shape(variables)[0]  # number of keys in dataframe
# Create a matrix full of zeros to put the correlations
correlations = np.zeros([n_var, n_var])
XXXXXXXXXX

# Q30 ## Plot the Matrix of correlations of all variables
fig = plt.figure()
ax = fig.add_subplot()
cax = ax.XXXXXXXXXX
plt.xticks(range(0, n_var))  # Add a tick for each variable
plt.yticks(range(0, n_var))
ax.tick_params(axis='x', rotation=90)  # Add tilt to ticks
ax.set_xticklabels(variables)
ax.set_yticklabels(variables)
cbar = fig.colorbar(cax)


################# ################# #################
# The Simpson paradox
################# ################# #################
# Q31 ## Open the Data_vaccine.csv file and save it in .csv format
vacc = XXXXXXXXXX

# Q32 ## Look at the mortality rate of vaccinated and non-vaccinated people
with_vacc = XXXXXXXXXX
without_vacc = XXXXXXXXXX
plt.bar(['Vacc', 'noVacc'], [with_vacc, without_vacc], color=['g', 'b'])
# -> What do you conclude ? Is the vaccine really dangerous ?


# Q33 ## Now, look at vaccination rate as a function of age
vacc_rate_young = XXXXXXXXXX
vacc_rate_old = XXXXXXXXXX

# Q34 ## And look at mortality rate as a function of age
mortality_rate_young = XXXXXXXXXX
mortality_rate_old = XXXXXXXXXX

# Q35 ## make a bar plot with mortality rate separated by vaccination status and age
mortality_rate_young_vacc = XXXXXXXXXX
mortality_rate_young_novacc = XXXXXXXXXX
mortality_rate_old_vacc = XXXXXXXXXX
mortality_rate_old_novacc = XXXXXXXXXX

plt.bar(['Young_Vacc', 'Young_noVacc', 'Old_Vacc', 'Old_noVacc'], [mortality_rate_young_vacc,
        mortality_rate_young_novacc, mortality_rate_old_vacc, mortality_rate_old_novacc], color=['g', 'b', 'g', 'b'])
# -> This is an example of a confounding factor creating the Simpson Paradox (https://fr.wikipedia.org/wiki/Paradoxe_de_Simpson)


""
