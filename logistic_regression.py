import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import statsmodels.api as sm
import math

loansData_clean = pd.read_csv('/Users/kuttush/Desktop/Spongebob/Thinkful/DataScience/Unit2/LogisticRegression/loansData_clean.csv')

#printing again to see if cleaning took place or not
#print (loansData_clean['Interest.Rate'][0:5])
#print (loansData_clean['Loan.Length'][0:5])
#print (loansData_clean['FICO.Score'][0:5])
#print (loansData_clean['Amount.Requested'][0:5])

#add a column indicating whether the interest rate is < 12%
loansData_clean['Low Interest'] = loansData_clean['Interest.Rate'] < 12
#print(loansData_clean['Low Interest'][0:20])

#some spot checking
#loansData_clean[loansData_clean['Interest.Rate'] == 10].head()
#loansData_clean[loansData_clean['Interest.Rate'] == 13].head()

#adding intercept column
loansData_clean['Constant Intercept'] = 1

#create a list of column names for indep. variables
ind_vars = ['Constant Intercept', 'FICO.Score', 'Amount.Requested']
#print(ind_vars)

#create the logistic regression model
intrate = loansData_clean['Low Interest']
loanamt = loansData_clean['Amount.Requested']
fico = loansData_clean['FICO.Score']

y = np.matrix(intrate).transpose()  #dependent variable
x1 = np.matrix(fico).transpose()    #independent variable
x2 = np.matrix(loanamt).transpose() #independent variable

#take the independent matrix and create an input matrix, 1 col for each variable
x = np.column_stack([x1,x2])

#creating the linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()
print(f.params)
print(f.params[0])
print(f.params[1])
print(f.params[2])

#the interest rate is 
int_rate = f.params[0] + f.params[1]*fico + f.params[2]*loanamt
#print(int_rate)

#logistic_function is
px=[]
for j in int_rate:
	denominator = math.exp(j)
	p_x = 1/(1+denominator)
	px.append(p_x)

print(px)
plt.plot(fico, px)
plt.plot(loanamt, px)

'''the plots are not coming out looking like logistic function plots. It could be due to 
various reasons: (1). f.params[1] and f.params[2] actually should be going with loanamt and fico
respectively, and not what I've. (2). I'm not sure if f.params[0] is the intercept or not; it 
could very well be either f.params[1] or f.params[2]. (3). the points -- fico, px or loanamt, px -- does not any functional relationship, that is, they are not functions.'''

#amt = 10,000, int_rate < 12, fico = 720
int_rate = f.params[0] + f.params[1]*720 + f.params[2]*10000
denominator = math.exp(int_rate)
p_x = 1/(1 + denominator)
print(p_x)

'''Thus for the given amount and the fico_score, the probability of obtaining the loan, p_x,
is about 0.361. Thus, the loan will not be given as it is below the threshhold of 0.7
