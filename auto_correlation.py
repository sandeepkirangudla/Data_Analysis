#!/usr/bin/env python
# coding: utf-8

# ### Regression Analysis of correlated data using OLS.
# 
# #### We know that OLS gives the best linear unbiased estimators when all the [Gauss Markov Assumptions](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem) are met. One such assumption is random sampling. Let us see how the estimators behave when this assumption is violated.
# 
# 
# 
# One of the way to violate this error is creating serial dependencies on dependent and independent variables.
# 
# Let us begin by generating dependent and independent variables using the following data generating processes.
# 
# $$x_t = x_{t-1,i} + u_{t,i} ,  u_{t,i} \sim iid \mathcal{N} (0, \sigma^2) $$ 
# for $t = 1,2,....T $ and $i = 1,2,.....n$.
# 
# Let's use a similar process to generate another feature, say {$y_{t}$}
# $$y_t = y_{t-1} + e_t ,  e_t \sim iid \mathcal{N} (0, \sigma^2) $$ 
# for $t = 1,2,....T$

# In[2]:


#importing the necessary packages
import numpy as np
import pandas as pd
import random
from statsmodels import api as sm
import matplotlib.pyplot as plt
import scipy as sc


# #### Let's generate $x_{t,i}$ for $t = 1$ and $i = 1$, $x_1 = x_{0} + u_{1i} ,  u_{1i} \sim iid \mathcal{N} (0, \sigma^2) $  and <br>
# $y_{t}$ for $t = 1$, $x_1 = y_{0} + e_{1} ,  e_{1} \sim iid \mathcal{N} (0, \sigma^2) $
# 

# In[2]:


t=1
e=[]
y=[]
u=[]
x=[]
#initial y gen
y.append(random.normalvariate(4,1))
#initial x gen
x.append(random.normalvariate(10,1))
#generate errors
simn = 1000
random.seed(2)
for isim in range(0,simn):
    #error e gen
    e.append(random.normalvariate(0,1))
    #error u gen
    u.append(random.normalvariate(0,1))
for isim in range(1,simn):
    #generate y
    y.append(y[isim-t]+e[isim])
    #generate x
    x.append(x[isim-t]+u[isim])
x=np.array(x).reshape(-1,1)
y=np.array(y)
ones = np.ones((len(x),1))
x = np.hstack((ones,x))


# In[3]:


# the histogram of the error
n, bins, patches = plt.hist(e, 10, density=True, facecolor='g', alpha=0.75)
plt.grid(True)
plt.show()


# #### Here, we can see that the error is normally distributed about zero.
# 
# #### Let's analyze y

# In[4]:


# the histogram of the y
n, bins, patches = plt.hist(y, 5, density=True, facecolor='g', alpha=0.75)
plt.grid(True)
plt.show()


# #### We can see that y is not normally distributed.
# 
# #### Similarly, let's see the histograms of u and x.

# In[5]:


# the histogram of the u
n, bins, patches = plt.hist(u, 10, density=True, facecolor='g', alpha=0.75)
plt.grid(True)
plt.show()


# In[6]:


# the histogram of the x
n, bins, patches = plt.hist(x[:,1], 5, density=True, facecolor='g', alpha=0.75)
plt.grid(True)
plt.show()


# #### Let's build the following model using OLS
# 
# #### $y_t = \beta_0 + \beta_1 x_t + \epsilon_t$

# In[7]:


# regression of x on y
model1 = sm.OLS(y,x).fit()
print(model1.summary())


# #### We can see that the [Durbin-Watson](https://www.investopedia.com/terms/d/durbin-watson-statistic.asp) score is 0.061 which suggests that there is positive auto correlation

# In[8]:


# the histogram of the data
residuals=np.array(model1.resid).reshape(-1,1)
n, bins, patches = plt.hist(residuals, 20, density=True, facecolor='g', alpha=0.75)
#plt.axis([-1, 1, 0, 20])
plt.grid(True)
plt.show()


# Here, we can see that the error terms are not normally distributed, there for the assumptions are voilated and the OLS estimators are baised. 
# 
# Lets look into the correlation of error terms.

# In[9]:


model1 = sm.OLS(y,x).fit(cov_type='HC0')
print(model1.summary())


# #### Let's plot residual plots to get a better understanding of the error distribution.

# In[10]:


plt.scatter(x[:,1],residuals.flatten())


# #### We can see that there is heterogenity in error terms. This leads to bias in our estimators and therefore, the OLS estimators are no longer BLUE (best linear unbiased estimators).

# In[11]:


sc.stats.levene(y,x[:,1])


# In[12]:


sc.stats.ttest_ind(y,x[:,1])


# #### Based on the [Levene Test](https://en.wikipedia.org/wiki/Levene%27s_test) and [Welch's t-test](https://en.wikipedia.org/wiki/Welch%27s_t-test), the null hypothesis that the random sampling distribution has equal variance can be rejected.

# #### Let's analyze the case for Multivariate Linear Regressions

# In[23]:


t=1
e=[]
y=[]
u=[]
x=[]
#initial y gen
n = 2
y.append(random.normalvariate(4,1))
#initial x gen
x.append(np.random.normal(10,1,n))
#generate errors
simn = 1000
random.seed(2)
for isim in range(0,simn):
    #error e gen
    e.append(random.normalvariate(0,1))
    #error u gen
    u.append(np.random.normal(0,1,n))
for isim in range(1,simn):
    #generate y
    y.append(y[isim-t]+e[isim])
    #generate x
    x.append(x[isim-t]+u[isim])
x=np.array(x).reshape(-1,n)
y=np.array(y)
ones = np.ones((len(x),1))
x = np.hstack((ones,x))


# In[24]:


x.shape


# In[25]:


# regression of x on y
model1 = sm.OLS(y,x).fit()
print(model1.summary())


# In[41]:


model1 = sm.OLS(y,x).fit(cov_type='HC0')
print(model1.summary())


# In[42]:


plt.scatter(x[:,1],residuals.flatten())


# In[43]:


plt.scatter(x[:,2],residuals.flatten())


# In[44]:


sc.stats.levene(y,x[:,1],x[:,2])


# In[47]:


[sc.stats.ttest_ind(y,x[:,1]),sc.stats.ttest_ind(y,x[:,2])]


# Now let's analyze how the number of samples effect the overall R-square

# In[68]:


t=1
e=[]
y=[]
u=[]
x=[]
#initial y gen
y.append(random.normalvariate(4,1))
#initial x gen
x.append(random.normalvariate(10,1))
#generate errors
simn = 1000
random.seed(2)
R_sq = [0]
for isim in range(0,simn):
    #error e gen
    e.append(random.normalvariate(0,1))
    #error u gen
    u.append(random.normalvariate(0,1))
for isim in range(1,simn):
    #generate y
    y.append(y[isim-t]+e[isim])
    #generate x
    x.append(x[isim-t]+u[isim])
    x_ar = np.array(x).reshape(-1,1)
    ones = np.ones((len(x),1))
    x_ar = np.hstack((ones,x_ar))
    y_ar=np.array(y)
    model1 = sm.OLS(y_ar,x_ar).fit()
    R_sq.append(model1.rsquared)
plt.plot(range(1,simn),R_sq[1:])


# In[75]:


t=1
e=[]
y=[]
u=[]
x=[]
#initial y gen
n = 2
y.append(random.normalvariate(4,1))
#initial x gen
x.append(np.random.normal(10,1,n))
#generate errors
simn = 1000
random.seed(2)
R_sq =[0]
for isim in range(0,simn):
    #error e gen
    e.append(random.normalvariate(0,1))
    #error u gen
    u.append(np.random.normal(0,1,n))
for isim in range(1,simn):
    #generate y
    y.append(y[isim-t]+e[isim])
    #generate x
    x.append(x[isim-t]+u[isim])
    x_ar = np.array(x).reshape(-1,n)
    ones = np.ones((len(x),1))
    x_ar = np.hstack((ones,x_ar))
    y_ar=np.array(y)
    model1 = sm.OLS(y_ar,x_ar).fit()
    R_sq.append(model1.rsquared)
plt.plot(range(1,simn),R_sq[1:])


# Lets start analyzing for different time lags starting from ${t = 1, 2, 3.....10}$

# In[78]:


np.shape(R_sq)


# In[47]:


e=[]
y=[]
u=[]
x=[]
#initial y gen
n = 1
y.append(random.normalvariate(4,1))
#initial x gen
x.append(np.random.normal(10,1,n))
#generate errors
simn = 1000
outp = []
time = []
R_sq = []
random.seed(2)
T = [1,5,10,15,20,25,30,35,40,45,50]
for t in T:
    for isim in range(0,simn):
        #error e gen
        e.append(random.normalvariate(0,1))
        #error u gen
        u.append(np.random.normal(0,1,n))
    for isim in range(1,simn):
        #generate y
        y.append(y[isim-t]+e[isim])
        #generate x
        x.append(x[isim-t]+u[isim])
        x_ar = np.array(x).reshape(-1,n)
        ones = np.ones((len(x),1))
        x_ar = np.hstack((ones,x_ar))
        y_ar=np.array(y)
        model1 = sm.OLS(y_ar,x_ar).fit()
        outp.append(model1.rsquared)
        #time.append(t)
    print("time lags - ",t)
    R_sq.append(model1.rsquared)
    plt.plot(range(1,simn),outp)
    plt.show()
    outp=[]


# In[48]:


R_sq


# #### Now consider the case where ${n = 1,...10}$ and $ t $  $\epsilon$  $[1,5,10,15,20,25,30,35,40,45,50]$

# In[52]:


simn = 1000
outp = []
time = []
R_sq = []
random.seed(2)
T = [1,5,10,15,20,25,30,35,40,45,50]
for n in range(1,10):
    #generate errors
    e=[]
    y=[]
    u=[]
    x=[]
    #initial y gen
    y.append(random.normalvariate(4,1))
    #initial x gen
    x.append(np.random.normal(10,1,n))
    for t in T:
        for isim in range(0,simn):
            #error e gen
            e.append(random.normalvariate(0,1))
            #error u gen
            u.append(np.random.normal(0,1,n))
        for isim in range(1,simn):
            #generate y
            y.append(y[isim-t]+e[isim])
            #generate x
            x.append(x[isim-t]+u[isim])
            x_ar = np.array(x).reshape(-1,n)
            ones = np.ones((len(x),1))
            x_ar = np.hstack((ones,x_ar))
            y_ar=np.array(y)
            model1 = sm.OLS(y_ar,x_ar).fit()
            outp.append(model1.rsquared)
            #time.append(t)
        print("number of parameters - ",n)
        print("time lags - ",t)
        R_sq.append(model1.rsquared)
        plt.plot(range(1,simn),outp)
        plt.show()
        outp=[]     


# In[96]:


n = []
for i in range(1,10):
    c=0
    while c <11:
        n.append(i)
        c+=1


# In[163]:


time = []
for i in range(0,9):
    time.append(T)
time=np.array(time).flatten()


# In[166]:


fin = pd.DataFrame(data=np.hstack(([n],[time],[R_sq])).reshape(-1,99).T,columns=['Number of Features','Time-lags','R-Squared'])


# In[167]:


fin


# #### In coming posts, we will discuss on how to fix the serial correlation issue in OLS.
