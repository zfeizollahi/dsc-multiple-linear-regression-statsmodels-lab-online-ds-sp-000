
# Multiple Linear Regression in Statsmodels - Lab

## Introduction
In this lab, you'll practice fitting a multiple linear regression model on our Boston Housing Data set!

## Objectives
You will be able to:
* Run linear regression on Boston Housing dataset with all the predictors
* Interpret the parameters of the multiple linear regression model

## The Boston Housing Data

We pre-processed the Boston Housing Data again. This time, however, we did things slightly different:
- We dropped "ZN" and "NOX" completely
- We categorized "RAD" in 3 bins and "TAX" in 4 bins
- We used min-max-scaling on "B", "CRIM" and "DIS" (and logtransformed all of them first, except "B")
- We used standardization on "AGE", "INDUS", "LSTAT" and "PTRATIO" (and logtransformed all of them first, except for "AGE") 


```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
boston = load_boston()

boston_features = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_features = boston_features.drop(["NOX","ZN"],axis=1)

# first, create bins for based on the values observed. 3 values will result in 2 bins
bins = [0,6,  24]
bins_rad = pd.cut(boston_features['RAD'], bins)
bins_rad = bins_rad.cat.as_unordered()

# first, create bins for based on the values observed. 4 values will result in 3 bins
bins = [0, 270, 360, 712]
bins_tax = pd.cut(boston_features['TAX'], bins)
bins_tax = bins_tax.cat.as_unordered()

tax_dummy = pd.get_dummies(bins_tax, prefix="TAX")
rad_dummy = pd.get_dummies(bins_rad, prefix="RAD")
boston_features = boston_features.drop(["RAD","TAX"], axis=1)
boston_features = pd.concat([boston_features, rad_dummy, tax_dummy], axis=1)
```


```python
age = boston_features["AGE"]
b = boston_features["B"]
logcrim = np.log(boston_features["CRIM"])
logdis = np.log(boston_features["DIS"])
logindus = np.log(boston_features["INDUS"])
loglstat = np.log(boston_features["LSTAT"])
logptratio = np.log(boston_features["PTRATIO"])

# minmax scaling
boston_features["B"] = (b-min(b))/(max(b)-min(b))
boston_features["CRIM"] = (logcrim-min(logcrim))/(max(logcrim)-min(logcrim))
boston_features["DIS"] = (logdis-min(logdis))/(max(logdis)-min(logdis))

#standardization
boston_features["AGE"] = (age-np.mean(age))/np.sqrt(np.var(age))
boston_features["INDUS"] = (logindus-np.mean(logindus))/np.sqrt(np.var(logindus))
boston_features["LSTAT"] = (loglstat-np.mean(loglstat))/np.sqrt(np.var(loglstat))
boston_features["PTRATIO"] = (logptratio-np.mean(logptratio))/(np.sqrt(np.var(logptratio)))
```


```python
boston_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>RAD_(0, 6]</th>
      <th>RAD_(6, 24]</th>
      <th>TAX_(0, 270]</th>
      <th>TAX_(270, 360]</th>
      <th>TAX_(360, 712]</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>-1.704344</td>
      <td>0.0</td>
      <td>6.575</td>
      <td>-0.120013</td>
      <td>0.542096</td>
      <td>-1.443977</td>
      <td>1.000000</td>
      <td>-1.275260</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.153211</td>
      <td>-0.263239</td>
      <td>0.0</td>
      <td>6.421</td>
      <td>0.367166</td>
      <td>0.623954</td>
      <td>-0.230278</td>
      <td>1.000000</td>
      <td>-0.263711</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.153134</td>
      <td>-0.263239</td>
      <td>0.0</td>
      <td>7.185</td>
      <td>-0.265812</td>
      <td>0.623954</td>
      <td>-0.230278</td>
      <td>0.989737</td>
      <td>-1.627858</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.171005</td>
      <td>-1.778965</td>
      <td>0.0</td>
      <td>6.998</td>
      <td>-0.809889</td>
      <td>0.707895</td>
      <td>0.165279</td>
      <td>0.994276</td>
      <td>-2.153192</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.250315</td>
      <td>-1.778965</td>
      <td>0.0</td>
      <td>7.147</td>
      <td>-0.511180</td>
      <td>0.707895</td>
      <td>0.165279</td>
      <td>1.000000</td>
      <td>-1.162114</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



## Run an linear model in Statsmodels


```python
import statsmodels.api as sm
from statsmodels.formula.api import ols
```


```python
boston_features['MEDV'] = boston.target
```


```python
boston_features.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>RAD_0</th>
      <th>RAD_1</th>
      <th>TAX_0</th>
      <th>TAX_1</th>
      <th>TAX_2</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>-1.704344</td>
      <td>0.0</td>
      <td>6.575</td>
      <td>-0.120013</td>
      <td>0.542096</td>
      <td>-1.443977</td>
      <td>1.000000</td>
      <td>-1.275260</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.153211</td>
      <td>-0.263239</td>
      <td>0.0</td>
      <td>6.421</td>
      <td>0.367166</td>
      <td>0.623954</td>
      <td>-0.230278</td>
      <td>1.000000</td>
      <td>-0.263711</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.153134</td>
      <td>-0.263239</td>
      <td>0.0</td>
      <td>7.185</td>
      <td>-0.265812</td>
      <td>0.623954</td>
      <td>-0.230278</td>
      <td>0.989737</td>
      <td>-1.627858</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.171005</td>
      <td>-1.778965</td>
      <td>0.0</td>
      <td>6.998</td>
      <td>-0.809889</td>
      <td>0.707895</td>
      <td>0.165279</td>
      <td>0.994276</td>
      <td>-2.153192</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.250315</td>
      <td>-1.778965</td>
      <td>0.0</td>
      <td>7.147</td>
      <td>-0.511180</td>
      <td>0.707895</td>
      <td>0.165279</td>
      <td>1.000000</td>
      <td>-1.162114</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>




```python
outcome = 'MEDV'
predictors = boston_features.columns
f = '+'.join(predictors)
formula = outcome + '~' + f
stats_model = ols(formula=formula, data=boston_features).fit()
```


```python
stats_model.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>          <td>MEDV</td>       <th>  R-squared:         </th>  <td>   1.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   1.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>1.968e+31</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 16 Jul 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>20:59:04</td>     <th>  Log-Likelihood:    </th>  <td>  15471.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>-3.091e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   492</td>      <th>  BIC:               </th> <td>-3.085e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    13</td>      <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>-1.243e-14</td> <td> 5.32e-15</td> <td>   -2.339</td> <td> 0.020</td> <td>-2.29e-14</td> <td>-1.99e-15</td>
</tr>
<tr>
  <th>CRIM</th>      <td> 7.105e-15</td> <td> 6.25e-15</td> <td>    1.136</td> <td> 0.256</td> <td>-5.18e-15</td> <td> 1.94e-14</td>
</tr>
<tr>
  <th>INDUS</th>     <td> 8.882e-16</td> <td> 1.08e-15</td> <td>    0.826</td> <td> 0.409</td> <td>-1.23e-15</td> <td>    3e-15</td>
</tr>
<tr>
  <th>CHAS</th>      <td>-3.331e-15</td> <td> 2.38e-15</td> <td>   -1.401</td> <td> 0.162</td> <td>   -8e-15</td> <td> 1.34e-15</td>
</tr>
<tr>
  <th>RM</th>        <td> 4.718e-15</td> <td> 1.26e-15</td> <td>    3.759</td> <td> 0.000</td> <td> 2.25e-15</td> <td> 7.18e-15</td>
</tr>
<tr>
  <th>AGE</th>       <td>-1.887e-15</td> <td> 1.04e-15</td> <td>   -1.817</td> <td> 0.070</td> <td>-3.93e-15</td> <td> 1.53e-16</td>
</tr>
<tr>
  <th>DIS</th>       <td>-3.553e-15</td> <td> 5.64e-15</td> <td>   -0.629</td> <td> 0.529</td> <td>-1.46e-14</td> <td> 7.54e-15</td>
</tr>
<tr>
  <th>PTRATIO</th>   <td> 8.049e-16</td> <td>  7.4e-16</td> <td>    1.088</td> <td> 0.277</td> <td>-6.48e-16</td> <td> 2.26e-15</td>
</tr>
<tr>
  <th>B</th>         <td>-1.221e-15</td> <td> 2.96e-15</td> <td>   -0.413</td> <td> 0.680</td> <td>-7.03e-15</td> <td> 4.59e-15</td>
</tr>
<tr>
  <th>LSTAT</th>     <td> 1.138e-15</td> <td> 1.29e-15</td> <td>    0.885</td> <td> 0.376</td> <td>-1.39e-15</td> <td> 3.66e-15</td>
</tr>
<tr>
  <th>RAD_0</th>     <td>-5.995e-15</td> <td> 2.43e-15</td> <td>   -2.463</td> <td> 0.014</td> <td>-1.08e-14</td> <td>-1.21e-15</td>
</tr>
<tr>
  <th>RAD_1</th>     <td>-5.773e-15</td> <td> 3.19e-15</td> <td>   -1.808</td> <td> 0.071</td> <td> -1.2e-14</td> <td> 5.01e-16</td>
</tr>
<tr>
  <th>TAX_0</th>     <td>-4.385e-15</td> <td> 2.14e-15</td> <td>   -2.047</td> <td> 0.041</td> <td>-8.59e-15</td> <td>-1.77e-16</td>
</tr>
<tr>
  <th>TAX_1</th>     <td>-3.664e-15</td> <td> 2.09e-15</td> <td>   -1.756</td> <td> 0.080</td> <td>-7.76e-15</td> <td> 4.35e-16</td>
</tr>
<tr>
  <th>TAX_2</th>     <td>-4.885e-15</td> <td> 2.02e-15</td> <td>   -2.416</td> <td> 0.016</td> <td>-8.86e-15</td> <td>-9.13e-16</td>
</tr>
<tr>
  <th>MEDV</th>      <td>    1.0000</td> <td> 1.33e-16</td> <td> 7.52e+15</td> <td> 0.000</td> <td>    1.000</td> <td>    1.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>82.703</td> <th>  Durbin-Watson:     </th> <td>   0.166</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 140.111</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.986</td> <th>  Prob(JB):          </th> <td>3.76e-31</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 4.660</td> <th>  Cond. No.          </th> <td>2.01e+17</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 7.91e-30. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.




```python
boston_features_constant = sm.add_constant(boston_features)
stats_model2 =  sm.OLS(boston.target, boston_features_constant).fit()
stats_model2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>            <td>y</td>        <th>  R-squared:         </th>  <td>   1.000</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th>  <td>   1.000</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th>  <td>1.941e+31</td>
</tr>
<tr>
  <th>Date:</th>             <td>Tue, 16 Jul 2019</td> <th>  Prob (F-statistic):</th>   <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>21:02:33</td>     <th>  Log-Likelihood:    </th>  <td>  15467.</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>   506</td>      <th>  AIC:               </th> <td>-3.091e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>   492</td>      <th>  BIC:               </th> <td>-3.085e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>    13</td>      <th>                     </th>      <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>      <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
     <td></td>        <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>   <td>-1.243e-14</td> <td> 5.35e-15</td> <td>   -2.323</td> <td> 0.021</td> <td> -2.3e-14</td> <td>-1.92e-15</td>
</tr>
<tr>
  <th>CRIM</th>    <td> 7.105e-15</td> <td>  6.3e-15</td> <td>    1.129</td> <td> 0.260</td> <td>-5.26e-15</td> <td> 1.95e-14</td>
</tr>
<tr>
  <th>INDUS</th>   <td> 8.882e-16</td> <td> 1.08e-15</td> <td>    0.820</td> <td> 0.413</td> <td>-1.24e-15</td> <td> 3.02e-15</td>
</tr>
<tr>
  <th>CHAS</th>    <td>-3.331e-15</td> <td> 2.39e-15</td> <td>   -1.392</td> <td> 0.165</td> <td>-8.03e-15</td> <td> 1.37e-15</td>
</tr>
<tr>
  <th>RM</th>      <td> 4.718e-15</td> <td> 1.26e-15</td> <td>    3.733</td> <td> 0.000</td> <td> 2.24e-15</td> <td>  7.2e-15</td>
</tr>
<tr>
  <th>AGE</th>     <td>-1.887e-15</td> <td> 1.05e-15</td> <td>   -1.805</td> <td> 0.072</td> <td>-3.94e-15</td> <td> 1.68e-16</td>
</tr>
<tr>
  <th>DIS</th>     <td>-3.553e-15</td> <td> 5.68e-15</td> <td>   -0.625</td> <td> 0.532</td> <td>-1.47e-14</td> <td> 7.61e-15</td>
</tr>
<tr>
  <th>PTRATIO</th> <td> 8.049e-16</td> <td> 7.45e-16</td> <td>    1.081</td> <td> 0.280</td> <td>-6.58e-16</td> <td> 2.27e-15</td>
</tr>
<tr>
  <th>B</th>       <td>-1.221e-15</td> <td> 2.98e-15</td> <td>   -0.410</td> <td> 0.682</td> <td>-7.07e-15</td> <td> 4.63e-15</td>
</tr>
<tr>
  <th>LSTAT</th>   <td> 1.138e-15</td> <td> 1.29e-15</td> <td>    0.879</td> <td> 0.380</td> <td>-1.41e-15</td> <td> 3.68e-15</td>
</tr>
<tr>
  <th>RAD_0</th>   <td>-5.995e-15</td> <td> 2.45e-15</td> <td>   -2.446</td> <td> 0.015</td> <td>-1.08e-14</td> <td>-1.18e-15</td>
</tr>
<tr>
  <th>RAD_1</th>   <td>-5.773e-15</td> <td> 3.22e-15</td> <td>   -1.795</td> <td> 0.073</td> <td>-1.21e-14</td> <td> 5.44e-16</td>
</tr>
<tr>
  <th>TAX_0</th>   <td>-4.385e-15</td> <td> 2.16e-15</td> <td>   -2.033</td> <td> 0.043</td> <td>-8.62e-15</td> <td>-1.48e-16</td>
</tr>
<tr>
  <th>TAX_1</th>   <td>-3.664e-15</td> <td>  2.1e-15</td> <td>   -1.744</td> <td> 0.082</td> <td>-7.79e-15</td> <td> 4.63e-16</td>
</tr>
<tr>
  <th>TAX_2</th>   <td>-4.885e-15</td> <td> 2.04e-15</td> <td>   -2.399</td> <td> 0.017</td> <td>-8.88e-15</td> <td>-8.85e-16</td>
</tr>
<tr>
  <th>MEDV</th>    <td>    1.0000</td> <td> 1.34e-16</td> <td> 7.47e+15</td> <td> 0.000</td> <td>    1.000</td> <td>    1.000</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>75.198</td> <th>  Durbin-Watson:     </th> <td>   0.114</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td> <th>  Jarque-Bera (JB):  </th> <td> 146.708</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.846</td> <th>  Prob(JB):          </th> <td>1.39e-32</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.025</td> <th>  Cond. No.          </th> <td>2.01e+17</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The smallest eigenvalue is 7.91e-30. This might indicate that there are<br/>strong multicollinearity problems or that the design matrix is singular.



## Run the same model in Scikit-learn


```python
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
scikit_model = linreg.fit(boston_features, boston_features['MEDV'])
```


```python
scikit_model.intercept_
```




    -3.552713678800501e-15




```python
scikit_model.coef_
```




    array([ 2.87137407e-15,  2.22295919e-15, -4.95903487e-15, -2.14867577e-15,
            9.51114330e-17,  7.15399269e-15, -1.11605471e-16, -2.81607561e-15,
            1.20797223e-15,  4.97889559e-16, -5.04828453e-16,  1.14448395e-16,
           -1.79024486e-16,  6.27854721e-17,  1.00000000e+00])




```python
stats_model.params
```




    Intercept   -1.243450e-14
    CRIM         7.105427e-15
    INDUS        8.881784e-16
    CHAS        -3.330669e-15
    RM           4.718448e-15
    AGE         -1.887379e-15
    DIS         -3.552714e-15
    PTRATIO      8.049117e-16
    B           -1.221245e-15
    LSTAT        1.137979e-15
    RAD_0       -5.995204e-15
    RAD_1       -5.773160e-15
    TAX_0       -4.385381e-15
    TAX_1       -3.663736e-15
    TAX_2       -4.884981e-15
    MEDV         1.000000e+00
    dtype: float64



## Remove the necessary variables to make sure the coefficients are the same for Scikit-learn vs Statsmodels


```python
boston_features.drop('TAX_1', axis=1, inplace=True)
```

### Statsmodels


```python
outcome = 'MEDV'
predictors = boston_features.columns
f = '+'.join(predictors)
formula = outcome + '~' + f
stats_model = ols(formula=formula, data=boston_features).fit()
stats_model.params
```




    Intercept   -1.065814e-14
    CRIM         1.509903e-14
    INDUS        1.998401e-15
    CHAS        -1.998401e-15
    RM           1.443290e-15
    AGE          8.881784e-16
    DIS          2.664535e-15
    PTRATIO     -4.996004e-16
    B           -1.332268e-15
    LSTAT       -1.582068e-15
    RAD_0       -5.551115e-15
    RAD_1       -6.883383e-15
    TAX_0        1.054712e-15
    TAX_2       -1.776357e-15
    MEDV         1.000000e+00
    dtype: float64



### Scikit-learn


```python
linreg = LinearRegression()
scikit_model = linreg.fit(boston_features, boston_features['MEDV'])
print(scikit_model.intercept_)
print(scikit_model.coef_)
```

    1.7763568394002505e-14
    [-1.76115074e-14  3.31473577e-15 -1.48570731e-15 -1.61094393e-15
      4.36370224e-16  3.88011781e-15 -1.33449918e-15  4.30480898e-15
     -2.19597439e-15 -1.81204825e-16  1.67327037e-16  1.04211187e-15
      8.84005400e-16  1.00000000e+00]


## Interpret the coefficients for PTRATIO, PTRATIO, LSTAT

- CRIM: per capita crime rate by town
- INDUS: proportion of non-retail business acres per town
- CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
- RM: average number of rooms per dwelling
- AGE: proportion of owner-occupied units built prior to 1940
- DIS: weighted distances to five Boston employment centres
- RAD: index of accessibility to radial highways
- TAX: full-value property-tax rate per $10,000
- PTRATIO: pupil-teacher ratio by town
- B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
- LSTAT: % lower status of the population

## Predict the house price given the following characteristics (before manipulation!!)

Make sure to transform your variables as needed!

- CRIM: 0.15
- INDUS: 6.07
- CHAS: 1        
- RM:  6.1
- AGE: 33.2
- DIS: 7.6
- PTRATIO: 17
- B: 383
- LSTAT: 10.87
- RAD: 8
- TAX: 284


```python
x = {'CRIM': 0.15, 'INDUS': 6.07, 'CHAS': 1, 'RM':  6.1, 'AGE': 33.2, 'DIS': 7.6, 'PTRATIO': 17
 ,'B': 383, 'LSTAT': 10.87, 'RAD': 8, 'TAX': 284}
# (go through and change all values, but you need also min/max of o
#other .. etc. x_transform = [y['CRIM'] ]
#linreg.predict(y_predict)
```

## Summary
Congratulations! You've fitted your first multiple linear regression model on the Boston Housing Data.
