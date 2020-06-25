
# Logistic Regression

## Learning goals

1. Compare predicting a continuous outcome to predicting a class
2. Compare linear to logistic regression as classification models
3. Understand how the sigmoid function translates the linear equation to a probability
4. Describe why logistic regression is a descriminative, parametric algorithm
5. Learn how to interpret a trained logistic model's coefficients
6. Explore the C (inverse regularization) paramater and hyperparameter tune
7. Learn how to adjust the threshold of a logistic model
8. Describe the assumptions of linear regression

Logistic regression is a good model to usher us into the world of classification. It takes a concept we are familiar with, a linear equation, and translates it into a form fit for predicting a class.  

It generally can't compete with the best supervised learning algorithms, but it is simple, fast, and interpretable.  

As we will see in mod 4, it will also serve as a segue into our lessons on neural nets.

# 1. Compare predicting a continuous outcome to predicting a class

Thus far, we have worked to predict continuous target variables using linear regression. 

  - Continous target variables:
        - Sales price of a home
        - MPG of a car
        - Price of Google stock
        - Number of rides per day on the El
        - HS graduation rates
        
We will now transition into another category of prediction: classification. Instead of continous target variables, we will be predicting whether records from are data are labeled as a particular class.  Whereas the output for the linear regression model can be any number, the output of our classification algorithms can only be a value designated by a set of discrete outcomes.

  - Categorical target variables:
        - Whether an employee will stay at a company or leave (churn)
        - Whether a tumor is cancerous or benign
        - Whether a flower is a rose, a dandelion, a tulip, or a daffodil
        - Whether a voter is Republican, Democrat, or Independent
        
What are some other categorical target variables can you think of?

![discuss](https://media.giphy.com/media/l0MYIAUWRmVVzfHag/giphy.gif)



### We are still dealing with **labeled data**.

![labels](https://media.giphy.com/media/26Ff5evMweBsENWqk/giphy.gif)


This is still supervised learning. 

But now, instead of the label being a continuous value, such as house price, the label is the category.  This can be either binary or multiclass.  But we still need the labels to train our models.




```python
from sklearn.datasets import load_iris

data = load_iris()
# Here, in our familiar iris dataset, we see that the target variable is one of three classes labeled 0, 1, 2 
# relating to setosa, versicolor, and virginica
print(data.target)
print(data.target_names)
```

    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2]
    ['setosa' 'versicolor' 'virginica']


# 2. Compare linear to logistic regression as classification models


The goal of logistic regression, and any classification problem, is to build a model which accurately separates the classes based on independent variables.  

We are already familiar with how linear regression finds a best-fit "line".  It uses the MSE cost function to minimize the difference between true and predicted values.  

A natural thought would be to use that "line" to descriminate between classes: Everything with an output greater than a certain point is classified as a 1, everything below is classified as a 0.

# Glass Data
Take the glass data set from the UCI Machine Learning Dataset.  

It is composed of a set of features describing the physical makeup of different glass types.  

Glass types 1,2,3 represent window glass.
Glass types 4,5,6 represent household glass.

We will try to predict whether a record is window glass or household glass. 


```python
# glass identification dataset
import pandas as pd
import numpy as np
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data'
col_names = ['id','ri','na','mg','al','si','k','ca','ba','fe','glass_type']
glass = pd.read_csv(url, names=col_names, index_col='id')
glass.sort_values('al', inplace=True)
# types 1, 2, 3 are window glass
# types 5, 6, 7 are household glass
glass['household'] = glass.glass_type.map({1:0, 2:0, 3:0, 5:1, 6:1, 7:1})
glass.head()
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
      <th>ri</th>
      <th>na</th>
      <th>mg</th>
      <th>al</th>
      <th>si</th>
      <th>k</th>
      <th>ca</th>
      <th>ba</th>
      <th>fe</th>
      <th>glass_type</th>
      <th>household</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>1.51966</td>
      <td>14.77</td>
      <td>3.75</td>
      <td>0.29</td>
      <td>72.02</td>
      <td>0.03</td>
      <td>9.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>185</th>
      <td>1.51115</td>
      <td>17.38</td>
      <td>0.00</td>
      <td>0.34</td>
      <td>75.41</td>
      <td>0.00</td>
      <td>6.65</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1.52213</td>
      <td>14.21</td>
      <td>3.82</td>
      <td>0.47</td>
      <td>71.77</td>
      <td>0.11</td>
      <td>9.57</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1.52213</td>
      <td>14.21</td>
      <td>3.82</td>
      <td>0.47</td>
      <td>71.77</td>
      <td>0.11</td>
      <td>9.57</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1.52320</td>
      <td>13.72</td>
      <td>3.72</td>
      <td>0.51</td>
      <td>71.75</td>
      <td>0.09</td>
      <td>10.06</td>
      <td>0.0</td>
      <td>0.16</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Let's look at the relationship between aluminum content and glass type. 
There appears to be a relationship between the two, where more aluminum correlates with household glass. 


```python
import matplotlib.pyplot as plt
plt.scatter(glass.al, glass.household)
plt.xlabel('al')
plt.ylabel('household')
```




    Text(0, 0.5, 'household')




![png](index_files/index_11_1.png)


We could fit a linear regression model to this data


```python
from sklearn.linear_model import LinearRegression
# fit a linear regression model and store the predictions

feature_cols = ['al']
X = glass[feature_cols]
y = glass.household

lr = LinearRegression()
lr.fit(X, y)
glass['household_pred'] = lr.predict(X)
```


```python
# scatter plot that includes the regression line
plt.scatter(glass.al, glass.household)
plt.plot(glass.al, glass.household_pred, color='red')
plt.xlabel('al')
plt.ylabel('household')
```




    Text(0, 0.5, 'household')




![png](index_files/index_14_1.png)


## What are some issues with the graph above?

![talk amongst yourselves](https://media.giphy.com/media/3o6Zt44rlujPePNVVC/giphy.gif)

If al=3, what class do we predict for household?

If al=1.5, what class do we predict for household?

We predict the 0 class for lower values of al, and the 1 class for higher values of al. What's our cutoff value? Around al=2, because that's where the linear regression line crosses the midpoint between predicting class 0 and class 1.

Therefore, we'll say that if household_pred >= 0.5, we predict a class of 1, else we predict a class of 0.


```python
# transform household_pred to 1 or 0
glass['household_pred_class'] = np.where(glass.household_pred >= 0.5, 1, 0)
glass.head()
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
      <th>ri</th>
      <th>na</th>
      <th>mg</th>
      <th>al</th>
      <th>si</th>
      <th>k</th>
      <th>ca</th>
      <th>ba</th>
      <th>fe</th>
      <th>glass_type</th>
      <th>household</th>
      <th>household_pred</th>
      <th>household_pred_class</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>1.51966</td>
      <td>14.77</td>
      <td>3.75</td>
      <td>0.29</td>
      <td>72.02</td>
      <td>0.03</td>
      <td>9.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>-0.340495</td>
      <td>0</td>
    </tr>
    <tr>
      <th>185</th>
      <td>1.51115</td>
      <td>17.38</td>
      <td>0.00</td>
      <td>0.34</td>
      <td>75.41</td>
      <td>0.00</td>
      <td>6.65</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>6</td>
      <td>1</td>
      <td>-0.315436</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1.52213</td>
      <td>14.21</td>
      <td>3.82</td>
      <td>0.47</td>
      <td>71.77</td>
      <td>0.11</td>
      <td>9.57</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>-0.250283</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1.52213</td>
      <td>14.21</td>
      <td>3.82</td>
      <td>0.47</td>
      <td>71.77</td>
      <td>0.11</td>
      <td>9.57</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>-0.250283</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1.52320</td>
      <td>13.72</td>
      <td>3.72</td>
      <td>0.51</td>
      <td>71.75</td>
      <td>0.09</td>
      <td>10.06</td>
      <td>0.0</td>
      <td>0.16</td>
      <td>1</td>
      <td>0</td>
      <td>-0.230236</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# plot the class predictions
plt.scatter(glass.al, glass.household)
plt.plot(glass.al, glass.household_pred_class, color='red')
plt.xlabel('al')
plt.ylabel('household')
```




    Text(0, 0.5, 'household')




![png](index_files/index_18_1.png)



```python
# Let's see what happens if we have class imbalance.

from sklearn.linear_model import LinearRegression
# fit a linear regression model and store the predictions

feature_cols = ['al']

# Add data with high aluminum content associated with household glass
X_more = np.random.normal(4.5, .25, 20)
y_more = np.full(20, 1,)
y_aug = np.append(y, y_more)
X_aug = np.append(X['al'], X_more).reshape(-1,1)
```


```python
# fit our regression
lr = LinearRegression()
lr.fit(X_aug, y_aug.reshape(-1,1))
y_hat = lr.predict(X_aug)

```


```python
# The line is shifts dramatically with this outliers.
plt.scatter(X_aug, y_aug)
plt.plot(X_aug, y_hat, color='red')
plt.xlabel('al')
plt.ylabel('household')
```




    Text(0, 0.5, 'household')




![png](index_files/index_21_1.png)



```python
# With class imbalance

pred_class = np.where(y_hat >= 0.5, 1, 0)
np.unique(pred_class, return_counts = True)

# plot the class predictions
plt.scatter(X_aug, y_aug)
plt.plot(X_aug, pred_class, color='red')
plt.xlabel('al')
plt.ylabel('household')
```




    Text(0, 0.5, 'household')




![png](index_files/index_22_1.png)


## How about we use logistic logistic regression instead?


Logistic regression performs a similar function as above:


```python
# fit a logistic regression model and store the class predictions
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
feature_cols = ['al']
X = glass[feature_cols]
y = glass.household
logreg.fit(X, y)
glass['household_pred_class'] = logreg.predict(X)
```


```python
# plot the class predictions
plt.scatter(glass.al, glass.household)
plt.plot(glass.al, glass.household_pred_class, color='red')
plt.xlabel('al')
plt.ylabel('household')
```




    Text(0, 0.5, 'household')




![png](index_files/index_26_1.png)


Not only do we have class predictions:


```python
logreg.predict(X)
```




    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])




```python
# fit a logistic regression model and store the class predictions
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)

logreg.fit(X_aug, y_aug)
log_aug_pred = logreg.predict(X_aug)
```


```python
# plot the class predictions
plt.scatter(X_aug, y_aug)
plt.plot(X_aug, log_aug_pred, color='red')
plt.xlabel('al')
plt.ylabel('household')
```




    Text(0, 0.5, 'household')




![png](index_files/index_30_1.png)


# 3. Understand how the sigmoid function translates the linear equation to a probability

## Predict Proba

Let's take a closer look into what fitting the logistic model results in.

Along with the class predictions, we have probabilities associated with each record:


```python
logreg.predict_proba(X)[:10]
```




    array([[0.99850141, 0.00149859],
           [0.99815364, 0.00184636],
           [0.99682475, 0.00317525],
           [0.99682475, 0.00317525],
           [0.99624893, 0.00375107],
           [0.99538086, 0.00461914],
           [0.99538086, 0.00461914],
           [0.99498004, 0.00501996],
           [0.99328482, 0.00671518],
           [0.99300013, 0.00699987]])



## Probabilities and the sigmoid functions
Here, we can start digging deeper into how logistic regression works.

The idea behind logistic regression is to model the conditional probability of a class given a set of independent features.

For the binary case, it is the probability of a 0 or 1 based on a set of independent features X.

$\Large P(G = 1|X = x)$

Since the total probability must be equal to 1:

$\Large P(G = 0|X = x) = 1 - P(G = 1|X = x)$ 

In order to realize such a goal, we have to somehow translate our linear output into a probability.  As we know, probability takes on a value between 0 and 1,  whereas the linear equation can output any value from $-\infty$ to $\infty$.

![sigmoid](https://media.giphy.com/media/GtKtQ9Gb064uY/giphy.gif)

In comes the sigmoid function to the rescue.


<img src='https://cdn-images-1.medium.com/max/1600/1*RqXFpiNGwdiKBWyLJc_E7g.png' />

If ‘Z’ goes to infinity, Y(predicted) will become 1 and if ‘Z’ goes to negative infinity, Y(predicted) will become 0.



Using the sigmoid function above, if X = 1, the estimated probability would be 0.8. This tells that there is 80% chance that this observation would fall in the positive class.




```python
import numpy as np
def sigmoid(any_number):
    
    "Input any number, return a number between 0 and 1"
    
    return 1/(1+ np.e**(-any_number))
    
```

In your head, work through the approximate output of the function for:
  - z = 0
  - z = 1000
  - z = -1000


```python
# Now, input numbers into the function to see that it is functioning correcting.

sigmoid()
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-21-32c3939704e4> in <module>
          1 # Now, input numbers into the function to see that it is functioning correcting.
          2 
    ----> 3 sigmoid()
    

    TypeError: sigmoid() missing 1 required positional argument: 'any_number'


## Linking it to the linear equation

Now for the fun part.  The input of the sigmoid is our trusty old linear equation.

$$ \hat y = \hat\beta_0 + \hat\beta_1 x_1 + \hat\beta_2, x_2 +\ldots + \hat\beta_n x_n $$

The linear equation is passed into the sigmoid function to produce a probability between 0 and 1
$$\displaystyle\frac{1}{1+e^{-(\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n)}}$$

Remember, the goal of logistic regression is to model the conditional of a class using a transformation of the linear equation.

In other words:

$$\Large P(G = 1|X = x_1, x_2...x_n) = \frac{1}{1+e^{-(\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n)}}$$


Now, with some arithmetic:

You can show that, by multiplying both numerator and denominator by $e^{(\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n)}$


$$ \Large P(G = 1|X = x) = \displaystyle \frac{e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n}}{1+e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n}}$$

As a result, you can compute:

$$ \Large P(G = 0|X =x) = 1- \displaystyle \frac{e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n}}{1+e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n}}= \displaystyle \frac{1}{1+e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n}}$$


#### Odds ratio

This doesn't seem to be very spectacular, but combining these two results leads to an easy interpretation of the model parameters, triggered by the *odds*, which equal p/(1-p):

$$ \Large \dfrac{ P(G = 1|X = x) }{P(G = 0|X =x)} = e^{\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n} $$

This expression can be interpreted as the *odds in favor of class 1*.  

Taking the log of both sides leads to:
<br><br>
    $\ln{\dfrac{ P(G = 1|X = x) }{P(G = 0|X =x)}} = \beta_0 + \beta_1*X_1 + \beta_2*X_2...\beta_n*X_n$
    
Here me can see why we call it logisitic regression.

Our linear function calculates the log of the probability we predict 1, divided by the probability of predicting 0.  In other words, the linear equation is calculating the **log of the odds** that we predict a class of 1.
    

# Pair
### Those are a lot of formulas to take in.  

To help us reinforce how logistic regression works, let's do an exercise where we reproduce the predicted probabilities by using our coefficients.  Below is model we fit above, predicting whether glass was window or household glass.



```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
feature_cols = ['al']
X = glass[feature_cols]
y = glass.household
logreg.fit(X, y)
glass['household_pred_class'] = logreg.predict(X)
glass.head()
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
      <th>ri</th>
      <th>na</th>
      <th>mg</th>
      <th>al</th>
      <th>si</th>
      <th>k</th>
      <th>ca</th>
      <th>ba</th>
      <th>fe</th>
      <th>glass_type</th>
      <th>household</th>
      <th>household_pred</th>
      <th>household_pred_class</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>1.51966</td>
      <td>14.77</td>
      <td>3.75</td>
      <td>0.29</td>
      <td>72.02</td>
      <td>0.03</td>
      <td>9.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>-0.340495</td>
      <td>0</td>
    </tr>
    <tr>
      <th>185</th>
      <td>1.51115</td>
      <td>17.38</td>
      <td>0.00</td>
      <td>0.34</td>
      <td>75.41</td>
      <td>0.00</td>
      <td>6.65</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>6</td>
      <td>1</td>
      <td>-0.315436</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1.52213</td>
      <td>14.21</td>
      <td>3.82</td>
      <td>0.47</td>
      <td>71.77</td>
      <td>0.11</td>
      <td>9.57</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>-0.250283</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1.52213</td>
      <td>14.21</td>
      <td>3.82</td>
      <td>0.47</td>
      <td>71.77</td>
      <td>0.11</td>
      <td>9.57</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>-0.250283</td>
      <td>0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>1.52320</td>
      <td>13.72</td>
      <td>3.72</td>
      <td>0.51</td>
      <td>71.75</td>
      <td>0.09</td>
      <td>10.06</td>
      <td>0.0</td>
      <td>0.16</td>
      <td>1</td>
      <td>0</td>
      <td>-0.230236</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Just like linear regression, logistic regression calculates parameters associated with features and intercept. In the fit model above, we have one coefficient associated with aluminum, and one associate with the intercept.

In the cell below, use those coefficients along with the original data to calculate an array repressenting the logodds.


```python
log_odds = None
```

Now, take that array, and feed it into the sigmoid function to get the probabilities of class 1.


```python
# Confirm that the output of predict_proba matches the probabilities you calculated
logreg.predict_proba(X)[:10]
```




    array([[0.99850073, 0.00149927],
           [0.99815284, 0.00184716],
           [0.99682354, 0.00317646],
           [0.99682354, 0.00317646],
           [0.99624756, 0.00375244],
           [0.99537926, 0.00462074],
           [0.99537926, 0.00462074],
           [0.99497834, 0.00502166],
           [0.99328271, 0.00671729],
           [0.99299796, 0.00700204]])



# 4. Describe why logistic regression is a descriminative, parametric algorithm


A decision boundary is a pretty simple concept. Logistic regression is a classification algorithm, the output should be a category: Yes/No, True/False, Red/Yellow/Orange. Our prediction function however returns a probability score between 0 and 1. A decision boundary is a threshold or tipping point that helps us decide which category to choose based on probability.

Logistic regression is a parametric, discriminative model.  

In other words, its decisions are made via trained parameters: our beta coefficients. The hyperplane that these coefficients define is a boundary by which we can discriminate between the classes.    

![](img/decision_boundary_2.jpg)

# 5. Interpreting Coefficients

What does our coefficient calculated above mean?


```python
logreg.coef_
```




    array([[4.18041341]])



**Interpretation:** A 1 unit increase in 'al' is associated with a 4.18 unit increase in the log-odds of 'household'.

**Bottom line:** Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).

# Note on optimizing the coefficients

Instead of optimizing the coefficients based on mean squared error, logistic regression looks to maximize the likelihood of seeing the probabilities given the class.

Because we are dealing with a binary outcome, our likelihood equation comes from the Bernouli distribution:

$$ \Large Likelihood=\prod\limits_{i=0}^N p_i^{y_i}(1-p_i)^{1-y_i}$$

Taking the log of both sides leads to the log_likelihood equation:

$$ \Large loglikelihood = \sum\limits_{i=1}^N y_i\log{p_i} + (1-y_i)\log(1-p_i) $$

The goal of MLE is to maximize log-likelihood.

Or, as we are generally look for minimums, we minimize the negative loglikelihood, which is our cost function:

$$ \Large negative\ loglikelihood = \sum\limits_{i=1}^N - y_i\log{p_i} - (1-y_i)\log(1-p_i) $$

When solving for the optimal coefficients of a logistic regression model, Log-Loss is the cost function that is used.


The general idea is to start with a set of betas, calculate the probabilities, calculate the log-likelihood, adjust the Betas in the direction of which gradient is heading towards higher likelihood.

There is no closed form solution like the normal equation in linear regression, so we have to use stocastic gradient descent.  To do so we take the derivative of the negative loglikelihood and set it to zero to find the gradient of the loglikelihood, then update our coefficients. Just like in linear regression SGD, we use a learning rate when updating them.


https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a

http://wiki.fast.ai/index.php/Log_Loss

Here is a good Youtube video on MLE: https://www.youtube.com/watch?v=BfKanl1aSG0

Math behind the gradient of log-likelihood is ESL section 4.4.1: https://web.stanford.edu/~hastie/ElemStatLearn//.




# 6. Hyperparameter Tuning the C Variable

We have discussed 'L1' (lasso)  and 'L2' (ridge) regularization.  If you looked at the docstring of Sklearn's Logistic Regression function, you may have noticed that we can specify different types of regularization when fitting the model via the `penalty` parameter.

We can also specificy the strength of the regularization via the `C` parameter. `C` is the inverse regularization strength.  So, a low `C` means high regularization strength.


```python
from sklearn.linear_model import LogisticRegression

# Ridge regularization with low strength
logr = LogisticRegression(penalty='l2', C=10**8)

```


```python
diabetes = pd.read_csv('data/diabetes.csv')
diabetes.head()
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
      <th>Pregnancies</th>
      <th>Glucose</th>
      <th>BloodPressure</th>
      <th>SkinThickness</th>
      <th>Insulin</th>
      <th>BMI</th>
      <th>DiabetesPedigreeFunction</th>
      <th>Age</th>
      <th>Outcome</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>148</td>
      <td>72</td>
      <td>35</td>
      <td>0</td>
      <td>33.6</td>
      <td>0.627</td>
      <td>50</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>85</td>
      <td>66</td>
      <td>29</td>
      <td>0</td>
      <td>26.6</td>
      <td>0.351</td>
      <td>31</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8</td>
      <td>183</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>23.3</td>
      <td>0.672</td>
      <td>32</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>89</td>
      <td>66</td>
      <td>23</td>
      <td>94</td>
      <td>28.1</td>
      <td>0.167</td>
      <td>21</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>137</td>
      <td>40</td>
      <td>35</td>
      <td>168</td>
      <td>43.1</td>
      <td>2.288</td>
      <td>33</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
logr = LogisticRegression(penalty='l2', C=10**8)
y = diabetes.Outcome
X = diabetes.drop("Outcome", axis=1)
logr.fit(X, y)

print(logr.coef_)
print(logr.intercept_)
print(logr.score(X,y))
```

    [[ 1.18217114e-01  3.50280937e-02 -1.35333258e-02  7.25847638e-04
      -1.20213666e-03  9.05742512e-02  9.52539204e-01  1.60020353e-02]]
    [-8.42344453]
    0.7825520833333334


    /Users/johnmaxbarry/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)



```python
# Same result as 'none'
logr = LogisticRegression(penalty='none', C=10**8, max_iter=10000)
y = diabetes.Outcome
X = diabetes.drop("Outcome", axis=1)
logr.fit(X, y)

for coef, feature in zip(list(logr.coef_[0]), X.columns):
    print(round(coef,4), feature)
print(logr.intercept_)
print(logr.score(X,y))
```

    0.1232 Pregnancies
    0.0352 Glucose
    -0.0133 BloodPressure
    0.0006 SkinThickness
    -0.0012 Insulin
    0.0897 BMI
    0.9456 DiabetesPedigreeFunction
    0.0149 Age
    [-8.40952187]
    0.7825520833333334


    /Users/johnmaxbarry/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:1505: UserWarning: Setting penalty='none' will ignore the C and l1_ratio parameters
      "Setting penalty='none' will ignore the C and l1_ratio "



```python
# With a high L2 regularization, the coefficients shrink.
logr = LogisticRegression(penalty='l2', C=.0001)
y = diabetes.Outcome
X = diabetes.drop("Outcome", axis=1)
logr.fit(X, y)

for coef, feature in zip(list(logr.coef_[0]), X.columns):
    print(round(coef,4), feature)
print(logr.intercept_)
print(logr.score(X,y))
```

    0.0134 Pregnancies
    0.0324 Glucose
    -0.0055 BloodPressure
    0.006 SkinThickness
    -0.0008 Insulin
    0.0313 BMI
    0.0012 DiabetesPedigreeFunction
    0.018 Age
    [-6.02705282]
    0.7643229166666666



```python
# try with a strongL1. They are all shrunk to zero!
logr = LogisticRegression(penalty='l1', C=.0001, solver='liblinear')
y = diabetes.Outcome
X = diabetes.drop("Outcome", axis=1)
logr.fit(X, y)

for coef, feature in zip(list(logr.coef_[0]), X.columns):
    print(round(coef,4), feature)
print(logr.intercept_)
print(logr.score(X,y))
```

    0.0 Pregnancies
    0.0 Glucose
    0.0 BloodPressure
    0.0 SkinThickness
    0.0 Insulin
    0.0 BMI
    0.0 DiabetesPedigreeFunction
    0.0 Age
    [0.]
    0.6510416666666666


How do we choose between them? We iterate over possible parameters and judge the success based on our metric of choice.  We will eventually move towards grid search, which will help us be more thorough with our tuning.  For now, we will work through how to tune our C parameter with an Ridge regularization.

For now, let's judge on accuracy, which can be accessed via the `score()` method of a trained model.



```python
# The parameters for C can be anything above 0.  
# Set up a list of possible values to try out.
# Start with 1000 numbers above 0
c_candidates = None

```


```python
# Split your data into training and test data with a random state of 42 
# and a test size of .3
from sklearn.model_selection import train_test_split

# Your code here
```


```python
# Train the no regularization logistic model on the train set, 
# and return the accuracy as measured on the test

logr = LogisticRegression(penalty='l2', C=10**8, max_iter=10000)
y = diabetes.Outcome
X = diabetes.drop("Outcome", axis=1)
logr.fit(X_train, y_train)

print(logr.coef_)
print(logr.intercept_)
print(logr.score(X_test,y_test))
```

    [[ 0.05806895  0.03587482 -0.0108455  -0.00152413 -0.00099075  0.10902572
       0.42182975  0.03594817]]
    [-9.44606271]
    0.7359307359307359



```python
# Create a for loop which runs through all of the possible values of C,
# fits the model on the train set, and scores the model on test set.
# Add the accuracies into a dictionary or a list, whichever you prefer
# Use 'l2'

c_scores = {}
for c in c_candidates:
    pass

best_c = max(c_scores, key=c_scores.get)
best_c
    
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-106-b7eebe9cade7> in <module>
          8     pass
          9 
    ---> 10 best_c = max(c_scores, key=c_scores.get)
         11 best_c
         12 


    ValueError: max() arg is an empty sequence



```python
# Once you have the best C for the range of values between 0 and 1000,
# narrow in even further.  Choose a new range of 100 C-values between your best C
# and the next closer integer.
import math
c_candidates = np.linspace(math.floor(best_c)+.01,math.floor(best_c)+1, 100)
```

We improved our test R^2 from .740 to .745. Not too much gain. 
Sometimes hyperparameter tuning can have a large effect, sometimes not. 
Don't rely on hyperparameter tuning to fix your model.  
Treat it as a necessary step in the process which if you are lucky, may increase the predictive power of your model.

In future lessons, we will use Grid Search to automate our hyperparamater tuning, and make it more thorough.


## 7. Threshold

Because logistic regression calculates the probability of a given class, we can easily change the threshold of what is categorized as a 1 or a 0.   

Let's use our best c from above, and use predict_proba() to output probabilities.


```python
regr = LogisticRegression(penalty='l2', C=c)
regr.fit(X_train, y_train)
probas = regr.predict_proba(X_test)
probas
```




    array([[0.73448964, 0.26551036],
           [0.81663515, 0.18336485],
           [0.87880443, 0.12119557],
           [0.84386409, 0.15613591],
           [0.50388651, 0.49611349],
           [0.55417737, 0.44582263],
           [0.98640773, 0.01359227],
           [0.38353854, 0.61646146],
           [0.44359444, 0.55640556],
           [0.20688072, 0.79311928],
           [0.77188623, 0.22811377],
           [0.09960853, 0.90039147],
           [0.61558465, 0.38441535],
           [0.71437818, 0.28562182],
           [0.93046452, 0.06953548],
           [0.64019335, 0.35980665],
           [0.87284577, 0.12715423],
           [0.93133688, 0.06866312],
           [0.14131971, 0.85868029],
           [0.40641923, 0.59358077],
           [0.78804599, 0.21195401],
           [0.92446224, 0.07553776],
           [0.52986444, 0.47013556],
           [0.90654853, 0.09345147],
           [0.4564225 , 0.5435775 ],
           [0.11356232, 0.88643768],
           [0.89170448, 0.10829552],
           [0.96976598, 0.03023402],
           [0.72707198, 0.27292802],
           [0.8859791 , 0.1140209 ],
           [0.08577806, 0.91422194],
           [0.11940044, 0.88059956],
           [0.19446235, 0.80553765],
           [0.17623291, 0.82376709],
           [0.3606312 , 0.6393688 ],
           [0.31454081, 0.68545919],
           [0.0452017 , 0.9547983 ],
           [0.77140866, 0.22859134],
           [0.51918911, 0.48081089],
           [0.27040316, 0.72959684],
           [0.93723944, 0.06276056],
           [0.41661771, 0.58338229],
           [0.44573304, 0.55426696],
           [0.68472919, 0.31527081],
           [0.97167786, 0.02832214],
           [0.47071468, 0.52928532],
           [0.3917095 , 0.6082905 ],
           [0.79440448, 0.20559552],
           [0.66340286, 0.33659714],
           [0.03603383, 0.96396617],
           [0.95598421, 0.04401579],
           [0.34075161, 0.65924839],
           [0.17477983, 0.82522017],
           [0.74132017, 0.25867983],
           [0.8926453 , 0.1073547 ],
           [0.96262389, 0.03737611],
           [0.20995711, 0.79004289],
           [0.99590968, 0.00409032],
           [0.59599864, 0.40400136],
           [0.2220079 , 0.7779921 ],
           [0.27553141, 0.72446859],
           [0.65488957, 0.34511043],
           [0.77054217, 0.22945783],
           [0.7939636 , 0.2060364 ],
           [0.9180347 , 0.0819653 ],
           [0.38202982, 0.61797018],
           [0.95516759, 0.04483241],
           [0.21616069, 0.78383931],
           [0.96318711, 0.03681289],
           [0.21083105, 0.78916895],
           [0.3044171 , 0.6955829 ],
           [0.93360253, 0.06639747],
           [0.83152467, 0.16847533],
           [0.88093397, 0.11906603],
           [0.91050754, 0.08949246],
           [0.49425782, 0.50574218],
           [0.8399165 , 0.1600835 ],
           [0.88482845, 0.11517155],
           [0.86126514, 0.13873486],
           [0.73691508, 0.26308492],
           [0.33029405, 0.66970595],
           [0.86348794, 0.13651206],
           [0.94428648, 0.05571352],
           [0.59527863, 0.40472137],
           [0.73421246, 0.26578754],
           [0.13527446, 0.86472554],
           [0.09754939, 0.90245061],
           [0.68656665, 0.31343335],
           [0.88744933, 0.11255067],
           [0.91616052, 0.08383948],
           [0.9378747 , 0.0621253 ],
           [0.7420584 , 0.2579416 ],
           [0.99699002, 0.00300998],
           [0.49566077, 0.50433923],
           [0.47727056, 0.52272944],
           [0.35620501, 0.64379499],
           [0.64891581, 0.35108419],
           [0.8804961 , 0.1195039 ],
           [0.31696977, 0.68303023],
           [0.92094959, 0.07905041],
           [0.27567292, 0.72432708],
           [0.93780694, 0.06219306],
           [0.20924864, 0.79075136],
           [0.46045482, 0.53954518],
           [0.33602521, 0.66397479],
           [0.76796238, 0.23203762],
           [0.75384868, 0.24615132],
           [0.25165018, 0.74834982],
           [0.88760793, 0.11239207],
           [0.51192644, 0.48807356],
           [0.91307805, 0.08692195],
           [0.64206582, 0.35793418],
           [0.98578232, 0.01421768],
           [0.22908243, 0.77091757],
           [0.81541793, 0.18458207],
           [0.68082214, 0.31917786],
           [0.24703858, 0.75296142],
           [0.77724461, 0.22275539],
           [0.93778172, 0.06221828],
           [0.45642043, 0.54357957],
           [0.94066963, 0.05933037],
           [0.73526882, 0.26473118],
           [0.78291046, 0.21708954],
           [0.9331154 , 0.0668846 ],
           [0.71565279, 0.28434721],
           [0.6070472 , 0.3929528 ],
           [0.97129524, 0.02870476],
           [0.13691968, 0.86308032],
           [0.03232322, 0.96767678],
           [0.25776838, 0.74223162],
           [0.29250286, 0.70749714],
           [0.14084784, 0.85915216],
           [0.9105725 , 0.0894275 ],
           [0.58611539, 0.41388461],
           [0.161801  , 0.838199  ],
           [0.88206124, 0.11793876],
           [0.83530604, 0.16469396],
           [0.12030205, 0.87969795],
           [0.1973698 , 0.8026302 ],
           [0.9884428 , 0.0115572 ],
           [0.91202144, 0.08797856],
           [0.96095232, 0.03904768],
           [0.7800487 , 0.2199513 ],
           [0.54922969, 0.45077031],
           [0.87011032, 0.12988968],
           [0.74108079, 0.25891921],
           [0.8828019 , 0.1171981 ],
           [0.98094143, 0.01905857],
           [0.59249147, 0.40750853],
           [0.22838805, 0.77161195],
           [0.90147561, 0.09852439],
           [0.5211832 , 0.4788168 ],
           [0.73869919, 0.26130081],
           [0.83498853, 0.16501147],
           [0.99539938, 0.00460062],
           [0.58266239, 0.41733761],
           [0.68679347, 0.31320653],
           [0.35139478, 0.64860522],
           [0.23889093, 0.76110907],
           [0.85497748, 0.14502252],
           [0.48619602, 0.51380398],
           [0.35638455, 0.64361545],
           [0.81827383, 0.18172617],
           [0.98437037, 0.01562963],
           [0.87798117, 0.12201883],
           [0.11800231, 0.88199769],
           [0.95296749, 0.04703251],
           [0.70366246, 0.29633754],
           [0.20095442, 0.79904558],
           [0.45024235, 0.54975765],
           [0.37696556, 0.62303444],
           [0.82695131, 0.17304869],
           [0.65615804, 0.34384196],
           [0.23445147, 0.76554853],
           [0.35824207, 0.64175793],
           [0.91265245, 0.08734755],
           [0.70319609, 0.29680391],
           [0.74996434, 0.25003566],
           [0.78201433, 0.21798567],
           [0.67922025, 0.32077975],
           [0.49282494, 0.50717506],
           [0.41049691, 0.58950309],
           [0.60774864, 0.39225136],
           [0.22081729, 0.77918271],
           [0.35573603, 0.64426397],
           [0.90971712, 0.09028288],
           [0.95601041, 0.04398959],
           [0.87819494, 0.12180506],
           [0.15269383, 0.84730617],
           [0.63366366, 0.36633634],
           [0.93913987, 0.06086013],
           [0.91530767, 0.08469233],
           [0.11413536, 0.88586464],
           [0.74844187, 0.25155813],
           [0.92521747, 0.07478253],
           [0.94579229, 0.05420771],
           [0.9961893 , 0.0038107 ],
           [0.93757734, 0.06242266],
           [0.76018382, 0.23981618],
           [0.3063606 , 0.6936394 ],
           [0.86976076, 0.13023924],
           [0.88826324, 0.11173676],
           [0.65462468, 0.34537532],
           [0.72899025, 0.27100975],
           [0.3014781 , 0.6985219 ],
           [0.90824438, 0.09175562],
           [0.90106051, 0.09893949],
           [0.75161894, 0.24838106],
           [0.09902713, 0.90097287],
           [0.37897441, 0.62102559],
           [0.72382684, 0.27617316],
           [0.78898929, 0.21101071],
           [0.85974489, 0.14025511],
           [0.88536914, 0.11463086],
           [0.27039747, 0.72960253],
           [0.91524768, 0.08475232],
           [0.19063715, 0.80936285],
           [0.70929191, 0.29070809],
           [0.65451459, 0.34548541],
           [0.12021553, 0.87978447],
           [0.39279522, 0.60720478],
           [0.89679068, 0.10320932],
           [0.92657739, 0.07342261],
           [0.83985673, 0.16014327],
           [0.93769284, 0.06230716],
           [0.25276423, 0.74723577],
           [0.67609522, 0.32390478],
           [0.73267162, 0.26732838],
           [0.70352636, 0.29647364],
           [0.8171564 , 0.1828436 ],
           [0.89906029, 0.10093971]])




```python
y_hat = regr.predict(X_test)
y_hat
```




    array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0,
           0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0,
           0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1,
           0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
           0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0,
           0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
           0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
           0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0,
           0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0,
           0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1,
           1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])



Compare the output of predict and predict proba. Write out below how the output of predict_proba is related to the predict output.  

Now, isolate one of the columns of predict_proba, and create an area of booleans which returns True if the proba is above .4


```python
# Your code here
lower_threshold = None
```

Then, use the astype method to convert the array to integers: True will become 1, and False will become 0


```python
# Your code here

predictions = None
```

While the accuracy of the model will fall by increasing the threshold, we are protecting against a certain type of error. What type of error are we reducing? Of the metrics that we have learned, what score will increase? Why might protecting against such errors be smart in a model that deals with a life-threatening medical condition?



```python
# Your answer here
```


```python
from sklearn.metrics import accuracy_score, recall_score, precision_score
# check that logic in code.
```



# 8. Assumptions of Logistic Regression

Logistic regression does not make many of the key assumptions of linear regression and general linear models that are based on ordinary least squares algorithms – particularly regarding linearity, normality, and homoscedasticity.

First, logistic regression does not require a linear relationship between the dependent and independent variables.  Second, the error terms (residuals) do not need to be normally distributed.  Third, homoscedasticity is not required.  

**The following assumptions still apply:**

1.  Binary logistic regression requires the dependent variable to be binary and ordinal logistic regression requires the dependent variable to be ordinal.

2. Logistic regression requires the observations to be independent of each other.  In other words, the observations should not come from repeated measurements or matched data.

3. Logistic regression requires there to be little or no multicollinearity among the independent variables.  This means that the independent variables should not be too highly correlated with each other.

4. Logistic regression assumes linearity of independent variables and log odds.  although this analysis does not require the dependent and independent variables to be related linearly, it requires that the independent variables are linearly related to the log odds.

5. Logistic regression typically requires a large sample size.  A general guideline is that you need at minimum of 10 cases with the least frequent outcome for each independent variable in your model. For example, if you have 5 independent variables and the expected probability of your least frequent outcome is .10, then you would need a minimum sample size of 500 (10*5 / .10).


```python

```
