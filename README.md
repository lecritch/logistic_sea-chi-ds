
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

# Why logistic 1st of our classifiers?

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




![png](index_files/index_18_1.png)



```python
# Back to our original dataset

plt.scatter(glass.al, glass.household)
plt.plot(glass.al, glass.household_pred, color='red')
plt.xlabel('al')
plt.ylabel('household')
```




    Text(0, 0.5, 'household')




![png](index_files/index_19_1.png)


We predict the 0 class for lower values of al, and the 1 class for higher values of al. What's our cutoff value? Around al=2, because that's where the linear regression line crosses the midpoint between predicting class 0 and class 1.

Therefore, we'll say that if household_pred >= 0.5, we predict a class of 1, else we predict a class of 0.

If al=3, what class do we predict for household?

If al=1.5, what class do we predict for household?




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




![png](index_files/index_22_1.png)



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




![png](index_files/index_23_1.png)


## How about we use logistic logistic regression instead?


Which performs a similar threshold decision


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




![png](index_files/index_27_1.png)


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




![png](index_files/index_31_1.png)


# 3. Understand how the sigmoid function translates the linear equation to a probability

## Predict Proba

Let's take a closer look into what fitting the logistic model results in.

Along with the class predictions, we have probabilities associated with each record:


```python
logreg.predict_proba(X)[:10]
```




    array([[0.99850163, 0.00149837],
           [0.9981539 , 0.0018461 ],
           [0.99682514, 0.00317486],
           [0.99682514, 0.00317486],
           [0.99624938, 0.00375062],
           [0.99538139, 0.00461861],
           [0.99538139, 0.00461861],
           [0.99498061, 0.00501939],
           [0.99328552, 0.00671448],
           [0.99300085, 0.00699915]])



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

    <ipython-input-64-32c3939704e4> in <module>
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
# and a test size of .25
from sklearn.model_selection import train_test_split

# Your code here
```

Since we are still getting used to train test split, let's now just put aside the test set and not touch it again for this lesson.

We will perform a second train test split on the train set, and hyperparameter tune our C on it.


```python
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, 
                                                    test_size=.25)
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
print(logr.score(X_val,y_val))
```

    [[-2.57301586e-03  4.15764506e-02 -1.06355703e-02 -4.29656343e-04
      -2.11472441e-03  1.03939431e-01  4.91470380e-01  3.00925692e-02]]
    [-9.52387315]
    0.7592592592592593



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

    <ipython-input-44-b7eebe9cade7> in <module>
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

    /Users/johnmaxbarry/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/_logistic.py:940: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG)





    array([[0.79912525, 0.20087475],
           [0.75964267, 0.24035733],
           [0.84738361, 0.15261639],
           [0.89184558, 0.10815442],
           [0.50720745, 0.49279255],
           [0.6429583 , 0.3570417 ],
           [0.98599785, 0.01400215],
           [0.42622018, 0.57377982],
           [0.39736641, 0.60263359],
           [0.20412125, 0.79587875],
           [0.79606015, 0.20393985],
           [0.08098182, 0.91901818],
           [0.58196534, 0.41803466],
           [0.80946559, 0.19053441],
           [0.93719196, 0.06280804],
           [0.66581474, 0.33418526],
           [0.86980466, 0.13019534],
           [0.93843801, 0.06156199],
           [0.09653207, 0.90346793],
           [0.34085826, 0.65914174],
           [0.79761096, 0.20238904],
           [0.92105388, 0.07894612],
           [0.61499103, 0.38500897],
           [0.89244726, 0.10755274],
           [0.51478288, 0.48521712],
           [0.08858361, 0.91141639],
           [0.88229252, 0.11770748],
           [0.96077702, 0.03922298],
           [0.69374401, 0.30625599],
           [0.90658872, 0.09341128],
           [0.06642278, 0.93357722],
           [0.13322155, 0.86677845],
           [0.15038522, 0.84961478],
           [0.14359532, 0.85640468],
           [0.43739109, 0.56260891],
           [0.3932071 , 0.6067929 ],
           [0.02892683, 0.97107317],
           [0.7759153 , 0.2240847 ],
           [0.61624789, 0.38375211],
           [0.21386184, 0.78613816],
           [0.94079223, 0.05920777],
           [0.43089343, 0.56910657],
           [0.51661289, 0.48338711],
           [0.78857836, 0.21142164],
           [0.97194586, 0.02805414],
           [0.47519245, 0.52480755],
           [0.57732967, 0.42267033],
           [0.80282881, 0.19717119],
           [0.77267999, 0.22732001],
           [0.04488328, 0.95511672],
           [0.94829263, 0.05170737],
           [0.44196752, 0.55803248],
           [0.29072475, 0.70927525],
           [0.69753977, 0.30246023],
           [0.84658709, 0.15341291],
           [0.9634376 , 0.0365624 ],
           [0.19260947, 0.80739053],
           [0.99528985, 0.00471015],
           [0.61572349, 0.38427651],
           [0.17067537, 0.82932463],
           [0.2755024 , 0.7244976 ],
           [0.59753945, 0.40246055],
           [0.84167563, 0.15832437],
           [0.91385805, 0.08614195],
           [0.90857403, 0.09142597],
           [0.26918867, 0.73081133],
           [0.95163204, 0.04836796],
           [0.20046458, 0.79953542],
           [0.95754114, 0.04245886],
           [0.22153631, 0.77846369],
           [0.38620272, 0.61379728],
           [0.93805047, 0.06194953],
           [0.87977289, 0.12022711],
           [0.85444514, 0.14555486],
           [0.91215731, 0.08784269],
           [0.54460212, 0.45539788],
           [0.8556695 , 0.1443305 ],
           [0.91689644, 0.08310356],
           [0.84238428, 0.15761572],
           [0.71801921, 0.28198079],
           [0.30800781, 0.69199219],
           [0.84252192, 0.15747808],
           [0.94894018, 0.05105982],
           [0.50475759, 0.49524241],
           [0.72013393, 0.27986607],
           [0.10426969, 0.89573031],
           [0.08877251, 0.91122749],
           [0.72342202, 0.27657798],
           [0.88271615, 0.11728385],
           [0.90824413, 0.09175587],
           [0.93379507, 0.06620493],
           [0.75099792, 0.24900208],
           [0.99635463, 0.00364537],
           [0.52361862, 0.47638138],
           [0.47488009, 0.52511991],
           [0.29312492, 0.70687508],
           [0.72100249, 0.27899751],
           [0.89527414, 0.10472586],
           [0.32949804, 0.67050196],
           [0.94918627, 0.05081373],
           [0.20723907, 0.79276093],
           [0.93345021, 0.06654979],
           [0.31340648, 0.68659352],
           [0.40442911, 0.59557089],
           [0.32286864, 0.67713136],
           [0.74305076, 0.25694924],
           [0.72636349, 0.27363651],
           [0.2457509 , 0.7542491 ],
           [0.93126857, 0.06873143],
           [0.64399684, 0.35600316],
           [0.91181031, 0.08818969],
           [0.49746042, 0.50253958],
           [0.98563054, 0.01436946],
           [0.21837879, 0.78162121],
           [0.81034247, 0.18965753],
           [0.62081987, 0.37918013],
           [0.19685189, 0.80314811],
           [0.78232883, 0.21767117],
           [0.93797494, 0.06202506],
           [0.49971524, 0.50028476],
           [0.92100678, 0.07899322],
           [0.74363521, 0.25636479],
           [0.7870174 , 0.2129826 ],
           [0.95212794, 0.04787206],
           [0.59157948, 0.40842052],
           [0.7152243 , 0.2847757 ],
           [0.97380908, 0.02619092],
           [0.25409181, 0.74590819],
           [0.05106005, 0.94893995],
           [0.19630515, 0.80369485],
           [0.27259148, 0.72740852],
           [0.22232317, 0.77767683],
           [0.9244404 , 0.0755596 ],
           [0.63554516, 0.36445484],
           [0.16461266, 0.83538734],
           [0.86623774, 0.13376226],
           [0.83337585, 0.16662415],
           [0.14306475, 0.85693525],
           [0.24482972, 0.75517028],
           [0.98796424, 0.01203576],
           [0.88840766, 0.11159234],
           [0.95477884, 0.04522116],
           [0.70184916, 0.29815084],
           [0.56146609, 0.43853391],
           [0.8484922 , 0.1515078 ],
           [0.80769616, 0.19230384],
           [0.92890826, 0.07109174],
           [0.97855356, 0.02144644],
           [0.56958619, 0.43041381],
           [0.22681423, 0.77318577],
           [0.93322907, 0.06677093],
           [0.64465586, 0.35534414],
           [0.71137745, 0.28862255],
           [0.89982494, 0.10017506],
           [0.99464265, 0.00535735],
           [0.74284386, 0.25715614],
           [0.64216291, 0.35783709],
           [0.27682878, 0.72317122],
           [0.28985978, 0.71014022],
           [0.82282632, 0.17717368],
           [0.4435381 , 0.5564619 ],
           [0.38230322, 0.61769678],
           [0.80414227, 0.19585773],
           [0.98181586, 0.01818414],
           [0.87798482, 0.12201518],
           [0.0920669 , 0.9079331 ],
           [0.9471322 , 0.0528678 ],
           [0.63277546, 0.36722454],
           [0.13825265, 0.86174735],
           [0.58933667, 0.41066333],
           [0.34643623, 0.65356377],
           [0.85593702, 0.14406298],
           [0.62804248, 0.37195752],
           [0.2880839 , 0.7119161 ],
           [0.43039835, 0.56960165],
           [0.91341189, 0.08658811],
           [0.65025185, 0.34974815],
           [0.84348202, 0.15651798],
           [0.84757357, 0.15242643],
           [0.64893232, 0.35106768],
           [0.57611574, 0.42388426],
           [0.33257774, 0.66742226],
           [0.66392549, 0.33607451],
           [0.15812911, 0.84187089],
           [0.43638191, 0.56361809],
           [0.89995487, 0.10004513],
           [0.9579845 , 0.0420155 ],
           [0.88778985, 0.11221015],
           [0.1120891 , 0.8879109 ],
           [0.70343252, 0.29656748],
           [0.94143569, 0.05856431],
           [0.9011293 , 0.0988707 ]])




```python
y_hat = regr.predict(X_val)
y_hat
```




    array([0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0,
           0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
           1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1,
           0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0])



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

While the accuracy of the model will fall by increasing the threshold, we are protecting against a certain type of error. What type of error are we reducing? Think back to Type 1 and Type 2 errors. Why might protecting against such errors be smart in a model that deals with a life-threatening medical condition?



```python
# Your answer here
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
