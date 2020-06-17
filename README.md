
# Logistic Regression

## Learning goals

You will be able to:
1. Describe the need for logistic regression
2. Explore how the sigmoid function "links" the linear equation to a probabilistic model
3. Explain the connection of logodds to the linear model
4. Differentiate between the new type of loss function and OLS
5. Explore the C (inverse regularization) paramater and hyperparameter tune

## 1. Describe the need for logistic regression

## Linear to Logistic regression
![img](img/linear_vs_logistic_regression.jpg)

For linear regression, we use a set of features to predict a **continuous** target variable.  We have a set of assumptions, primary amongst them a fundamentally linear relationship between independent and dependent variables.  Linear Regression optimizes parameters using a cost function (OLS or gradient descent) which calculates the difference between predicted values and true values of a dataset.

But what if, instead of predicting a continous outcome, we want to predict a binary outcome?  How do we translate the linear equation to output a binary prediction?  One idea would be to simply set a threshold where any prediction below a certain value is categorized as 0 and above is categorized as 1.

What problems can we forsee in this approach?

![talk amongst yourselves](https://media.giphy.com/media/3o6Zt44rlujPePNVVC/giphy.gif)





```python
# Results of discussion
```

# 2. Explore how the sigmoid function "links" the linear equation to a probabilistic model

The goal of logistic regression is to model a conditional probability.  For the binary case, it is the probability of a 0 or 1 based on a set of independent features X.

$\Large P(G = 0|X = x)$  

$\Large P(G = 1|X = x)$

In order to realize such a goal, we have to somehow translate our linear output into a probability.  As we know, probability takes on a value between 0 and 1,  whereas the linear equation can output any value from $-\infty$ to $\infty$.

In comes the sigmoid function to the rescue.

$$ \displaystyle \frac{1}{1+e^{-z}}$$


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

    <ipython-input-5-32c3939704e4> in <module>
          1 # Now, input numbers into the function to see that it is functioning correcting.
          2 
    ----> 3 sigmoid()
    

    TypeError: sigmoid() missing 1 required positional argument: 'any_number'


The sigmoid gives us an s-shaped output, shown below:

![sigmoid](img/SigmoidFunction_701.gif)

Notice how the curve crosses the y-axis at .5, and how the values are between 0 and 1.

## 3. Explain the connection of logodds to the linear model

Now for the fun part.  The input of the sigmoid is our trusty old linear equation.

$$ \hat y = \hat\beta_0 + \hat\beta_1 x_1 + \hat\beta_2, x_2 +\ldots + \hat\beta_n x_n $$

The linear equation is passed into the sigmoid function to produce a number between 0 and 1
$$\displaystyle \frac{1}{1+e^{-(\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n)}}$$

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
    

## Generalized Linear Model
The strategy is to *generalize* the notion of linear regression; regression will become a special case. In particular, we'll keep the idea of the regression best-fit line, but now **we'll allow the model to be constructed from the dependent variable through some (non-trivial) function of the linear predictor**. 
This function is standardly called the **link function**. 

The equation from above: 
$\large\ln\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1x_1 + ... + \beta_nx_n$
<br>

The characteristic link function is this logit function.

# Decision Boundary

Logistic regression is a parametric, discriminative model.  
In other words, its decisions are made via trained parameters: our beta coefficients. And the hyperplane that these coefficients creates defines a boundary by which we can discriminate between the classes.    

![](img/decision_boundary_1.jpg)
![](img/decision_boundary_2.jpg)

## Pair Program: Created Income Data

Let's manufacture some data created from a linear relationship between age and income, with some normally distributed noise.


```python
# create data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

age = np.random.uniform(18, 65, 100)
income = np.random.normal(age/10, 0.5)
age = age.reshape(-1,1)
```


```python
fig = plt.figure(figsize=(8,6))
fig.suptitle('age vs income', fontsize=16)
plt.scatter(age, income)
plt.xlabel("age", fontsize=14)
plt.ylabel("monthly income", fontsize=14)
plt.show()
```


![png](index_files/index_25_0.png)



```python
# Then convert this data to a binary

income_bin = income > 4
income_bin
```




    array([False,  True, False,  True,  True, False, False,  True,  True,
            True, False,  True,  True,  True, False,  True,  True, False,
            True,  True, False,  True, False, False,  True,  True, False,
            True, False,  True,  True, False,  True, False,  True,  True,
           False,  True, False,  True, False, False, False,  True,  True,
            True, False,  True, False, False, False,  True, False, False,
            True,  True,  True,  True,  True,  True, False,  True, False,
           False, False, False, False,  True, False, False,  True,  True,
           False, False, False,  True, False, False, False, False,  True,
            True,  True,  True,  True, False,  True,  True,  True, False,
           False,  True, False,  True, False, False,  True,  True, False,
            True])




```python
fig = plt.figure(figsize=(8,6))
fig.suptitle('age vs binary income', fontsize=16)
plt.scatter(age, income_bin)
plt.xlabel("age", fontsize=14)
plt.ylabel("monthly income (> or < 4000)", fontsize=14)
plt.show()
```


![png](index_files/index_27_0.png)


We could fit a linear regression model to this, but the line doesn't capture the shape of the data well.


```python
from sklearn.linear_model import LinearRegression

# create linear regression object
lin_reg = LinearRegression()
lin_reg.fit(age, income_bin)
# store the coefficients
coef = lin_reg.coef_
interc = lin_reg.intercept_
# create the line
lin_income = (interc + age * coef)
```


```python
fig = plt.figure(figsize=(8,6))
fig.suptitle('linear regression', fontsize=16)
plt.scatter(age, income_bin)
plt.xlabel("age", fontsize=14)
plt.ylabel("monthly income", fontsize=14)
plt.plot(age, lin_income, c = "black")
plt.show()
```


![png](index_files/index_30_0.png)


Logistic Regression is fit much the same way as linear regression:


```python
# Create logistic regression object
regr = LogisticRegression()
# Train the model using the training sets
regr.fit(age, income_bin)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



The trained object comes with a coef_ parameter which stores the $\beta$s and an intercept_ paramater. These parameters can be combined to create the linear equation.

### Now, in pairs: 
  1. create an array which is the result of computing the linear equation. That represents the log odds.
  2. Pass those results into the sigmoid function defined above.
  
Along with the predict method, the regr object comes with the predict_proba() method, which outputs probabities associated with each class.

  3. As a check, make sure that the output of the sigmoid function matches the probabilities output by  `regr.predict_proba(age)`

#### Interpretting coefficients

This result, in combination with mathematical properties of exponential functions, leads to the fact that, applied to our example:

if *age* goes up by 1, the odds are multiplied by $e^{\beta_1}$


```python
print(f"""As age goes up by 1, the odds of having a salary about 40k goes up by 
              {np.e**(regr.coef_)[0][0]}""")
```

    As age goes up by 1, the odds of having a salary about 40k goes up by 
                  2.0164407557970443


In our example, there is a positive relationship between age and income, resulting in a positive $\beta_1 > 0$, so $e^{\beta_1}>1$, and the odds will increase as *age* increases.

# 4. Differentiate between the new type of loss function and OLS

Ordinary least squares does not make sense with regards to odds and binary outcomes.  The odds of the true value, 1, equals 1/(1-1). Instead of OLS, we frame the discussion as likelihood.  What is the likelihood that we see the labels given the features and the hypothesis. 

To maximize likelihood, we need to choose a probability distribution.  In this case, since the labels are binary, we use the Bernouli distribution. The likelihood equation for the Bernouli distribution is:

$ Likelihood=\prod\limits_{i=0}^N p_i^{y_i}(1-p_i)^{1-y_i}$

Taking the log of both sides leads to the log_likelihood equation:

$loglikelihood = \sum\limits_{i=1}^N y_i\log{p_i} + (1-y_i)\log(1-p_i) $

The goal of MLE is to maximize log-likelihood



![Maximum Likelihood](img/MLE.png)


There is no closed form solution like the normal equation in linear regression, so we have to use stocastic gradient descent.  To do so we take the derivative of the loglikelihood and set it to zero to find the gradient of the loglikelihood, then update our coefficients. Just like linear regression, we use a learning rate when updating them.

Math behind the gradient of log-likelihood is ESL section 4.4.1: https://web.stanford.edu/~hastie/ElemStatLearn//.

# Hyperparameter Tuning the C Variable

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
diabetes.corr()
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
      <th>Pregnancies</th>
      <td>1.000000</td>
      <td>0.129459</td>
      <td>0.141282</td>
      <td>-0.081672</td>
      <td>-0.073535</td>
      <td>0.017683</td>
      <td>-0.033523</td>
      <td>0.544341</td>
      <td>0.221898</td>
    </tr>
    <tr>
      <th>Glucose</th>
      <td>0.129459</td>
      <td>1.000000</td>
      <td>0.152590</td>
      <td>0.057328</td>
      <td>0.331357</td>
      <td>0.221071</td>
      <td>0.137337</td>
      <td>0.263514</td>
      <td>0.466581</td>
    </tr>
    <tr>
      <th>BloodPressure</th>
      <td>0.141282</td>
      <td>0.152590</td>
      <td>1.000000</td>
      <td>0.207371</td>
      <td>0.088933</td>
      <td>0.281805</td>
      <td>0.041265</td>
      <td>0.239528</td>
      <td>0.065068</td>
    </tr>
    <tr>
      <th>SkinThickness</th>
      <td>-0.081672</td>
      <td>0.057328</td>
      <td>0.207371</td>
      <td>1.000000</td>
      <td>0.436783</td>
      <td>0.392573</td>
      <td>0.183928</td>
      <td>-0.113970</td>
      <td>0.074752</td>
    </tr>
    <tr>
      <th>Insulin</th>
      <td>-0.073535</td>
      <td>0.331357</td>
      <td>0.088933</td>
      <td>0.436783</td>
      <td>1.000000</td>
      <td>0.197859</td>
      <td>0.185071</td>
      <td>-0.042163</td>
      <td>0.130548</td>
    </tr>
    <tr>
      <th>BMI</th>
      <td>0.017683</td>
      <td>0.221071</td>
      <td>0.281805</td>
      <td>0.392573</td>
      <td>0.197859</td>
      <td>1.000000</td>
      <td>0.140647</td>
      <td>0.036242</td>
      <td>0.292695</td>
    </tr>
    <tr>
      <th>DiabetesPedigreeFunction</th>
      <td>-0.033523</td>
      <td>0.137337</td>
      <td>0.041265</td>
      <td>0.183928</td>
      <td>0.185071</td>
      <td>0.140647</td>
      <td>1.000000</td>
      <td>0.033561</td>
      <td>0.173844</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>0.544341</td>
      <td>0.263514</td>
      <td>0.239528</td>
      <td>-0.113970</td>
      <td>-0.042163</td>
      <td>0.036242</td>
      <td>0.033561</td>
      <td>1.000000</td>
      <td>0.238356</td>
    </tr>
    <tr>
      <th>Outcome</th>
      <td>0.221898</td>
      <td>0.466581</td>
      <td>0.065068</td>
      <td>0.074752</td>
      <td>0.130548</td>
      <td>0.292695</td>
      <td>0.173844</td>
      <td>0.238356</td>
      <td>1.000000</td>
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



```python
# Same result as 'none'
logr = LogisticRegression(penalty='none', C=10**8)
y = diabetes.Outcome
X = diabetes.drop("Outcome", axis=1)
logr.fit(X, y)

for coef, feature in zip(list(logr.coef_[0]), X.columns):
    print(round(coef,4), feature)
print(logr.intercept_)
print(logr.score(X,y))
```

    0.1186 Pregnancies
    0.035 Glucose
    -0.0136 BloodPressure
    0.0007 SkinThickness
    -0.0012 Insulin
    0.0906 BMI
    0.9504 DiabetesPedigreeFunction
    0.016 Age
    [-8.42319193]
    0.7825520833333334



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

logr = LogisticRegression(penalty='l2', C=10**8)
y = diabetes.Outcome
X = diabetes.drop("Outcome", axis=1)
logr.fit(X_train, y_train)

print(logr.coef_)
print(logr.intercept_)
print(logr.score(X_test,y_test))
```

    [[ 0.05846675  0.03573097 -0.010864   -0.00178508 -0.00099074  0.10821316
       0.52958357  0.03573258]]
    [-9.43814156]
    0.7402597402597403



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

    <ipython-input-326-b7eebe9cade7> in <module>
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


## Threshold

Because logistic regression calculates the probability of a given class, we can easily change the threshold of what is categorized as a 1 or a 0.   

Let's use our best c from above, and use predict_proba() to output probabilities.


```python
regr = LogisticRegression(penalty='l2', C=c)
regr.fit(X_train, y_train)
probas = regr.predict_proba(X_test)
probas
```


```python
y_hat = regr.predict(X_test)
y_hat
```

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


```python

```
