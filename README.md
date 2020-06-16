
# Logistic Regression

## Learning goals

You will be able to:
1. Describe the need for logistic regression
2. Explore how the sigmoid function "links" the linear equation to a probabilistic model
3. Explain the connection of logodds to the linear model
4. Differentiate between the new type of loss function and OLS

## 1. Describe the need for logistic regression

## Linear to Logistic regression
![img](img/linear_vs_logistic_regression.jpg)

For linear regression, we use a set of features to predict a **continuous** target variable.  We have a set of assumptions, primary amongst them a fundamentally linear relationship between independent and dependent variables.  Linear Regression optimized parameters using a cost function (OLS or gradient descent) which calculates the difference between predicted values and true values of a dataset.

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

    <ipython-input-65-32c3939704e4> in <module>
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

Remember, the goal of logistic regression is to model the conditional of a class using a transformation of the linear equation.

In other words:

$$\Large P(G = 1|X = x_1, x_2...x_n) = \frac{1}{1+e^{-(\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n)}}$$

Now, with some arithmetic: 

$$ \Large P(G = 1|X = x) = \displaystyle \frac{1}{1+e^{-(\hat \beta_o+\hat \beta_1 x_1 + \hat \beta_2 x_2...\hat\beta_n x_n))}}$$

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


![png](index_files/index_26_0.png)



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


![png](index_files/index_28_0.png)


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


![png](index_files/index_31_0.png)


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

In pairs, 
  1. create an array which is the result of computing the linear equation. That represents the log odds.
  2. Pass those results into the sigmoid function defined above.
  
Along with the predict method, the regr object comes with the predict_proba() method, which outputs probabities associated with each class.

  3. As a check, make sure that the output of the sigmoid function matches the probabilities output by  `regr.predict_proba(age)`


```python
#__SOLUTION__
regr.coef_
regr.intercept_

# store the coefficients
coef = regr.coef_
interc = regr.intercept_
# create the linear predictor
log_odds= (age * coef + interc)

probs = [sigmoid(odd) for odd in log_odds]
probs
```




    [array([0.00002236]),
     array([0.97026438]),
     array([0.06962443]),
     array([0.99985897]),
     array([0.99983159]),
     array([0.00032357]),
     array([0.0003676]),
     array([0.99991816]),
     array([0.99999953]),
     array([0.99999288]),
     array([0.00534339]),
     array([0.37589488]),
     array([0.99596064]),
     array([0.99845536]),
     array([0.0080285]),
     array([0.81417651]),
     array([0.39217428]),
     array([0.00000006]),
     array([0.99978685]),
     array([0.99999429]),
     array([0.00673598]),
     array([0.96317144]),
     array([0.00000049]),
     array([0.00766249]),
     array([0.99999892]),
     array([0.98845648]),
     array([0.01929754]),
     array([0.9998738]),
     array([0.00138955]),
     array([0.84617412]),
     array([0.99999108]),
     array([0.06637792]),
     array([0.9999189]),
     array([0.00000463]),
     array([0.99796085]),
     array([0.99798223]),
     array([0.00005494]),
     array([0.99999858]),
     array([0.07965762]),
     array([0.99999763]),
     array([0.00000029]),
     array([0.00001762]),
     array([0.00000019]),
     array([0.99464703]),
     array([0.92951589]),
     array([0.63603166]),
     array([0.00000017]),
     array([0.81535501]),
     array([0.00211962]),
     array([0.39126059]),
     array([0.00000162]),
     array([0.95228335]),
     array([0.83670346]),
     array([0.00000005]),
     array([0.96548991]),
     array([0.99999784]),
     array([0.99988105]),
     array([0.99999985]),
     array([0.99999954]),
     array([0.99988656]),
     array([0.00049103]),
     array([0.97282165]),
     array([0.22064719]),
     array([0.00002564]),
     array([0.01190353]),
     array([0.00000024]),
     array([0.10587276]),
     array([0.99999978]),
     array([0.00000241]),
     array([0.00000207]),
     array([0.99933996]),
     array([0.9119718]),
     array([0.18620209]),
     array([0.00000138]),
     array([0.00007747]),
     array([0.99999677]),
     array([0.03612978]),
     array([0.65519552]),
     array([0.00000005]),
     array([0.00081525]),
     array([0.0678634]),
     array([0.9591806]),
     array([0.99999823]),
     array([0.97352721]),
     array([0.99807408]),
     array([0.00000566]),
     array([0.99948514]),
     array([0.99996868]),
     array([0.97953076]),
     array([0.07087774]),
     array([0.00000619]),
     array([0.84750359]),
     array([0.59641196]),
     array([0.99999941]),
     array([0.23375548]),
     array([0.38806785]),
     array([0.66279917]),
     array([0.99995378]),
     array([0.00000027]),
     array([0.99359841])]



#### Interpretting coefficients

This result, in combination with mathematical properties of exponential functions, leads to the fact that, applied to our example:

if *age* goes up by 1, the odds are multiplied by $e^{\beta_1}$

In our example, there is a positive relationship between age and income, this will lead a positive $\beta_1 > 0$, so $e^{\beta_1}>1$, and the odds will increase as *age* increases.

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


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```

## What do we know about linear regression?

- What are the requirements for the variables types?
- What assumptions do we have?
- How do we interpret the coefficients?
- What metrics do we use to evaluate our model?

And how will logistic regression be different?

![log](https://media.giphy.com/media/m8DnDYfRwEtvG/giphy.gif)

Some continous target variables students have used linear regression in the past are: 

1. carbon offset 
2. NFL draft position 
3. home prices 
4. used car prices
5. Spotify streams

In exploring possible subjects, you almost surely came across data meant to predict classifications. The target was a binary variable: 

1. A patient has heart disease or not. 
2. A baby will be a boy or a girl.
3. A visa application will be approved or not.
4. A released prisoner will be imprisoned again or not.

## Scenarios 

*We will return to the scenarios below with real data at the bottom of the notebook*

#### Scenario 1: Predict income bracket
In this example, we want to find a relationship between age and monthly income. It is definitely reasonable to assume that, on average, older people have a higher income than younger people who are newer to the job market and have less experience.

#### Scenario 2: Predict likelihood of diabetes
This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. [reference here](https://www.kaggle.com/uciml/pima-indians-diabetes-database)

## Created income data


```python
# create data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)

age = np.random.uniform(18, 65, 100)
income = np.random.normal((age/10), 0.5)
age = age.reshape(-1,1)
income.shape
```




    (100,)



Plot it!


```python
fig = plt.figure(figsize=(8,6))
fig.suptitle('age vs income', fontsize=16)
plt.scatter(age, income)
plt.xlabel("age", fontsize=14)
plt.ylabel("monthly income", fontsize=14)
plt.show()
```


![png](index_files/index_55_0.png)


In linear regression, you would try to find a relationship between age and monthly income. Conceptually, this means fitting a line that represents the relationship between age and monthly income, as shown below.


```python
fig = plt.figure(figsize=(8,6))
fig.suptitle('linear regression', fontsize=16)
plt.scatter(age, income)
plt.plot(age, age/10, c = "black")
plt.xlabel("age", fontsize=14)
plt.ylabel("monthly income", fontsize=14)
plt.show()
```


![png](index_files/index_57_0.png)


The idea is that you could use this line to make predictions in the future. In this case, the relationship is modeled as follows: the expected monthly income for someone who is, say, 40 years old, is 3000 (3 on the y-axis). Of course, the actual income will most likely be different, but this gives an indication of what the model predicts as the salary value.

## So how is this related to logistic regression?

Now, imagine you get a data set where no information on exact income is given (after all, people don't like to talk about how much they earn!), but you only have information on whether or not they earn more than 4000 USD per month. Starting from the generated data we used before, the new variable `income_bin` was transformed to 1 when someone's income is over 4000 USD, and 0 when the income is less than 4000 USD.


```python
# Your turn: Add code that transforms the income to a binary

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



Let's have a look at what happens when we plot this.


```python
fig = plt.figure(figsize=(8,6))
fig.suptitle('age vs binary income', fontsize=16)
plt.scatter(age, income_bin)
plt.xlabel("age", fontsize=14)
plt.ylabel("monthly income (> or < 4000)", fontsize=14)
plt.show()
```


![png](index_files/index_63_0.png)


You can already tell that fitting a straight line will not be exactly desired here, but let's still have a look at what happens when you fit a regression line to these data. 


```python
from sklearn.linear_model import LogisticRegression
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


![png](index_files/index_66_0.png)


You can see that this doesn't make a lot of sense. This straight line cannot grasp the true structure of what is going on when using a linear regression model. Now, without going into the mathematical details for now, let's look at a logistic regression model and fit that to the dataset.


```python
# Create logistic regression object
regr = LogisticRegression(C=1e5)
# Train the model using the training sets
regr.fit(age, income_bin)
```




    LogisticRegression(C=100000.0, class_weight=None, dual=False,
                       fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                       max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
# store the coefficients
coef = regr.coef_
interc = regr.intercept_
# create the linear predictor
lin_pred= (age * coef + interc)
# perform the log transformation
mod_income = 1 / (1 + np.exp(-lin_pred))
#sort the numbers to make sure plot looks right
age_ordered, mod_income_ordered = zip(*sorted(zip(age ,mod_income.ravel()),key=lambda x: x[0]))
```


```python
mod_income
```




    array([[0.0000142 ],
           [0.97407755],
           [0.06261011],
           [0.99990227],
           [0.99988242],
           [0.00022988],
           [0.00026256],
           [0.99994457],
           [0.99999974],
           [0.99999564],
           [0.00427498],
           [0.36975725],
           [0.99677498],
           [0.99881615],
           [0.00653735],
           [0.82264087],
           [0.38661943],
           [0.00000003],
           [0.99984971],
           [0.99999654],
           [0.00544327],
           [0.96757421],
           [0.00000026],
           [0.00622668],
           [0.99999939],
           [0.99035512],
           [0.0163345 ],
           [0.99991295],
           [0.00104971],
           [0.85463266],
           [0.9999945 ],
           [0.05954926],
           [0.99994509],
           [0.00000275],
           [0.99841865],
           [0.99843593],
           [0.00003623],
           [0.99999919],
           [0.07211754],
           [0.99999862],
           [0.00000015],
           [0.00001108],
           [0.0000001 ],
           [0.99567427],
           [0.93598283],
           [0.64028457],
           [0.00000009],
           [0.82382483],
           [0.00163013],
           [0.38567237],
           [0.00000092],
           [0.95747163],
           [0.84519858],
           [0.00000002],
           [0.96970743],
           [0.99999874],
           [0.99991815],
           [0.99999992],
           [0.99999975],
           [0.9999221 ],
           [0.00035502],
           [0.97640448],
           [0.21085107],
           [0.00001638],
           [0.00986122],
           [0.00000013],
           [0.09725813],
           [0.99999989],
           [0.00000139],
           [0.00000119],
           [0.99951196],
           [0.91916099],
           [0.17628616],
           [0.00000078],
           [0.00005183],
           [0.99999809],
           [0.03147654],
           [0.66013303],
           [0.00000002],
           [0.00060216],
           [0.06094881],
           [0.96388675],
           [0.99999898],
           [0.97704466],
           [0.99851007],
           [0.00000339],
           [0.99962327],
           [0.99997962],
           [0.9824564 ],
           [0.06379385],
           [0.00000373],
           [0.85595448],
           [0.59915009],
           [0.99999967],
           [0.2240883 ],
           [0.38236363],
           [0.66799778],
           [0.99996943],
           [0.00000014],
           [0.99478684]])




```python
regr.predict_proba(age)
```




    array([[0.9999858 , 0.0000142 ],
           [0.02592245, 0.97407755],
           [0.93738989, 0.06261011],
           [0.00009773, 0.99990227],
           [0.00011758, 0.99988242],
           [0.99977012, 0.00022988],
           [0.99973744, 0.00026256],
           [0.00005543, 0.99994457],
           [0.00000026, 0.99999974],
           [0.00000436, 0.99999564],
           [0.99572502, 0.00427498],
           [0.63024275, 0.36975725],
           [0.00322502, 0.99677498],
           [0.00118385, 0.99881615],
           [0.99346265, 0.00653735],
           [0.17735913, 0.82264087],
           [0.61338057, 0.38661943],
           [0.99999997, 0.00000003],
           [0.00015029, 0.99984971],
           [0.00000346, 0.99999654],
           [0.99455673, 0.00544327],
           [0.03242579, 0.96757421],
           [0.99999974, 0.00000026],
           [0.99377332, 0.00622668],
           [0.00000061, 0.99999939],
           [0.00964488, 0.99035512],
           [0.9836655 , 0.0163345 ],
           [0.00008705, 0.99991295],
           [0.99895029, 0.00104971],
           [0.14536734, 0.85463266],
           [0.0000055 , 0.9999945 ],
           [0.94045074, 0.05954926],
           [0.00005491, 0.99994509],
           [0.99999725, 0.00000275],
           [0.00158135, 0.99841865],
           [0.00156407, 0.99843593],
           [0.99996377, 0.00003623],
           [0.00000081, 0.99999919],
           [0.92788246, 0.07211754],
           [0.00000138, 0.99999862],
           [0.99999985, 0.00000015],
           [0.99998892, 0.00001108],
           [0.9999999 , 0.0000001 ],
           [0.00432573, 0.99567427],
           [0.06401717, 0.93598283],
           [0.35971543, 0.64028457],
           [0.99999991, 0.00000009],
           [0.17617517, 0.82382483],
           [0.99836987, 0.00163013],
           [0.61432763, 0.38567237],
           [0.99999908, 0.00000092],
           [0.04252837, 0.95747163],
           [0.15480142, 0.84519858],
           [0.99999998, 0.00000002],
           [0.03029257, 0.96970743],
           [0.00000126, 0.99999874],
           [0.00008185, 0.99991815],
           [0.00000008, 0.99999992],
           [0.00000025, 0.99999975],
           [0.0000779 , 0.9999221 ],
           [0.99964498, 0.00035502],
           [0.02359552, 0.97640448],
           [0.78914893, 0.21085107],
           [0.99998362, 0.00001638],
           [0.99013878, 0.00986122],
           [0.99999987, 0.00000013],
           [0.90274187, 0.09725813],
           [0.00000011, 0.99999989],
           [0.99999861, 0.00000139],
           [0.99999881, 0.00000119],
           [0.00048804, 0.99951196],
           [0.08083901, 0.91916099],
           [0.82371384, 0.17628616],
           [0.99999922, 0.00000078],
           [0.99994817, 0.00005183],
           [0.00000191, 0.99999809],
           [0.96852346, 0.03147654],
           [0.33986697, 0.66013303],
           [0.99999998, 0.00000002],
           [0.99939784, 0.00060216],
           [0.93905119, 0.06094881],
           [0.03611325, 0.96388675],
           [0.00000102, 0.99999898],
           [0.02295534, 0.97704466],
           [0.00148993, 0.99851007],
           [0.99999661, 0.00000339],
           [0.00037673, 0.99962327],
           [0.00002038, 0.99997962],
           [0.0175436 , 0.9824564 ],
           [0.93620615, 0.06379385],
           [0.99999627, 0.00000373],
           [0.14404552, 0.85595448],
           [0.40084991, 0.59915009],
           [0.00000033, 0.99999967],
           [0.7759117 , 0.2240883 ],
           [0.61763637, 0.38236363],
           [0.33200222, 0.66799778],
           [0.00003057, 0.99996943],
           [0.99999986, 0.00000014],
           [0.00521316, 0.99478684]])



### Look at dataset predictions

It is the **probability** of being in the target class


```python
np.set_printoptions(suppress=True)
print(mod_income[:6])
```

    [[0.0000142 ]
     [0.97407755]
     [0.06261011]
     [0.99990227]
     [0.99988242]
     [0.00022988]]


### Plot it!


```python
fig = plt.figure(figsize=(8,6))
fig.suptitle('logistic regression', fontsize=16)
plt.scatter(age, income_bin)
plt.xlabel("age", fontsize=14)
plt.ylabel("monthly income", fontsize=14)
plt.plot(age_ordered, mod_income_ordered, c = "black")
plt.show()
```


![png](index_files/index_75_0.png)


#### Review the new shape

This already looks a lot better! You can see that this function has an S-shape which plateaus to 0 in the left tale and 1 to the right tale. This is exactly what we needed here. Hopefully this example was a good way of showing why logistic regression is useful. Now, it's time to dive into the mathematics that make logistic regression possible.

That **S-shape** is what's known as a **sigmoid function**

![sigmoid](img/SigmoidFunction_701.gif)

## Logistic regression model formulation

### The model

As you might remember from the linear regression lesson, a linear regression model can be written as:

$$ \hat y = \hat\beta_0 + \hat\beta_1 x_1 + \hat\beta_2, x_2 +\ldots + \beta_n x_n $$

When there are $n$ predictors $x_1,\ldots,x_n$ and $n+1$ parameter estimates that are estimated by the model $\hat\beta_0, \hat\beta_1,\ldots, \hat\beta_n$. $ \hat y $ is an estimator for the outcome variable.

Translating this model formulation to our example, this boils down to:

$$ \text{income} = \beta_0 + \beta_1 \text{age} $$

When you want to apply this to a binary dataset, what you actually want to do is perform a **classification** of your data in one group versus another one. In our case, we want to classify our observations (the 100 people in our data set) as good as possible in "earns more than 4k" and "earns less than 4k". A model will have to make a guess of what the **probability** is of belonging to one group versus another. And that is exactly what logistic regression models can do! 

### Transformation

Essentially, what happens is, the linear regression is *transformed* in a way that the outcome takes a value between 0 and 1. This can then be interpreted as a probability (e.g., 0.2 is a probability of 20%). Applied to our example, the expression for a logistic regression model would look like this:

$$ P(\text{income} > 4000) = \displaystyle \frac{1}{1+e^{-(\hat \beta_0+\hat \beta_1 \text{age})}}$$

Note that the outcome is written as $P(\text{income} > 4000)$. This means that the output should be interpreted as *the probability that the monthly income is over 4000 USD*.

It is important to note that this is the case because the income variable was relabeled to be equal to 1 when the income is bigger than 4000, and 0 when smaller than 4000. In other words, The outcome variable should be interpreted as *the* **probability** *of the class label to be equal to 1*.

### Interpretation - with a side of more math
#### What are the odds?

As mentioned before, the probability of an income over 4000 can be calculated using:

$$ P(\text{income} > 4000) = \displaystyle \frac{1}{1+e^{-(\hat \beta_o+\hat \beta_1 \text{age})}}$$

You can show that, by multiplying both numerator and denominator by $e^{(\hat \beta_0+\hat \beta_1 \text{age})}$


$$ P(\text{income} > 4000) = \displaystyle \frac{e^{\hat \beta_0+\hat \beta_1 \text{age}}}{1+e^{\hat \beta_o+\hat \beta_1 \text{age}}}$$

As a result, you can compute $P(\text{income} \leq 4000)$ as:

$$ P(\text{income} < 4000) = 1- \displaystyle \frac{e^{\hat \beta_0+\hat \beta_1 \text{age}}}{1+e^{\hat \beta_o+\hat \beta_1 \text{age}}}= \displaystyle \frac{1}{1+e^{\hat \beta_0+\hat \beta_1 \text{age}}}$$



#### Odds ratio

This doesn't seem to be very spectacular, but combining these two results leads to an easy interpretation of the model parameters, triggered by the *odds*

$$ \dfrac{P(\text{income} > 4000)}{P(\text{income} < 4000)} = e^{\hat \beta_0+\hat \beta_1 \text{age}} $$

This expression can be interpreted as the *odds in favor of an income greater than 4000 USD*.

Taking the log of both sides leads to:
<br><br>
    $\ln{\dfrac{P(\text{income} > 4000)}{P(\text{income} < 4000)}} = \beta_0 + \beta_1*X_1 + \beta_2*X_2...\beta_n*X_n$
    
Here me can see why we call it logisitic regression.

Our linear function calculates the log of the probability we predict 1, divided by the probability of predicting 0.  In other words, the linear equation is calculating the **log of the odds** that we predict a class of 1.

## Generalized Linear Model
The strategy is to *generalize* the notion of linear regression; regression will become a special case. In particular, we'll keep the idea of the regression best-fit line, but now **we'll allow the model to be constructed from the dependent variable through some (non-trivial) function of the linear predictor**. 
This function is standardly called the **link function**. 

The equation from above: 
$\large\ln\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1x_1 + ... + \beta_nx_n$
<br>
is the characteristic link function is this logit function.

# Decision Boundary


![](img/decision_boundary_1.jpg)
![](img/decision_boundary_2.jpg)


```python
def sigmoid(log_odds):
    
    prob_class_1 = 1/(1+np.e**(log_odds*-1))
    
    return prob_class_1

sigmoid(0)
```




    0.5



#### Interpretting coefficients

This result, in combination with mathematical properties of exponential functions, leads to the fact that, applied to our example:

if *age* goes up by 1, the odds are multiplied by $e^{\beta_1}$

In our example, there is a positive relationship between age and income, this will lead a positive $\beta_1 > 0$, so $e^{\beta_1}>1$, and the odds will increase as *age* increases.

## Fitting the Model


Ordinary least squares does not make sense with regards to odds and binary outcomes.  The odds of the true value, 1, equals 1/(1-1). Instead of OLS, we frame the discussion as likelihood.  What is the likelihood that we see the labels given the features and the hypothesis. 

To maximize likelihood, we need to choose a probability distribution.  In this case, since the labels are binary, we use the Bernouli distribution. The likelihood equation for the Bernouli distribution is:

$ Likelihood=\prod\limits_{i=0}^N p_i^{y_i}(1-p_i)^{1-y_i}$

Taking the log of both sides leads to the log_likelihood equation:

$loglikelihood = \sum\limits_{i=1}^N y_i\log{p_i} + (1-y_i)\log(1-p_i) $

The goal of MLE is to maximize log-likelihood



![Maximum Likelihood](img/MLE.png)


There is no closed form solution like the normal equation in linear regression, so we have to use stocastic gradient descent.  To do so we take the derivative of the loglikelihood and set it to zero to find the gradient of the loglikelihood, then update our coefficients. Just like linear regression, we use a learning rate when updating them.

Math behind the gradient of log-likelihood is ESL section 4.4.1: https://web.stanford.edu/~hastie/ElemStatLearn//.

### Assumptions

- Binary logistic regression requires the dependent variable to be binary.
- For a binary regression, the factor level 1 of the dependent variable should represent the desired outcome.
- Only the meaningful variables should be included.
- The independent variables should be independent of each other. That is, the model should have little or no multicollinearity.
- The independent variables are linearly related to the log odds.
- Logistic regression requires quite large sample sizes.

# A real data example: Salaries (statsmodels)


```python
import statsmodels as sm
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy import stats
```


```python
salaries = pd.read_csv("salaries_final.csv", index_col = 0)
salaries.Target.value_counts()
```


```python
# !pip install patsy
```


```python
from patsy import dmatrices
y, X = dmatrices('Target ~ Age  + Race + Sex',
                  salaries, return_type = "dataframe")
```

#### Statsmodels method
[statsmodels logit documentation](https://www.statsmodels.org/dev/generated/statsmodels.discrete.discrete_model.Logit.html)


```python
import statsmodels.api as sm
logit_model = sm.Logit(y.iloc[:,1], X)
result = logit_model.fit()
```


```python
# statsmodels has a nice summary function - remember this
result.summary()
```


```python
# translate the coefficients to reflect odds
np.exp(result.params)
```


```python
pred = result.predict(X) > .7
sum(pred.astype(int))
```

Once you **get** a model with `Logit` you can use `LogitResults` to evaluate it's performance more in depth. 
[documentation](http://www.statsmodels.org/devel/generated/statsmodels.discrete.discrete_model.LogitResults.html)

## A Real Data Example: Diabetes (sklearn)




```python
diabetes = pd.read_csv('diabetes.csv')
diabetes.shape
```


```python
diabetes.head()
```


```python
import seaborn as sns
sns.pairplot(diabetes)
```


```python
diabetes.corr()
```


```python
diabetes.dtypes
```


```python
import sklearn.preprocessing as preprocessing

scaler = preprocessing.StandardScaler()
```


```python
X = diabetes.iloc[:,:-1]
```


```python
X.head()
```


```python
Y = diabetes.Outcome
```


```python
Y.head()
```


```python
X_scaled = scaler.fit_transform(X)
```


```python
type(X_scaled)
```


```python
X.columns
```


```python
X.corr()
```


```python
x_Df = pd.DataFrame(X_scaled, columns=X.columns)
```


```python
x_Df.corr()
```

## C parameter

Logistic regression in sklearn allows tuning of the regularization strength, i.e. Lasso/Ridge, via the C parameter.  

Like in regression, except now in MLE, the lasso adds a  term to the equation which penalizes models with too many coefficients, and ridge penalizes models with large coefficients. 

The strength of the penalty is the $\lambda$ term

C is the inverse of $\lambda$, so a small C results in a large penalty.


```python
logreg = LogisticRegression(C = 1**1000, penalty='l2')
model_log = logreg.fit(x_Df, Y)
model_log
```


```python
logreg.score(X_scaled, Y)
```


```python
#we can iterate through values of C to find the optimal parameter.
import warnings
warnings.filterwarnings('ignore')

best = 0
best_score = 0
for c in np.arange(.001, 2, .001):
    lr = LogisticRegression(C=c, penalty='l1')
    lr.fit(X_scaled, Y)
    if lr.score(X_scaled, Y) > best_score:
        best = c
        best_score = lr.score(X_scaled, Y)
print(best)
print(best_score)
```


```python
lr = LogisticRegression(C=.001, penalty='l2')
lr.fit(X_scaled, Y)
```


```python
lr.coef_
```


```python
#Or we can use grid-search.

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.arange(.1, 100, .5)
print(C)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
```


```python
from sklearn.model_selection import GridSearchCV
# Create grid search using 5-fold cross validation
lr = LogisticRegression()
clf = GridSearchCV(lr, hyperparameters, cv=5, verbose=0)

grid = clf.fit(X_scaled, Y)
print(grid.best_estimator_.get_params()['penalty'])
print(grid.best_estimator_.get_params()['C'])
```


```python
lr = LogisticRegression(C=.6, penalty='l2')
lr.fit(X_scaled, Y)
lr.score(X_scaled, Y)
```


```python
model_log.coef_
```


```python
y_pred = lr.predict(X_scaled)
# Returns the probabilitis instead of the rounded predictions
y_proba =lr.predict_proba(X_scaled)
# Returns the accuracy
# lr.score(X_scaled, Y)
y_proba[:,1] > .5
```


```python
from sklearn.metrics import accuracy_score
```


```python
accuracy_score(Y, y_pred)
```


```python

```
