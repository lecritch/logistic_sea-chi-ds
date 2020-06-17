
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




# 2. Explore how the sigmoid function "links" the linear equation to a probabilistic model

The goal of logistic regression is to model a conditional probability.  For the binary case, it is the probability of a 0 or 1 based on a set of independent features X.

$\Large P(G = 0|X = x)$  

$\Large P(G = 1|X = x)$

In order to realize such a goal, we have to somehow translate our linear output into a probability.  As we know, probability takes on a value between 0 and 1,  whereas the linear equation can output any value from $-\infty$ to $\infty$.

In comes the sigmoid function to the rescue.

$$ \displaystyle \frac{1}{1+e^{-z}}$$

In your head, work through the approximate output of the function for:
  - z = 0
  - z = 1000
  - z = -1000
 

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

We could fit a linear regression model to this, but the line doesn't capture the shape of the data well.

Logistic Regression is fit much the same way as linear regression:

The trained object comes with a coef_ parameter which stores the $\beta$s and an intercept_ paramater. These parameters can be combined to create the linear equation.

### Now, in pairs: 
  1. create an array which is the result of computing the linear equation. That represents the log odds.
  2. Pass those results into the sigmoid function defined above.
  
Along with the predict method, the regr object comes with the predict_proba() method, which outputs probabities associated with each class.

  3. As a check, make sure that the output of the sigmoid function matches the probabilities output by  `regr.predict_proba(age)`


```python

# store the coefficients
coef = regr.coef_
interc = regr.intercept_
# create the linear predictor
log_odds= (age * coef + interc)

probs = [sigmoid(odd) for odd in log_odds]
probs
```




    [array([2.23611783e-05]),
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
     array([6.38126685e-08]),
     array([0.99978685]),
     array([0.99999429]),
     array([0.00673598]),
     array([0.96317144]),
     array([4.86323987e-07]),
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
     array([4.63339003e-06]),
     array([0.99796085]),
     array([0.99798223]),
     array([5.49417759e-05]),
     array([0.99999858]),
     array([0.07965762]),
     array([0.99999763]),
     array([2.91074578e-07]),
     array([1.76182058e-05]),
     array([1.93072168e-07]),
     array([0.99464703]),
     array([0.92951589]),
     array([0.63603166]),
     array([1.69048274e-07]),
     array([0.81535501]),
     array([0.00211962]),
     array([0.39126059]),
     array([1.62042433e-06]),
     array([0.95228335]),
     array([0.83670346]),
     array([5.06565353e-08]),
     array([0.96548991]),
     array([0.99999784]),
     array([0.99988105]),
     array([0.99999985]),
     array([0.99999954]),
     array([0.99988656]),
     array([0.00049103]),
     array([0.97282165]),
     array([0.22064719]),
     array([2.56439344e-05]),
     array([0.01190353]),
     array([2.39350318e-07]),
     array([0.10587276]),
     array([0.99999978]),
     array([2.41050059e-06]),
     array([2.0739717e-06]),
     array([0.99933996]),
     array([0.9119718]),
     array([0.18620209]),
     array([1.38477631e-06]),
     array([7.74734808e-05]),
     array([0.99999677]),
     array([0.03612978]),
     array([0.65519552]),
     array([4.9737341e-08]),
     array([0.00081525]),
     array([0.0678634]),
     array([0.9591806]),
     array([0.99999823]),
     array([0.97352721]),
     array([0.99807408]),
     array([5.65912259e-06]),
     array([0.99948514]),
     array([0.99996868]),
     array([0.97953076]),
     array([0.07087774]),
     array([6.19383897e-06]),
     array([0.84750359]),
     array([0.59641196]),
     array([0.99999941]),
     array([0.23375548]),
     array([0.38806785]),
     array([0.66279917]),
     array([0.99995378]),
     array([2.66344804e-07]),
     array([0.99359841])]



#### Interpretting coefficients

This result, in combination with mathematical properties of exponential functions, leads to the fact that, applied to our example:

if *age* goes up by 1, the odds are multiplied by $e^{\beta_1}$

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

How do we choose between them? We iterate over possible parameters and judge the success based on our metric of choice.  We will eventually move towards grid search, which will help us be more thorough with our tuning.  For now, we will work through how to tune our C parameter with an Ridge regularization.

For now, let's judge on accuracy, which can be accessed via the `score()` method of a trained model.



```python
# The parameters for C can be anything above 0.  
# Set up a list of possible values to try out.
# Start with numbers above 0 up to 1000
c_candidates = np.linspace(1,1000,1000)
c_candidates = np.insert(c_candidates, 0, .0001)
c_candidates
```




    array([1.00e-04, 1.00e+00, 2.00e+00, ..., 9.98e+02, 9.99e+02, 1.00e+03])




```python
# Split your data into training and test data with a random state of 42 
# and a test size of .3
X = diabetes.drop('Outcome', axis=1)
y = diabetes.Outcome
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, 
                                                    test_size=.3)
```


```python
# Create a for loop which runs through all of the possible values of C,
# fits the model on the train set, and scores the model on test set.
# Add the accuracies into a dictionary or a list, whichever you prefer
# Use 'l2'

c_scores = {}
for c in c_candidates:
    regr = LogisticRegression(penalty='l2', C=c)
    regr.fit(X_train, y_train)
    accuracy = regr.score(X_test, y_test)
    c_scores[c] = accuracy
    
# the best accuracy score comes with the highest regularization
best_c = max(c_scores, key=c_scores.get)
best_c
    
```


```python
c_scores = {}
for c in c_candidates:
    regr = LogisticRegression(penalty='l2', C=c)
    regr.fit(X_train, y_train)
    accuracy = regr.score(X_test, y_test)
    c_scores[c] = accuracy
    
# the best accuracy score comes with the highest regularization
best_c = max(c_scores, key=c_scores.get)
print(best_c, c_scores[best_c])
```

We improved our test R^2 from .740 to .745. Not too much gain. 
Sometimes hyperparameter tuning can have a large effect, sometimes not. 
Don't rely on hyperparameter tuning to fix your model.  
Treat it as a necessary step in the process which if you are lucky, may increase the predictive power of your model.

In future lessons, we will use Grid Search to automate our hyperparamater tuning, and make it more thorough.


## Threshold

Because logistic regression calculates the probability of a given class, we can easily change the threshold of what is categorized as a 1 or a 0.   

Let's use our best c from above, and use predict_proba() to output probabilities.

Compare the output of predict and predict proba. Write out below how the output of predict_proba is related to the predict output.  


```python
"""The default threshold is .5. Any value in the predict_proba first column above .5 is categorized as a 0. 
Any value above .5 in the second column is categorized as 1."""
```

Now, isolate one of the columns of predict_proba, and create an area of booleans which returns True if the proba is above .4

Then, use the astype method to convert the array to integers: True will become 1, and False will become 0

While the accuracy of the model will fall by increasing the threshold, we are protecting against a certain type of error. What type of error are we reducing? Of the metrics that we have learned, what score will increase? Why might protecting against such errors be smart in a model that deals with a life-threatening medical condition?



```python
higher_threshold = probas[:,1] > .7
y_hat_lower = higher_threshold.astype(int)

"""By increasing the threshold, we are protecting against false negatives. Our recall score will increase. 
In the case of heart disease, we should err on the side of caution.  We would rather have a false positive 
mistake, since the individual would still be flagged for intervention. A false negative could result in death"""

print(f"Recall score with default .5 threshold: {recall_score(y_hat, y_test)}")
print(f"Recall score with default .4 threshold: {recall_score(y_hat_lower, y_test)}")
```
