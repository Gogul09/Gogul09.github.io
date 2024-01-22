---
layout: post
category: software
class: Machine Learning
title: Logistic Regression from Scratch
description: Understand how to solve a classification problem using logistic regression from scratch using python and numpy.
permalink: software/ml/logistic-regression-from-scratch
image: https://drive.google.com/uc?id=1rjTumTjtBj7nRdADfGiXs22NcX8xBwe2
--- 

<div class="sidebar_tracker" id="sidebar_tracker">
   <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
   <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
   <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#dataset">Dataset</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#supervised-learning">Supervised Learning</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#linear-predictor">Linear Predictor (score)</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#link-function">Link Function</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#compute-likelihood">Compute Likelihood</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#compute-derivative">Compute Derivative</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#gradient-ascent">Gradient Ascent</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#split-the-dataset">Split the Dataset</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#train-the-classifier">Train the Classifier</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#test-the-classifier">Test the Classifier</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_11" href="#reduce-overfitting">Reduce Overfitting</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_12" href="#l2-regularization">L2 Regularization</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_13" href="#conclusion">Conclusion</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_14" href="#references">References</a></li>
  </ul>
</div>

During my journey as a Machine Learning (ML) practitioner, I found it has become ultimately easy for any human with limited knowledge on algorithms to take advantage of free python libraries such as [scikit-learn](https://scikit-learn.org/){:target="_blank"} to solve a ML problem. Truth be said, it's easy and sometimes no brainer to achieve this, as there are so many codes available in GitHub, Medium, Kaggle etc., You just need some amount of time looking at these codes to arrive at a solution to a problem of your choice. 

But, what if we learn every algorithm or procedures behind each machine learning pipeline that does all the heavy lifting for us inside these amazing libraries. In this blog post and the series of blog posts to come, I will be focusing on implementing machine learning algorithms from scratch using python and numpy.

<div class="note">
<p>Sure you might argue with me for the first paragraph. But learning how each algorithm work behind the scenes is very important to use these algorithms and bring in customized features in any domain (say ASIC design).</p>
</div>

In this blog post, we will implement logistic regression from scratch using python and numpy to a binary classification problem.

I assume that you have knowledge on python programming and scikit-learn, because at the end we will compare our implementation (from scratch) with scikit-learn's implementation.

<h3 id="dataset">Dataset</h3>

We will use the [breast cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html){:target="_blank"} from scikit-learn for this implementation. This is a binary classification problem i.e., each data point in the training data belong to one of two classes. Below code loads the dataset and prints out the important information we seek.

<div class="code-head">logistic_regression.py<span>code</span></div>

```python
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

print(data.keys())
print("No.of.data points (rows) : {}".format(len(data.data)))
print("No.of.features (columns) : {}".format(len(data.feature_names)))
print("No.of.classes            : {}".format(len(data.target_names)))
print("Class names              : {}".format(list(data.target_names)))
```

```
dict_keys(['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename'])

No.of.data points (rows) : 569
No.of.features (columns) : 30
No.of.classes            : 2
Class names              : ['malignant', 'benign']
```
{: .code-out}


If you wish to know more about the dataset, use <span class="coding">data.DESCR</span> to print out the entire description of the dataset.

As you can see, the dataset has <span class="coding">data</span> and <span class="coding">target</span> keys from which we can access the data in this dataset. There are 569 rows and 30 columns. To describe the dataset mathematically, we will use this notation \\( [x_i, h(x_i), y_i] \\).

where 

* \\( x_i \\) denotes a single data point in the dataset.
* \\( h(x_i) \\) is th feature vector for that single data point which has \\( [h_1(x_i), h_2(x_i) ... h_{30}(x_i)] \\).
* \\( y_i \\) is the target class for that single data point.

We can easily convert this dataset into a pandas dataframe for better data analysis. To view the datatype of each column, we use the below code. As every column is numeric (float64), we don't want to perform any data preprocessing here.

<div class="code-head">logistic_regression.py<span>code</span></div>

```python
import pandas as pd
df = pd.DataFrame(data.data)
print(df.dtypes)
```

```
0     float64
1     float64
2     float64
3     float64
4     float64
5     float64
6     float64
7     float64
8     float64
9     float64
10    float64
11    float64
12    float64
13    float64
14    float64
15    float64
16    float64
17    float64
18    float64
19    float64
20    float64
21    float64
22    float64
23    float64
24    float64
25    float64
26    float64
27    float64
28    float64
29    float64
dtype: object
```
{: .code-out}

<h3 id="supervised-learning">Supervised Learning</h3>

In a nutshell, what we try to solve in this problem is - 
* We have some training data <span class="coding">X_train</span> along with its class names <span class="coding">y_train</span>.
* We train a model (set of algorithms) with this training data (the magic happens here and we are yet to see it!)
* We use the trained model to predict the class <span class="coding">y_test</span> of unseen data point <span class="coding">X_test</span>.

This is called supervised learning problem (if you don't know yet) because we use a training dataset with class names already made available to us (nicely).

The flowchart that you could expect before diving into logistic regression implementation might look like the one shown below.

<figure>
  <img src="https://drive.google.com/uc?id=1U0Jfye7G3zKZRAoHVpHYSrhj25_hnJl-">
  <figcaption>Figure 1. Supervised Learning using Logistic Regression.</figcaption>
</figure>

Logistic regression is a type of [generalized linear classification algorithm](https://en.wikipedia.org/wiki/Generalized_linear_model){:target="_blank"} which follows a beautiful procedure to learn from data. To learn means,

* **Weight**: We define a weight value (parameter) for each feature (column) in the dataset.
* **Linear Predictor (score)**: We compute weighted sum for each data point in the dataset. 
* **Link Function**: We use a [link function](https://en.wikipedia.org/wiki/Generalized_linear_model#Link_function){:target="_blank"} to transform that weighted sum to the probability range \\( [0, 1] \\).
* **Log-Likelihood**: We use the log-likelihood function as the *quality metric* to evaluate the prediction of the model i.e., how well the model has predicted <span class="coding">y_predict</span> when compared with ground truth <span class="coding">y_train</span>.
* **Gradient Ascent**: We use gradient ascent algorithm to *update the weights (parameters)* by trying to maximize the likelihood.
* **Prediction**: We take these learned weights and make predictions when new data point is given to the model.

<div class="note">
  <p><b>Note</b>: To understand how the above steps work, we need to have some knowledge on probability, statistics, calculus and linear algebra.</p>
</div>

<h3 id="linear-predictor">Linear Predictor (score)</h3>

First, we define a weight value for each column (feature) in our dataset. As we have 30 features (columns) in the breast cancer dataset, we will have 30 weights [ \\( \mathbf W_1, \mathbf W_2 ... \mathbf W_{30}\\) ]. We compute the score (weighted sum) for each data point as follows.

<div class="math-cover">
$$
\begin{align}
score & = \mathbf W_0 + (\mathbf W_1 * h_1(x_i)) + (\mathbf W_2 * h_2(x_i)) + ... + (\mathbf W_{30} * h_{30}(x_i)) \\
& = \mathbf W^T h(x_i)
\end{align}
$$
</div>

Notice we have \\( \mathbf W_0 \\) with no coefficient which is called the *bias* or *intercept* which must be learnt from the training data. If you need to understand what bias is, please watch [this](https://www.youtube.com/watch?v=EuBBz3bI-aA){:target="_blank"} excellent video.

 As we have numeric values, the score for each data point might fall within a range \\( [-\infty, +\infty]\\). Recall that our aim is to predict *"given a new data point, tell me whether it's malignant (0) or benign (1)"*. This means, prediction from the ML model must be either 0 or 1. How are we going to achieve this? The answer is *link function*.

 <h3 id="link-function">Link Function</h3>

If you give any input to a link function (say sigmoid), it transforms that input value to a range \\( [0, 1] \\). In our case, anything below 0.5 is assumed to be malignant, and anything above or equal to 0.5 is assumed to be benign.

<figure>
  <img src="https://drive.google.com/uc?id=1NJSBDykPDbXtasFD6RP93M6auznJviDS">
  <figcaption>Figure 2. Link Function over a linear predictor (score).</figcaption>
</figure>

Below code is used to find the sigmoid value for a given input score.

<div class="code-head">logistic_regression.py<span>code</span></div>

```python
def sigmoid(score):
  return (1 / (1 + np.exp(-score)))

def predict_probability(features, weights):
  score = np.dot(features, weights)
  return sigmoid(score)
```

In the above code, <span class="coding">features</span>, <span class="coding">weights</span> and <span class="coding">score</span> correspond to the matrices shown below.

<div class="math-cover">
$$
\begin{align}
[features] = \begin{bmatrix}
    h(x_1)^T \\
    h(x_2)^T \\
    . \\
    . \\
    . \\
    h(x_{569})^T \\
    \end{bmatrix}
    = \begin{bmatrix}
    h_0(x_1) & h_1(x_1) & . & . & . & h_{30}(x_1) \\
    h_0(x_2) & h_1(x_2) & . & . & . & h_{30}(x_2) \\
    . & . & . & . & . & . \\
    . & . & . & . & . & . \\
    . & . & . & . & . & . \\
    h_0(x_{569}) & h_1(x_{569}) & . & . & . & h_{30}(x_{569}) \\
    \end{bmatrix}
\end{align}
$$
</div>

<div class="math-cover">
$$
[score] = [features] \mathbf w
   = \begin{bmatrix}
    h(x_1)^T \\
    h(x_2)^T \\
    . \\
    . \\
    . \\
    h(x_{569})^T 
    \end{bmatrix} \mathbf w
    = \begin{bmatrix}
    h(x_1)^T \mathbf w \\
    h(x_2)^T \mathbf w \\
    . \\
    . \\
    . \\
    h(x_{569})^T \mathbf w
    \end{bmatrix} 
    = \begin{bmatrix}
    \mathbf w^T h(x_1) \\
    \mathbf w^T h(x_2) \\
    . \\
    . \\
    . \\
    \mathbf w^T h(x_{569})
    \end{bmatrix}
$$
</div>

But wait! how will the output value of this link function be the same as the ground truth value for a particular data point? It can't be as we are randomizing the weights for the features which will throw out some random value as the prediction.

The whole point in learning algorithm is to *adjust these weights* based on the training data to arrive at a *sweet spot* that makes the ML model have *low bias* and *low variance*.

<div class="note">
  <p>Training the classifier = Learning the weight coefficients (with low bias and low variance).</p>
</div>

How do we adjust these weights? We need to define a *quality metric* that compares the output prediction of the ML model with the original ground truth class value. 

After evaluating the quality metric, we use *gradient ascent algorithm* to update the weights in a way that the quality metric reaches a global optimum value. Interesting isn't it?

<h3 id="compute-likelihood">Compute Likelihood</h3>

How do we measure "how well the classifier fits the training data"? Using [likelihood](https://en.wikipedia.org/wiki/Likelihood_function){:target="_blank"}. We need to choose weight coefficients \\(\mathbf w\\) that maximizes likelihood given below.

<div class="math-cover">
$$
\prod_{i=1}^N P(y_i | \mathbf x_i, \mathbf w)
$$
</div>

For a binary classification problem, it turns out that we can use [log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood){:target="_blank"} as the quality metric which makes computations and derivatives simpler.

<div class="math-cover">
$$
l(\mathbf w) = ln \prod_{i=1}^N P(y_i | \mathbf x_i, \mathbf w)
$$
</div>

After picking the log-likelihood function, we must know it's derivative with respect to a weight coefficient so that we can use gradient ascent to update that weight.

We use the below equation to calculate the log-likelihood for the classifier.

<div class="math-cover">
$$
ll(\mathbf w) = \sum_{i=1}^N ((\mathbf 1[y_i = +1] - 1) \mathbf w^T h(\mathbf w_i) - ln(1 + exp(-\mathbf w^T h(x_i))))
$$
</div>

We will understand the formation of these equations in a separate post. But for now, let us focus on implementing everything in code.

We define the below function to compute log-likelihood. Notice that we sum over all the training examples.

<div class="code-head">logistic_regression.py<span>code</span></div>

```python
def compute_log_likelihood(features, label, weights):
  indicator = (label==+1)
  scores    = np.dot(features, weights)
  ll        = np.sum((np.transpose(np.array([indicator]))-1)*scores - np.log(1. + np.exp(-scores)))
  return ll
```

<h3 id="compute-derivative">Compute Derivative</h3>

Once we have the log-likelihood equation, we can compute its derivative with respect to a single weight coefficient using the below formula.

<div class="math-cover">
$$
\frac{\partial l}{\partial w_j} = \sum_{i=1}^N h_j(\mathbf x_i) (\mathbf 1[y_i = +1] - P(y_i = +1|\mathbf x_i, \mathbf w))
$$
</div>

The above equation might look scary. But its easy to write in code.

* The term \\((\mathbf 1[y_i = +1] - P(y_i = +1\|\mathbf x_i, \mathbf w)\\) is nothing but the difference between <span class="coding">indicators</span> and <span class="coding">predictions</span> which is equal to <span class="coding">errors</span>.
* \\(h_j(\mathbf x_i)\\) is the feature value of a training example \\(\mathbf x_i \\) for a single column \\(j\\).

We find the derivative of log-likelihood with respect to each of the weight coefficient \\( \mathbf w \\) which in turn depends on its feature column. 

Notice that we sum over all the training examples, and the derivative that we return is a single number.

<div class="code-head">logistic_regression.py<span>code</span></div>

```python
def feature_derivative(errors, feature):
  derivative = np.dot(np.transpose(errors), feature)
  return derivative
```

<h3 id="gradient-ascent">Gradient Ascent</h3>

Now, we have all the ingredients to perform gradient ascent. The magic of this tutorial happens here!

Think of gradient ascent similar to hill-climbing. To reach the top of the hill (which is the global maximum), we choose a parameter called *learning-rate*. This defines the *step-size* that we need to take each iteration before we update the weight coefficients.

The steps that we will perform in gradient ascent are as follows.

1. Initialize weights vector \\( \mathbf w \\) to random values or zero using <span class="coding">np.zeros()</span>.
2. Predict the class probability \\( P(y_i = +1\|\mathbf x_i, \mathbf w) \\) for all training examples using <span class="coding">predict_probability</span> function and save to a variable named <span class="coding">predictions</span>. The shape of this variable would be <span class="coding">y_train.shape</span>.
3. Calculate the indicator value for all training examples by comparing the label against \\( +1 \\) and save it to a variable named <span class="coding">indicators</span>. The shape of this variable would also be <span class="coding">y_train.shape</span>.
4. Calculate the errors as the difference between <span class="coding">indicators</span> and <span class="coding">predictions</span> and save it to a variable named <span class="coding">errors</span>.
5. **Important step**: For each \\( j^{th} \\) weight coefficient, compute it's derivative using <span class="coding">feature_derivative</span> function with the \\( j^{th} \\) column of features. Increment the \\( j^{th} \\) coefficient using \\( lr * derivative\\) where \\( lr \\) is the learning rate for this algorithm which we handpick.
6. Do steps 2 to 5 for <span class="coding">epochs</span> times (number of iterations) and return the learned weight coefficients.

Below is the code to perform logistic regression using gradient ascent optimization algorithm.

<div class="code-head">logistic_regression.py<span>code</span></div>

```python
# logistic regression without L2 regularization
def logistic_regression(features, labels, lr, epochs):

  # add bias (intercept) with features matrix
  bias      = np.ones((features.shape[0], 1))
  features  = np.hstack((bias, features))

  # initialize the weight coefficients
  weights = np.zeros((features.shape[1], 1))

  logs = []

  # loop over epochs times
  for epoch in range(epochs):

    # predict probability for each row in the dataset
    predictions = predict_probability(features, weights)

    # calculate the indicator value
    indicators = (labels==+1)

    # calculate the errors
    errors = np.transpose(np.array([indicators])) - predictions

    # loop over each weight coefficient
    for j in range(len(weights)):

      # calculate the derivative of jth weight cofficient
      derivative = feature_derivative(errors, features[:,j])
      weights[j] += lr * derivative

    # compute the log-likelihood
    ll = compute_log_likelihood(features, labels, weights)
    logs.append(ll)

  import matplotlib.pyplot as plt
  x = np.linspace(0, len(logs), len(logs))
  fig = plt.figure()
  plt.plot(x, logs)
  fig.suptitle('Training the classifier (without L2)')
  plt.xlabel('Epoch')
  plt.ylabel('Log-likelihood')
  fig.savefig('train_without_l2.jpg')
  plt.show()

  return weights
```

<h3 id="split-the-dataset">Split the dataset</h3>

To test our classifier's performance, we will split the original dataset into training and testing. We choose a <span class="coding">test_size</span> parameter value to split the dataset into <span class="coding">train</span> and <span class="coding">test</span> using scikit-learn's <span class="coding">train_test_split</span> function as shown below.


<div class="code-head">logistic_regression.py<span>code</span></div>

```python
from sklearn.model_selection import train_test_split
# split the dataset into training and testing 
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.20, random_state=9)

print("X_train : " + str(X_train.shape))
print("y_train : " + str(y_train.shape))
print("X_test : " + str(X_test.shape))
print("y_test : " + str(y_test.shape))
```

```
X_train : (455, 30)
y_train : (455,)
X_test : (114, 30)
y_test : (114,)
```
{: .code-out}

<h3 id="train-the-classifier">Train the classifier</h3>

As we already learnt, training the classifier means learning the weight coefficients. To train the classifier, we
* Add intercept or bias to the feature matrix. 
* Initialize the weight coefficients to zeros.
* Handpick the hyper-parameters *learning rate* and *epochs*.
* Use <span class="coding">logistic_regression()</span> function that we have just built and pass in the ingredients.

<div class="code-head">logistic_regression.py<span>code</span></div>

```python
# hyper-parameters
learning_rate = 1e-7
epochs        = 500

# perform logistic regression
learned_weights = logistic_regression(X_train, y_train, learning_rate, epochs)
```

<figure>
  <img src="https://drive.google.com/uc?id=1e9REgzv_FIBdyW235udH9y-BUVfkxqvC">
  <figcaption>Figure 3. Increasing log-likelihood during training (without L2 regularization).</figcaption>
</figure>

<h3 id="test-the-classifier">Test the classifier</h3>

To make predictions using the trained classifier, we use <span class="coding">X_test</span> data (testing data), <span class="coding">learned_weights</span> and <span class="coding">predict_probability()</span> function. 

To find the accuracy between ground truth class values <span class="coding">y_test</span> and logistic regression predicted class values <span class="coding">predictions</span>, we use scikit-learn's <span class="coding">accuracy_score()</span> function as shown below.

<div class="code-head">logistic_regression.py<span>code</span></div>

```python
from sklearn.metrics import accuracy_score
# make predictions using learned weights on testing data
bias_train     = np.ones((X_train.shape[0], 1))
bias_test      = np.ones((X_test.shape[0], 1))
features_train = np.hstack((bias_train, X_train))
features_test  = np.hstack((bias_test, X_test))

test_predictions  = (predict_probability(features_test, learned_weights).flatten()>0.5)
train_predictions = (predict_probability(features_train, learned_weights).flatten()>0.5)
print("Accuracy of our LR classifier on training data: {}".format(accuracy_score(np.expand_dims(y_train, axis=1), train_predictions)))
print("Accuracy of our LR classifier on testing data: {}".format(accuracy_score(np.expand_dims(y_test, axis=1), test_predictions)))
```

```
Accuracy of our LR classifier on training data: 0.9164835164835164
Accuracy of our LR classifier on testing data: 0.9298245614035088
```
{: .code-out}

<h3 id="reduce-overfitting">Reduce Overfitting</h3>

Overfitting is a mandatory problem that we need to solve when it comes to machine learning. After training, we have the learned weight coefficients which must not *overfit* the training dataset. 

When the decision boundary traced by the learned weight coefficients fits the training data extremely well, we have this overfitting problem. Often, overfitting is associated with very large estimated weight coefficients. This leads to overconfident predictions which is not very good for a real-world classifier.

To solve this, we need to measure the magnitude of weight coefficients. There are two approaches to measure it.

**L1 norm**: Sum of absolute value 
  
\\(\lVert \mathbf w \rVert _1 = \|\mathbf w_0\| + \|\mathbf w_1\| + \|\mathbf w_2\| ... + \|\mathbf w_N\| \\)

**L2 norm**: Sum of squares 
  
\\(\lVert \mathbf w \rVert _2^2 = \mathbf w_0^2 + \mathbf w_1^2 + \mathbf w_2^2 ... + \mathbf w_N^2 \\)

<h3 id="l2-regularization">L2 Regularization</h3>

We will use L2 norm (sum of squares) to reduce overshooting weight coefficients. It turns out that, instead of using likelihood function alone as the quality metric, what if we subtract \\(\lambda \lVert \mathbf w \rVert _2^2\\) from it, where \\(\lambda\\) is a hyper-parameter to control bias-variance tradeoff due to this regularization.

So, our new quality metric with regularization to combat overconfidence problem would be

<div class="math-cover">
$$
l(w) - \lambda \lVert \mathbf w \rVert _2^2
$$
</div>

* Large \\(\lambda \\): High bias, low variance.
* Small \\(\lambda \\): Low bias, high variance.

Recall to perform gradient ascent, we need to know the derivative of quality metric to update the weight coefficients. Thus, the new derivative equation would be

<div class="math-cover">
$$
\frac{\partial l(\mathbf w)}{\partial \mathbf w_j} - 2 \lambda \mathbf w_j
$$
</div>

Let's understand the regularization impact on penalizing weight coefficients.

* If \\( \mathbf w_j > 0\\), then \\(- 2 \lambda \mathbf w_j < 0\\), thus it decreases \\( \mathbf w_j > 0\\) resulting in \\( \mathbf w_j \\) closer to 0.
* If \\( \mathbf w_j < 0\\), then \\(- 2 \lambda \mathbf w_j > 0\\), thus it increases \\( \mathbf w_j > 0\\) resulting in \\( \mathbf w_j \\) closer to 0.

When it comes to code, we need to update <span class="coding">feature_derivative()</span> function, <span class="coding">compute_log_likelihood()</span> function and <span class="coding">logistic_regression()</span> function with whatever we have learnt so far about L2 regularization as shown below.

<div class="code-head">logistic_regression.py<span>code</span></div>

```python
# feature derivative computation with L2 regularization
def l2_feature_derivative(errors, feature, weight, l2_penalty, feature_is_constant):
  derivative = np.dot(np.transpose(errors), feature)
  
  if not feature_is_constant:
    derivative -= 2 * l2_penalty * weight

  return derivative

# log-likelihood computation with L2 regularization
def l2_compute_log_likelihood(features, labels, weights, l2_penalty):
  indicator = (label==+1)
  scores    = np.dot(features, weights)
  ll        = np.sum((np.transpose(np.array([indicator]))-1)*scores - np.log(1. + np.exp(-scores))) - (l2_penalty * np.sum(weights[1:]**2))
  return ll

# logistic regression with L2 regularization
def l2_logistic_regression(features, labels, lr, epochs, l2_penalty):

  # add bias (intercept) with features matrix
  bias      = np.ones((features.shape[0], 1))
  features  = np.hstack((bias, features))

  # initialize the weight coefficients
  weights = np.zeros((features.shape[1], 1))

  logs = []

  # loop over epochs times
  for epoch in range(epochs):

    # predict probability for each row in the dataset
    predictions = predict_probability(features, weights)

    # calculate the indicator value
    indicators = (labels==+1)

    # calculate the errors
    errors = np.transpose(np.array([indicators])) - predictions

    # loop over each weight coefficient
    for j in range(len(weights)):

      isIntercept = (j==0)

      # calculate the derivative of jth weight cofficient
      derivative = l2_feature_derivative(errors, features[:,j], weights[j], l2_penalty, isIntercept)
      weights[j] += lr * derivative

    # compute the log-likelihood
    ll = l2_compute_log_likelihood(features, labels, weights, l2_penalty)
    logs.append(ll)

  import matplotlib.pyplot as plt
  x = np.linspace(0, len(logs), len(logs))
  fig = plt.figure()
  plt.plot(x, logs)
  fig.suptitle('Training the classifier (with L2)')
  plt.xlabel('Epoch')
  plt.ylabel('Log-likelihood')
  fig.savefig('train_with_l2.jpg')
  plt.show()

  return weights
```

Now, we can perform logistic regression with L2 regularization on this dataset using the below code.

<div class="code-head">logistic_regression.py<span>code</span></div>

```python
# logistic regression with regularization
def lr_with_regularization():
  # hyper-parameters
  learning_rate = 1e-7
  epochs        = 300000
  l2_penalty    = 0.001

  # perform logistic regression and get the learned weights
  learned_weights = l2_logistic_regression(X_train, y_train, learning_rate, epochs, l2_penalty)

  # make predictions using learned weights on testing data
  bias_train     = np.ones((X_train.shape[0], 1))
  bias_test      = np.ones((X_test.shape[0], 1))
  features_train = np.hstack((bias_train, X_train))
  features_test  = np.hstack((bias_test, X_test))

  test_predictions  = (predict_probability(features_test, learned_weights).flatten()>0.5)
  train_predictions = (predict_probability(features_train, learned_weights).flatten()>0.5)
  print("Accuracy of our LR classifier on training data: {}".format(accuracy_score(np.expand_dims(y_train, axis=1), train_predictions)))
  print("Accuracy of our LR classifier on testing data: {}".format(accuracy_score(np.expand_dims(y_test, axis=1), test_predictions)))

  # using scikit-learn's logistic regression classifier
  model = LogisticRegression(random_state=9)
  model.fit(X_train, y_train)
  sk_test_predictions  = model.predict(X_test)
  sk_train_predictions = model.predict(X_train)
  print("Accuracy of scikit-learn's LR classifier on training data: {}".format(accuracy_score(y_train, sk_train_predictions)))
  print("Accuracy of scikit-learn's LR classifier on testing data: {}".format(accuracy_score(y_test, sk_test_predictions)))

  visualize_weights(np.squeeze(learned_weights), 'weights_with_l2.jpg')

# visualize weight coefficients
def visualize_weights(weights, title):
  import matplotlib.pyplot as plt
  x = np.linspace(0, len(weights), len(weights))

  fig = plt.figure()
  plt.bar(x, weights, align='center', alpha=0.5)
  plt.xlabel("Weight Index (Feature Column Number)")
  plt.ylabel("Weight Coefficient")
  plt.title('Visualizing Weights')
  plt.tight_layout()
  fig.savefig(title)

  plt.show()

lr_without_regularization()
```

```
Accuracy of our LR classifier on training data: 0.9406593406593406
Accuracy of our LR classifier on testing data: 0.9385964912280702
Accuracy of scikit-learn's LR classifier on training data: 0.9648351648351648
Accuracy of scikit-learn's LR classifier on testing data: 0.9385964912280702
```
{: .code-out}

<figure>
  <img src="https://drive.google.com/uc?id=1NZq0sfCxxoiw7BIkldIZ8-EdxkixUtBm">
  <figcaption>Figure 4. Visualizing learnt weight coefficients after training.</figcaption>
</figure>

<figure>
  <img src="https://drive.google.com/uc?id=1HLvXeICcqJ2eaf21f4mLCkShGj1jOEWs">
  <figcaption>Figure 5. Increasing log-likelihood during training (with L2 regularization).</figcaption>
</figure>

<h3 id="conclusion">Conclusion</h3>

Thus, we have implemented our very own logistic regression classifier using python and numpy with/without L2 regularization, and compared it with scikit-learn's implementation.

We have achieved the **same test accuracy as scikit-learn's implementation** and what a way to achieve it on our own!

One key take away from this post is that, we still need to manually tune these hyper-parameters (<span class="coding">learning_rate</span>, <span class="coding">epochs</span> and <span class="coding">l2_penalty</span>) to reach the global maximum. If you found some approach to automate this task, please leave it out in the comments so that I as well as others can learn it.

<div class="references">
<h3 id="references">References</h3>

<ul>
  <li><a href="https://www.coursera.org/learn/ml-classification" target="_blank">Coursera's Machine Learning: Classification by Professor Carlos Guestrin</a></li>
  <li><a href="http://www.cs.cmu.edu/~guestrin/Class/10701-S05/slides/LogRegress-1-24-05.pdf" target="_blank">Logistic Regression, Generative and Discriminative Classifiers</a></li>
</ul>

</div>