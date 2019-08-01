---
layout: post
category: software
class: Keras
title: 6 Steps to Create Your First Deep Neural Network using Keras and Python
description: Learn how to create your first Deep Neural Network in few lines of code using Keras and Python
author: Gogul Ilango
permalink: software/first-neural-network-keras
image: https://drive.google.com/uc?id=1MS7idWK7YUQAGFjjdTj1VcCxznFancht
cardimage: https://drive.google.com/uc?id=1KqcxcuO1_x3Tq7DT0XZ92daV0iVHzJDA
---

<div class="git-showcase">
  <div>
    <a class="github-button" href="https://github.com/Gogul09" data-show-count="true" aria-label="Follow @Gogul09 on GitHub">Follow @Gogul09</a>
  </div>

  <div>
    <a class="github-button" href="https://github.com/Gogul09/deep-learning-fundamentals/fork" data-icon="octicon-repo-forked" data-show-count="true" aria-label="Fork Gogul09/deep-learning-fundamentals on GitHub">Fork</a>
  </div>

  <div>
    <a class="github-button" href="https://github.com/Gogul09/deep-learning-fundamentals" data-icon="octicon-star" data-show-count="true" aria-label="Star Gogul09/deep-learning-fundamentals on GitHub">Star</a>
  </div>  
</div>

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#deep-learning-environment-setup">Deep Learning Environment Setup</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#deep-learning-flowchart">Deep Learning Flowchart</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#analyse-the-dataset">1. Analyse the Dataset</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#prepare-the-dataset">2. Prepare the dataset</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#create-the-model">3. Create the Model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#compile-the-model">4. Compile the model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#fit-the-model">5. Fit the model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#evaluate-the-model">6. Evaluate the model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#complete-code">Complete code</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#summary">Summary</a></li>
  </ul>
</div>

**When I began to study Neural Networks during my coursework, I realized the complexity involved in representing those large layers of neurons in code. But when I read about [Keras](https://keras.io/){:target="_blank"} and started experimenting with it, everything was super easy to understand and perfectly made sense.** 

To all those who want to actually write some code to build a Deep Neural Network, but don't know where to begin, I highly suggest you to visit [Keras](https://keras.io/){:target="_blank"} website as well as it's [github](https://github.com/fchollet/keras){:target="_blank"} page. 

> **Update**: As Python2 faces [end of life](https://pythonclock.org/), the below code only supports **Python3**.

In this post, we will learn the simple 6 steps with which we can create our first deep neural network using Keras and Python. Let's get started!

<h3 class="code-head">Objectives</h3>

```
After reading this post, we will understand

* How to setup environment to create Deep Neural Nets?
* How to analyze and understand a training dataset?
* How to load and work with .csv file in Python?
* How to work with Keras, NumPy and Python for Deep Learning?
* How to create a Deep Neural Net in less than 5 minutes?
* How to split training and testing data for evaluating our model?
* How to display metrics to analyze performance of our model?
```

### Deep Learning Environment Setup

Before getting into concept and code, we need some libraries to get started with Deep Learning in Python. Copy and paste the below commands line-by-line to install all the dependencies needed for Deep Learning using Keras in Linux. I used Ubuntu 14.04 LTS 64-bit architecture as my OS and I didn't use any GPU to speed up computations for this tutorial.

<div class="code-head"><span>cmd</span>dependencies</div>
```
sudo pip install numpy
sudo pip install scipy
sudo pip install matplotlib
sudo pip install seaborn
sudo pip install scikit-learn
sudo pip install pillow
sudo pip install h5py
sudo pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
sudo pip install tensorflow
sudo pip install keras
```

Keras needs **Theano** or **TensorFlow** as its backend to perform numerical computations. TensorFlow won't work in Ubuntu 32-bit architecture (at the time of writing this tutorial). So, better have a 64-bit OS to properly install Keras with both TensorFlow and Theano backend.

If you need to work with Deep Learning on a Windows machine, please visit my post on environment setup for Windows [here](https://gogul09.github.io/software/deep-learning-windows){:target="_blank"}.

### Deep Learning flowchart

Let's talk about the classic [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning){:target="_blank"} problem. 

<figure>
  <img src="/images/software/first-neural-network-keras/supervised-learning.png">
  <figcaption>Figure 1. Supervised Learning</figcaption>
</figure>

Supervised Learning means we have a set of **training data** along with its outcomes (**classes** or **real values**). We train a deep learning model with the training data so that the model will be in a position to predict the outcome (**class** or **real value**) of future unseen data (or test data). 

This problem has two sub-divisions namely **Classification** and **Regression**. 

* **Classification** - If the output variable to be predicted by our model is a **label** or a **category**, then it is a Classification problem. *Ex: Predicting the name of a flower species.*
* **Regression** - If the output variable to be predicted by our model is a **real** or **continuous** value (integer, float), then it is a Regression problem. *Ex: Predicting the stock price of a company.*

We will concentrate on a Supervised Learning Classification problem and learn how to implement a Deep Neural Network in code using Keras. 

<div class="note"><p>
<b>Note</b>: To learn more about Deep Learning theory, I highly suggest you to register in Andrew NG's <a href="https://www.coursera.org/learn/machine-learning" target="_blank">machine learning course</a> and <a href="https://www.coursera.org/specializations/deep-learning" target="_blank">deep learning course</a> at Coursera or visit <a href="http://deeplearning.stanford.edu/tutorial/" target="_blank">Stanford University's</a> awesome website.
</p></div>

Download the below image for your future reference. Any complex problem related to Supervised Learning can be solved using this flowchart. We will walk through each step one-by-one in detail.

<figure>
	<img src="/images/software/first-neural-network-keras/Deep Learning flow.jpg">
	<figcaption>Figure 2. Keras Flow Chart for Deep Learning</figcaption>
</figure>

### Analyse the Dataset

> Deep Learning is all about **Data** and **Computational Power**.

You read it right! The first step in any Deep Learning problem is to collect more data to work with, analyse the data thoroughly and understand the various parameters/attributes in it. Attributes (also called as features) to look at in any dataset are as follows.

* **Dataset characteristics** - Multivariate, Univariate, Sequential, Time-series, Text.
* **Attribute characteristics** - Real, Integers, Strings, Float, Binary.
* **Number of Instances** - Total number of rows.
* **Number of Attributes** - Total number of columns.
* **Associated Tasks** - Classification, Regression, Clustering, Other problem.

For this tutorial, we will use the [Pima Indian Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database/data){:target="_blank"} from Kaggle. This dataset contains the patient medical record data for Pima Indians and tell us whether they had an onset of diabetes within 5 years or not (last column in the dataset). It is typically a **binary classification** problem where

* **1 = yes!** the patient had an onset of diabetes in 5 years.
* **0 = no!** the patient had no onset of diabetes in 5 years.

In this dataset, there are 8 attributes (i.e 8 columns) that describes each patient (i.e a single row) and a total of 768 instances (i.e total number of rows or number of patients).

Go to this [link](https://www.kaggle.com/uciml/pima-indians-diabetes-database/data){:target="_blank"}, register/login, download the dataset, save it inside a folder named **pima-indians-diabetes** and rename it as **dataset.csv**. Below is the folder structure to follow.

<div class="code-head"><span>rule</span>folder structure</div>
```python
|--pima-indians-diabetes
|--|--dataset.csv
|--|--train.py
```

### Prepare the dataset

> Data must be represented in a **structured way** for computers to understand.

Representing our analyzed data is the next step to do in Deep Learning. Data will be represented as an **n-dimensional matrix** in most of the cases (whether it is numerical or images or videos). Some of the common file-formats to store matrices are [csv](https://en.wikipedia.org/wiki/Comma-separated_values){:target="_blank"}, [cPickle](https://docs.python.org/2.2/lib/module-cPickle.html){:target="_blank"} and [h5py](http://www.h5py.org/){:target="_blank"}. If you have millions of data (say millions of images), **h5py** is the file-format of choice. We will stick with **CSV** file-format in this tutorial.

CSV (Comma Separated Values) file formats can easily be loaded in Python in two ways. One using [NumPy](http://www.numpy.org/){:target="_blank"} and other using [Pandas](https://pandas.pydata.org/){:target="_blank"}. We will use **NumPy**. 

Below is the code to import all the necessary libraries and load the pima-indians diabetes dataset.

<div class="code-head"><span>code</span>train.py</div>

```python
# organize imports
from keras.models import Sequential
from keras.models import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# seed for reproducing same results
seed = 9
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)

# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)
```

* Line (1-3) handles the imports to build our deep learning model using Keras.
* Line (6-7) fix a seed for reproducing same results if we wish to train and evaluate our network more than once.
* Line (10) loads the dataset from the **.csv file** saved on disk. It uses an argument <span class="coding">delimiter</span> to inform NumPy that all attributes are separated by **,** in the **.csv file**.
* Line (13-14) splits the dataset into input and output variables.
  * X has all the 8 attributes of the patients
  * Y has whether they have an onset of diabetes or not.
* Line (17) splits the dataset into training (67%) and testing (33%) using the scikit-learn's <span class="coding">train_test_split</span> function. By performing this split, we can easily verify the performance of our model in an unseen test dataset.

### Create the Model

In Keras, a model is created using <span class="coding">Sequential</span>. You may wanna recall that Neural Networks holds large number of neurons residing inside several sequential layers. 

We will create a model that has fully connected layers, which means all the neurons are connected from one layer to its next layer. This is achieved in Keras with the help of <span class="coding">Dense</span> function.

<figure>
	<img src="/images/software/first-neural-network-keras/Network Architecture.jpg">
	<figcaption>Figure 3. Our First Deep Neural Network Architecture</figcaption>
</figure>

We will use the above Deep Neural Network architecture which has a **single input layer**, **2 hidden layers** and a **single output layer**. 

The input data which is of size 8 is sent to the first hidden layer that has randomly initialized 8 neurons. This is a very useful approach, if we don't have any clue about the no.of.neurons to specify at the very first attempt. From here, we can easily perform trial-and-error procedure to increase the network architecture to produce good results. The next hidden layer has 6 neurons and the final output layer has 1 neuron that outputs whether the patient has an onset of diabetes or not. 

Figure 4 shows our Deep Neural Network with an input layer, a hidden layer having 8 neurons, a hidden layer having 6 neurons and an output layer with a single neuron. 

<figure>
  <img src="/images/software/first-neural-network-keras/mlp-1.png">
  <figcaption>Figure 4. Our First Deep Neural Network (Multi-Layer Perceptron)</figcaption>
</figure>

Below are the four lines of code to create the above architecture.

<div class="code-head"><span>code</span>train.py</div>

```python
# create the model
model = Sequential()
model.add(Dense(8, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(6, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
```

* Line (1) creates a <span class="coding">Sequential</span> model to add layers one at a time.
* Line (2), the first layer expects four arguments:
  * <span class="coding">8</span>: No.of.neurons present in that layer.
  * <span class="coding">input_dim</span>: specify the dimension of the input data.
  * <span class="coding">init</span>: specify whether <span class="coding">uniform</span> or  <span class="coding">normal</span> distribution of weights to be initialized.
  * <span class="coding">activation</span>: specify whether <span class="coding">relu</span> or <span class="coding">sigmoid</span> or <span class="coding">tanh</span> activation function to be used for each neuron in that layer.
* Line (3), the next hidden layer has 6 neurons with an <span class="coding">uniform</span> initialization of weights and <span class="coding">relu</span> activation function
* Line (4), the output layer has only one neuron as this is a binary classification problem. The activation function at output is <span class="coding">sigmoid</span> because it outputs a probability in the range 0 and 1 so that we could easily discriminate output by assigning a threshold.

### Compile the model

After creating the model, three parameters are needed to compile the model in Keras.
* <span class="coding">loss</span>: This is used to evaluate a set of weights. It is needed to reduce the error between actual output and expected output. It could be <span class="coding">binary_crossentropy</span> or <span class="coding">categorical_crossentropy</span> depending on the problem. As we are dealing with a binary classification problem, we need to pick <span class="coding">binary_crossentropy</span>. [Here](https://keras.io/losses/){:target="_blank"} is the list of loss functions available in Keras.
* <span class="coding">optimizer</span>: This is used to search through different weights for the network. It could be <span class="coding">adam</span> or <span class="coding">rmsprop</span> depending on the problem. [Here](https://keras.io/optimizers/){:target="_blank"} is the list of optimizers available in Keras.
* <span class="coding">metrics</span>: This is used to collect the report during training. Normally, we pick <span class="coding">accuracy</span> as our performance metric. [Here](https://keras.io/metrics/){:target="_blank"} is the list of metrics available in Keras.

These parameters are to be tuned according to the problem as our model needs some optimization in the background (which is taken care by **Theano** or **TensorFlow**) so that it learns from the data during each epoch (which means reducing the error between actual output and predicted output). 

<span class="coding">epoch</span> is the term used to denote the number of iterations involved during the training process of a neural network.


<div class="code-head"><span>code</span>train.py</div>

```python
# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

* Line (2) chooses a <span class="coding">binary_crossentropy</span> loss function and the famous Stochastic Gradient Descent (SGD) optimizer <span class="coding">adam</span>. It also collects the <span class="coding">accuracy</span> metric for training.

### Fit the model

After compiling the model, the dataset must be fitted with the model. The <span class="coding">fit()</span> function in Keras expects five arguments -

* <span class="coding">X_train</span>: the input training data.
* <span class="coding">Y_train</span>: the output training classes.
* <span class="coding">validation_data</span>: Tuple of testing or validation data used to check the performance of our network.
* <span class="coding">nb_epoch</span>: how much iterations should the training process take place.
* <span class="coding">batch_size</span>: No.of.instances that are evaluated before performing a weight update in the network.

<div class="code-head"><span>code</span>train.py</div>

```python
# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=100, batch_size=5)
```

* Line (2) chooses **100 iterations** to be performed by the deep neural network with a <span class="coding">batch_size</span> of **5**. 

### Evaluate the model

After fitting the dataset to the model, the model needs to be evaluated. Evaluating the trained model with an unseen test dataset shows how our model predicts output on unseen data. The <span class="coding">evaluate()</span> function in Keras expects two arguments.

* <span class="coding">X</span> - the input data.
* <span class="coding">Y</span> - the output data.

<div class="code-head"><span>code</span>train.py</div>

```python
# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" %(scores[1]*100))
```

### Complete code
Excluding comments and empty new lines, we have combined all the 6 steps together and created our first Deep Neural Network model using 18 lines of code. Below is the entire code of the model we have just built.

<div class="code-head"><span>code</span>train.py</div>

```python
# organize imports
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np

# seed for reproducing same results
seed = 9
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt('dataset.csv', delimiter=',', skiprows=1)

# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

# split the data into training (67%) and testing (33%)
(X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.33, random_state=seed)

# create the model
model = Sequential()
model.add(Dense(8, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(6, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=200, batch_size=5, verbose=0)

# evaluate the model
scores = model.evaluate(X_test, Y_test)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

* Save the code with a filename <span class="coding">train.py</span> in the same folder as the dataset.
* Open up a command prompt and go to that folder.
* Type <span class="coding">python train.py</span>

I got an accuracy of **72.83%** which is pretty good in our first try without any hyperparameter tuning or changing the network architecture. 

Kindly post your results in the comments after tuning around with <span class="coding">nb_epoch</span>, <span class="coding">batch_size</span>, <span class="coding">activation</span>, <span class="coding">init</span>, <span class="coding">optimizer</span> and <span class="coding">loss</span>. You can also increase the number of hidden layers and the number of neurons in each hidden layer.

### Summary
Thus, we have built our first Deep Neural Network (Multi-layer Perceptron) using Keras and Python in a matter of minutes. Note that we haven't even touched any math involved behind these Deep Neural Networks as it needs a separate post to understand. We have strictly focused on how to solve a supervised learning classification problem using Keras and Python.