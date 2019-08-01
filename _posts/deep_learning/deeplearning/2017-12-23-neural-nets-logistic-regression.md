---
layout: post
category: software
class: Deep Learning
title: Logistic Regression with a Neural Network mindset using NumPy and Python
description: Build a binary classifier logistic regression model with a neural network mindset using numpy and python.
author: Gogul Ilango
permalink: software/neural-nets-logistic-regression
image: https://drive.google.com/uc?id=1XHhFGbejMAuNJlY5PUMtodBjaHhZ9rQd
cardimage: https://drive.google.com/uc?id=1lJRANugElZw2CAIJPTRUabWNqDNO_8ud
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
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#image-as-a-vector">Image as a vector</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#dataset">Dataset</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#logistic-regression-pipeline">Logisitic Regression pipeline</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#logistic-regression-concept">Logistic Regression concept</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#vectorization">Vectorization</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#gradient-descent">Gradient Descent</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#mathematical-equations">Mathematical equations</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#math-to-code">Math to Code</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#fitting-it-all-together">Fitting it all together</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#training-the-model">Training the model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_11" href="#testing-the-trained-model">Testing the trained model (optional)</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_12" href="#resources">Resources</a></li>
  </ul>
</div>

Before understanding the math behind a Deep Neural Network and implementing it in code, it is better to get a mindset of how Logistic Regression could be modelled as a simple Neural Network that actually learns from data.

> **Update**: As Python2 faces [end of life](https://pythonclock.org/), the below code only supports **Python3**.

<div class="code-head">Objectives</div>

```
After reading this post, we will understand

* How to convert an image into a vector?
* How to preprocess an existing image dataset to do Deep Learning?
* How to represent images and labels as numpy arrays?
* How to use just one for-loop to train a logistic regression model?
```

A look at what we will be building at the end of this tutorial is shown below. A binary classifier that will classify an image as either <span class="coding">airplane</span> or <span class="coding">bike</span>.

<figure>
  <img src="/images/software/logistic-regression/out.gif" class="typical-image">
  <figcaption>Figure 1. Binary Classification using Logistic Regression Neural Network model</figcaption>
</figure>

<h3 id="image-as-a-vector">Image as a vector</h3> 

The input to the logistic regression model is an image. An image is a three-dimensional matrix that holds pixel intensity values of Red, Green and Blue channels. In Deep Learning, what we do first is that we convert this image (3d-matrix) to a 1d-matrix (also called as a vector). 

For example, if our image is of dimension [640, 480, 3] where 640 is the width, 480 is the height and 3 is the number of channels, then the flattened version of the image or 1-d representation of the image will be [1, 921600].

Notice that in the above vector dimension, we represent the image as a row vector having 921600 columns.

To better understand this, look at the image below.

<figure>
  <img src="/images/software/logistic-regression/image-to-vector.jpg" class="typical-image">
  <figcaption>Figure 2. Image (3-d) to Vector (1-d)</figcaption>
</figure>

<h3 id="dataset">Dataset</h3>
We will use the [CALTECH-101](http://www.vision.caltech.edu/Image_Datasets/Caltech101/){:target="_blank"} dataset which has images belonging to 101 categories such as airplane, bike, elephant etc. As we are dealing with a binary classification problem, we will specifically use images from two categories <span class="coding">airplane</span> and <span class="coding">bike</span>. 

Download this dataset and make sure you follow the below folder structure.

<div class="code-head">Folder Structure<span>rule</span></div>

```python
|--logistic-regression
|--|--dataset
|--|--|--train
|--|--|--|--airplane
|--|--|--|--|--image_1.jpg
|--|--|--|--|--image_2.jpg
|--|--|--|--|--....
|--|--|--|--|--image_750.jpg
|--|--|--|--bike
|--|--|--|--|--image_1.jpg
|--|--|--|--|--image_2.jpg
|--|--|--|--|--....
|--|--|--|--|--image_750.jpg
|--|--|--test
|--|--|--|--airplane
|--|--|--|--|--image_1.jpg
|--|--|--|--|--image_2.jpg
|--|--|--|--|--....
|--|--|--|--|--image_50.jpg
|--|--|--|--bike
|--|--|--|--|--image_1.jpg
|--|--|--|--|--image_2.jpg
|--|--|--|--|--....
|--|--|--|--|--image_50.jpg
|--|--train.py
```

Inside the train folder, you need to create two sub-folders namely <span class="coding">airplane</span> and <span class="coding">bike</span>. You have to manually copy 750 images from <span class="coding">airplane</span> folder in "CALTECH-101" dataset to our <span class="coding">airplane</span> folder. Similarly, you have to manually copy 750 images from <span class="coding">motorbikes</span> folder in "CALTECH-101" dataset to our <span class="coding">bike</span> folder.

Inside the test folder, you have to do the same process, but now, you will be having 50 images in <span class="coding">airplane</span> and 50 images in <span class="coding">bike</span>.

Before starting anything, make sure you have the following number of images in each folder.
* dataset -> train -> airplane -> 750 images
* dataset -> train -> bike     -> 750 images
* dataset -> test  -> airplane -> 50  images
* dataset -> test  -> bike     -> 50  images

#### Prepare the dataset

Let's fix the input image size with dimensions \\([64, 64, 3]\\), meaning \\(64\\) is the width and height with \\(3\\) channels. The flattened vector will then have dimension \\([1, 12288]\\). We will also need to know the total number of training images that we are going to use so that we can build an empty numpy array with that dimension and then fill it up after flattening every image. In our case, total number of train images <span class="coding">num_train_images</span> = \\(1500\\) and total number of test images <span class="coding">num_test_images</span> = \\(100\\).

We need to define four numpy arrays filled with zeros.
* Array of dimension \\([12288, 1500]\\) to hold our train images. 
* Array of dimension \\([12288, 100]\\) to hold our test images. 
* Array of dimension \\([1, 1500]\\) to hold our train labels. 
* Array of dimension \\([1, 100]\\) to hold our test labels. 

For each image in the dataset: 
* Convert the image into a matrix of fixed size using <span class="coding">load_img()</span> in Keras - \\([64, 64, 3]\\).
* Convert the image into a row vector using <span class="coding">flatten()</span> in NumPy - \\([12288,]\\)
* Expand the dimensions of the above vector using <span class="coding">np.expand_dims()</span> in NumPy - \\([1, 12288]\\)
* Concatenate this vector to a numpy array <span class="coding">train_x</span> of dimension \\([12288, 1500]\\).
* Concatenate this vector's label to a numpy array  <span class="coding">train_y</span> of dimension \\([1, 1500]\\).

We need to perform the above procedure for test data to get <span class="coding">test_x</span> and <span class="coding">test_y</span>. 

We then standardize <span class="coding">train_x</span> and <span class="coding">test_x</span> by dividing each pixel intensity value by 255. This is because normalizing the image matrix makes our learning algorithm better.

Also, we will assign "0" as the label to <span class="coding">airplane</span> and "1" as the label to <span class="coding">bike</span>. This is very important as computers work only with numbers.

Finally, we can save all our four numpy arrays locally using <span class="coding">h5py</span> library.

Below is the code snippet to do all the above steps before building our logistic regression neural network model.

<div class="code-head">train.py<span>code</span></div>

```python
#-------------------
# organize imports
#-------------------
import numpy as np
import os
import h5py
import glob
import cv2
from keras.preprocessing import image

#------------------------
# dataset pre-processing
#------------------------
train_path   = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\train"
test_path    = "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test"
train_labels = os.listdir(train_path)
test_labels  = os.listdir(test_path) 

# tunable parameters
image_size       = (64, 64)
num_train_images = 1500
num_test_images  = 100
num_channels     = 3

# train_x dimension = {(64*64*3), 1500}
# train_y dimension = {1, 1500}
# test_x dimension  = {(64*64*3), 100}
# test_y dimension  = {1, 100}
train_x = np.zeros(((image_size[0]*image_size[1]*num_channels), num_train_images))
train_y = np.zeros((1, num_train_images))
test_x  = np.zeros(((image_size[0]*image_size[1]*num_channels), num_test_images))
test_y  = np.zeros((1, num_test_images))

#----------------
# TRAIN dataset
#----------------
count = 0
num_label = 0
for i, label in enumerate(train_labels):
	cur_path = train_path + "\\" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x   = image.img_to_array(img)
		x   = x.flatten()
		x   = np.expand_dims(x, axis=0)
		train_x[:,count] = x
		train_y[:,count] = num_label
		count += 1
	num_label += 1

#--------------
# TEST dataset
#--------------
count = 0 
num_label = 0 
for i, label in enumerate(test_labels):
	cur_path = test_path + "\\" + label
	for image_path in glob.glob(cur_path + "/*.jpg"):
		img = image.load_img(image_path, target_size=image_size)
		x   = image.img_to_array(img)
		x   = x.flatten()
		x   = np.expand_dims(x, axis=0)
		test_x[:,count] = x
		test_y[:,count] = num_label
		count += 1
	num_label += 1

#------------------
# standardization
#------------------
train_x = train_x/255.
test_x  = test_x/255.

print("train_labels : " + str(train_labels))
print("train_x shape: " + str(train_x.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x shape : " + str(test_x.shape))
print("test_y shape : " + str(test_y.shape))

#-----------------
# save using h5py
#-----------------
h5_train = h5py.File("train_x.h5", 'w')
h5_train.create_dataset("data_train", data=np.array(train_x))
h5_train.close()

h5_test = h5py.File("test_x.h5", 'w')
h5_test.create_dataset("data_test", data=np.array(test_x))
h5_test.close()
```

```
train_labels : ['airplane', 'bike']
train_x shape: (12288, 1500)
train_y shape: (1, 1500)
test_x shape : (12288, 100)
test_y shape : (1, 100)
```
{: .code-output}

<h3 id="logistic-regression-pipeline">Logistic Regression pipeline</h3>

<figure>
  <img src="/images/software/logistic-regression/logistic-regression-neural-network.jpg" class="typical-image">
  <figcaption>Figure 3. Logistic Regression - A Simple Neural Network</figcaption>
</figure>

By looking at the above figure, the problem that we are going to solve is this -

> Given an input image, our model must be able to figure out the label by telling whether it is an <span class="coding">airplane</span> or a <span class="coding">bike</span>. 

#### A simple neuron

An artificial neuron (shown as purple circle in Figure 3) is a biologically inspired representation of a neuron inside human brain. Similar to a neuron in human brain, an artificial neuron accepts inputs from some other neurons and fires a value to the next set of artificial neurons. Inside a single neuron, two computations are performed.
1. Weighted sum
2. Activation

#### Weighted sum
Every input value \\(x_{(i)}\\) to an artificial neuron has a weight \\(w_{(i)}\\) associated with it which tells about the relative importance of that input with other inputs. Each weight \\(w_{(i)}\\) is multiplied with its corresponding input \\(x_{(i)}\\) and gets summed up to produce a single value \\(z\\).

#### Activation
After computing the weighted sum \\(z\\), an activation function \\(a = g(z)\\) is applied to this weighted sum \\(z\\). Activation function is a simple mathematical transformation of an input value to an output value by introducing a non-linearity. This is necessary because real-world inputs are non-linear and we need our neural network to learn this non-linearity somehow.   

<h3 id="logistic-regression-concept">Logistic Regression concept</h3>
1. Initialize the weights <span class="coding">w</span> and biases <span class="coding">b</span> to random values (say 0 or using random distribution). 
2. Foreach training sample in the dataset -
   * Calculate the output value \\(a^{(i)}\\) for an input sample \\(x^{(i)}\\).
   	 * First: find out the weighted sum \\(z^{(i)}\\).
     * Second: compute the activation value \\(a^{(i)}\\) = \\(y'^{(i)}\\) = \\(g(z^{(i)})\\) for the weighted sum \\(z^{(i)}\\).
   * As we know the true label for this input training sample \\(y^{(i)}\\), we use that to find the loss \\(L(a^{(i)},y^{(i)})\\).
3. Calculate the cost function \\(J\\) which is the sum of all losses divided by the number of training examples \\(m\\) i.e., \\(\frac{1}{m}\sum_{i=1}^m L(a^{(i)},y^{(i)})\\).
4. To minimize the cost function, compute the gradients for parameters \\(\frac{dJ}{dw}\\) and \\(\frac{dJ}{db}\\) using chain rule of calculus.
5. Use gradient descent to update the parameters <span class="coding">w</span> and <span class="coding">b</span>.
6. Perform the above procedure till the cost function becomes minimum.

<h3 id="vectorization">Vectorization</h3>
One interesting thing in the above algorithm is that we will not be using the for loop (2nd point) in code; rather we will use vectorization offered by numpy to speed up computations in an efficient way.

After successfully pre-processing the dataset, please look at the below image to visually understand how the dimensions of our numpy arrays look like.

<figure>
  <img src="/images/software/logistic-regression/dimensions.jpg" class="typical-image">
  <figcaption>Figure 4. Dimensions of weights, train image matrix, biases and labels.</figcaption>
</figure>

As you can see, the weights array has a dimension of shape \\([1, 12288]\\) and biases array has a dimension of shape \\([1, 1500]\\). But you will see that we will be initializing bias as a single value. A concept called <span class="coding">broadcasting</span> automatically applies the single <span class="coding">b</span> value to the matrix of shape \\([1, 1500]\\).

<h3 id="gradient-descent">Gradient Descent</h3>
<figure>
  <img src="/images/software/logistic-regression/computation-graph-1.jpg" class="typical-image">
  <figcaption>Figure 5. Computing derivatives of parameters "w" and "b" with respect to loss function "L" on one training example.</figcaption>
</figure>

The above figure shows us how to visualize forward propagation and backpropagation as a computation graph for one training example. 
* Forward propagation (for a single training example)
  * Calculate the weighted sum \\(z = w_1x_1 + w_2x_2 + b\\).
  * Calculate the activation \\(a = \sigma(z)\\).
  * Compute the loss \\(L(a,y) = -ylog(a)+(1-y)log(1-a)\\).
* Backpropagation (for a single training example)
  * Compute the derivatives of parameters \\(\frac{dL}{dw1}\\), \\(\frac{dL}{dw2}\\) and \\(\frac{dL}{db}\\) using \\(\frac{dL}{da}\\) and \\(\frac{dL}{dz}\\).
  * Use update rule to update the parameters.
    * \\(w1 = w1 -\alpha \frac{dL}{dw1}\\)
    * \\(w2 = w2 -\alpha \frac{dL}{dw2}\\)
    * \\(b = b -\alpha \frac{dL}{db}\\)

In code, we will be denoting \\(\frac{dL}{dw1}\\) as <span class="coding">dw1</span>, \\(\frac{dL}{dw2}\\) as <span class="coding">dw2</span>, \\(\frac{dL}{db}\\) as <span class="coding">db</span>, \\(\frac{dL}{da}\\) as <span class="coding">da</span> and \\(\frac{dL}{dz}\\) as <span class="coding">dz</span>.

But in our problem we don't just have 1 training example, rather \\(m\\) training examples. This is where cost function \\(J\\) comes into picture. So, we calculate losses for all the training examples, sum it up and divide by the number of training examples \\(m\\).

<h3 id="mathematical-equations">Mathematical equations</h3>

There are 7 mathematical equations to build a logistic regression model with a neural network mindset. Everything else is vectorization. So, the core concept in building neural networks is to understand these equations thoroughly.

*Weighted Sum of \\(i^{th}\\) training example*
<div class="math-cover">
$$
z^{(i)} = w^Tx^{(i)} + b
$$
</div>

*Activation of \\(i^{th}\\) training example* (using sigmoid)
<div class="math-cover">
$$
y'^{(i)} = a^{(i)} = \sigma(z^{(i)}) = \frac{1}{1+e^{-z^{(i)}}}
$$
</div>

*Loss function of \\(i^{th}\\) training example*
<div class="math-cover">
$$
L(a^{(i)},y^{(i)}) = -y^{(i)}log(a^{(i)}) - (1-y^{(i)})log(1-a^{(i)}))
$$
</div>

*Cost function for all training examples*
<div class="math-cover">
$$
J  = \frac{1}{m}\sum_{i=1}^m L(a^{(i)},y^{(i)}) \\
J  = -\frac{1}{m}\sum_{i=1}^m y^{(i)}log(a^{(i)}) + (1-y^{(i)})log(1-a^{(i)}))
$$
</div>

*Gradient Descent w.r.t cost function, weights and bias*
<div class="math-cover">
$$
\frac{dJ}{dw} = \frac{1}{m} X(A-Y)^T \\
\frac{dJ}{db} = \frac{1}{m} \sum_{i=1}^m (a^{(i)} - y^{(i)})
$$
</div>

*Parameters update rule*
<div class="math-cover">
$$
w = w - \alpha \frac{dJ}{dw} \\
b = b - \alpha \frac{dJ}{db}
$$
</div>

<h3 id="math-to-code">Math to Code</h3>

We will be using the below functions to create and train our logistic regression neural network model.

##### 1. <span class="coding">sigmoid()</span>
   * *Input*  - a number or a numpy array.
   * *Output* - sigmoid of the number or the numpy array.

<div class="code-head">train.py<span>code</span></div>

```python
def sigmoid(z):
	return (1/(1+np.exp(-z)))
```

##### 2. <span class="coding">init_params()</span>
   * *Input*  - dimension for weights (every value in an image's vector has a weight associated with it).
   * *Output* - weight vector <span class="coding">w</span> and bias <span class="coding">b</span>

<div class="code-head">train.py<span>code</span></div>

```python
def init_params(dimension):
	w = np.zeros((dimension, 1))
	b = 0
	return w, b
```

##### 3. <span class="coding">propagate()</span>
   * *Input*  - weight vector <span class="coding">w</span>, bias <span class="coding">b</span>, image matrix <span class="coding">X</span> and label vector <span class="coding">Y</span>.
   * *Output* - gradients <span class="coding">dw</span>, <span class="coding">db</span> and cost function <span class="coding">costs</span> for every 100 iterations.

<div class="code-head">train.py<span>code</span></div>

```python
def propagate(w, b, X, Y):
	# num of training samples
	m = X.shape[1]

	# forward pass
	A    = sigmoid(np.dot(w.T,X) + b)
	cost = (-1/m)*(np.sum(np.multiply(Y,np.log(A)) + np.multiply((1-Y),np.log(1-A))))

	# back propagation
	dw = (1/m)*(np.dot(X, (A-Y).T))
	db = (1/m)*(np.sum(A-Y))

	cost = np.squeeze(cost)

	# gradient dictionary
	grads = {"dw": dw, "db": db}

	return grads, cost
```

##### 4. <span class="coding">optimize()</span>
   * *Input*  - weight vector <span class="coding">w</span>, bias <span class="coding">b</span>, image matrix <span class="coding">X</span>, label vector <span class="coding">Y</span>, number of iterations for gradient descent <span class="coding">epochs</span> and learning rate <span class="coding">lr</span>.
   * *Output* - parameter dictionary <span class="coding">params</span> holding updated <span class="coding">w</span> and <span class="coding">b</span>, gradient dictionary <span class="coding">grads</span> holding <span class="coding">dw</span> and <span class="coding">db</span>, and list of cost function <span class="coding">costs</span> after every 100 iterations.

<div class="code-head">train.py<span>code</span></div>

```python
def optimize(w, b, X, Y, epochs, lr):
	costs = []
	for i in range(epochs):
		# calculate gradients
		grads, cost = propagate(w, b, X, Y)

		# get gradients
		dw = grads["dw"]
		db = grads["db"]

		# update rule
		w = w - (lr*dw)
		b = b - (lr*db)

		if i % 100 == 0:
			costs.append(cost)
			print("cost after %i epochs: %f" %(i, cost))

	# param dict
	params = {"w": w, "b": b}

	# gradient dict
	grads  = {"dw": dw, "db": db}

	return params, grads, costs
```

##### 5. <span class="coding">predict()</span>
   * *Input*  - updated parameters <span class="coding">w</span>, <span class="coding">b</span> and image matrix <span class="coding">X</span>.
   * *Output* - predicted labels <span class="coding">Y_predict</span> for the image matrix <span class="coding">X</span>.

<div class="code-head">train.py<span>code</span></div>

```python
def predict(w, b, X):
	m = X.shape[1]
	Y_predict = np.zeros((1,m))
	w = w.reshape(X.shape[0], 1)

	A = sigmoid(np.dot(w.T, X) + b)

	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:
			Y_predict[0, i] = 0
		else:
			Y_predict[0,i]  = 1

	return Y_predict
```

##### 6. <span class="coding">predict_image()</span>
   * *Input*  - updated parameters <span class="coding">w</span>, <span class="coding">b</span> and a single image vector <span class="coding">X</span>.
   * *Output* - predicted label <span class="coding">Y_predict</span> for the single image vector <span class="coding">X</span>.

<div class="code-head">train.py<span>code</span></div>

```python
def predict_image(w, b, X):
	Y_predict = None
	w = w.reshape(X.shape[0], 1)
	A = sigmoid(np.dot(w.T, X) + b)
	for i in range(A.shape[1]):
		if A[0, i] <= 0.5:
			Y_predict = 0
		else:
			Y_predict = 1

	return Y_predict
```

<h3 id="fitting-it-all-together">Fitting it all together</h3>

We will use all the above functions into a main function named <span class="coding">model()</span>.
* *Input* - Training image matrix <span class="coding">X_train</span>, Training image labels <span class="coding">Y_train</span>, Testing image matrix <span class="coding">X_test</span>, Test image labels <span class="coding">Y_test</span>, number of iterations for gradient descent <span class="coding">epochs</span> and learning rate <span class="coding">lr</span>.
* *Output* - Logistic regression model dictionary having the parameters (w,b), predictions, costs, learning rate, epochs.
 
<div class="code-head">train.py<span>code</span></div>

```python
def model(X_train, Y_train, X_test, Y_test, epochs, lr):
	w, b = init_params(X_train.shape[0])
	params, grads, costs = optimize(w, b, X_train, Y_train, epochs, lr)

	w = params["w"]
	b = params["b"]

	Y_predict_train = predict(w, b, X_train)
	Y_predict_test  = predict(w, b, X_test)

	print("train_accuracy: {} %".format(100-np.mean(np.abs(Y_predict_train - Y_train)) * 100))
	print("test_accuracy : {} %".format(100-np.mean(np.abs(Y_predict_test  - Y_test)) * 100))

	log_reg_model = {"costs": costs,
				     "Y_predict_test": Y_predict_test, 
					 "Y_predict_train" : Y_predict_train, 
					 "w" : w, 
					 "b" : b,
					 "learning_rate" : lr,
					 "epochs": epochs}

	return log_reg_model
```

<h3 id="training-the-model">Training the model</h3>

Finally, we can train our model using the below code. This produces the train accuracy and test accuracy for the dataset.

<div class="code-head">train.py<span>code</span></div>

```python
# activate the logistic regression model
myModel = model(train_x, train_y, test_x, test_y, epochs, lr)
```

```
Using TensorFlow backend.
train_labels : ['airplane', 'bike']
train_x shape: (12288, 1500)
train_y shape: (1, 1500)
test_x shape : (12288, 101)
test_y shape : (1, 101)
cost after 0 epochs: 0.693147
cost after 100 epochs: 0.136297
cost after 200 epochs: 0.092398
cost after 300 epochs: 0.076973
cost after 400 epochs: 0.067062
cost after 500 epochs: 0.059735
cost after 600 epochs: 0.053994
cost after 700 epochs: 0.049335
cost after 800 epochs: 0.045456
cost after 900 epochs: 0.042167
cost after 1000 epochs: 0.039335
cost after 1100 epochs: 0.036868
cost after 1200 epochs: 0.034697
cost after 1300 epochs: 0.032769
cost after 1400 epochs: 0.031045
cost after 1500 epochs: 0.029493
cost after 1600 epochs: 0.028088
cost after 1700 epochs: 0.026810
cost after 1800 epochs: 0.025642
cost after 1900 epochs: 0.024570
train_accuracy: 99.66666666666667 %
test_accuracy : 100.0 %
```
{: .code-output}

<h3 id="testing-the-trained-model">Testing the trained model (optional)</h3>

We can use OpenCV to visualize our model's performance on test dataset. Below code snipped takes it four images, our model predicts the label for these four images and displays it on the screen.

<div class="code-head">train.py<span>code</span></div>

```python
test_img_paths = ["G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\airplane\\image_0723.jpg",
                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\airplane\\image_0713.jpg",
                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\image_0782.jpg",
                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\image_0799.jpg",
                  "G:\\workspace\\machine-intelligence\\deep-learning\\logistic-regression\\dataset\\test\\bike\\test_1.jpg"]

for test_img_path in test_img_paths:
	img_to_show    = cv2.imread(test_img_path, -1)
	img            = image.load_img(test_img_path, target_size=image_size)
	x              = image.img_to_array(img)
	x              = x.flatten()
	x              = np.expand_dims(x, axis=1)
	predict        = predict_image(myModel["w"], myModel["b"], x)
	predict_label  = ""

	if predict == 0:
		predict_label = "airplane"
	else:
		predict_label = "bike"

	# display the test image and the predicted label
	cv2.putText(img_to_show, predict_label, (30,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
	cv2.imshow("test_image", img_to_show)
	key = cv2.waitKey(0) & 0xFF
	if (key == 27):
		cv2.destroyAllWindows()
```

<figure>
  <img src="/images/software/logistic-regression/out.gif" class="typical-image">
  <figcaption>Figure 5. Making predictions using OpenCV on test data</figcaption>
</figure>

<h3 id="resources">Resources</h3>

1. [Coursera-Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning/home/welcome){:target="_blank"}
2. [But what is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk){:target="_blank"}
3. [Gradient Descent-How Neural Networks learn](https://www.youtube.com/watch?v=IHZwWFHWa-w){:target="_blank"}
4. [What is Backpropagation really doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U){:target="_blank"}
5. [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8){:target="_blank"}
6. [Keras](https://keras.io/){:target="_blank"}
7. [NumPy](http://www.numpy.org/){:target="_blank"}