---
layout: post
category: software
class: Keras
title: Handwritten Digit Recognition using Deep Learning, Keras and Python
description: Learn how to recognize handwritten digit using a Deep Neural Network called Multi-layer Perceptron (MLP).
author: Gogul Ilango
permalink: software/digits-recognition-mlp
image: https://drive.google.com/uc?id=1M3Elnv5HDaS_KR5MVmNxFP6JoQOboaxK
cardimage: https://drive.google.com/uc?id=1MmLXQwBFhCSBMxnrvyJFI8ZkQb0GUfx3
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
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#dependencies">Dependencies</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#pipeline">Pipeline</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#dataset-summary">Dataset summary</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#organize-imports">Organize imports</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#define-user-inputs">Define user inputs</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#prepare-the-dataset">Prepare the dataset</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#one-hot-encoding">One Hot Encoding</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#create-the-model">Create the model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#compile-the-model">Compile the model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#fit-the-model">Fit the model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_11" href="#evaluate-the-model">Evaluate the model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_12" href="#results">Results</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_13" href="#complete-code">Complete code</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_14" href="#testing-the-model">Testing the model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_15" href="#summary">Summary</a></li>
  </ul>
</div>

**In this tutorial, we will learn how to recognize handwritten digit using a simple Multi-Layer Perceptron (MLP) in Keras. We will also learn how to build a near state-of-the-art deep neural network model using Python and Keras. A quick Google search about this dataset will give you tons of information - [MNIST](https://en.wikipedia.org/wiki/MNIST_database){:target="_blank"}.**

> **Update**: As Python2 faces [end of life](https://pythonclock.org/), the below code only supports **Python3**.

This tutorial is intended for beginners exploring how to implement a neural network using Keras. We will strictly focus towards the implementation of a neural network without concentrating too much on the theory part. To learn more Deep Learning theory, please visit [this](http://deeplearning.stanford.edu/tutorial/){:target="_blank"} awesome website.

<div class="note" style="margin-bottom: 0px !important">
	<p><b>Note</b>: Before implementing anything in code, we need to setup our environment to do Deep Learning. Make sure you use the below links to do that and then come here.</p>
	<ul style="margin-bottom: 0px !important">
		<li><a href="https://gogul09.github.io/software/deep-learning-windows" target="_blank">Deep Learning Environment Setup (Windows)</a></li>
		<li><a href="https://gogul09.github.io/software/deep-learning-linux" target="_blank">Deep Learning Environment Setup for (Linux)</a></li>
	</ul>
</div>

<div class="code-head">Objectives</div>

```
After reading this post, we will understand
* What is MNIST dataset?
* How to load and pre-process MNIST dataset in Keras?
* How to use one-hot encoding in Keras?
* How to create a multi-layer perceptron in Keras?
* How to perform Deep Learning on MNIST dataset using Keras?
* How to save history of model and visualize metrics using Keras and matplotlib?
* How to recognize digits in real-images by segmenting each digit and then classifying it?
```

### Dependencies

You will need the following software packages and libraries to follow this tutorial.

* Python
* NumPy
* Matplotlib
* Theano or TensorFlow
* Keras

### Pipeline

Any deep learning implementation in Keras follows the below pipeline. It is highly recommended that you go through this first before writing any code so that you get a clear sense of what you are going to achieve through Deep Learning.

<figure>
  <img src="/images/software/first-neural-network-keras/Deep Learning flow.jpg" class="typical-image" />
  <figcaption>Figure 1. Pipeline for Deep Learning</figcaption>
</figure>

### Dataset summary
* MNIST - Mixed National Institute of Standards and Technology database.
* Created by - Yaan LeCun, Corinna Cortes, Christopher Burges.
* Image size - 28 x 28 pixel square (784 pixels in total).
* Training images - 60,000
* Testing images - 10,000
* Top error rate - 0.21 (achieved by CNN)
* Dataset size in Keras - 14.6 MB

<div class="note"><p>
<b>Note:</b> To look at state-of-the-art accuracies on standard datasets such as MNIST, CIFAR-10 etc., please visit <a href="http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354" target="_blank">this</a> excellent website by Rodrigo Benenson.
</p></div>

### Organize imports
We will need the following python libraries to build our neural network.
* [NumPy](http://www.numpy.org/){:target="_blank"} - To perform matrix/vector operations as we are working with Images (3D data).
* [Matplotlib](https://matplotlib.org/){:target="_blank"} - To visualize what's happening with our neural network model.
* [Keras](https://keras.io/){:target="_blank"} - To create the neural network model with neurons, layers and other utilities.

<div class="code-head"><span>code</span>train.py</div>

```python
# organize imports
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils

# fix a random seed for reproducibility
np.random.seed(9)
```

### Define user inputs
In a neural network, there are some variables/parameters that could be tuned to obtain good results. These variables are indeed user inputs which needs some experience to pick up the right one. Such user-defined inputs are given below - 

* <span class="coding">nb_epoch</span> - Number of iterations needed for the network to minimize the loss function, so that it learns the weights.
* <span class="coding">num_classes</span> - Total number of class labels or classes involved in the classification problem.
* <span class="coding">batch_size</span> - Number of images given to the model at a particular instance.
* <span class="coding">train_size</span> - Number of training images to train the model.
* <span class="coding">test_size</span> - Number of testing images to test the model.
* <span class="coding">v_length</span> - Dimension of flattened input image size i.e. if input image size is [28x28], then v_length = 784.

<div class="code-head"><span>code</span>train.py</div>

```python
# user inputs
nb_epoch = 25
num_classes = 10
batch_size = 128
train_size = 60000
test_size = 10000
v_length = 784
```

### Prepare the dataset 

* Loading the MNIST dataset is done simply by calling <span class="coding">mnist.load_data()</span> function in Keras. It returns two tuples holding (train data and train label) in one tuple, and (test data and test label) in another tuple.
* If you run this function for the first time, it will download the MNIST dataset to a local folder <span class="coding">~/.keras/datasets/mnist.pkl.gz</span> which is 14.6 MB in size.
* After loading the dataset, we need to make it in a way that our model can understand. It means we need to analyze and pre-process the dataset.
	* **Reshaping** - This is needed because, in Deep Learning, we provide the raw pixel intensities of images as inputs to the neural nets. If you check the shape of original data and label, you see that each image has the dimension of [28x28]. If we flatten it, we will get 28x28=784 pixel intensities. This is achieved by using NumPy's reshape function.
	* **Data type** - After reshaping, we need to change the pixel intensities to <span class="coding">float32</span> datatype so that we have a uniform representation throughout the solution. As grayscale image pixel intensities are integers in the range [0-255], we can convert them to floating point representations using <span class="coding">.astype</span> function provided by NumPy.
	* **Normalize** - Also, we normalize these floating point values in the range (0-1) to improve computational efficiency as well as to follow the standards.

<div class="code-head"><span>code</span>train.py</div>

```python
# split the mnist data into train and test
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()
print("[INFO] train data shape: {}".format(trainData.shape))
print("[INFO] test data shape: {}".format(testData.shape))
print("[INFO] train samples: {}".format(trainData.shape[0]))
print("[INFO] test samples: {}".format(testData.shape[0]))
```

<div class="code-out">
[INFO] train data shape: (60000L, 28L, 28L) <br>
[INFO] test data shape: (10000L, 28L, 28L)  <br>
[INFO] train samples: 60000  <br>
[INFO] test samples: 10000 <br>
</div>

<div class="code-head"><span>code</span>train.py</div>

```python
# reshape the dataset
trainData = trainData.reshape(train_size, v_length)
testData = testData.reshape(test_size, v_length)
trainData = trainData.astype("float32")
testData = testData.astype("float32")
trainData /= 255
testData /= 255

print("[INFO] train data shape: {}".format(trainData.shape))
print("[INFO] test data shape: {}".format(testData.shape))
print("[INFO] train samples: {}".format(trainData.shape[0]))
print("[INFO] test samples: {}".format(testData.shape[0]))
```

<div class="code-out">
[INFO] train data shape: (60000L, 784L)  <br>
[INFO] test data shape: (10000L, 784L)  <br>
[INFO] train samples: 60000  <br>
[INFO] test samples: 10000 <br>
</div>

### One Hot Encoding
The class labels for our neural network to predict are numeric digits ranging from (0-9). As this is a multi-label classification problem, we need to represent these numeric digits into a binary form representation called as one-hot encoding. 

It simply means that if we have a digit, say 8, then we form a table of 10 columns (as we have 10 digits), and make all the cells zero, except 8. In Keras, we can easily transform numeric value to one-hot encoded representation using <span class="coding">np_utils.to_categorical</span> function, which takes in labels and number of class labels as input. 

For better understanding, see the one-hot encoded representation of 5, 8 and 3 below.

| d | b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8 | b9 |
|---|----|----|----|----|----|----|----|----|----|----|
| 5 | 0  | 0  | 0  | 0  | 0  | 1  | 0  | 0  | 0  | 0  |
| 8 | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 1  | 0  |
| 3 | 0  | 0  | 0  | 1  | 0  | 0  | 0  | 0  | 0  | 0  |

<div class="code-head"><span>code</span>train.py</div>

```python
# convert class vectors to binary class matrices --> one-hot encoding
mTrainLabels = np_utils.to_categorical(trainLabels, num_classes)
mTestLabels = np_utils.to_categorical(testLabels, num_classes)
```

### Create the model
We will use a simple Multi-Layer Perceptron (MLP) as our neural network model with 784 input neurons. 

Two hidden layers are used with 512 neurons in hidden layer 1 and 256 neurons in hidden layer 2, followed by a fully connected layer of 10 neurons for taking the probabilities of all the class labels. 

ReLU is used as the activation function for hidden layers and softmax is used as the activation function for output layer. 

After creating the model, a summary of the model is presented with different parameters involved.

We are still allowed to tune these parameters (called as hyperparameters) based on the model's performance. In fact, there are algorithms to get the best possible hyper-parameters for our model which could be read [here](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)){:target="_blank"}.

<div class="code-head"><span>code</span>train.py</div>

```python
# create the model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# summarize the model
model.summary()
```

### Compile the model
After creating the model, we need to compile the model for optimization and learning. We will use <span class="coding">categorical_crossentropy</span> as the loss function (as this is a multi-label classification problem), <span class="coding">adam</span> (gradient descent algorithm) as the optimizer and <span class="coding">accuracy</span> as our performance metric.

<div class="code-head"><span>code</span>train.py</div>

```python
# compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```

### Fit the model 
After compiling the model, we need to fit the model with the MNIST dataset. Using <span class="coding">model.fit</span> function, we can easily fit the created model. This function requires some arguments that we created above. Train data and train labels goes into 1st and 2nd position, followed by the <span class="coding">validation_data/validation_split</span>. Then comes <span class="coding">nb_epoch, batch_size</span> and <span class="coding">verbose</span>. Verbose is for debugging purposes. To view the history of our model or to analyse how our model gets trained with the dataset, we can use history object provided by Keras.

<div class="code-head"><span>code</span>train.py</div>

```python
# fit the model
history = model.fit(trainData, mTrainLabels, validation_data=(testData, mTestLabels), batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)
```

### Evaluate the model 
After fitting the model, the model can be evaluated on the unseen test data. Using <span class="coding">model.evaluate</span> function in Keras, we can give test data and test labels to the model and make predictions. We can also use matplotlib to visualize how our model reacts at different epochs on both training and testing data.

<div class="code-head"><span>code</span>train.py</div>

```python
# print the history keys
print(history.history.keys())

# evaluate the model
scores = model.evaluate(testData, mTestLabels, verbose=0)

# history plot for accuracy
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# history plot for accuracy
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# print the results
print("[INFO] test score - {}".format(scores[0]))
print("[INFO] test accuracy - {}".format(scores[1]))
```

<div class="code-out">
['acc', 'loss', 'val_acc', 'val_loss'] <br>
[INFO] test score - 0.0850750412192 <br>
[INFO] test accuracy - 0.9815 <br>
</div>

### Results
As you can see, our simple MLP model with just two hidden layers achieves a test accuracy of 98.15%, which is a great thing to achieve on our first attempt. Normally, training accuracy reaches around 90%, but test accuracy determines how well your model generalizes. So, it is important that you get good test accuracies for your problem.

<figure>
  <img src="/images/software/digits-recognition-mlp/results.jpg" class="typical-image" />
  <figcaption>Figure 2 a) Model Accuracy b) Model Loss</figcaption>
</figure>

### Complete code 

<div class="code-head"><span>code</span>train.py</div>

```python
# organize imports
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils

# fix a random seed for reproducibility
np.random.seed(9)

# user inputs
nb_epoch = 25
num_classes = 10
batch_size = 128
train_size = 60000
test_size = 10000
v_length = 784

# split the mnist data into train and test
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()
print("[INFO] train data shape: {}".format(trainData.shape))
print("[INFO] test data shape: {}".format(testData.shape))
print("[INFO] train samples: {}".format(trainData.shape[0]))
print("[INFO] test samples: {}".format(testData.shape[0]))

# reshape the dataset
trainData = trainData.reshape(train_size, v_length)
testData = testData.reshape(test_size, v_length)
trainData = trainData.astype("float32")
testData = testData.astype("float32")
trainData /= 255
testData /= 255

print("[INFO] train data shape: {}".format(trainData.shape))
print("[INFO] test data shape: {}".format(testData.shape))
print("[INFO] train samples: {}".format(trainData.shape[0]))
print("[INFO] test samples: {}".format(testData.shape[0]))

# convert class vectors to binary class matrices --> one-hot encoding
mTrainLabels = np_utils.to_categorical(trainLabels, num_classes)
mTestLabels = np_utils.to_categorical(testLabels, num_classes)

# create the model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# summarize the model
model.summary()

# compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# fit the model
history = model.fit(trainData, mTrainLabels, validation_data=(testData, mTestLabels), batch_size=batch_size, nb_epoch=nb_epoch, verbose=2)

# print the history keys
print(history.history.keys())

# evaluate the model
scores = model.evaluate(testData, mTestLabels, verbose=0)

# history plot for accuracy
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# history plot for accuracy
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "test"], loc="upper left")
plt.show()

# print the results
print("[INFO] test score - {}".format(scores[0]))
print("[INFO] test accuracy - {}".format(scores[1]))
```

### Testing the model
In order to test the model, we can use some images from the testing dataset. These are taken from the testing dataset because these images are unknown to our model, so that we can test our model's performance easily. We will grab few images from those 10,000 test images and make predictions using <span class="coding">model.predict_classes</span> function which takes in the flattened raw pixel intensities of the test image. 

<div class="code-head"><span>code</span>train.py</div>

```python
import matplotlib.pyplot as plt

# grab some test images from the test data
test_images = testData[1:5]

# reshape the test images to standard 28x28 format
test_images = test_images.reshape(test_images.shape[0], 28, 28)
print("[INFO] test images shape - {}".format(test_images.shape))

# loop over each of the test images
for i, test_image in enumerate(test_images, start=1):
	# grab a copy of test image for viewing
	org_image = test_image
	
	# reshape the test image to [1x784] format so that our model understands
	test_image = test_image.reshape(1,784)
	
	# make prediction on test image using our trained model
	prediction = model.predict_classes(test_image, verbose=0)
	
	# display the prediction and image
	print("[INFO] I think the digit is - {}".format(prediction[0]))
	plt.subplot(220+i)
	plt.imshow(org_image, cmap=plt.get_cmap('gray'))

plt.show()
```

<div class="code-out">
[INFO] test images shape - (4L, 28L, 28L) <br>
[INFO] I think the digit is - 2 <br>
[INFO] I think the digit is - 1 <br>
[INFO] I think the digit is - 0 <br>
[INFO] I think the digit is - 4 <br>
</div>

<figure>
  <img src="/images/software/digits-recognition-mlp/output.jpg" class="typical-image" />
  <figcaption>Figure 3. Testing MLP model using random test images</figcaption>
</figure>

### Summary

Thus, we have built a simple Multi-Layer Perceptron (MLP) to recognize handwritten digit (using MNIST dataset). Note that we haven't used Convolutional Neural Networks (CNN) yet. As I told earlier, this tutorial is to make us get started with Deep Learning. 

To build a system that is capable of recognizing our own hand-writing, we must take one step further to apply this approach i.e. we need to segment each digit in our hand-written image and then make our model to predict each segmented digit. 