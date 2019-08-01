---
layout: post
category: software
class: Keras
title: Using Keras Pre-trained Deep Learning models for your own dataset
description: Learn how to use state-of-the-art Deep Learning neural network architectures trained on ImageNet such as VGG16, VGG19, Inception-V3, Xception, ResNet50 for your own dataset with/without GPU acceleration.
permalink: software/flower-recognition-deep-learning
image: https://drive.google.com/uc?id=1fd6p-iNVWNKuPA88TSb0anIrIcv-d6d0
cardimage: https://drive.google.com/uc?id=1p_P1leDBQaVQqo4BmGwNywt9N_tAZpFD
---

<div class="git-showcase">
  <div>
  <a class="github-button" href="https://github.com/Gogul09" data-show-count="true" aria-label="Follow @Gogul09 on GitHub">Follow @Gogul09</a>
  </div>

  <div>
  <a class="github-button" href="https://github.com/Gogul09/flower-recognition/fork" data-icon="octicon-repo-forked" data-show-count="true" aria-label="Fork Gogul09/flower-recognition on GitHub">Fork</a>
  </div>

  <div>
  <a class="github-button" href="https://github.com/Gogul09/flower-recognition" data-icon="octicon-star" data-show-count="true" aria-label="Star Gogul09/flower-recognition on GitHub">Star</a>
  </div>  
</div>

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#beautiful-keras">Beautiful Keras</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#feature-extraction-using-convnets">Feature Extraction using ConvNets</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#keras-pre-trained-models">Keras Pre-trained models</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#gpu-acceleration">GPU Acceleration</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#dependencies">Dependencies</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#5-simple-steps-for-deep-learning">5 simple steps for Deep Learning</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#folder-structure">Folder Structure</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#training-dataset">Training Dataset</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#deep-learning-pipeline">Deep Learning Pipeline</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#show-me-the-numbers">Show me the numbers</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_11" href="#testing-on-new-images">Testing on new images</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_12" href="#issues-and-workarounds">Issues and Workarounds</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_13" href="#references">References</a></li>
  </ul>
</div>

<p class="hundred-days"><span>#100DaysOfMLCode</span></p>

<p class="intro-para">
In this blog post, we will quickly understand how to use state-of-the-art Deep Learning models in <a href="https://keras.io/" target="_blank">Keras</a> to solve a supervised image classification problem using our own dataset with/without GPU acceleration.
</p>

We will be using the pre-trained Deep Neural Nets trained on the ImageNet challenge that are made publicly available in Keras. We will specifically use [FLOWERS17](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/){:target="_blank"} dataset from the University of Oxford. 

The pre-trained models we will consider are VGG16, VGG19, Inception-v3, Xception, ResNet50, InceptionResNetv2 and MobileNet. Instead of creating and training deep neural nets from scratch (which takes time and involves many iterations), what if we use the pre-trained weights of these deep neural net architectures (trained on ImageNet dataset) and use it for our own dataset? 

Let's start feature extraction using Deep Convolutional Neural Networks! What we will be making at the end of this tutorial is shown below.

<figure>
  <img src="/images/software/pretrained-models/out.gif">
  <figcaption>Figure 1. Flower Species Recognition using Pretrained Deep Learning models.</figcaption>
</figure>

<div class="note">
  <p><b>Update (16/12/2017):</b> After installing Anaconda with Python 3.6 to work with TensorFlow in Windows 10, I found two additional pretrained models added to Keras applications module - InceptionResNetV2 and MobileNet. I have updated my code accordingly to enable these models to work for our own dataset.
  </p>
</div>

<div class="note">
  <p><b>Update (10/06/2018)</b>: If you use <a href="https://github.com/keras-team/keras/releases" target="_blank">Keras 2.2.0</a> version, then you will not find the <span class="coding">applications</span> module inside keras installed directory. Keras has externalized the <span class="coding">applications</span> module to a separate directory called <a href="https://github.com/keras-team/keras-applications" target="_blank">keras_applications</a> from where all the pre-trained models will now get imported. To make changes to any &lt;pre-trained_model&gt;.py file, simply go to the below directory where you will find all the pre-trained models .py files.
  </p>
</div>

<div class="note">
  <p><b>Update (16/12/2017):</b> You could also see the new MobileNet architecture achieves the best accuracy compared to other architectures. In addition, I found that MobileNet uses DepthwiseConvolution layers and has lesser number of parameters, reduced weights size and depth. More details about this can be found at - <a href="https://arxiv.org/pdf/1704.04861.pdf" target="_blank">MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications</a>.
  </p>
</div>

<h3 id="beautiful-keras">Beautiful Keras</h3>
[Keras](https://keras.io/){:target="_blank"} is an amazing library to quickly start Deep Learning for people entering into this field. Developed by François Chollet, it offers simple understandable functions and syntax to start building Deep Neural Nets right away instead of worrying too much on the programming part. Keras is a wrapper for Deep Learning libraries namely Theano and TensorFlow. I found the documentation and GitHub repo of Keras well maintained and easy to understand. If you know some technical details regarding Deep Neural Networks, then you will find the Keras documentation as the best place to learn.

<h3 id="feature-extraction-using-convnets">Feature Extraction using ConvNets</h3>
Traditional machine learning approach uses feature extraction for images using Global feature descriptors such as Local Binary Patterns (LBP), Histogram of Oriented Gradients (HoG), Color Histograms etc. or Local descriptors such as SIFT, SURF, ORB etc. These are hand-crafted features that requires domain level expertise.

But here comes Convolutional Neural Networks (CNN)! Instead of using hand-crafted features, Deep Neural Nets automatically learns these features from images in a hierarchical fashion. Lower layers learn low-level features such as Corners, Edges whereas middle layers learn color, shape etc. and higher layers learn high-level features representing the object in the image.

Instead of making a CNN as a model to classify images, what if we use it as a Feature Extractor by taking the activations available before the last fully connected layer of the network (i.e. *before* the final softmax classifier). These activations will be acting as the feature vector for a machine learning model (classifier) which further learns to classify it. This type of approach is well suited for Image Classification problems, where instead of training a CNN from scratch (which is time-consuming and tedious), a pre-trained CNN could be used as a Feature Extractor - [Transfer Learning](http://cs231n.github.io/transfer-learning/){:target="_blank"}.

<h3 id="keras-pre-trained-models">Keras Pre-trained Models</h3>
The Deep Neural Net architectures that won the ImageNet challenge are made publicly available in Keras including the model weights. Please check [this](https://keras.io/applications/){:target="_blank"} page for more details regarding each neural network architecture. Please be aware of the input <span class="coding">image_size</span> that are given to each model as we will be transforming our input images to these sizes. Below are the pre-trained models available in Keras at the time of writing this post.

* Xception
* VGG16
* VGG19
* ResNet50
* InceptionV3
* InceptionResNetV2
* MobileNet

<div class="note">
<p>
<b>Update (16/12/2017):</b> After installing Anaconda with Python 3.6 to work with TensorFlow in Windows 10, I found two additional pretrained models added to Keras applications module - <b>InceptionResNetV2</b> and <b>MobileNet</b>. I have updated my code accordingly to enable these models to work for our own dataset.
</p>
</div>

The [applications](https://github.com/fchollet/keras/tree/master/keras/applications){:target="_blank"} module of Keras provides all the necessary functions needed to use these pre-trained models right away.

Below is the table that shows image size, weights size, top-1 accuracy, top-5 accuracy, no.of.parameters and depth of each deep neural net architecture available in Keras.

<table class="tg">
  <tr>
    <th class="tg-yw4l">Model</th>
    <th class="tg-yw4l">Image size</th>
    <th class="tg-yw4l">Weights size</th>
    <th class="tg-yw4l">Top-1 accuracy</th>
    <th class="tg-yw4l">Top-5 accuracy</th>
    <th class="tg-yw4l">Parameters</th>
    <th class="tg-yw4l">Depth</th>
  </tr>
  <tr>
    <td class="tg-yw4l">Xception</td>
    <td class="tg-yw4l">299 x 299</td>
    <td class="tg-yw4l">88 MB</td>
    <td class="tg-yw4l">0.790</td>
    <td class="tg-yw4l">0.945</td>
    <td class="tg-yw4l">22,910,480</td>
    <td class="tg-yw4l">126</td>
  </tr>
  <tr>
    <td class="tg-yw4l">VGG16</td>
    <td class="tg-yw4l">224 x 224</td>
    <td class="tg-yw4l">528 MB</td>
    <td class="tg-yw4l">0.715</td>
    <td class="tg-yw4l">0.901</td>
    <td class="tg-yw4l">138,357,544</td>
    <td class="tg-yw4l">23</td>
  </tr>
  <tr>
    <td class="tg-yw4l">VGG19</td>
    <td class="tg-yw4l">224 x 224</td>
    <td class="tg-yw4l">549 MB</td>
    <td class="tg-yw4l">0.727</td>
    <td class="tg-yw4l">0.910</td>
    <td class="tg-yw4l">143,667,240</td>
    <td class="tg-yw4l">26</td>
  </tr>
  <tr>
    <td class="tg-yw4l">ResNet50</td>
    <td class="tg-yw4l">224 x 224 </td>
    <td class="tg-yw4l">99 MB</td>
    <td class="tg-yw4l">0.759</td>
    <td class="tg-yw4l">0.929</td>
    <td class="tg-yw4l">25,636,712</td>
    <td class="tg-yw4l">168</td>
  </tr>
  <tr>
    <td class="tg-yw4l">InceptionV3</td>
    <td class="tg-yw4l">299 x 299</td>
    <td class="tg-yw4l">92 MB</td>
    <td class="tg-yw4l">0.788</td>
    <td class="tg-yw4l">0.944</td>
    <td class="tg-yw4l">23,851,784</td>
    <td class="tg-yw4l">159</td>
  </tr>
  <tr>
    <td class="tg-yw4l">Inception<br>ResNetV2</td>
    <td class="tg-yw4l">299 x 299</td>
    <td class="tg-yw4l">215 MB</td>
    <td class="tg-yw4l" style="font-weight: bold;">0.804</td>
    <td class="tg-yw4l" style="font-weight: bold;">0.953</td>
    <td class="tg-yw4l">55,873,736</td>
    <td class="tg-yw4l">572</td>
  </tr>
  <tr>
    <td class="tg-yw4l">MobileNet</td>
    <td class="tg-yw4l">224 x 224</td>
    <td class="tg-yw4l" style="font-weight: bold;">17 MB</td>
    <td class="tg-yw4l">0.665</td>
    <td class="tg-yw4l">0.871</td>
    <td class="tg-yw4l" style="font-weight: bold;">4,253,864</td>
    <td class="tg-yw4l">88</td>
  </tr>
</table>

<div class="note" style="margin-top: 20px;"><p>
<b>Note:</b> All the above architectures can be created using either Theano or TensorFlow except <b>Xception</b> and <b>MobileNet</b> (as they depend on Separable Convolutions and Depthwise Convolutions which is available only in TensorFlow).</p>
</div>

<h3 id="gpu-acceleration">GPU acceleration</h3>
GPUs are the beasts when it comes to Deep Learning and no wonder if you enable GPU in your computer, you can speed up feature extraction as well as training process. Steps to activate GPU acceleration to train deep neural nets in Windows 10 are provided in my [blog post](https://gogul09.github.io/software/deep-learning-windows){:target="_blank"}.

<h3 id="dependencies">Dependencies</h3>
You will need the following Python packages to run the code provided in this tutorial.

* Theano or TensorFlow
* Keras
* NumPy
* scikit-learn
* matplotlib
* seaborn
* h5py

<div class="note" style="margin-bottom: 0px !important">
  <p><b>Note</b>: If you don't have an environment to do Deep Learning in Windows or Linux, please make sure you use the below two links to do that and then follow on.</p>
  <ul style="margin-bottom: 0px !important">
    <li><a href="https://gogul09.github.io/software/deep-learning-windows" target="_blank">Deep Learning Environment Setup (Windows)</a></li>
    <li><a href="https://gogul09.github.io/software/deep-learning-linux" target="_blank">Deep Learning Environment Setup for (Linux)</a></li>
  </ul>
</div>

<h3 id="5-simple-steps-for-deep-learning">5 simple steps for Deep Learning</h3>

1. Prepare the training dataset with flower images and its corresponding labels.
2. Specify your own configurations in <span class="coding">conf.json</span> file.
3. Extract and store features from the last fully connected layers (or intermediate layers) of a pre-trained Deep Neural Net (CNN) using <span class="coding">extract_features.py</span>.
4. Train a Machine Learning model such as Logisitic Regression using these CNN extracted features and labels using <span class="coding">train.py</span>.
5. Evaluate the trained model on unseen data and make further optimizations if necessary.

<h3 id="folder-structure">Folder structure</h3>

<div class="code-head"><span>rule</span>flower recognition</div>
```python
|--flower_recognition
|--|--conf
|--|--|--conf.json
|--|--dataset
|--|--|--train
|--|--|--test
|--|--output
|--|--|--flower_17
|--|--|--|--inceptionv3
|--|--|--|--|--classifier.cPickle
|--|--|--|--|--labels.h5
|--|--|--|--|--features.h5
|--|--|--|--|--results.txt
|--|--|--|--vgg16
|--|--|--|--|--classifier.cPickle
|--|--|--|--|--labels.h5
|--|--|--|--|--features.h5
|--|--|--|--|--results.txt
|--|--|--|--vgg19
|--|--|--|--|--classifier.cPickle
|--|--|--|--|--labels.h5
|--|--|--|--|--features.h5
|--|--|--|--|--results.txt
|--|--|--|--resnet50
|--|--|--|--|--classifier.cPickle
|--|--|--|--|--labels.h5
|--|--|--|--|--features.h5
|--|--|--|--|--results.txt
|--|--|--|--xception
|--|--|--|--|--classifier.cPickle
|--|--|--|--|--labels.h5
|--|--|--|--|--features.h5
|--|--|--|--|--results.txt
|--|--|--|--inceptionresnetv2
|--|--|--|--|--classifier.cPickle
|--|--|--|--|--labels.h5
|--|--|--|--|--features.h5
|--|--|--|--|--results.txt
|--|--|--|--mobilenet
|--|--|--|--|--classifier.cPickle
|--|--|--|--|--labels.h5
|--|--|--|--|--features.h5
|--|--|--|--|--results.txt
|--|--extract_features.py
|--|--train.py
```

<h3 id="training-dataset">Training dataset</h3>

Download the FLOWER17 dataset from [this](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/){:target="_blank"} website. Unzip the file and you will see all the 1360 images listed in one single folder named ***.jpg**. The FLOWERS17 dataset has 1360 images of 17 flower species classes with 80 images per class. 

To build our training dataset, we need to create a master folder named **dataset**, inside which we need to create two more folders namely **train** and **test**. Inside **train** folder, we need to create 17 folders corresponding to the flower species labels. 

To automate this task, I have a [script](https://github.com/Gogul09/flower-recognition/blob/master/organize_flowers17.py){:target="_blank"} that takes in input path that has all the 1360 images and dumps 17 folders inside **train** folder. In those 17 folders, each folder will be having 80 flower images belonging to that folder name. Below is the screenshot of how to organize our training dataset as well as the output folder to store features, labels, results and classifier.

<figure>
  <img src="/images/software/pretrained-models/organize-dataset.png">
  <figcaption>Figure 2. Organizing FLOWER17 Training Dataset</figcaption>
</figure>

Script to organize training dataset is given below. Please be aware of the <span class="coding">input_path</span> and <span class="coding">output_path</span> that you give to create folders and store the images.

<div class="code-head"><span>code</span>organize_flowers17.py</div>

```python
# organize imports
import os
import glob
import datetime

# print start time
print("[INFO] program started on - " + str(datetime.datetime.now))

# get the input and output path
input_path  = "G:\\workspace\\machine-intelligence\\deep-learning\\flower-recognition\\17flowers\\jpg"
output_path = "G:\\workspace\\machine-intelligence\\deep-learning\\flower-recognition\\dataset\\train"

# get the class label limit
class_limit = 17

# take all the images from the dataset
image_paths = glob.glob(input_path + "\\*.jpg")

# variables to keep track
label = 0
i = 0
j = 80

# flower17 class names
class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
			   "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
			   "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
			   "windflower", "pansy"]

# change the current working directory
os.chdir(output_path)

# loop over the class labels
for x in range(1, class_limit+1):
	# create a folder for that class
	os.system("mkdir " + class_names[label])
	# get the current path
	cur_path = output_path + "\\" + class_names[label] + "\\"
	# loop over the images in the dataset
	for image_path in image_paths[i:j]:
		original_path = image_path
		image_path = image_path.split("\\")
		image_path = image_path[len(image_path)-1]
		os.system("copy " + original_path + " " + cur_path + image_path)
	i += 80
	j += 80
	label += 1

# print end time
print("[INFO] program ended on - " + str(datetime.datetime.now))
```

<h3 id="deep-learning-pipeline">Deep Learning pipeline</h3>

#### 1. conf.json
This is the configuration file or the settings file we will be using to provide inputs to our system. This is just a <span class="coding">json</span> file which is a key-value pair file format to store data effectively.

* The <span class="coding">model</span> key takes in any of these parameters - <span class="coding">inceptionv3</span>, <span class="coding">resnet50</span>, <span class="coding">vgg16</span>, <span class="coding">vgg19</span>, <span class="coding">xception</span>, <span class="coding">inceptionresnetv2</span> and <span class="coding">mobilenet</span>.
* The <span class="coding">weights</span> key takes the value <span class="coding">imagenet</span> specifying that we intend to use weights from imagenet. You can also set this to <span class="coding">None</span> if you wish to train the network from scratch.
* The <span class="coding">include_top</span> key takes the value <span class="coding">false</span> specifying that we are going to take the features from any intermediate layer of the network. You can set this to <span class="coding">true</span> if you want to extract features before the fully connected layers.
* The <span class="coding">test_size</span> key takes the value in the range (0.10 - 0.90). This is to make a split between your overall data into training and testing.
* The <span class="coding">seed</span> key takes any value to reproduce same results everytime you run the code.
* The <span class="coding">num_classes</span> specifies the number of classes or labels considered for the image classification problem.

<div class="code-head"><span>code</span>conf.json</div>

```json
{
  "model"           : "inceptionv3",
  "weights"         : "imagenet",
  "include_top"     : false,

  "train_path"      : "dataset/train",
  "test_path"       : "dataset/test",
  "features_path"   : "output/flowers_17/inceptionv3/features.h5",
  "labels_path"     : "output/flowers_17/inceptionv3/labels.h5",
  "results"         : "output/flowers_17/inceptionv3/results.txt",
  "classifier_path" : "output/flowers_17/inceptionv3/classifier.pickle",
  "model_path"      : "output/flowers_17/inceptionv3/model",

  "test_size"       : 0.10,
  "seed"            : 9,
  "num_classes"     : 17
}
```

Here, I have decided to use <span class="coding">inceptionv3</span> architecture of GoogleNet pre-trained on <span class="coding">imagenet</span> including the top layers. You can extract features from any arbitrary layer using the layer name (eg: <span class="coding">flatten</span>), by checking the <span class="coding">.py</span> file of each of the model residing inside the <span class="coding">applications</span> directory of Keras.

<div class="note">
<p><b>Update (10/06/2018)</b>: If you use <a href="https://github.com/keras-team/keras/releases" target="_blank">Keras 2.2.0</a> version, then you will not find the <span class="coding">applications</span> module inside keras installed directory. Keras has externalized the <span class="coding">applications</span> module to a separate directory called <a href="https://github.com/keras-team/keras-applications" target="_blank">keras_applications</a> from where all the pre-trained models will now get imported. To make changes to any <b>&lt;pre-trained_model&gt;.py</b> file, simply go to the below directory where you will find all the pre-trained models <b>.py</b> files.</p>
</div>

<div class="code-head"><span>path</span>New applications module (Keras 2.2.0)</div>
```
"python_installation_directory" -> "Lib" -> "site-packages" -> "keras_applications"
```

As we are using FLOWERS17 dataset from the University of Oxford, I have specified the <span class="coding">num_classes</span> as 17. We will have a <span class="coding">test_size</span> of 0.10, which means we use 90% of data for training and 10% for testing.

<figure>
  <img src="/images/software/plants-species/flowers17_data.jpg">
  <figcaption>Figure 3. FLOWERS17 dataset from University of Oxford</figcaption>
</figure>

#### 2. Feature Extraction using ConvNets

<div class="code-head"><span>code</span>extract_features.py</div>

```python
# filter warnings
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.mobilenet import MobileNet, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input

# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import datetime
import time

# load the user configs
with open('conf/conf.json') as f:    
  config = json.load(f)

# config variables
model_name    = config["model"]
weights     = config["weights"]
include_top   = config["include_top"]
train_path    = config["train_path"]
features_path   = config["features_path"]
labels_path   = config["labels_path"]
test_size     = config["test_size"]
results     = config["results"]
model_path    = config["model_path"]

# start time
print("[STATUS] start time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
start = time.time()

# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not
if model_name == "vgg16":
  base_model = VGG16(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
  image_size = (224, 224)
elif model_name == "vgg19":
  base_model = VGG19(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('fc1').output)
  image_size = (224, 224)
elif model_name == "resnet50":
  base_model = ResNet50(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('flatten').output)
  image_size = (224, 224)
elif model_name == "inceptionv3":
  base_model = InceptionV3(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
  model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
  image_size = (299, 299)
elif model_name == "inceptionresnetv2":
  base_model = InceptionResNetV2(include_top=include_top, weights=weights, input_tensor=Input(shape=(299,299,3)))
  model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
  image_size = (299, 299)
elif model_name == "mobilenet":
  base_model = MobileNet(include_top=include_top, weights=weights, input_tensor=Input(shape=(224,224,3)), input_shape=(224,224,3))
  model = Model(input=base_model.input, output=base_model.get_layer('custom').output)
  image_size = (224, 224)
elif model_name == "xception":
  base_model = Xception(weights=weights)
  model = Model(input=base_model.input, output=base_model.get_layer('avg_pool').output)
  image_size = (299, 299)
else:
  base_model = None

print("[INFO] successfully loaded base model and model...")

# path to training dataset
train_labels = os.listdir(train_path)

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
le.fit([tl for tl in train_labels])

# variables to hold features and labels
features = []
labels   = []

# loop over all the labels in the folder
count = 1
for i, label in enumerate(train_labels):
  cur_path = train_path + "/" + label
  count = 1
  for image_path in glob.glob(cur_path + "/*.jpg"):
    img = image.load_img(image_path, target_size=image_size)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = model.predict(x)
    flat = feature.flatten()
    features.append(flat)
    labels.append(label)
    print("[INFO] processed - " + str(count))
    count += 1
  print("[INFO] completed label - " + label)

# encode the labels using LabelEncoder
le = LabelEncoder()
le_labels = le.fit_transform(labels)

# get the shape of training labels
print("[STATUS] training labels: {}".format(le_labels))
print("[STATUS] training labels shape: {}".format(le_labels.shape))

# save features and labels
h5f_data = h5py.File(features_path, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(features))

h5f_label = h5py.File(labels_path, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(le_labels))

h5f_data.close()
h5f_label.close()

# save model and weights
model_json = model.to_json()
with open(model_path + str(test_size) + ".json", "w") as json_file:
  json_file.write(model_json)

# save weights
model.save_weights(model_path + str(test_size) + ".h5")
print("[STATUS] saved model and weights to disk..")

print("[STATUS] features and labels saved..")

# end time
end = time.time()
print("[STATUS] end time - {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
```

* **How to run this script?**<br>
  Open up a command prompt and go into the folder where you saved this file. Type <span class="coding">python extract_features.py</span>. It will extract all the features from the images in your dataset and store it in HDF5 format locally.

* **What this script does?**<br>
  The pre-trained models are loaded from the <span class="coding">application</span> module of Keras library and the model is constructed based on the user specified configurations in the <span class="coding">conf.json</span> file. After that, features are extracted from the user-specified layer in the model pre-trained with ImageNet dataset. These features along with its labels are stored locally using HDF5 file format. Also, the model and the weights are saved just to show that these could also be done in Keras.

The below table shows the **feature vector size** for each image for a particular deep neural net model that I used.

| Model             | Feature vector size |
|-------------------|---------------------|
| VGG16             | (1, 4096)           |
| VGG19             | (1, 4096)           |
| InceptionV3       | (1, 131072)         |
| ResNet50          | (1, 2048)           |
| InceptionResNetV2 | (1, 98304)          |
| Xception          | (1, 2048)           |
| MobileNet         | (1, 50176)          |

#### 3. Training a machine learning model

<div class="code-head"><span>code</span>train.py</div>

```python
# organize imports
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# load the user configs
with open('conf/conf.json') as f:    
  config = json.load(f)

# config variables
test_size     = config["test_size"]
seed      = config["seed"]
features_path   = config["features_path"]
labels_path   = config["labels_path"]
results     = config["results"]
classifier_path = config["classifier_path"]
train_path    = config["train_path"]
num_classes   = config["num_classes"]
classifier_path = config["classifier_path"]

# import features and labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print("[INFO] features shape: {}".format(features.shape))
print("[INFO] labels shape: {}".format(labels.shape))

print("[INFO] training started...")
# split the training and testing data
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size,
                                                                  random_state=seed)

print("[INFO] splitted train and test data...")
print("[INFO] train data  : {}".format(trainData.shape))
print("[INFO] test data   : {}".format(testData.shape))
print("[INFO] train labels: {}".format(trainLabels.shape))
print("[INFO] test labels : {}".format(testLabels.shape))

# use logistic regression as the model
print("[INFO] creating model...")
model = LogisticRegression(random_state=seed)
model.fit(trainData, trainLabels)

# use rank-1 and rank-5 predictions
print("[INFO] evaluating model...")
f = open(results, "w")
rank_1 = 0
rank_5 = 0

# loop over test data
for (label, features) in zip(testLabels, testData):
  # predict the probability of each class label and
  # take the top-5 class labels
  predictions = model.predict_proba(np.atleast_2d(features))[0]
  predictions = np.argsort(predictions)[::-1][:5]

  # rank-1 prediction increment
  if label == predictions[0]:
    rank_1 += 1

  # rank-5 prediction increment
  if label in predictions:
    rank_5 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100

# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data
preds = model.predict(testData)

# write the classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# dump classifier to file
print("[INFO] saving model...")
pickle.dump(model, open(classifier_path, 'wb'))

# display the confusion matrix
print("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))

# plot the confusion matrix
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm,
            annot=True,
            cmap="Set2")
plt.show()
```

* **How to run this script?**<br>
  Open up a command prompt and go to the folder where you saved this file. Type <span class="coding">python train.py</span>. It will train the Logistic Regression classifier with the features and labels extracted and stored locally. Finally, it prints the RANK-1 and RANK-5 accuracies of the model on unseen test data.

* **What this script does?**<br>
  The features and labels extracted from your dataset are loaded. Logistic Regression model is created to train these features and labels. The trained model could then be used to predict the label of unseen images. I have added some code to visualize the confusion matrix of the trained model on unseen test data splitted using scikit-learn and seaborn.

<h3 id="show-me-the-numbers">Show me the numbers</h3>

The below tables shows the accuracies obtained for each pretrained model used to extract features from FLOWERS17 dataset using different configuration settings.

#### Result-1

  * test_size  : 0.10
  * classifier : Logistic Regression

| Model        | Rank-1 accuracy | Rank-5 accuracy |
|--------------|-----------------|-----------------|
| Xception     | 97.06%      	 | 99.26%      	   |
| InceptionV3 | 96.32%          | 99.26%          |
| VGG16        | 85.29%          | 98.53%          |
| VGG19        | 88.24%          | 99.26%          |
| ResNet50     | 56.62%          | 90.44%          |
| MobileNet     | <b>98.53%</b>          | <b>100.00%</b>         |
| Inception<br>ResNetV2     | 91.91%          | 98.53%          |

#### Result-2

  * test_size  : 0.30
  * classifier : Logistic Regression

| Model        | Rank-1 accuracy | Rank-5 accuracy |
|--------------|-----------------|-----------------|
| Xception     | 93.38%          | <b>99.75%</b>          |
| InceptionV3 | <b>96.81%</b>          | 99.51%          |
| VGG16        | 88.24%          | 99.02%          |
| VGG19        | 88.73%          | 98.77%          |
| ResNet50     | 59.80%          | 86.52%          |
| MobileNet     | 96.32%         | <b>99.75%</b>         |
| Inception<br>ResNetV2     | 88.48%          | 99.51%          |

Notice how InceptionV3 outperforms the other Deep Neural Net architectures. This could mainly be due to the presence of 9 network-in-a-network modules codenamed as Inception modules which applies different convolutional filter sizes parallely to an input volume and concatenates the result at output. More details about this can be found in this astounding paper by C. Szegedy et al. - [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842){:target="_blank"}.

<div class="note">
<p>
<b>Update (16/12/2017):</b> You could also see the new MobileNet architecture achieves the best accuracy compared to other architectures. In addition, I found that MobileNet uses DepthwiseConvolution layers and has lesser number of parameters, reduced weights size and depth. More details about this can be found at - <a href="https://arxiv.org/pdf/1704.04861.pdf" target="_blank">MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications</a>.	
</p>
</div>

Thus, we have built our own Flower Species Recognition System using Deep Neural Nets and Keras. Our system is able to obtain much higher accuracies than state-of-the-art accuracies (which mostly used hand-crafted features for shape, color and texture representations) in this FLOWERS17 dataset. Using this procedure, you could use these pretrained models for your own image dataset and reduce the time consumed to construct a deep neural net from scratch.

<h3 id="testing-on-new-images">Testing on new images</h3>

To test on new flower images, we need to have some test images in **dataset/test** folder. Please use [this](https://github.com/Gogul09/flower-recognition/blob/master/test.py){:target="_blank"} script to make predictions on unseen test images.

---

<h3 id="issues-and-workarounds">Issues and Workarounds <span style="font-size:12px; font-weight: 100; color: #a7a5a5;"> <br> (Updated on 10/06/2018)</span></h3>

* <b style="color: #cf2321">Intel MKL FATAL ERROR: Cannot load mkl_intel_thread.dll</b><br>
When I installed Anaconda with Python 3.6 to enable TensorFlow on Windows 10, I got this error. Tried googling the error and this [link](https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/SnY1Uazkcew){:target="_blank"} gave me a working solution.

* <b style="color: #cf2321">no such layer: flatten or custom</b><br>
This error is the common one found when trying to add <span class="coding">Flatten()</span> in any of the model's .py file. Please update Keras to the latest version. Completely uninstall Keras and reinstall it using pip. Now, get into the similar directory shown below where you have installed Anaconda.

<div class="code-head"><span>path</span>Anaconda install dir</div>
```python
C:\deeplearning\anaconda2\Lib\site-packages\keras\applications\
```

Inside this directory, you can see all the pre-trained models <span class="coding">.py</span> file. If you use InceptionV3 as the model, then open <span class="coding">inception_v3.py</span>. 

Don't forget to add the below code on top where imports are written. 

<div class="code-head"><span>code</span>Add in "&lt;model&gt;.py"</div>
```python
from ..layers import Flatten
```

Next, go to the place where you can find the final <span class="coding">Dense()</span> layer. Normally, we need to perform <span class="coding">Flatten()</span> before the last fully connected layer. This is because the final <span class="coding">Dense()</span> layer has the number of classes in ImageNet challenge which is typically 1000. We could take these 1000 activations as (1, 1000) feature vector for a single image. But taking features from intermediate layers makes our classifier learn better.

This is how I inserted <span class="coding">Flatten()</span> layer to get features from InceptionV3 model. Notice that I have set <span class="coding">include_top</span> as <span class="coding">False</span>. This gave me a feature vector of size (1, 131072). 

<div class="code-head"><span>code</span>Add in "&lt;model&gt;.py"</div>
```python
...
...
if include_top:
        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        x = Flatten(name='custom')(x)
...
...
```

* <b style="color: #cf2321">couldn't see Dense() layer in "model_name".py applications folder</b><br>
This issue is seen in [Keras 2.2.0](https://github.com/keras-team/keras/releases){:target="_blank"} version update. Keras has externalized the <span class="coding">applications</span> module to "[keras_applications](https://github.com/keras-team/keras-applications){:target="_blank"}" from where all the pre-trained models are getting imported. To make changes to the <span class="coding">pre-trained_model.py</span> file, simply go to the below directory where you will find all the pre-trained models ".py" files.

<div class="code-head"><span>path</span>New applications module</div>
```
"python_installation_directory" -> "Lib" -> "site-packages" -> "keras_applications"
```

<figure>
  <img src="/images/software/pretrained-models/keras_version_update.png">
  <figcaption>Figure 4. Keras 2.2.0 version update</figcaption>
</figure>

In case if you want to keep the previous Keras version, simply do the following two commands.

<div class="code-head"><span>cmd</span>Reinstall Keras</div>
```
pip uninstall keras
pip install keras==2.1.2
```

### References

1. [Keras - Official Documentation](https://keras.io/){:target="_blank"}
2. [Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html){:target="_blank"}
3. [A Comprehensive guide to Fine-tuning Deep Learning Models in Keras (Part I)](https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html){:target="_blank"}
4. [A Comprehensive guide to Fine-tuning Deep Learning Models in Keras (Part II)](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html){:target="_blank"}
5. [17 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/){:target="_blank"}
6. [102 Category Flower Dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/){:target="_blank"} 