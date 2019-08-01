---
layout: post
category: software
class: Computer Vision
title: Image Classification using Python and Scikit-learn
description: Learn how to use Global Feature Descriptors such as RGB Color Histograms, Hu Moments and Haralick Texture to classify Flower species using different Machine Learning classifiers available in scikit-learn.
author: Gogul Ilango
permalink: software/image-classification-python
image: https://drive.google.com/uc?id=1QMdxBIeMhIfMgBmyuod1aWNaIim54rba
--- 

<div class="git-showcase">
  <div>
    <a class="github-button" href="https://github.com/Gogul09" data-show-count="true" aria-label="Follow @Gogul09 on GitHub">Follow @Gogul09</a>
  </div>

  <div>
    <a class="github-button" href="https://github.com/Gogul09/image-classification-python/fork" data-icon="octicon-repo-forked" data-show-count="true" aria-label="Fork Gogul09/image-classification-python on GitHub">Fork</a>
  </div>

  <div>
    <a class="github-button" href="https://github.com/Gogul09/image-classification-python" data-icon="octicon-star" data-show-count="true" aria-label="Star Gogul09/image-classification-python on GitHub">Star</a>
  </div>  
</div>

<div class="sidebar_tracker" id="sidebar_tracker">
   <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
   <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
   <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#project-idea">Project Idea</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#classification-problem">Classification Problem</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#feature-extraction">Feature Extraction</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#global-feature-descriptors">Global Feature Descriptors</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#local-feature-descriptors">Local Feature Descriptors</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#combining-global-features">Combining Global Features</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#flowers-17-dataset">FLOWERS-17 dataset</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#global-feature-extraction">Global Feature Extraction</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#organizing-dataset">Organizing Dataset</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#functions-for-global-feature-descriptors">Functions for global feature descriptors</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_11" href="#training-classifiers">Training classifiers</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_12" href="#testing-the-best-classifier">Testing the best classifier</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_13" href="#improving-classifier-accuracy">Improving classifier accuracy</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_14" href="#resources">Resources</a></li>
  </ul>
</div>

**The ability of a machine learning model to classify or label an image into its respective class with the help of learned features from hundreds of images is called as Image Classification.**

<div class="note">
  <p><b>Note</b>: This tutorial is specific to <b>Windows</b> environment. Please modify code accordingly to work in other environments such as Linux and Max OS.</p>
</div>

This is typically a supervised learning problem where we humans must provide training data (set of images along with its labels) to the machine learning model so that it learns how to discriminate each image (by learning the pattern behind each image) with respect to its label.

> **Update (03/07/2019)**: As Python2 faces [end of life](https://pythonclock.org/){:target="_blank"}, the below code only supports **Python3**.

In this post, we will look into one such image classification problem namely **Flower Species Recognition**, which is a hard problem because there are millions of flower species around the world. As we know machine learning is all about learning from past data, we need huge dataset of flower images to perform real-time flower species recognition. Without worrying too much on real-time flower recognition, we will learn how to perform a simple image classification task using computer vision and machine learning algorithms with the help of Python.

A short clip of what we will be making at the end of the tutorial 😊

<figure>
    <img src="/images/software/plants-species/output.gif">
    <figcaption style="text-align: center;">Flower Species Recognition - Watch the full video <a href="https://www.youtube.com/watch?v=_XHNPN1hzfk" target="_blank">here</a></figcaption>
</figure>

<div class="note">
<p><b>Update</b>: After reading this post, you could look into my <a href="https://gogul09.github.io/software/flower-recognition-deep-learning" target="_blank">post</a> on how to use state-of-the-art pretrained deep learning models such as Inception-V3, Xception, VGG16, VGG19, ResNet50, InceptionResNetv2 and MobileNet to this flower species recognition problem.</p>
</div> 

<figure>
    <img src="/images/software/plants-species/garden.jpg">
    <figcaption>Figure 1. Can we label each flower with it's name?<br>
        <span style="font-size: 9px">Image taken from <a href="http://www.artistic-law.com/garden-tag-beautiful-nature-day-water-park-hd-background-for-wallpapers-page-misc-desktop/colorful-garden-flowers-hd-free-wallpapers/" target="_blank">here</a></span>
    </figcaption>
</figure>

### Project Idea

What if

* You build an intelligent system that was trained with massive dataset of flower/plant images.
* Your system predicts the label/class of the flower/plant using Computer Vision techniques and Machine Learning algorithms.
* Your system searches the web for all the flower/plant related data after predicting the label/class of the captured image.
* Your system helps gardeners and farmers to increase their productivity and yield with the help of automating tasks in garden/farm.
* Your system applies the recent technological advancements such as Internet of Things (IoT) and Machine Learning in the agricultural domain.
* You build such a system for your home or your garden to monitor your plants using a Raspberry Pi.

> All the above scenarios need a common task to be done at the first place - **Image Classification**. 

Yeah! It is **classifying** a flower/plant into it's corresponding class or category. For example, when our awesome intelligent assistant looks into a Sunflower image, it must label or classify it as a "Sunflower".

### Classification Problem
Plant or Flower Species Classification is one of the most challenging and difficult problems in Computer Vision due to a variety of reasons.

<b>Availability of plant/flower dataset</b><br>
Collecting plant/flower dataset is a time-consuming task. You can visit the <a href="#dataset">links</a> provided at the bottom of this post where I have collected all the publicly available plant/flower datasets around the world. Although traning a machine with these dataset might help in some scenerios, there are still more problems to be solved.

<b>Millions of plant/flower species around the world</b><br>
This is something very interesting to look from a machine learning point of view. When I looked at the numbers in this [link](https://www.currentresults.com/Environment-Facts/Plants-Animals/estimate-of-worlds-total-number-of-species.php){:target="_blank"}, I was frightened. Because, to accomodate every such species, we need to train our model with such large number of images with its labels. We are talking about 6 digit class labels here for which we need tremendous computing power (GPU farms).

<b>High inter-class as well as intra-class variation</b><br>
What we mean here is that "Sunflower" might be looking similar to a "Daffodil" in terms of color. This becomes an inter-class variation problem. Similarly, sometimes a single "Sunflower" image might have differences within it's class itself, which boils down to intra-class variation problem.

<b>Fine-grained classification problem</b><br>
It means our model must not look into the image or video sequence and find <i>"Oh yes! there is a flower in this image"</i>. It means our model must tell <i>"Yeah! I found a flower in this image and I can tell you it's a tulip"</i>.

<b>Segmentation, View-point, Occlusion, Illumination and the list goes on..</b><br>
Segmenting the plant/flower region from an image is a challenging task. This is because we might need to remove the unwanted background and take only the foreground object (plant/flower) which is again a difficult thing due to the shape of plant/flower.

<figure>
    <img src="https://farm5.staticflickr.com/4018/4651874181_2dbe64b3fb_b.jpg">
    <figcaption>Figure 2. How will our model segment and classify flowers here?<br>
        <span style="font-size: 9px">Image taken from <a href="https://www.flickr.com/photos/fourseasonsgarden/4651874181" target="_blank">here</a></span>
    </figcaption>
</figure>

### Feature Extraction

Features are the information or list of numbers that are extracted from an image. These are real-valued numbers (integers, float or binary). There are a wider range of feature extraction algorithms in Computer Vision.

When deciding about the features that could quantify plants and flowers, we could possibly think of <span class="light">Color, Texture</span> and <span class="light">Shape</span> as the primary ones. This is an obvious choice to globally quantify and represent the plant or flower image. 

But this approach is less likely to produce good results, if we choose only one feature vector, as these species have many attributes in common like **sunflower** will be similar to **daffodil** in terms of color and so on. So, we need to quantify the image by combining different feature descriptors so that it describes the image more **effectively**.

<figure>
    <img src="/images/software/plants-species/feature_extraction.jpg">
    <figcaption>Figure 3. Feature Extraction</figcaption>
</figure>

### Global Feature Descriptors

These are the feature descriptors that quantifies an image globally. These don't have the concept of [interest points](https://en.wikipedia.org/wiki/Interest_point_detection){:target="_blank"} and thus, takes in the entire image for processing. Some of the commonly used global feature descriptors are
* **Color**   - Color Channel Statistics (Mean, Standard Deviation) and [Color Histogram](https://en.wikipedia.org/wiki/Color_histogram){:target="_blank"}
* **Shape**   - [Hu Moments](https://en.wikipedia.org/wiki/Image_moment){:target="_blank"}, [Zernike Moments](https://en.wikipedia.org/wiki/Zernike_polynomials){:target="_blank"}
* **Texture** - [Haralick Texture](http://haralick.org/journals/TexturalFeatures.pdf){:target="_blank"}, [Local Binary Patterns](https://en.wikipedia.org/wiki/Local_binary_patterns){:target="_blank"} (LBP)
* **Others**  - [Histogram of Oriented Gradients](https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients){:target="_blank"} (HOG), Threshold Adjancency Statistics (TAS)


### Local Feature Descriptors

These are the feature descriptors that quantifies local regions of an image. Interest points are determined in the entire image and image patches/regions surrounding those interest points are considered for analysis. Some of the commonly used local feature descriptors are
* [SIFT](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform){:target="_blank"} (Scale Invariant Feature Transform)
* [SURF](https://en.wikipedia.org/wiki/Speeded_up_robust_features){:target="_blank"} (Speeded Up Robust Features)
* [ORB](https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF){:target="_blank"} (Oriented Fast and Rotated BRIEF)
* [BRIEF](http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_brief/py_brief.html){:target="_blank"} (Binary Robust Independed Elementary Features)

### Combining Global Features

There are two popular ways to combine these feature vectors.
* For global feature vectors, we just concatenate each feature vector to form a single global feature vector. This is the approach we will be using in this tutorial.
* For local feature vectors as well as combination of global and local feature vectors, we need something called as [Bag of Visual Words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision){:target="_blank"} (BOVW). This approach is not discussed in this tutorial, but there are lots of resources to learn this technique. Normally, it uses Vocabulory builder, K-Means clustering, Linear SVM, and Td-Idf vectorization.

<figure>
    <img src="/images/software/plants-species/global_features.jpg">
    <figcaption>Figure 4. Global Features to quantify a flower image.</figcaption>
</figure>

### FLOWERS-17 dataset

We will use the FLOWER17 dataset provided by the University of Oxford, Visual Geometry group. This dataset is a highly challenging dataset with 17 classes of flower species, each having 80 images. So, totally we have 1360 images to train our model. For more information about the dataset and to download it, kindly visit [this](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/){:target="_blank"} link.

<figure>
    <img src="/images/software/plants-species/flowers17_data.jpg">
    <figcaption>Figure 5. FLOWER17 dataset from the University of Oxford, Visual Geometry group</figcaption>
</figure>

### Organizing Dataset

The folder structure for this example is given below. 

<h3 class="code-head">folder structure</h3>

```shell
|--image-classification (folder)
|--|--dataset (folder)
|--|--|--train (folder)
|--|--|--|--cowbell (folder)
|--|--|--|--|--image_1.jpg
|--|--|--|--|--image_2.jpg
|--|--|--|--|--...
|--|--|--|--tulip (folder)
|--|--|--|--|--image_1.jpg
|--|--|--|--|--image_2.jpg
|--|--|--|--|--...
|--|--|--test (folder)
|--|--|--|--image_1.jpg
|--|--|--|--image_2.jpg
|--|--output (folder)
|--|--|--data.h5
|--|--|--labels.h5
|--|--global.py
|--|--train_test.py

```

<div class="note">
<p><b>Update (03/07/2019):</b> To create the above folder structure and organize the training dataset folder, I have created a script for you - <a href="https://github.com/Gogul09/image-classification-python/blob/master/organize_flowers17.py" target="_blank">organize_flowers17.py</a>. Please use this script first before calling any other script in this tutorial.</p>
</div>

<h3 class="code-head">organize_flowers17.py<span>code</span></h3>

```python
#-----------------------------------------
# DOWNLOAD AND ORGANIZE FLOWERS17 DATASET
#-----------------------------------------
import os
import glob
import datetime
import tarfile
import urllib.request

def download_dataset(filename, url, work_dir):
  if not os.path.exists(filename):
    print("[INFO] Downloading flowers17 dataset....")
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    print("[INFO] Succesfully downloaded " + filename + " " + str(statinfo.st_size) + " bytes.")
    untar(filename, work_dir)

def jpg_files(members):
  for tarinfo in members:
    if os.path.splitext(tarinfo.name)[1] == ".jpg":
      yield tarinfo

def untar(fname, path):
  tar = tarfile.open(fname)
  tar.extractall(path=path, members=jpg_files(tar))
  tar.close()
  print("[INFO] Dataset extracted successfully.")

#-------------------------
# MAIN FUNCTION
#-------------------------
if __name__ == '__main__':
  flowers17_url  = "http://www.robots.ox.ac.uk/~vgg/data/flowers/17/"
  flowers17_name = "17flowers.tgz"
  train_dir      = "dataset"

  if not os.path.exists(train_dir):
    os.makedirs(train_dir)

  download_dataset(flowers17_name, flowers17_url, train_dir)
  if os.path.exists(train_dir + "\\jpg"):
    os.rename(train_dir + "\\jpg", train_dir + "\\train")


  # get the class label limit
  class_limit = 17

  # take all the images from the dataset
  image_paths = glob.glob(train_dir + "\\train\\*.jpg")

  # variables to keep track
  label = 0
  i = 0
  j = 80

  # flower17 class names
  class_names = ["daffodil", "snowdrop", "lilyvalley", "bluebell", "crocus",
             "iris", "tigerlily", "tulip", "fritillary", "sunflower", 
             "daisy", "coltsfoot", "dandelion", "cowslip", "buttercup",
             "windflower", "pansy"]

  # loop over the class labels
  for x in range(1, class_limit+1):
    # create a folder for that class
    os.makedirs(train_dir + "\\train\\" + class_names[label])
    
    # get the current path
    cur_path = train_dir + "\\train\\" + class_names[label] + "\\"
    
    # loop over the images in the dataset
    for index, image_path in enumerate(image_paths[i:j], start=1):
      original_path   = image_path
      image_path      = image_path.split("\\")
      image_file_name = str(index) + ".jpg"
      os.rename(original_path, cur_path + image_file_name)
    
    i += 80
    j += 80
    label += 1
```

### Global Feature Extraction

Ok! It's time to code!

We will use a simpler approach to produce a baseline accuracy for our problem. It means everything should work somehow without any error. 

Our three global feature descriptors are 
1. **Color Histogram** that quantifies **color** of the flower.
2. **Hu Moments** that quantifies **shape** of the flower.
3. **Haralick Texture** that quantifies **texture** of the flower.

As you might know images are matrices, we need an efficient way to store our feature vectors locally. Our script takes one image at a time, extract three global features, concatenates the three global features into a single global feature and saves it along with its label in a [HDF5 file format](https://en.wikipedia.org/wiki/Hierarchical_Data_Format){:target="_blank"}. 

Insted of using HDF5 file-format, we could use ".csv" file-format to store the features. But, as we will be working with large amounts of data in future, becoming familiar with HDF5 format is worth it.

<h3 class="code-head">global.py<span>code</span></h3>

```python
#-----------------------------------
# GLOBAL FEATURE EXTRACTION
#-----------------------------------
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py

#--------------------
# tunable-parameters
#--------------------
images_per_class = 80
fixed_size       = tuple((500, 500))
train_path       = "dataset/train"
h5_data          = 'output/data.h5'
h5_labels        = 'output/labels.h5'
bins             = 8
```

* Lines 4 - 10 imports the necessary libraries we need to work with.
* Line 16 used to convert the input image to a fixed size of (500, 500).
* Line 17 is the path to our training dataset.
* Lines 18 - 19 stores our global features and labels in <span class="coding">output</span> directory.
* Line 20 is the number of bins for color histograms.

### Functions for global feature descriptors

#### 1. Hu Moments
To extract Hu Moments features from the image, we use <span class="coding">cv2.HuMoments()</span> function provided by OpenCV. The argument to this function is the moments of the image <span class="coding">cv2.moments()</span> flatenned. It means we compute the moments of the image and convert it to a vector using <span class="coding">flatten()</span>. Before doing that, we convert our color image into a grayscale image as moments expect images to be grayscale.

<h3 class="code-head">global.py<span>code</span></h3>
```python
# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
```

#### 2. Haralick Textures
To extract Haralick Texture features from the image, we make use of mahotas library. The function we will be using is <span class="coding">mahotas.features.haralick()</span>. Before doing that, we convert our color image into a grayscale image as haralick feature descriptor expect images to be grayscale. 

<h3 class="code-head">global.py<span>code</span></h3>
```python
# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick
```

#### 3. Color Histogram
To extract Color Histogram features from the image, we use <span class="coding">cv2.calcHist()</span> function provided by [OpenCV](https://docs.opencv.org/3.2.0/d1/db7/tutorial_py_histogram_begins.html){:target="_blank"}. The arguments it expects are the image, channels, mask, histSize (bins) and ranges for each channel [typically 0-256). We then normalize the histogram using <span class="coding">normalize()</span> function of OpenCV and return a flattened version of this normalized matrix using <span class="coding">flatten()</span>.

<h3 class="code-head">global.py<span>code</span></h3>
```python
# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()
```

<div class="note"><p>
<b>Important</b>: To get the list of training labels associated with each image, under our training path, we are supposed to have folders that are named with the labels of the respective flower species name inside which all the images belonging to that label are kept. Please keep a note of this as you might get errors if you don't have a proper folder structure. 
</p>
</div>

<h3 class="code-head">global.py<span>code</span></h3>

```python
# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels          = []
```

<div class="code-out">
<p>
['bluebell', 'buttercup', 'coltsfoot', 'cowslip', 'crocus', 'daffodil', 'daisy', 'dandelion', 'fritillary', 'iris', 'lilyvalley', 'pansy', 'snowdrop', 'sunflower', 'tigerlily', 'tulip', 'windflower']
</p>
</div>

For each of the training label name, we iterate through the corresponding folder to get all the images inside it. For each image that we iterate, we first resize the image into a fixed size. Then, we extract the three global features and concatenate these three features using NumPy's <span class="coding">np.hstack()</span> function. We keep track of the feature with its label using those two lists we created above - <span class="coding">labels</span> and <span class="coding">global_features</span>. You could even use a dictionary here. Below is the code snippet to do these.

<h3 class="code-head">global.py<span>code</span></h3>

```python
# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for x in range(1,images_per_class+1):
        # get the image file name
        file = dir + "/" + str(x) + ".jpg"

        # read the image and resize it to a fixed-size
        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        ####################################
        # Global Feature extraction
        ####################################
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)

        ###################################
        # Concatenate global features
        ###################################
        global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")
```

<div class="code-out">
<p>
[STATUS] processed folder: bluebell<br>
[STATUS] processed folder: buttercup<br>
[STATUS] processed folder: coltsfoot<br>
[STATUS] processed folder: cowslip<br>
[STATUS] processed folder: crocus<br>
[STATUS] processed folder: daffodil<br>
[STATUS] processed folder: daisy<br>
[STATUS] processed folder: dandelion<br>
[STATUS] processed folder: fritillary<br>
[STATUS] processed folder: iris<br>
[STATUS] processed folder: lilyvalley<br>
[STATUS] processed folder: pansy<br>
[STATUS] processed folder: snowdrop<br>
[STATUS] processed folder: sunflower<br>
[STATUS] processed folder: tigerlily<br>
[STATUS] processed folder: tulip<br>
[STATUS] processed folder: windflower<br>
[STATUS] completed Global Feature Extraction...
</p>
</div>

After extracting features and concatenating it, we need to save this data locally. Before saving this data, we use something called <span class="coding">LabelEncoder()</span> to encode our labels in a proper format. This is to make sure that the labels are represented as unique numbers. As we have used different global features, one feature might dominate the other with respect to it's value. In such scenarios, it is better to normalize everything within a range (say 0-1). Thus, we normalize the features using scikit-learn's <span class="coding">MinMaxScaler()</span> function. After doing these two steps, we use h5py to save our features and labels locally in <span class="coding">.h5</span> file format. Below is the code snippet to do these.

<h3 class="code-head">global.py<span>code</span></h3>

```python
# get the overall feature vector size
print("[STATUS] feature vector size {}".format(np.array(global_features).shape))

# get the overall training label size
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

# scale features in the range (0-1)
scaler            = MinMaxScaler(feature_range=(0, 1))
rescaled_features = scaler.fit_transform(global_features)
print("[STATUS] feature vector normalized...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

# save the feature vector using HDF5
h5f_data = h5py.File(h5_data, 'w')
h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

h5f_label = h5py.File(h5_labels, 'w')
h5f_label.create_dataset('dataset_1', data=np.array(target))

h5f_data.close()
h5f_label.close()

print("[STATUS] end of training..")
```

<div class="code-out">
<p>
[STATUS] feature vector size (1360, 532)<br>
[STATUS] training Labels (1360,)<br>
[STATUS] training labels encoded...<br>
[STATUS] feature vector normalized...<br>
[STATUS] target labels: [ 0  0  0 ..., 16 16 16]<br>
[STATUS] target labels shape: (1360,)<br>
[STATUS] end of training..
</p>
</div>

Notice that there are 532 columns in the global feature vector which tells us that when we concatenate color histogram, haralick texture and hu moments, we get a single row with 532 columns. So, for 1360 images, we get a feature vector of size (1360, 532). Also, you could see that the target labels are encoded as integer values in the range (0-16) denoting the 17 classes of flower species.

### Training classifiers

After extracting, concatenating and saving global features and labels from our training dataset, it's time to train our system. To do that, we need to create our Machine Learning models. For creating our machine learning model's, we take the help of [scikit-learn](http://scikit-learn.org/){:target="_blank"}. 

We will choose Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Trees, Random Forests, Gaussian Naive Bayes and Support Vector Machine as our machine learning models. To understand these algorithms, please go through Professor Andrew NG's amazing Machine Learning [course](https://www.coursera.org/learn/machine-learning){:target="_blank"} at Coursera or you could look into this awesome [playlist](https://www.youtube.com/playlist?list=PLea0WJq13cnCS4LLMeUuZmTxqsqlhwUoe) of Dr.Noureddin Sadawi.

Furthermore, we will use <span class="coding">train_test_split</span> function provided by <span class="coding">scikit-learn</span> to split our training dataset into train_data and test_data. By this way, we train the models with the train_data and test the trained model with the unseen test_data. The split size is decided by the <span class="coding">test_size</span> parameter.

We will also use a technique called [K-Fold Cross Validation](https://www.youtube.com/watch?v=TIgfjmp-4BA){:target="_blank"}, a model-validation technique which is the best way to predict ML model's accuracy. In short, if we choose K = 10, then we split the entire data into 9 parts for training and 1 part for testing uniquely over each round upto 10 times. To understand more about this, go through [this link](https://en.wikipedia.org/wiki/Cross-validation_(statistics)){:target="_blank"}.

We import all the necessary libraries to work with and create a <span class="coding">models</span> list. This list will have all our machine learning models that will get trained with our locally stored features. During import of our features from the locally saved <span class="coding">.h5</span> file-format, it is always a good practice to check its shape. To do that, we make use of <span class="coding">np.array()</span> function to convert the <span class="coding">.h5</span> data into a numpy array and then print its shape.

<h3 class="code-head">train_test.py<span>code</span></h3>

```python
#-----------------------------------
# TRAINING OUR MODEL
#-----------------------------------
import h5py
import numpy as np
import os
import glob
import cv2
import warnings
from matplotlib import pyplot
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.externals import joblib

warnings.filterwarnings('ignore')

#--------------------
# tunable-parameters
#--------------------
num_trees = 100
test_size = 0.10
seed      = 9
train_path = "dataset/train"
test_path  = "dataset/test"
h5_data    = 'output/data.h5'
h5_labels  = 'output/labels.h5'
scoring    = "accuracy"

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()

if not os.path.exists(test_path):
    os.makedirs(test_path)

# create all the machine learning models
models = []
models.append(('LR', LogisticRegression(random_state=seed)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=seed)))
models.append(('RF', RandomForestClassifier(n_estimators=num_trees, random_state=seed)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(random_state=seed)))

# variables to hold the results and names
results = []
names   = []

# import the feature vector and trained labels
h5f_data  = h5py.File(h5_data, 'r')
h5f_label = h5py.File(h5_labels, 'r')

global_features_string = h5f_data['dataset_1']
global_labels_string   = h5f_label['dataset_1']

global_features = np.array(global_features_string)
global_labels   = np.array(global_labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of the feature vector and labels
print("[STATUS] features shape: {}".format(global_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")
```

<div class="code-out">
<p>
[STATUS] features shape: (1360, 532)<br>
[STATUS] labels shape: (1360,)<br>
[STATUS] training started...
</p>
</div>

As I already mentioned, we will be splitting our training dataset into train_data as well as test_data. <span class="coding">train_test_split()</span> function does that for us and it returns four variables as shown below.

<h3 class="code-head">train_test.py<span>code</span></h3>

```python
# split the training and testing data
(trainDataGlobal, testDataGlobal, trainLabelsGlobal, testLabelsGlobal) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=test_size,
                                                                                          random_state=seed)

print("[STATUS] splitted train and test data...")
print("Train data  : {}".format(trainDataGlobal.shape))
print("Test data   : {}".format(testDataGlobal.shape))
print("Train labels: {}".format(trainLabelsGlobal.shape))
print("Test labels : {}".format(testLabelsGlobal.shape))
```

<div class="code-out">
<p>
[STATUS] splitted train and test data...<br>
Train data  : (1224, 532)<br>
Test data   : (136, 532)<br>
Train labels: (1224,)<br>
Test labels : (136,)
</p>
</div>

Notice we have decent amount of train_data and less test_data. We always want to train our model with more data so that our model generalizes well. So, we keep <span class="coding">test_size</span> variable to be in the range (0.10 - 0.30). Not more than that. 

Finally, we train each of our machine learning model and check the cross-validation results. Here, we have used only our train_data.

<h3 class="code-head">train_test.py<span>code</span></h3>

```python
# 10-fold cross validation
for name, model in models:
    kfold = KFold(n_splits=10, random_state=seed)
    cv_results = cross_val_score(model, trainDataGlobal, trainLabelsGlobal, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Machine Learning algorithm comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()
```

<div class="code-out">
<p>
LR:   0.501719 (0.051735)<br>
LDA:  0.441197 (0.034820)<br>
KNN:  0.362742 (0.025958)<br>
CART: 0.474690 (0.041314)<br>
RF:   0.643809 (0.029491)<br>
NB:   0.361102 (0.034966)<br>
SVM:  0.043343 (0.027239)
</p>
</div>    

<figure>
    <img src="/images/software/plants-species/classification.png">
    <figcaption>Figure 6. Comparison chart of different machine learning classifiers used (Y-axis: Accuracy)</figcaption>
</figure>

As you can see, the accuracies are **not** so good. Random Forests (RF) gives the maximum accuracy of **64.38%**. This is mainly due to the number of images we use per class. We need large amounts of data to get better accuracy. For example, for a single class, we atleast need around 500-1000 images which is indeed a time-consuming task. But, in this post, I have provided you with the steps, tools and concepts needed to solve an image classification problem.

### Testing the best classifier

Let's quickly try to build a Random Forest model, train it with the training data and test it on some unseen flower images.

<h3 class="code-head">train_test.py<span>code</span></h3>

```python
#-----------------------------------
# TESTING OUR MODEL
#-----------------------------------

# to visualize results
import matplotlib.pyplot as plt

# create the model - Random Forests
clf  = RandomForestClassifier(n_estimators=num_trees, random_state=seed)

# fit the training data to the model
clf.fit(trainDataGlobal, trainLabelsGlobal)

# loop through the test images
for file in glob.glob(test_path + "/*.jpg"):
    # read the image
    image = cv2.imread(file)

    # resize the image
    image = cv2.resize(image, fixed_size)

    ####################################
    # Global Feature extraction
    ####################################
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)

    ###################################
    # Concatenate global features
    ###################################
    global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

    # scale features in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_feature = scaler.fit_transform(global_feature)

    # predict label of test image
    prediction = clf.predict(rescaled_feature.reshape(1,-1))[0]

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

    # display the output image
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
```

<figure>
    <img src="/images/software/plants-species/result1.png">
    <figcaption>Figure 7. Prediction 1 - Sunflower (Correct)</figcaption>
</figure>

<figure>
    <img src="/images/software/plants-species/result2.png">
    <figcaption>Figure 8. Prediction 2 - Bluebell (Correct)</figcaption>
</figure>

<figure>
    <img src="/images/software/plants-species/result3.png">
    <figcaption>Figure 9. Prediction 3 - Pansy (Correct)</figcaption>
</figure>

<figure>
    <img src="/images/software/plants-species/result4.png">
    <figcaption>Figure 10. Prediction 4 - Buttercup (Wrong)</figcaption>
</figure>

As we can see, our approach seems to do pretty good at recognizing flowers. But it also predicted wrong label like the **last one**. Instead of sunflower, our model predicted buttercup.

You can download the entire code used in this post [here](https://github.com/Gogul09/image-classification-python){:target="_blank"}.

### Improving classifier accuracy

So, how are we going to improve the accuracy further? Fortunately, there are multiple techniques to achieve better accuracy. Some of them are listed below.

1. Gather more data for each class. (500-1000) images per class.
2. Use Data Augmentation to generate more images per class.
3. Global features along with local features such as SIFT, SURF or DENSE could be used along with Bag of Visual Words (BOVW) technique.
4. Local features alone could be tested with BOVW technique.
5. [Convolutional Neural Networks](https://en.wikipedia.org/wiki/Convolutional_neural_network){:target="_blank"} - State-of-the-art models when it comes to Image Classification and Object Recognition.

Some of the state-of-the-art Deep Learning CNN models are mentioned below.

* AlexNet
* Inception-V3
* Xception
* VGG16
* VGG19
* OverFeat
* ZeilerNet
* MSRA

But to apply CNN to this problem, the size of our dataset must be large enough and also to process those tremendous amount of data it is always recommended to use GPUs.

### Resources

#### Research papers
1. [A Visual Vocabulary for Flower Classification](http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback06.pdf){:target="_blank"}
2. [Delving into the whorl of flower segmentation](http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback07.pdf){:target="_blank"}
3. [Automated flower classification over a large number of classes](http://www.robots.ox.ac.uk/~vgg/publications/papers/nilsback08.pdf){:target="_blank"}
4. [Fine-Grained Plant Classification Using Convolutional Neural Networks for Feature Extraction](http://ceur-ws.org/Vol-1180/CLEF2014wn-Life-SunderhaufEt2014.pdf){:target="_blank"}
5. [Fine-tuning Deep Convolutional Networks for Plant Recognition](http://ceur-ws.org/Vol-1391/121-CR.pdf){:target="_blank"}
6. [Plant species classification using deep convolutional neural network](http://www.sciencedirect.com/science/article/pii/S1537511016301465){:target="_blank"}
7. [Plant classification using convolutional neural networks](http://ieeexplore.ieee.org/document/7577698/){:target="_blank"}
8. [Deep-plant: Plant identification with convolutional neural networks](http://ieeexplore.ieee.org/document/7350839/){:target="_blank"}
9. [Deep Neural Networks Based Recognition of Plant Diseases by Leaf Image Classification](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4934169/){:target="_blank"}
10. [Plant Leaf Identification via A Growing Convolution Neural Network with Progressive Sample Learning](http://vigir.missouri.edu/~gdesouza/Research/Conference_CDs/ACCV_2014/pages/PDF/825.pdf){:target="_blank"}

#### Libraries and Tools
1. [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/install.html){:target="_blank"}
2. [OpenCV](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_tutorials.html){:target="_blank"}
3. [Scikit-learn](http://scikit-learn.org/stable/){:target="_blank"}
4. [Mahotas](http://mahotas.readthedocs.io/en/latest/){:target="_blank"}
5. [NumPy](http://www.numpy.org/){:target="_blank"}
6. [SciPy](http://matplotlib.org/){:target="_blank"}
7. [h5Py](http://www.h5py.org/){:target="_blank"}

#### Dataset
1. [LeafSnap](http://leafsnap.com/dataset/){:target="_blank"}
2. [ImageCLEF](http://www.imageclef.org/lifeclef/2016/plant){:target="_blank"}
3. [PlantVillage](https://www.plantvillage.org/){:target="_blank"}
4. [FLOWERS17](http://www.robots.ox.ac.uk/~vgg/data/flowers/17/){:target="_blank"}
5. [FLOWERS102](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/){:target="_blank"}{:target="_blank"}
6. [Plant Image Analysis](http://www.plant-image-analysis.org/){:target="_blank"}