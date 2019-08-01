---
layout: post
category: software
class: Computer Vision
title: Texture Recognition using Haralick Texture and Python
description: Learn how to quantify images globally using Haralick Textures and classify images based on Textures. Understand the concept of Gray-Level Co-occurance Matrix (GLCM) used when computing Haralick Textures in Python.
author: Gogul Ilango
permalink: software/texture-recognition
image: https://drive.google.com/uc?id=1_Ge8_Na4-OBRq58vgsbmFlMObMLb2Hdd
---

<div class="git-showcase">
  <div>
    <a class="github-button" href="https://github.com/Gogul09" data-show-count="true" aria-label="Follow @Gogul09 on GitHub">Follow @Gogul09</a>
  </div>

  <div>
    <a class="github-button" href="https://github.com/Gogul09/explore-computer-vision/fork" data-icon="octicon-repo-forked" data-show-count="true" aria-label="Fork Gogul09/explore-computer-vision on GitHub">Fork</a>
  </div>

  <div>
    <a class="github-button" href="https://github.com/Gogul09/explore-computer-vision" data-icon="octicon-star" data-show-count="true" aria-label="Star Gogul09/explore-computer-vision on GitHub">Star</a>
  </div>  
</div>

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#what-is-a-texture">What is a Texture?</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#what-is-haralick-texture">What is Haralick Texture?</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#what-is-glcm">What is GLCM?</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#haralick-texture-feature-vector">Haralick Texture Feature Vector</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#implementing-texture-recognition">Implementing Texture Recognition</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#import-the-necessary-packages">1. Import the necessary packages</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#load-the-training-dataset">2. Load the training dataset</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#extract-features-function">3. Extract features function</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#extract-features-for-all-images">4. Extract features for all images</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#create-the-machine-learning-classifier">5. Create the machine learning classifier</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_11" href="#test-the-classifier-on-testing-data">6. Test the classifier on testing data</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_12" href="#training-images">Training images</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_13" href="#testing-images">Testing images</a></li>
  </ul>
</div>

When it comes to Global Feature Descriptors (i.e feature vectors that quantifies the entire image), there are three major attributes to be considered - <span class="light">Color, Shape</span> and <span class="light">Texture</span>. All these three could be used separately or combined to quantify images. In this post, we will learn how to recognize texture in images. We will study a new type of global feature descriptor called Haralick Texture. Let's jump right into it!

<div class="code-head">Objectives</div>

```
After reading this tutorial, we will understand -

* How to label and organize our own dataset?
* What is Haralick Texture and how it is computed?
* How to use Haralick Texture module in mahotas library?
* How to recognize textures in images?
* How to use Linear SVM to train and classify images?
* How to make predictions using the created model on an unseen data?
```

### What is a Texture?

Texture defines the consistency of patterns and colors in an object/image such as bricks, school uniforms, sand, rocks, grass etc. To classify objects in an image based on texture, we have to look for the consistent spread of patterns and colors in the object's surface. Rough-Smooth, Hard-Soft, Fine-Coarse are some of the texture pairs one could think of, although there are many such pairs.

### What is Haralick Texture?

Haralick Texture is used to quantify an image based on texture. It was invented by Haralick in 1973 and you can read about it in detail [here](http://haralick.org/journals/TexturalFeatures.pdf){:target="_blank"}. The fundamental concept involved in computing Haralick Texture features is the Gray Level Co-occurrence Matrix or GLCM.

### What is GLCM?

Gray Level Co-occurrence matrix (GLCM) uses adjacency concept in images. The basic idea is that it looks for pairs of adjacent pixel values that occur in an image and keeps recording it over the entire image. Below figure explains how a GLCM is constructed.

<figure>
  <img src="/images/software/haralick-texture/GLCM.jpg" class="typical-image">
  <figcaption>Figure 1. Gray Level Co-occurance Matrix (GLCM)</figcaption>
</figure>

As you can see from the above image, gray-level pixel value 1 and 2 occurs twice in the image and hence GLCM records it as two. But pixel value 1 and 3 occurs only once in the image and thus GLCM records it as one. Of course, I have assumed the adjacency calculation only from left-to-right. Actually, there are four types of adjacency and hence four GLCM matrices are constructed for a single image. Four types of adjacency are as follows.

* Left-to-Right
* Top-to-Bottom
* Top Left-to-Bottom Right
* Top Right-to-Bottom Left

### Haralick Texture Feature Vector

From the four GLCM matrices, 14 textural features are computed that are based on some statistical theory. All these 14 statistical features needs a separate blog post. So, you can read in detail about those [here](http://haralick.org/journals/TexturalFeatures.pdf){:target="_blank"}. Normally, the feature vector is taken to be of 13-dim as computing 14th dim might increase the computational time.

### Implementing Texture Recognition

Ok, lets start with the code! 

Actually, it will take just 10-15 minutes to complete our texture recognition system using OpenCV, Python, sklearn and mahotas provided we have the training dataset.

<div class="note">
<p><b>Note</b>: In case if you don't have these packages installed, feel free to install these using my environment setup posts given below.</p>
<ul>
	<li><a href="https://gogul09.github.io/software/deep-learning-windows" target="_blank">Deep Learning Environment Setup (Windows)</a></li>
	<li><a href="https://gogul09.github.io/software/deep-learning-linux" target="_blank">Deep Learning Environment Setup (Linux)</a></li>
</ul>
</div>

### Import the necessary packages

<div class="code-head">train_test.py<span>code</span></div>

```python
import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC
```

### Load the training dataset

<div class="code-head">train_test.py<span>code</span></div>

```python
# load the training dataset
train_path = "dataset/train"
train_names = os.listdir(train_path)

# empty list to hold feature vectors and train labels
train_features = []
train_labels = []
```

* Line 2 is the path to training dataset.
* Line 3 gets the class names of the training data.
* Line 6-7 are empty lists to hold feature vectors and labels.

### Extract features function

<div class="code-head">train_test.py<span>code</span></div>

```python
def extract_features(image):
        # calculate haralick texture features for 4 types of adjacency
        textures = mt.features.haralick(image)

        # take the mean of it and return it
        ht_mean = textures.mean(axis=0)
        return ht_mean
```

* Line 1 is a function that takes an input image to compute haralick texture.
* Line 3 extracts the haralick features for all 4 types of adjacency.
* Line 6 finds the mean of all 4 types of GLCM.
* Line 7 returns the resulting feature vector for that image which describes the texture.

### Extract features for all images

<div class="code-head">train_test.py<span>code</span></div>

```python
# loop over the training dataset
print "[STATUS] Started extracting haralick textures.."
for train_name in train_names:
        cur_path = train_path + "/" + train_name
        cur_label = train_name
        i = 1
        for file in glob.glob(cur_path + "/*.jpg"):
                print "Processing Image - {} in {}".format(i, cur_label)
                # read the training image
                image = cv2.imread(file)

                # convert the image to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # extract haralick texture from the image
                features = extract_features(gray)

                # append the feature vector and label
                train_features.append(features)
                train_labels.append(cur_label)

                # show loop update
                i += 1
```

* Line 4 loops over the training labels we have just included from training directory.
* Line 5 is the path to current image class directory.
* Line 6 holds the current image class label.
* Line 8 takes all the files with .jpg as the extension and loops through each file one by one.
* Line 11 reads the input image that corresponds to a file.
* Line 14 converts the image to grayscale.
* Line 17 extracts haralick features for the grayscale image.
* Line 20 appends the 13-dim feature vector to the training features list.
* Line 21 appends the class label to training classes list.

<div class="code-head">train_test.py<span>code</span></div>

```python
# have a look at the size of our feature vector and labels
print "Training features: {}".format(np.array(train_features).shape)
print "Training labels: {}".format(np.array(train_labels).shape)
```

### Create the machine learning classifier

<div class="code-head">train_test.py<span>code</span></div>

```python
# create the classifier
print "[STATUS] Creating the classifier.."
clf_svm = LinearSVC(random_state=9)

# fit the training data and labels
print "[STATUS] Fitting data/label to model.."
clf_svm.fit(train_features, train_labels)
```

* Line 3 creates the Linear Support Vector Machine classifier.
* Line 7 fits the training features and labels to the classifier.

### Test the classifier on testing data

<div class="code-head">train_test.py<span>code</span></div>

```python
# loop over the test images
test_path = "dataset/test"
for file in glob.glob(test_path + "/*.jpg"):
        # read the input image
        image = cv2.imread(file)

        # convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # extract haralick texture from the image
        features = extract_features(gray)

        # evaluate the model and predict label
        prediction = clf_svm.predict(features.reshape(1, -1))[0]

        # show the label
        cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

       # display the output image
       cv2.imshow("Test_Image", image)
       cv2.waitKey(0)
```

* Line 2 gets the testing data path.
* Line 3 takes all the files with the .jpg extension and loops through each file one by one.
* Line 5 reads the input image.
* Line 8 converts the input image into grayscale image.
* Line 11 extract haralick features from grayscale image.
* Line 14 predicts the output label for the test image.
* Line 17 displays the output class label for the test image.
* Finally, Line 20 displays the test image with predicted label.

### Training images

These are the images from which we train our machine learning classifier to learn texture features. You can collect the images of your choice and include it under a label. For example, "Grass" images are collected and stored inside a folder named "grass". These images could either be taken from a simple google search (easy to do; but our model won't generalize well) or from your own camera/smart-phone (which is indeed time-consuming, but our model could generalize well due to real-world images).

As a demonstration, I have included my own training and testing images. I took 3 classes of training images which holds 3 images per class. Training images with their corresponding class/label are shown below.

<figure>
  <img src="/images/software/haralick-texture/Training Data.jpg" class="typical-image">
  <figcaption>Figure 2. Training Images</figcaption>
</figure>

### Testing images

These could be images or a video sequence from a smartphone/camera. We can test our model with this test data so that our model performs feature extraction on this text data and tries to come up with the best possible label/class.

Some of the test images for which we need to predict the class/label are shown below.

<div class="note">
<p><b>Note</b>: These test images won't have any label associated with them. Our model's purpose is to predict the best possible label/class for the image it sees.</p>
</div>

<figure>
  <img src="/images/software/haralick-texture/Testing Data.jpg" class="typical-image">
  <figcaption >Figure 3. Testing Images</figcaption>
</figure>

After running the code, our model was able to correctly predict the labels for the testing data as shown below.

<figure>
	<img src="/images/software/haralick-texture/test_image_1.jpg" class="typical-image">
	<figcaption>Figure 4. Test Image Prediction - 1</figcaption>
</figure>

<figure>
	<img src="/images/software/haralick-texture/test_image_2.jpg" class="typical-image">
	<figcaption>Figure 5. Test Image Prediction - 2</figcaption>
</figure>

<figure>
	<img src="/images/software/haralick-texture/test_image_3.jpg" class="typical-image">
	<figcaption>Figure 6. Test Image Prediction - 3</figcaption>
</figure>

Here is the entire code to build our texture recognition system.

<div class="code-head">train_test.py<span>code</span></div>

```python
import cv2
import numpy as np
import os
import glob
import mahotas as mt
from sklearn.svm import LinearSVC

# function to extract haralick textures from an image
def extract_features(image):
	# calculate haralick texture features for 4 types of adjacency
	textures = mt.features.haralick(image)

	# take the mean of it and return it
	ht_mean  = textures.mean(axis=0)
	return ht_mean

# load the training dataset
train_path  = "dataset/train"
train_names = os.listdir(train_path)

# empty list to hold feature vectors and train labels
train_features = []
train_labels   = []

# loop over the training dataset
print "[STATUS] Started extracting haralick textures.."
for train_name in train_names:
	cur_path = train_path + "/" + train_name
	cur_label = train_name
	i = 1

	for file in glob.glob(cur_path + "/*.jpg"):
		print "Processing Image - {} in {}".format(i, cur_label)
		# read the training image
		image = cv2.imread(file)

		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		# extract haralick texture from the image
		features = extract_features(gray)

		# append the feature vector and label
		train_features.append(features)
		train_labels.append(cur_label)

		# show loop update
		i += 1

# have a look at the size of our feature vector and labels
print "Training features: {}".format(np.array(train_features).shape)
print "Training labels: {}".format(np.array(train_labels).shape)

# create the classifier
print "[STATUS] Creating the classifier.."
clf_svm = LinearSVC(random_state=9)

# fit the training data and labels
print "[STATUS] Fitting data/label to model.."
clf_svm.fit(train_features, train_labels)

# loop over the test images
test_path = "dataset/test"
for file in glob.glob(test_path + "/*.jpg"):
	# read the input image
	image = cv2.imread(file)

	# convert to grayscale
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# extract haralick texture from the image
	features = extract_features(gray)

	# evaluate the model and predict label
	prediction = clf_svm.predict(features.reshape(1, -1))[0]

	# show the label
	cv2.putText(image, prediction, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
	print "Prediction - {}".format(prediction)

	# display the output image
	cv2.imshow("Test_Image", image)
	cv2.waitKey(0)
```

```
[STATUS] Started extracting haralick textures..
Processing Image - 1 in bricks
Processing Image - 2 in bricks
Processing Image - 3 in bricks
Processing Image - 1 in grass
Processing Image - 2 in grass
Processing Image - 3 in grass
Processing Image - 1 in rocks
Processing Image - 2 in rocks
Processing Image - 3 in rocks
Training features: (9, 13)
Training labels: (9,)
[STATUS] Creating the classifier..
[STATUS] Fitting data/label to model..
Prediction - grass
Prediction - bricks
Prediction - rocks
```
{: .code-output}

If you *copy-paste* the above code in any of your directory and run <span class="coding">python train_test.py</span>, you will get the following results.


Thus, we have implemented our very own Texture Recognition system using Haralick Textures, Python and OpenCV.