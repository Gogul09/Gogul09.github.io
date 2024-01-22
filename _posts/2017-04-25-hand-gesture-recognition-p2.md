---
layout: post
category: software
class: Computer Vision
title: Hand Gesture Recognition using Python and OpenCV - Part 2
description: Learn how to recognize hand gestures after applying background subtraction using OpenCV and Python.
author: Gogul Ilango
permalink: software/hand-gesture-recognition-p2
image: https://drive.google.com/uc?id=1keoqik_Zs2O56fPzB3T7lY2Juao5NyQh
---

<div class="git-showcase">
  <div>
    <a class="github-button" href="https://github.com/Gogul09" data-show-count="true" aria-label="Follow @Gogul09 on GitHub">Follow @Gogul09</a>
  </div>

  <div>
    <a class="github-button" href="https://github.com/Gogul09/gesture-recognition/fork" data-icon="octicon-repo-forked" data-show-count="true" aria-label="Fork Gogul09/gesture-recognition on GitHub">Fork</a>
  </div>

  <div>
    <a class="github-button" href="https://github.com/Gogul09/gesture-recognition" data-icon="octicon-star" data-show-count="true" aria-label="Star Gogul09/gesture-recognition on GitHub">Star</a>
  </div>  
</div>

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#count-my-fingers">Count My Fingers</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#four-intermediate-steps">Four Intermediate Steps</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#contours">Contours</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#bitwise-and">Bitwise AND</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#euclidean-distance">Euclidean Distance</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#convex-hull">Convex Hull</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#results">Results</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#segmenting-the-hand">Segmenting the hand</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#finding-the-count-of-fingers">Finding the count of fingers</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#summary">Summary</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_11" href="#references">References</a></li>
  </ul>
</div>

This is a follow-up post of my tutorial on Hand Gesture Recognition using OpenCV and Python. Please read the first part of the tutorial <a href="https://gogul09.github.io/software/hand-gesture-recognition-p1" target="_blank">here</a> and then come back. 

In the previous tutorial, we have used Background Subtraction, Motion Detection and Thresholding to segment our hand region from a live video sequence. In this tutorial, we will take one step further to recognize the fingers as well as predict the number of fingers (count) in the live video sequence.

<div class="note"><p>
<b>Note</b>: This tutorial assumes that you have knowledge in using OpenCV, Python, NumPy and some basics of Computer Vision and Image Processing. If you need to setup environment on your system, please follow the instructions posted <a href="https://gogul09.github.io/software/deep-learning-windows" target="_blank">here</a> and <a href="https://gogul09.github.io/software/deep-learning-linux" target="_blank">here</a>.</p></div>

### Count My Fingers
Having segmented the hand region from the live video sequence, we will make our system to count the fingers that are shown via a camera/webcam. We cannot use any template (provided by OpenCV) that is available to perform this, as it is indeed a challenging problem.

The entire code from my previous tutorial (Hand Gesture Recognition-Part 1) can be seen [here](https://github.com/Gogul09/gesture-recognition/blob/master/segment.py){:target="_blank"} for reference. Note that, we have used the concept of Background Subtraction, Motion Detection and Thresholding to segment the hand region from a live video sequence. 

We have obtained the segmented hand region by assuming it as the largest contour (i.e. contour with the maximum area) in the frame. If you bring in some large object inside this frame which is larger than your hand, then this algorithm fails. So, you must make sure that your hand occupies the majority of the region in the frame.

We will use the segmented hand region which was obtained in the variable <span class="coding">hand</span>. Remember, this <span class="coding">hand</span> variable is a tuple having <span class="coding">thresholded</span> (thresholded image) and <span class="coding">segmented</span> (segmented hand region). We are going to utilize these two variables to count the fingers shown. How are we going to do that?

There are various approaches that could be used to count the fingers, but we are going to see one such approach in this tutorial. This is a faster approach to perform hand gesture recognition as proposed by [Malima et.al](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.454.3689&rep=rep1&type=pdf){:target="_blank"}. The methodology to count the fingers (as proposed by Malima et.al) is shown in the figure below.

<figure>
	<img src="/images/software/gesture-recognition/count_fingers.png" class="typical-image">
	<figcaption>Figure 1. Hand-Gesture Recognition algorithm to count the fingers</figcaption>
</figure>

As you can see from the above image, there are four intermediate steps to count the fingers, given a segmented hand region. All these steps are shown with a corresponding output image (shown in the left) which we get, after performing that particular step.

### Four Intermediate Steps
1. Find the [convex hull](https://docs.opencv.org/3.4/d7/d1d/tutorial_hull.html){:target="_blank"} of the segmented hand region (which is a contour) and compute the most extreme points in the convex hull (Extreme Top, Extreme Bottom, Extreme Left, Extreme Right).
2. Find the center of palm using these extremes points in the convex hull.
3. Using the palm's center, construct a circle with the maximum [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance){:target="_blank"} (between the palm's center and the extreme points) as radius.
4. Perform bitwise AND operation between the thresholded hand image (frame) and the circular ROI (mask). This reveals the finger slices, which could further be used to calcualate the number of fingers shown. 

Below you could see the entire function used to perform the above four steps.

* Input - <span class="coding">thresholded</span> (thresholded image) and <span class="coding">segmented</span> (segmented hand region or contour)
* Output - <span class="coding">count</span> (Number of fingers).

<h3 class="code-head">recognize.py<span>code</span></h3>

```python
#--------------------------------------------------------------
# To count the number of fingers in the segmented hand region
#--------------------------------------------------------------
def count(thresholded, segmented):
    # find the convex hull of the segmented hand region
    chull = cv2.convexHull(segmented)

    # find the most extreme points in the convex hull
    extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
    extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
    extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
    extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

    # find the center of the palm
    cX = int((extreme_left[0] + extreme_right[0]) / 2)
    cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

    # find the maximum euclidean distance between the center of the palm
    # and the most extreme points of the convex hull
    distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
    maximum_distance = distance[distance.argmax()]

    # calculate the radius of the circle with 80% of the max euclidean distance obtained
    radius = int(0.8 * maximum_distance)

    # find the circumference of the circle
    circumference = (2 * np.pi * radius)

    # take out the circular region of interest which has 
    # the palm and the fingers
    circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
    # draw the circular ROI
    cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

    # take bit-wise AND between thresholded hand using the circular ROI as the mask
    # which gives the cuts obtained using mask on the thresholded hand image
    circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

    # compute the contours in the circular ROI
    (_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # initalize the finger count
    count = 0

    # loop through the contours found
    for c in cnts:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # increment the count of fingers only if -
        # 1. The contour region is not the wrist (bottom area)
        # 2. The number of points along the contour does not exceed
        #     25% of the circumference of the circular ROI
        if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
            count += 1

    return count
```

Each of the intermediate step requires some understanding of image processing fundamentals such as Contours, Bitwise-AND, Euclidean Distance and Convex Hull. 

### Contours 
The outline or the boundary of the object of interest. A contour is defined as the line that joins all the points along the boundary of an image that have the same intensity. <span class="coding">cv2.findContours()</span> function in OpenCV help us find contours in a binary image. If you pass a binary image to this function, it returns a list of all the contours in the image. Each of the elements in this list is a numpy array that represents the (x, y) coordinate of the boundary points of the contour (or the object).

### Bitwise-AND 
Performs bit-wise logical AND between two objects. You could visually think of this as using a mask and extracting the regions in an image that lie under this mask alone. OpenCV provides <span class="coding">cv2.bitwise_and()</span> function to perform this operation - [Bitwise AND](http://docs.opencv.org/trunk/d0/d86/tutorial_py_image_arithmetics.html){:target="_blank"}.

### Euclidean Distance 
This is the distance between two points given by the equation shown [here](https://bigsnarf.files.wordpress.com/2012/03/distance.jpg){:target="_blank"}. Scikit-learn provides a function called <span class="coding">pairwise.euclidean_distances()</span> to calculate the Euclidean distance from *one point* to *multiple points* in a single line of code - [Pairwise Euclidean Distance](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html){:target="_blank"}. After that, we take the *maximum* of all these distances using NumPy's <span class="coding">argmax()</span> function.

### Convex Hull 
You can think of convex hull as a dynamic, stretchable envelope that wraps around the object of interest. To read more about it, please visit [this](http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html){:target="_blank"} link.

---

<div class="note">
    <p>
        <b>Update</b>: I thought of adding some more explanation to the algorithm and code. Let's understand this algorithm and code with an image (one frame) so that each step is clear for us.
    </p>
</div>

### Segmenting the hand

Let's look at the following code snippet which is a python function that segments the hand region from the image (single frame). 

<h3 class="code-head">recognize-image.py<span>code</span></h3>

```python
#---------------------------------------------
# To segment the region of hand in the image
#---------------------------------------------
def segment(image, grayimage, threshold=75):
    # threshold the image to get the foreground which is the hand
    thresholded = cv2.threshold(grayimage, threshold, 255, cv2.THRESH_BINARY)[1]
    print("Original image shape - " + str(image.shape))
    print("Gray image shape - " + str(grayimage.shape))

    # show the thresholded image
    cv2.imshow("Thesholded", thresholded)

    # get the contours in the thresholded image
    (_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # analyze the contours
        print("Number of Contours found = " + str(len(cnts))) 
        cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
        cv2.imshow('All Contours', image) 
        
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        cv2.drawContours(image, segmented, -1, (0, 255, 0), 3)
        cv2.imshow('Max Contour', image) 
        
        return (thresholded, segmented)
```

* Line 4 initializes the function with input color image <span class="coding">image</span>, input grayscale image <span class="coding">grayimage</span> and a <span class="coding">threshold</span> variable that we can tune based on the lighting conditions of our image or video. 
* Line 6 thresholds the grayscale image with respect to the threshold that we have choosen. Anything above the threshold is transformed to white and anything below the threshold is transformed to black. You can read more on how to choose the threshold or what thresholding you should use [here](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html){:target="_blank"}.
* Line 7 and 8 prints the original color image and input gray image shapes for our reference.
* Line 11 displays the thresholded image.
* Line 14 is where the magic happens. We pass the thresholded image <span class="coding">thresholded</span> to OpenCV's <span class="coding">cv2.findContours()</span> function with some additional parameters that the function expects. This returns all the contours in the thresholded image. Notice how we unpack the return values. 
* <span class="coding">cnts</span> is a list of all the contours in the thresholded image. Each element in this list is a numpy array that represents the (x, y) coordinate of the boundary points of the contour (or the object).
* Line 17 checks the length of contours found, and if its 0, it simply returns nothing.
* Line 21 prints the number of contours found.
* Line 22 draws all the contours found in the thresholded image using OpenCV's <span class="coding">cv2.drawContours()</span> function.
* Line 23 displays all the contours present in the image.
* Line 26 finds the maximum contour in the image based on key <span class="coding">cv2.contourArea</span>. Here, we assume that our hand has the largest area in the image. If some object larger than our hand is found in the image, we need to handle this line of code.
* Line 27 draws the contour with maximum area which in our case is the hand.
* Line 28 displays the contour with maximum area in the image.
* Line 30 returns a tuple that has <span class="coding">thresholded</span> image and <span class="coding">segmented</span> contour (which is the hand).

<figure>
    <img src="/images/software/gesture-recognition/recognize-image-segment.jpg">
    <figcaption>Figure 2. Thresholding and Segmenting Hand region from an image (a single frame). <br> (a) Thresholded image (b) All Contours (c) Max Contour</figcaption>
</figure>

### Finding the count of fingers

Let's look at the following code snippets which is inside a python function that is used to count the number of fingers in the image (single frame).

I have splitted the code snippet into smaller pieces so that we can understand this line by line. Make sure to check the complete function [here](https://github.com/Gogul09/gesture-recognition/blob/master/recognize-image.py){:target='_blank'}.

Firstly, we pass the <span class="coding">segmented</span> contour which is a numpy array that contains all the boundary points (x,y) coordinates. To understand this, let's print the type and shape of the <span class="coding">segmented</span> contour with first 5 boundary points inside it.

<h3 class="code-head">recognize-image.py<span>code</span></h3>

```python
print("Type of Contour: " + str(type(segmented)))
print("Contour shape: " + str(segmented.shape))
print("First 5 points in contour: " + str(segmented[:5]))
```

```
Type of Contour: <class 'numpy.ndarray'>
Contour shape: (1196, 1, 2)
First 5 points in contour: 
[
 [[342   6]]
 [[341   7]]
 [[338   7]]
 [[337   8]]
 [[336   8]]
]
```
{: .code-out}

The next step is to find the extreme points in this contour. How? Here is another magic function in OpenCV - <span class="coding">cv2.convexHull()</span> which returns the farthest boundary points of a contour. To understand it clearly, let's pass in our <span class="coding">segmented</span> contour to <span class="coding">cv2.convexHull()</span> function.

<h3 class="code-head">recognize-image.py<span>code</span></h3>

```python
# find the convex hull of the segmented hand region
chull = cv2.convexHull(segmented)

print("Type of Convex hull: " + str(type(chull)))
print("Length of Convex hull: " + str(len(chull)))
print("Shape of Convex hull: " + str(chull.shape))

cv2.drawContours(image, [chull], -1, (0, 255, 0), 2)
cv2.imshow("Convex Hull", image)
```

```
Type of Convex hull: <class 'numpy.ndarray'>
Length of Convex hull: 30
Shape of Convex hull: (30, 1, 2)
```
{: .code-out}


<figure>
    <img src="/images/software/gesture-recognition/convex-hull.jpg" class="typical-image">
    <figcaption>Figure 3. Convex Hull of the segmented image</figcaption>
</figure>

For those of you new to python and numpy array, you can write a single line of code to find the extreme points after getting the convex hull. Concept of <span class="coding">argmin()</span> and <span class="coding">argmax()</span> is used to achieve this as shown below.

<h3 class="code-head">recognize-image.py<span>code</span></h3>

```python
print(chull[:,:,1])
print(chull[:,:,1].argmin())
print(chull[chull[:,:,1].argmin()])
print(chull[chull[:,:,1].argmin()][0])
print(tuple(chull[chull[:,:,1].argmin()][0]))
```

```
Type of Convex hull: <class 'numpy.ndarray'>
Length of Convex hull: 30
Shape of Convex hull: (30, 1, 2)
```
{: .code-out}

Above code sample is used to find extreme right boundary point in the convex hull. To find the extreme right boundary point, 

* We choose the x-axis column of the convex hull using <span class="coding">chull[:, :, 0]</span> where 0 indicates the first column.
* We then find the index of maximum number in x-axis column using <span class="coding">chull[:, :, 0].argmax()</span>.
* We then use that max index to grab the boundary point (x, y) using <span class="coding">chull[chull[:,:,1].argmin()]<span>.
* We then get the first indexed element and convert it to a tuple so that it doesn't get changed as tuples are immutable using <span class="coding">tuple(chull[chull[:,:,1].argmin()][0])</span>

Similar one line of code is used to find all the extreme points of the convex hull as shown below.

<h3 class="code-head">recognize-image.py<span>code</span></h3>

```python
# find the most extreme points in the convex hull
extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

print("Extreme Top : " + str(extreme_top))
print("Extreme Bottom : " + str(extreme_bottom))
print("Extreme Left : " + str(extreme_left))
print("Extreme Right : " + str(extreme_right))

cv2.drawContours(image, [chull], -1, (0, 255, 0), 2)
cv2.circle(image, extreme_top, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_bottom, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_left, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_right, radius=5, color=(0,0,255), thickness=5)
cv2.imshow("Extreme Points in Convex Hull", image)
```

```
Extreme Top    : (342, 6)
Extreme Bottom : (435, 548)
Extreme Left   : (121, 160)
Extreme Right  : (670, 322)
```
{: .code-out}

<figure>
    <img src="/images/software/gesture-recognition/extreme-points-convex-hull.jpg" class="typical-image">
    <figcaption>Figure 4. Extreme points in Convex Hull of the segmented image</figcaption>
</figure>

<h3 class="code-head">recognize-image.py<span>code</span></h3>

```python
# find the center of the palm
cX = int((extreme_left[0] + extreme_right[0]) / 2)
cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
print("Center point : " + str(tuple((cX,cY))))

cv2.drawContours(image, [chull], -1, (0, 255, 0), 2)
cv2.circle(image, (cX, cY), radius=5, color=(255,0,0), thickness=5)
cv2.circle(image, extreme_top, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_bottom, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_left, radius=5, color=(0,0,255), thickness=5)
cv2.circle(image, extreme_right, radius=5, color=(0,0,255), thickness=5)
cv2.imshow("Extreme Points in Convex Hull", image)
```

```
Center point : (395, 277)
```
{: .code-out}

<figure>
    <img src="/images/software/gesture-recognition/extreme-points-and-center-convex-hull.jpg" class="typical-image">
    <figcaption>Figure 5. Center point with Extreme points in Convex Hull of the segmented image</figcaption>
</figure>

Once we have the center point and extreme points, we need to find the euclidean distance from the center point to each of the extreme point. We do that in a single line of code using scikit-learn's <span class="coding">pairwise.euclidean_distances()</span>. After this, we find the maximum distance with respect to each of the four euclidean distances and form a circumference with radius as 80% of the maximum distance.

<h3 class="code-head">recognize-image.py<span>code</span></h3>

```python
# find the maximum euclidean distance between the center of the palm
# and the most extreme points of the convex hull
distances = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
max_distance = distances[distances.argmax()]

# calculate the radius of the circle with 80% of the max euclidean distance obtained
radius = int(0.8 * max_distance)

# find the circumference of the circle
circumference = (2 * np.pi * radius)

print("Euclidean Distances : " + str(distances))
print("Max Euclidean Distance : " + str(max_distance))
print("Radius : " + str(radius))
print("Circumference : " + str(circumference))
```

```
Euclidean Distance : [297.93455657 278.65749586 276.13402543 273.93612394]
Max Euclidean Distance : 297.9345565724124
Radius : 238
Circumference : 1495.3981031087415
```
{: .code-out}

Once we have the center of the convex hull, radius and circumference using the maximum distance, we then construct a numpy array filled with zeros called <span class="coding">circular_roi</span> using the <span class="coding">thresholded</span> image's shape. We then draw a circle in this <span class="coding">circular_roi</span> with the center of convex hull as the center and radius calculated above.

Next magic in our pipeline is OpenCV's <span class="coding">cv2.bitwise_and()</span> function which is a bitwise operation that you can read more about it [here](https://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html){:target="_blank"}. 

In simple terms, you have an image and a mask, and pass these to <span class="coding">cv2.bitwise_and()</span> function, the output will be the region where both image and mask had same intensity values. In our case, it displays the cuts in the fingers because we have used <span class="coding">thresholded</span> image and <span class="coding">circular_roi</span> as the mask.

<h3 class="code-head">recognize-image.py<span>code</span></h3>

```python
# initialize circular_roi with same shape as thresholded image
circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
print("Circular ROI shape : " + str(circular_roi.shape))
cv2.imshow("Thresholded", thresholded)

# draw the circular ROI with radius and center point of convex hull calculated above
cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
cv2.imshow("Circular ROI Circle", circular_roi)

# take bit-wise AND between thresholded hand using the circular ROI as the mask
# which gives the cuts obtained using mask on the thresholded hand image
circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
cv2.imshow("Bitwise AND", circular_roi)
```

<figure>
    <img src="/images/software/gesture-recognition/bitwise-and.jpg">
    <figcaption>Figure 6. Construct a circle with center point and radius obtained from convex hull and do bit-wise AND with thresholded image</figcaption>
</figure>

Now, we have a <span class="coding">circular_roi</span> which has the finger cuts obtained from the thresholded image. To calculate the count of the fingers, we need to exclude the wrist portion in the <span class="coding">circular_roi</span> so that we focus only on the fingers.

To achieve this, we make use of <span class="coding">cv2.findContours()</span> again to find all the contours in <span class="coding">circular_roi</span>. 

To remove the wrist, we can sort contours based on area and leave the first contour (as it will be the wrist).

<h3 class="code-head">recognize-image.py<span>code</span></h3>

```python
# compute the contours in the circular ROI
(_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of Contours found = " + str(len(cnts))) 
cv2.drawContours(image, cnts, -1, (0, 255, 0), 3)
cv2.imwrite("resources/count-contours.jpg", image)
```

<figure>
    <img src="/images/software/gesture-recognition/count-contours.jpg" class="typical-image">
    <figcaption>Figure 7. Contours of circular_roi to find out count</figcaption>
</figure>

Once we have the contours after bitwise-and done on <span class="coding">circular_roi</span>, we can easily find the count of fingers by two approaches.

#### Approach 1 
By sorting the contours based on area, the first contour will obviously be the wrist. So, we can ignore it and find the length of the remaining contours which will be the count of fingers shown in the image.

<h3 class="code-head">recognize-image.py<span>code</span></h3>

```python
cntsSorted = sorted(cnts, key=lambda x: cv2.contourArea(x))
print("Count of fingers : " + str(len(cntsSorted[1:])))
```

```
Count of fingers : 5
```
{: .code-out}

#### Approach 2 

For each of the contour, we can find the bounding rectangle of the contour using <span class="coding">cv2.boundingRect()</span> function. This returns 4 numbers where <span class="coding">(x,y)</span> is the top-left coordinate of the rectangle and <span class="coding">(w,h)</span> is the width and height of the rectangle.

**Check 1**: To filter out the wrist area, we can use vertical (y) axis to check whether <span class="coding">y + h</span> of a contour is lesser than center point <span class="coding">cY</span> of the convex hull. Sometimes, the thumb of our hand is flexible i.e it can move horizontally or vertically. To compensate for this, we marginally increase the <span class="coding">cY</span> of the convex hull by a factor 0.25 i.e. we use <span class="coding">cY + (cY * 0.25)</span> to check against <span class="coding">y + h</span>. You can still tune this value based on your use case. For me, 0.25 was good to go!

**Check 2**: To filter out the wrist area even more, we could also use the perimeter of the circle (circumference). When we loop over the contour, we find that a single contour is a numpy array. Inside this numpy array, we have the boundary points that forms the contour. We can check whether the number of boundary points of a contour <span class="coding">c.shape[0]</span> lie within a limit. Again, I have chosen this limit as <span class="coding">circumference * 0.25</span>. You can still tune this value based on your use case. For me, 0.25 was good to go!

<h3 class="code-head">recognize-image.py<span>code</span></h3>

```python
count = 0

# loop through the contours found
for c in cnts:
    print(type(c))
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)

    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(image,'C' + str(i), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255), 4)

    # increment the count of fingers only if -
    # 1. The contour region is not the wrist (bottom area)
    # 2. The number of points along the contour does not exceed
    #     25% of the circumference of the circular ROI
    if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
        count += 1

print("Count of fingers : " + str(len(cntsSorted[1:])))
cv2.imshow("Contours of fingers with box", image)
```

```
Count of fingers : 5
```
{: .code-out}

<figure>
    <img src="/images/software/gesture-recognition/count-contours-box.jpg" class="typical-image">
    <figcaption>Figure 8. Contours of circular_roi with bounding rectangle</figcaption>
</figure>

Figure 8 has all the intermediate steps that we have performed to build this pipeline. I have marked each of the intermediate step in different colors for clear understanding. 

### Results
You can download the entire code to perfom Hand Gesture Recognition [here](https://github.com/Gogul09/gesture-recognition){:target="_blank"}. Clone this repository using 

<h3 class="code-head">Command<span>shell</span></h3>

```shell
git clone https://github.com/Gogul09/gesture-recognition.git
```

in a Terminal/Command prompt. Then, get into the folder and type 

<h3 class="code-head">Command<span>shell</span></h3>

```shell
python recognize.py
```

<div class="note"><p>
<b>Note:</b> Do not shake your webcam during the calibration period of 30 frames. If shaken during the first 30 frames, the entire algorithm will not perform as we expect.
</p></div>

After that, you can bring in your hand into the bounding box, show gestures and the count of fingers will be displayed accordingly. I have included a demo of the entire pipeline below.

<figure>
	<img src="/images/software/gesture-recognition/demo.gif" class="typical-image" />
	<figcaption>Figure 2. Hand-Gesture Recognition | Counting the fingers | Demo</figcaption>
</figure>

### Summary
In this tutorial, we have learnt about recognizing hand gestures using Python and OpenCV. We have explored Background Subtraction, Thresholding, Segmentation, Contour Extraction, Convex Hull and Bitwise-AND operation on real-time video sequence. We have followed the methodology proposed by Malima et al. to quickly recognize hand gestures.

You could extend this idea by using the count of fingers to instruct a robot to perform some task like picking up an object, go forward, move backward etc. using Arduino or Raspberry Pi platforms. I have also made a simple demo for you by using the count of fingers to control a servo motor's rotation [here](https://www.youtube.com/watch?v=4lCjQ84EkSk){:target="_blank"}.

The entire algorithm assumes that the background is static i.e. the background does not change. If the background changes and new objects are brought into the frame, the algorithm will not perform well. Kindly let me know in the comments if there are ways to solve this limitation.

### References

1. [How to Use Background Subtraction Methods](https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html){:target="_blank"}
2. [Background Subtraction](https://www.youtube.com/watch?v=fn07iwCrvqQ){:target="_blank"}
3. [Pyimagesearch - Adrian Rosebrock](http://pyimagesearch.com/){:target="_blank"}
4. [A Fast Algorithm for Vision-based Hand Gesture Recognition For Robot Control](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.454.3689&rep=rep1&type=pdf){:target="_blank"}
5. [Arduino](https://www.arduino.cc/){:target="_blank"}
6. [Convex Hull using OpenCV in Python and C++](https://www.learnopencv.com/convex-hull-using-opencv-in-python-and-c/){:target="_blank"}
7. [Convex Hull - Set 1 (Jarvis’s Algorithm or Wrapping)](https://www.geeksforgeeks.org/convex-hull-set-1-jarviss-algorithm-or-wrapping/){:target="_blank"}
8. [Convex Hull - Brilliant](https://brilliant.org/wiki/convex-hull/){:target="_blank"}
9. [ConvexHull Documentation: OpenCV Docs](https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656){:target="_blank"}
10. [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_distance){:target="_blank"}
11. [Bitwise Operations - OpenCV](https://docs.opencv.org/trunk/d0/d86/tutorial_py_image_arithmetics.html){:target="_blank"}
12. [Find and Draw Contours using OpenCV - Python](https://www.geeksforgeeks.org/find-and-draw-contours-using-opencv-python/){:target="_blank"}
13. [Image Thresholding](https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html){:target="_blank"}
14. [Contour Features](https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html){:target="_blank"}