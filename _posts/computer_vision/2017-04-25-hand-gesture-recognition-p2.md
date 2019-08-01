---
layout: post
category: software
class: Computer Vision
title: Hand Gesture Recognition using Python and OpenCV - Part 2
description: Learn how to recognize hand gestures after applying background subtraction using OpenCV and Python.
author: Gogul Ilango
permalink: software/hand-gesture-recognition-p2
image: https://drive.google.com/uc?id=1n1IwjE8eRpKnnlw1JJthbsVtT_m3DFSr
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
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#summary">Summary</a></li>
  </ul>
</div>

This is a follow-up post of my tutorial on Hand Gesture Recognition using OpenCV and Python. Please read the first part of the tutorial <a href="https://gogul09.github.io/software/hand-gesture-recognition-p1" target="_blank">here</a> and then come back. 

In the previous tutorial, we have used Background Subtraction, Motion Detection and Thresholding to segment our hand region from a live video sequence. In this tutorial, we will take one step further to recognize the number of fingers shown in a live video sequence.

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
1. Find the convex hull of the segmented hand region (which is a contour) and compute the most extreme points in the convex hull (Extreme Top, Extreme Bottom, Extreme Left, Extreme Right).
2. Find the center of palm using these extremes points in the convex hull.
3. Using the palm's center, construct a circle with the maximum Euclidean distance (between the palm's center and the extreme points) as radius.
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
The outline or the boundary of the object of interest. This contour could easily be found using OpenCV's <span class="coding">cv2.findContours()</span> function. Be careful while unpacking the return value of this function, as we need three variables to unpack this tuple in OpenCV 3.1.0 - [Contours](http://docs.opencv.org/3.2.0/d4/d73/tutorial_py_contours_begin.html){:target="_blank"}.

### Bitwise-AND 
Performs bit-wise logical AND between two objects. You could visually think of this as using a mask and extracting the regions in an image that lie under this mask alone. OpenCV provides <span class="coding">cv2.bitwise_and()</span> function to perform this operation - [Bitwise AND](http://docs.opencv.org/trunk/d0/d86/tutorial_py_image_arithmetics.html){:target="_blank"}.

### Euclidean Distance 
This is the distance between two points given by the equation shown [here](https://bigsnarf.files.wordpress.com/2012/03/distance.jpg){:target="_blank"}. Scikit-learn provides a function called <span class="coding">pairwise.euclidean_distances()</span> to calculate the Euclidean distance from *one point* to *multiple points* in a single line of code - [Pairwise Euclidean Distance](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html){:target="_blank"}. After that, we take the *maximum* of all these distances using NumPy's <span class="coding">argmax()</span> function.

### Convex Hull 
You can think of convex hull as a dynamic, stretchable envelope that wraps around the object of interest. To read more about it, visit [this](http://docs.opencv.org/trunk/dd/d49/tutorial_py_contour_features.html){:target="_blank"} link.


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

After that, you can use bring in your hand into the bounding box, show gestures and the count of fingers will be displayed accordingly. I have included a demo of the entire pipeline below.

<figure>
	<img src="/images/software/gesture-recognition/demo.gif" class="typical-image" />
	<figcaption>Figure 2. Hand-Gesture Recognition | Counting the fingers | Demo</figcaption>
</figure>

### Summary
In this tutorial, we have learnt about recognizing hand gestures using Python and OpenCV. We have explored Background Subtraction, Thresholding, Segmentation, Contour Extraction, Convex Hull and Bitwise-AND operation on real-time video sequence. We have followed the methodology proposed by Malima et al. to quickly recognize hand gestures. 

You could extend this idea by using the count of fingers to instruct a robot to perform some task like picking up an object, go forward, move backward etc. using Arduino or Raspberry Pi platforms. I have also made a simple demo for you by using the count of fingers to control a servo motor's rotation [here](https://www.youtube.com/watch?v=4lCjQ84EkSk){:target="_blank"}.
