---
layout: post
category: software
class: TensorFlow.js
title: Recognizing Digits using TensorFlow.js in Google Chrome
description: Learn how to recognize handwritten digits based on user's drawing in Google Chrome using a Deep Neural Network (Convolutional Neural Network and Multi-Layer Perceptron)
permalink: software/digit-recognizer-tf-js
image: https://drive.google.com/uc?id=1jyOD9zWy9GFPJjsRtYYVTBQxNk7S4kAt
cardimage: https://drive.google.com/uc?id=13W0uytLtG9tcGJHuMaGfNScvoOY5UMAq
---

<div class="git-showcase">
  <div>
    <a class="github-button" href="https://github.com/Gogul09" data-show-count="true" aria-label="Follow @Gogul09 on GitHub">Follow @Gogul09</a>
  </div>

  <div>
    <a class="github-button" href="https://github.com/Gogul09/digit-recognizer-live/fork" data-icon="octicon-repo-forked" data-show-count="true" aria-label="Fork Gogul09/digit-recognizer-live on GitHub">Fork</a>
  </div>

  <div>
    <a class="github-button" href="https://github.com/Gogul09/digit-recognizer-live" data-icon="octicon-star" data-show-count="true" aria-label="Star Gogul09/digit-recognizer-live on GitHub">Star</a>
  </div>

  <div>
    <a href="https://www.youtube.com/watch?v=WTaXfYOhqmY" target="_blank"><img src="/images/icons/youtube-icon.png" class="youtube-showcase" /></a>
  </div>    
</div>

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#digit-recognizer-live-demo">Digit Recognizer using TensorFlow.js Demo</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#deep-neural-network-for-digit-recognition">Deep Neural Network for Digit Recognition</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#javaScript-handlers-for-mouse-and-touch">JavaScript handlers for Mouse and Touch</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#tensorFlow-js-handlers-for-model-related-operations">TensorFlow JS handlers for model related operations</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#simple-bar-chart-to-display-the-predictions">Simple Bar Chart to display the predictions</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#references">References</a></li>
  </ul>
</div>

<p class="hundred-days"><span>#100DaysOfMLCode</span></p>

**In this blog post, we will create a simple web application that provides a canvas (mobile + desktop + laptop + tablet ready) for the user to draw a digit and uses a deep neural network (MLP or CNN) to predict what digit the user had drawn.**

As we already know the capabilities offered by [TensorFlow.js](https://js.tensorflow.org/){:target="_blank"}, we will extend the ideas to create two Deep Neural Networks (MLP and CNN) in Keras Python environment to recognize digits and use TensorFlow.js to predict the user drawn digit on a canvas in a web browser. In this learning path, we will restrict the user to draw a single digit between [0, 9] and later we will extend the idea to multiple digits. 

Below is the interactive demo that you can use to draw a digit between [0, 9] and the Deep Neural Network (MLP or CNN) that is running in your browser will predict what that digit is in the form of a bar chart.

<div class="note">
<p><b>Important</b>: This is a highly experimental demo for mobile and tablet devices. Please try this demo in laptop or desktop if you encounter issues in mobile and tablet devices. Also, please use Google Chrome as the browser to try this demo. Other browsers not supported as of now.
</p>
</div>

<div class="digit-demo-container">
  <h3 id="digit-recognizer-live-demo">Handwritten Digit Recognizer using TensorFlow.js Demo</h3>
  <div class="flex-two" style="margin-top: 20px;">
    <button id="clear_canvas" class="material-button-pink" onclick="clearCanvas(this.id)">Clear</button>
    <select id="select_model">
      <option>CNN</option>
      <option>MLP</option>
    </select>
    <button id="predict_canvas" class="material-button-pink" onclick="predict(this.id)">Predict</button>
  </div>
  <div class="flex-two">
    <div id="canvas_box_wrapper" class="canvas-box-wrapper">
      <div id="canvas_box" class="canvas-box"></div>
      <div id="canvas_output" class="canvas-output">
        <p>You have drawn</p>
        <img id="canvas_image" class="canvas-image" />
      </div>
    </div>
    <div id="result_box">
      <canvas id="chart_box" width="100" height="100"></canvas>
    </div>
  </div>
</div>

<div class="note">
<p><b>Note</b>: To follow this tutorial, I assume you have basic knowledge of Python, HTML5, CSS3, Sass, JavaScript, jQuery and basic command line usage.
</p>
</div>

<div class="downloads">
  <span>Downloads</span>
  <div><button title="Download HTML" onclick="window.open('https://github.com/Gogul09/digit-recognizer-live/blob/master/index.html', '_blank')">HTML</button></div>
  <div><button title="Download CSS" onclick="window.open('https://github.com/Gogul09/digit-recognizer-live/blob/master/css/app.css', '_blank')">CSS</button></div>
  <div><button title="Download JavaScript" onclick="window.open('https://github.com/Gogul09/digit-recognizer-live/tree/master/js', '_blank')">JavaScript</button></div>
  <div><button title="Download Python" onclick="window.open('https://github.com/Gogul09/digit-recognizer-live/blob/master/mnist_mlp.py', '_blank')">Python</button></div>
</div>

<h3 id="deep-neural-network-for-digit-recognition">Deep Neural Network for Digit Recognition</h3>

I have already posted a tutorial a year ago on how to build Deep Neural Nets (specifically a Multi-Layer Perceptron) to recognize hand-written digits using Keras and Python [here](https://gogul09.github.io/software/digits-recognition-mlp){:target="_blank"}. I highly encourage you to read that post before proceeding here. 

I assume you have familiarity in using Keras before proceeding (else please read my other tutorials on Keras [here](https://gogul09.github.io/software/first-neural-network-keras){:target="_blank"} and [here](https://gogul09.github.io/software/flower-recognition-deep-learning){:target="_blank"}). If you are a beginner to Keras, I strongly encourage you to visit [Keras documentation](https://keras.io/){:target="_blank"} where you could find tons of information on how to use the library and learn from examples found [here](https://github.com/keras-team/keras/tree/master/examples){:target="_blank"}.

For this tutorial, we will learn to create two popular Deep Neural Networks (DNN) namely Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN). For this digit recognition problem, MLP achieves pretty good accuracy. But we will build a CNN too and compare both these model performances live.

##### Simple MLP using Keras and Python

We will simply use a Keras MLP model using Python, dump out the model and weights in Tf.js layers format (as we did [here](https://gogul09.github.io/software/mobile-net-tensorflow-js){:target="_blank"}). After that, we will load the Keras dumped model and weights in our browser and use TensorFlow.js to make predictions. 

To do this, here is the python code that fetches and loads MNIST dataset, trains a simple multi-layer perceptron with two hidden layers having 512 and 256 neurons respectively on a training data of 60000 images with labels, validates the trained model with 10000 unlabeled images and saves the model along with weights in Tf.js layers format in <span class="coding">model_save_path</span>.

<div class="code-head">mnist_mlp.py<span>code</span></div>

```python
# organize imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.utils import np_utils
import tensorflowjs as tfjs

# fix a random seed for reproducibility
np.random.seed(9)

# user inputs
nb_epoch            = 25
num_classes         = 10
batch_size          = 64
train_size          = 60000
test_size           = 10000
v_length            = 784
model_save_path     = "output/mlp"

# split the mnist data into train and test
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()

# reshape and scale the data
trainData   = trainData.reshape(train_size, v_length)
testData    = testData.reshape(test_size, v_length)
trainData   = trainData.astype("float32")
testData    = testData.astype("float32")
trainData  /= 255
testData   /= 255

# convert class vectors to binary class matrices --> one-hot encoding
mTrainLabels  = np_utils.to_categorical(trainLabels, num_classes)
mTestLabels   = np_utils.to_categorical(testLabels, num_classes)

# create the MLP model
model = Sequential()
model.add(Dense(512, input_shape=(v_length,)))
model.add(Activation("relu"))
model.add(Dense(256))
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# compile the model
model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

# fit the model
history = model.fit(trainData, 
                    mTrainLabels,
                    validation_data=(testData, mTestLabels),
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=2)

# evaluate the model
scores = model.evaluate(testData, mTestLabels, verbose=0)

# print the results
print ("[INFO] test score - {}".format(scores[0]))
print ("[INFO] test accuracy - {}".format(scores[1]))

# save tf.js specific files in model_save_path
tfjs.converters.save_keras_model(model, model_save_path)
```

```
[INFO] test score - 0.16604800749952142
[INFO] test accuracy - 0.9828
```
{: .code-output}

Two important things to watch carefully here are **image size** and **input vector size** to MLP. Keras function <span class="coding">mnist.load_data()</span> loads images of size of **[28, 28]**. We flatten this image into a vector of size **784** for the MLP. We also scale the pixel values between **[0, 1]** for the algorithm to perform better. These are the image preprocessing operations that we will do in the front-end too using javascript.

After training and validating the MLP, we save the model architecture and weights using <span class="coding">tensorflowjs</span> under <span class="coding">model_save_path</span>. We will upload this folder to our website from where we could easily load this Keras model in TensorFlow.js using HTTPS request to make predictions in the browser. 

##### Simple CNN using Keras and Python

One caveat on using MNIST dataset from Keras is that the dataset is well cleaned, images are centered, cropped and aligned perfectly. So, there is very minimal preprocessing work required from a developer. But, the MLP model that we created above, isn't well suited for cases like HTML5 canvas where different users have different handwriting. 

That's why we need to create a CNN (Convolutional Neural Network) which automatically learns from 2D matrix instead of vectorizing the canvas into a 1d vector.

Below is the python code snippet to create a simple CNN that fetches and loads MNIST dataset, trains a simple CNN with 

* One <span class="coding">Convolution2D()</span> layer with 32 feature maps with size [5, 5] and <span class="coding">relu</span> activation function that takes in the input canvas size of [28, 28, 1].
* One <span class="coding">MaxPooling2D()</span> layer with a pool size of [2, 2].
* One <span class="coding">Dropout()</span> layer with argument 0.2 (meaning randomly removes 20% of neurons to reduce overfitting).
* One <span class="coding">Flatten()</span> layer that converts the 2D array to 1d vector for next layer.
* One <span class="coding">Dense()</span> layer with 128 neurons activated by <span class="coding">relu</span>.
* Final <span class="coding">softmax</span> activated <span class="coding">dense</span> layer with <span class="coding">num_classes</span> neurons.

<div class="code-head">mnist_cnn.py<span>code</span></div>

```python
# organize imports
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.datasets import mnist
from keras.utils import np_utils
import tensorflowjs as tfjs

# fix a random seed for reproducibility
np.random.seed(9)

# user inputs
nb_epoch            = 10
num_classes         = 10
batch_size          = 200
train_size          = 60000
test_size           = 10000
model_save_path     = "output/cnn"

# split the mnist data into train and test
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()

# reshape and scale the data
trainData = trainData.reshape(trainData.shape[0], 28, 28, 1)
testData  = testData.reshape(testData.shape[0], 28, 28, 1)
trainData   = trainData.astype("float32")
testData    = testData.astype("float32")
trainData  /= 255
testData   /= 255

# convert class vectors to binary class matrices --> one-hot encoding
mTrainLabels  = np_utils.to_categorical(trainLabels, num_classes)
mTestLabels   = np_utils.to_categorical(testLabels, num_classes)

# create the CNN model
model = Sequential()
model.add(Convolution2D(32, (5, 5), border_mode='valid', input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['accuracy'])

# fit the model
history = model.fit(trainData, 
                    mTrainLabels,
                    validation_data=(testData, mTestLabels),
                    batch_size=batch_size,
                    nb_epoch=nb_epoch,
                    verbose=2)

# evaluate the model
scores = model.evaluate(testData, mTestLabels, verbose=0)

# print the results
print ("[INFO] test score - {}".format(scores[0]))
print ("[INFO] test accuracy - {}".format(scores[1]))

# save tf.js specific files in model_save_path
tfjs.converters.save_keras_model(model, model_save_path)
```

```
[INFO] test score - 0.03036247751080955
[INFO] test accuracy - 0.9904
```
{: .code-output}

Again, look carefully at the preprocessing steps that we do here for CNN and the input shape that we pass in to the <span class="coding">Convolution2D</span> layer. For this tutorial, we have used TensorFlow image ordering format.

Notice how the test accuracy jumped from **0.9828** (MLP) to **0.9904** using CNN. Similar to MLP, we use <span class="coding">tensorflowjs</span> to dump the CNN model + weights to the <span class="coding">model_save_path</span> and we can load it in our server or webpage to make predictions.

<h3 id="javaScript-handlers-for-mouse-and-touch">JavaScript handlers for Mouse and Touch</h3>

Now let's get into the front-end code for this tutorial. Before we start anything with JavaScript, we first need to load the following JS libraries for everything in this tutorial to work. We need to append these lines in the <span class="coding">head</span> tag of our HTML.

<div class="code-head">index.html<span>code</span></div>

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script type="text/javascript" src="https://code.jquery.com/jquery-2.1.1.min.js"></script>  
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.4.0/Chart.min.js"></script>
<script type="text/javascript" src="/js/app.js"></script>
```

For the user to draw a digit using mobile or desktop or tablet or laptop, we need to create a HTML5 element called <span class="coding">canvas</span>. Inside the <span class="coding">canvas</span>, the user will draw the digit. We will feed the user drawn digit into the deep neural network that we have created to make predictions.

Below is the HTML5 code to create the UI of our simple web app. We have two buttons namely <span class="coding">Clear</span> to clear the canvas and <span class="coding">Predict</span> to make predictions using our deep neural network model. We also have a select option to select any one of the two Deep Neural Nets that we have created (MLP or CNN).

<div class="code-head">index.html<span>code</span></div>

```html
<div class="digit-demo-container">
  <h3 id="digit-recognizer-live-demo">Digit Recognizer using TensorFlow.js Demo</h3>
  <div class="flex-two" style="margin-top: 20px;">
    <button id="clear_canvas" class="material-button-pink" onclick="clearCanvas(this.id)">Clear</button>
    <select id="select_model">
      <option>MLP</option>
      <option>CNN</option>
    </select>
    <button id="predict_canvas" class="material-button-pink" onclick="predict(this.id)">Predict</button>
  </div>
  <div class="flex-two">
    <div id="canvas_box_wrapper" class="canvas-box-wrapper">
      <div id="canvas_box" class="canvas-box"></div>
    </div>
    <div id="result_box">
      <canvas id="chart_box" width="100" height="100"></canvas>
    </div>
  </div>
</div>
```

For a smooth user experience, we will add in some style for the above created HTML5 code. Shown below is the <span class="coding">app.scss</span> code which we will convert to <span class="coding">app.css</span> using <span class="coding">sass app.scss app.css</span> command. If you are not familiar with Sass, please learn about Sass [here](https://sass-lang.com/guide){:target="_blank"}.

<div class="code-head">app.scss<span>code</span></div>

```css
.digit-demo-container {
  background-color: #4CAF50;
  border-radius: 5px;
  margin: 20px auto;

  h3 {
    width: 100%;
    font-size: 15px;
    background-color: #356937;
    padding: 10px;
    color: white;
    margin: 0px;
    border: 1px solid #295b2b;
    border-bottom: 0px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
  }

  select {
    margin-top: 0px;
    height: 30px;
    font-size: 12px;
  }
}

.flex-two {
  display: flex;
  flex-wrap: wrap;
  div {
    flex: 1;
    padding: 20px;
    text-align: center;
    @include mobile {
      padding: 5px;
    }
  }
}

.canvas-box-wrapper {
  display: block !important;

}

.material-button-pink {
  background-color: #FFD740;
  height: 30px;
  color: black;
  display: block;
  font-family: $font_body;
  font-weight: bold;
  padding: 5px 10px;
  cursor: pointer;
  border: 1px solid #a58e3a;
  font-size: 12px;
  transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); 
  margin: 0px auto;
  border-radius: 5px;
}

.material-button-pink:focus {
  outline: none;
}

.material-button-pink:hover {
  box-shadow: 0 1px 8px rgba(0,0,0,0.3), 0 5px 10px rgba(0,0,0,0.22);
}

#chart_box {
  background-color: white;
  padding: 20px;
  border-radius: 5px;
  @include mobile {
    padding: 5px;
  }
}
```

Drawing inside a canvas is a little bit tricky in mobile and desktop. We need to be aware of all the jQuery handlers that are available for <span class="coding">mouse</span> and <span class="coding">touch</span>. Below are the jQuery event handlers that we will be using.

##### For Desktop & Laptop
* <span class="coding">mousedown</span>
* <span class="coding">mousemove</span>
* <span class="coding">mouseup</span>
* <span class="coding">mouseleave</span>

##### For Tablet & Mobile
* <span class="coding">touchstart</span>
* <span class="coding">touchmove</span>
* <span class="coding">touchend</span>
* <span class="coding">touchleave</span>

Additionally, we will add two more user-defined functions namely <span class="coding">addUserGesture</span> and <span class="coding">drawOnCanvas</span> which I will be explaining shortly.

First, we will have a <span class="coding">div</span> with id <span class="coding">canvas_box</span> inside which we will dynamically create a <span class="coding">canvas</span>. Below is the html code and JavaScript code to create a <span class="coding">canvas</span> inside which the user will draw.

<div class="code-head">index.html<span>code</span></div>

```html
<div id="canvas_box" class="canvas-box">
  <button id="clear_canvas" class="material-button-pink" onclick="clearCanvas(this.id)">Clear</button>
</div>
```

<div class="code-head">app.js<span>code</span></div>

```javascript
// GLOBAL variables
var modelName = "digitrecognizermlp";
let model;

// canvas related variables
// you can change these variables
var canvasWidth             = 150;
var canvasHeight            = 150;
var canvasStrokeStyle       = "white";
var canvasLineJoin          = "round";
var canvasLineWidth         = 12;
var canvasBackgroundColor   = "black";
var canvasId                = "canvas";

// variables to hold coordinates and dragging boolean
var clickX = new Array();
var clickY = new Array();
var clickD = new Array();
var drawing;

document.getElementById('chart_box').innerHTML = "";
document.getElementById('chart_box').style.display = "none";

//---------------------
// Create canvas
//---------------------
var canvasBox = document.getElementById('canvas_box');
var canvas    = document.createElement("canvas");

canvas.setAttribute("width", canvasWidth);
canvas.setAttribute("height", canvasHeight);
canvas.setAttribute("id", canvasId);
canvas.style.backgroundColor = canvasBackgroundColor;
canvasBox.appendChild(canvas);
if(typeof G_vmlCanvasManager != 'undefined') {
  canvas = G_vmlCanvasManager.initElement(canvas);
}

ctx = canvas.getContext("2d");
```

Notice we get the context <span class="coding">ctx</span> of the canvas that we created dynamically using <span class="coding">canvas.getContext("2d")</span>.

When the user draws on the canvas, we need to register the position X and Y within the browser. To do that, we make use of <span class="coding">mousedown</span> and <span class="coding">touchstart</span> functions. For mobile and tablet devices, we need to tell JavaScript to prevent the default functionality of scroll if canvas is touched using <span class="coding">e.preventDefault()</span> function. 

When the user starts drawing, we pass the X and Y values to <span class="coding">addUserGesture()</span> function and set the <span class="coding">drawing</span> flag <span class="coding">true</span>. 

Below two code snippets does these functions for both mobile and desktop devices.

<div class="code-head">app.js<span>code</span></div>

```javascript
//---------------------
// MOUSE DOWN function
//---------------------
$("#canvas").mousedown(function(e) {
  var mouseX = e.pageX - this.offsetLeft;
  var mouseY = e.pageY - this.offsetTop;

  drawing = true;
  addUserGesture(mouseX, mouseY);
  drawOnCanvas();
});

//---------------------
// TOUCH START function
//---------------------
canvas.addEventListener("touchstart", function (e) {
  if (e.target == canvas) {
      e.preventDefault();
    }

  var rect  = canvas.getBoundingClientRect();
  var touch = e.touches[0];

  var mouseX = touch.clientX - rect.left;
  var mouseY = touch.clientY - rect.top;

  drawing = true;
  addUserGesture(mouseX, mouseY);
  drawOnCanvas();

}, false);
```

We have asked JavaScript to just start recording the positions. But the user normally move his finger or move cursor to draw something on the canvas. To record the movement, we use <span class="coding">mousemove</span> and <span class="coding">touchmove</span> functions.

Only if the <span class="coding">drawing</span> bool is set (i.e the user have started to draw), we record the position X, Y and send the drawing boolean to <span class="coding">addUserGesture()</span> function. Then, we call <span class="coding">drawOnCanvas()</span> function to update the user's drawing which I will explain in a while.

<div class="code-head">app.js<span>code</span></div>

```javascript
//---------------------
// MOUSE MOVE function
//---------------------
$("#canvas").mousemove(function(e) {
  if(drawing) {
    var mouseX = e.pageX - this.offsetLeft;
    var mouseY = e.pageY - this.offsetTop;
    addUserGesture(mouseX, mouseY, true);
    drawOnCanvas();
  }
});

//---------------------
// TOUCH MOVE function
//---------------------
canvas.addEventListener("touchmove", function (e) {
  if (e.target == canvas) {
      e.preventDefault();
    }
  if(drawing) {
    var rect = canvas.getBoundingClientRect();
    var touch = e.touches[0];

    var mouseX = touch.clientX - rect.left;
    var mouseY = touch.clientY - rect.top;

    addUserGesture(mouseX, mouseY, true);
    drawOnCanvas();
  }
}, false);
```

During all the other cases, we simply make the <span class="coding">drawing</span> variable <span class="coding">false</span>. Below is the code snippet to do that.

<div class="code-head">app.js<span>code</span></div>

```javascript
//---------------------
// MOUSE UP function
//---------------------
$("#canvas").mouseup(function(e) {
  drawing = false;
});

//---------------------
// TOUCH END function
//---------------------
canvas.addEventListener("touchend", function (e) {
  if (e.target == canvas) {
      e.preventDefault();
    }
  drawing = false;
}, false);

//----------------------
// MOUSE LEAVE function
//----------------------
$("#canvas").mouseleave(function(e) {
  drawing = false;
});

//---------------------
// TOUCH LEAVE function
//---------------------
canvas.addEventListener("touchleave", function (e) {
  if (e.target == canvas) {
      e.preventDefault();
    }
  drawing = false;
}, false);
```

Finally, we will understand what <span class="coding">drawOnCanvas()</span> function does. First, we clear the canvas during each move or touch and then refill it with the values of X and Y using the <span class="coding">ctx</span> we obtained eariler for our canvas. We make use of canvas attributes such as <span class="coding">strokeStyle</span>, <span class="coding">lineJoin</span> and <span class="coding">lineWidth</span>, and canvas functions such as <span class="coding">beginPath()</span>, <span class="coding">moveTo()</span>, <span class="coding">lineTo()</span>, <span class="coding">closePath()</span> and <span class="coding">stroke()</span> to visualize what the user had drawn.

To clear the canvas, we simply use <span class="coding">clearRect</span> function and pass in the width and height of the canvas, and we reinitialize the position arrays.

<div class="code-head">app.js<span>code</span></div>

```javascript
//----------------------
// ADD CLICK function
//----------------------
function addUserGesture(x, y, dragging) {
  clickX.push(x);
  clickY.push(y);
  clickD.push(dragging);
}

//----------------------
// RE DRAW function
//----------------------
function drawOnCanvas() {
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

  ctx.strokeStyle = canvasStrokeStyle;
  ctx.lineJoin    = canvasLineJoin;
  ctx.lineWidth   = canvasLineWidth;

  for (var i = 0; i < clickX.length; i++) {
    ctx.beginPath();
    if(clickD[i] && i) {
      ctx.moveTo(clickX[i-1], clickY[i-1]);
    } else {
      ctx.moveTo(clickX[i]-1, clickY[i]);
    }
    ctx.lineTo(clickX[i], clickY[i]);
    ctx.closePath();
    ctx.stroke();
  }
}

//----------------------
// CLEAR CANVAS function
//----------------------
function clearCanvas(id) {
  ctx.clearRect(0, 0, canvasWidth, canvasHeight);
  clickX = new Array();
  clickY = new Array();
  clickD = new Array();
}
```

That's it! We have all the super-power javascript functions to make the user draw on a canvas in all devices such as desktop, laptop, tablet and mobile. Now, let's move on to our deep neural network model that we dumped eariler.

<h3 id="tensorFlow-js-handlers-for-model-related-operations">TensorFlow.js handlers for model related operations</h3>

You can load the entire model folder dumped eariler into your web app or server. We write three important functions that are related to our model in JavaScript.

1. <span class="coding">loadModel()</span> with select element handler
2. <span class="coding">preprocessCanvas()</span>
3. <span class="coding">predict()</span>

##### Load Model with select element handler
First, we load the model trained in Keras Python environment using a simple HTTPS request and <span class="coding">tf.loadModel()</span> function. We check if the model has loaded or not by using <span class="coding">console.log()</span>. Below is the code snippet to load the trained Keras model using TensorFlow.js.

<div class="code-head">app.js<span>code</span></div>

```javascript
//-----------------------
// select model handler
//-----------------------
$("#select_model").change(function() {
    var select_model  = document.getElementById("select_model");
    var select_option = select_model.options[select_model.selectedIndex].value;

    if (select_option == "MLP") {
      modelName = "digitrecognizermlp";

    } else if (select_option == "CNN") {
      modelName = "digitrecognizercnn";

    } else {
      modelName = "digitrecognizermlp";
    }

    loadModel(modelName);
});

//-------------------------------------
// loader for digitrecognizermlp model
//-------------------------------------
async function loadModel() {
  console.log("model loading..");

  // clear the model variable
  model = undefined;
  
  // load the model using a HTTPS request (where you have stored your model files)
  model = await tf.loadLayersModel("https://gogul09.github.io/models/" + modelName + "/model.json");
  
  console.log("model loaded..");
}

loadModel();
```

##### Preprocess Canvas
After loading the model, we need to preprocess the canvas drawn by the user to feed it to the DNN (MLP or CNN) that we have trained using Keras. 

<div class="note">
<p><b>Warning</b>: Preprocessing the HTML5 canvas element is the crucial step in this application.</p>
</div>

##### Preprocessing for MLP
1. We use <span class="coding">tf.browser.fromPixels()</span> and pass in the HTML5 canvas element directly without any transformations.
2. We resize the canvas into our MLP input image size of **[28, 28]** using <span class="coding">tf.resizeNearestNeighbor()</span> function.
3. We transform the canvas image into a grayscale image which becomes two-dimensional using <span class="coding">tf.mean(2)</span> function.
4. We convert all the values in the canvas to float using <span class="coding">tf.toFloat()</span> function and reshape the two-dimensional matrix into a row vector of shape **[1, 784]** to feed it to our MLP model using <span class="coding">tf.reshape()</span>.
5. Finally, we return the tensor after dividing each value in it by **255.0** using <span class="coding">tf.div()</span> as we did earlier during MLP model training.

##### Preprocessing for CNN
1. We use <span class="coding">tf.browser.fromPixels()</span> and pass in the HTML5 canvas element directly without any transformations.
2. We resize the canvas into our CNN input image size of **[28, 28]** using <span class="coding">tf.resizeNearestNeighbor()</span> function.
3. We transform the canvas image into a grayscale image which becomes two-dimensional using <span class="coding">tf.mean(2)</span> function.
4. We then expand the dimensions of the grayscale image into 4 dimensions as CNN expects the input to be 4D. To do this, we use <span class="coding">tf.expandDims(2)</span> to get a 3D matrix of shape **[28, 28, 1]** and then we use <span class="coding">tf.expandDims()</span> to get a 4D matrix of shape **[1, 28, 28, 1]**.
4. We convert all the values in the canvas to float using <span class="coding">tf.toFloat()</span> function.
5. Finally, we return the tensor after dividing each value in it by **255.0** using <span class="coding">tf.div()</span> as we did earlier during CNN model training.

<div class="code-head">app.js<span>code</span></div>

```javascript
//-----------------------------------------------
// preprocess the canvas to be DNN friendly
//-----------------------------------------------
function preprocessCanvas(image, modelName) {

  // if model is not available, send the tensor with expanded dimensions
  if (modelName === undefined) {
    alert("No model defined..")
  } 

  // if model is digitrecognizermlp, perform all the preprocessing
  else if (modelName === "digitrecognizermlp") {
    
    // resize the input image to digitrecognizermlp's target size of (784, )
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([28, 28])
        .mean(2)
        .toFloat()
        .reshape([1 , 784]);
    return tensor.div(255.0);
  }

  // if model is digitrecognizercnn, perform all the preprocessing
  else if (modelName === "digitrecognizercnn") {
    // resize the input image to digitrecognizermlp's target size of (1, 28, 28, 1)
    let tensor = tf.browser.fromPixels(image)
        .resizeNearestNeighbor([28, 28])
        .mean(2)
        .expandDims(2)
        .expandDims()
        .toFloat();
    console.log(tensor.shape);
    return tensor.div(255.0);
  }

  // else throw an error
  else {
    alert("Unknown model name..")
  }
}
```

##### Predict

Finally, we are ready to predict what the user has drawn using our loaded DNN (MLP or CNN) model with preprocessed canvas tensor available. 

We use the method <span class="coding">model.predict()</span> and pass in our canvas tensor as the argument and get the predictions using <span class="coding">data()</span>. We convert the predictions into a JavaScript array and use <span class="coding">displayChart()</span> to display the predictions in a visually pleasing format.

Displaying the model predictions in the form of a chart is optional. By the way, the model predictions are now available in the variable <span class="coding">results</span>.

<div class="code-head">app.js<span>code</span></div>

```javascript
//----------------------------
// Bounding box for centering
//----------------------------
function boundingBox() {
  var minX = Math.min.apply(Math, clickX) - 20;
  var maxX = Math.max.apply(Math, clickX) + 20;
  
  var minY = Math.min.apply(Math, clickY) - 20;
  var maxY = Math.max.apply(Math, clickY) + 20;

  var tempCanvas = document.createElement("canvas"),
  tCtx = tempCanvas.getContext("2d");

  tempCanvas.width  = maxX - minX;
  tempCanvas.height = maxY - minY;

  tCtx.drawImage(canvas, minX, minY, maxX - minX, maxY - minY, 0, 0, maxX - minX, maxY - minY);

  var imgBox = document.getElementById("canvas_image");
  imgBox.src = tempCanvas.toDataURL();

  return tempCanvas;
}

//--------------------------------------------
// predict function for digit recognizer mlp
//--------------------------------------------
async function predict() {

  // get the user drawn region alone cropped
  croppedCanvas = boundingBox();

  // show the cropped image 
  document.getElementById("canvas_output").style.display = "block";

  // preprocess canvas
  let tensor = preprocessCanvas(croppedCanvas, modelName);

  // make predictions on the preprocessed image tensor
  let predictions = await model.predict(tensor).data();

  // get the model's prediction results
  let results = Array.from(predictions)

  // display the predictions in chart
  displayChart(results)

  console.log(results);
}
```

<h3 id="simple-bar-chart-to-display-the-predictions">Simple Bar Chart to display the predictions</h3>

This section of this tutorial is optional. It is for people like me who are obsessed with data visualization.

You can use the below lines of code to display the predictions of our DNN (MLP or CNN) model in the form of a bar chart. I have used [Chart.js](https://www.chartjs.org/){:target="_blank"} which is a open-source JavaScript charting library.

<div class="code-head">app.js<span>code</span></div>

```javascript
//------------------------------
// Chart to display predictions
//------------------------------
var chart = "";
var firstTime = 0;
function loadChart(label, data, modelSelected) {
  var context = document.getElementById('chart_box').getContext('2d');
  chart = new Chart(context, {
      // we are in need of a bar chart
      type: 'bar',

      // we feed in data dynamically using data variable
      // that is passed as an argument to this function
      data: {
          labels: label,
          datasets: [{
              label: modelSelected + " prediction",
              backgroundColor: '#f50057',
              borderColor: 'rgb(255, 99, 132)',
              data: data,
          }]
      },

      // you can also play around with options for the 
      // chart if you find time!
      options: {}
  });
}

//----------------------------
// display chart with updated
// drawing from canvas
//----------------------------
function displayChart(data) {
  var select_model  = document.getElementById("select_model");
  var select_option = select_model.options[select_model.selectedIndex].value;
  
  label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
  if (firstTime == 0) {
    loadChart(label, data, select_option);
    firstTime = 1;
  } else {
    chart.destroy();
    loadChart(label, data, select_option);
  }
  document.getElementById('chart_box').style.display = "block";
}
```

That's it! Finally, we have built something awesome using multiple programming languages such as JavaScript, Python, HTML5 and CSS3. It's all possible because of two amazing deep learning libraries such as Keras and TensorFlow.js.

### References

1. [MNIST Database](https://en.wikipedia.org/wiki/MNIST_database){:target="_blank"}
2. [MNIST For ML Beginners](https://www.tensorflow.org/versions/r1.0/get_started/mnist/beginners){:target="_blank"}
3. [Keras Official Documentation](https://keras.io/){:target="_blank"}
4. [TensorFlow.js Official Documentation](https://js.tensorflow.org/){:target="_blank"}
5. [Create a Drawing App with HTML5 Canvas and JavaScript](http://www.williammalone.com/articles/create-html5-canvas-javascript-drawing-app/){:target="_blank"}
6. [Multi-Layer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron){:target="_blank"}
7. [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network){:target="_blank"}

<script src="https://unpkg.com/@tensorflow/tfjs"></script>
<script type="text/javascript" src="https://code.jquery.com/jquery-2.1.1.min.js"></script>  
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.4.0/Chart.min.js"></script>
<script type="text/javascript" src="/js/digits-recognizer.js"></script>