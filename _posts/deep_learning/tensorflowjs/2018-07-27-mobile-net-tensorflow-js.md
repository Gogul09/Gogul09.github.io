---
layout: post
category: software
class: TensorFlow.js
title: Classifying images using Keras MobileNet and TensorFlow.js in Google Chrome
description: Learn how to perform image classification (recognition) using Keras MobileNet and TensorFlow.js.
permalink: software/mobile-net-tensorflow-js
image: https://drive.google.com/uc?id=1qiRsYptVoaqViIJKC7CmriYnG3YDbfpS
cardimage: https://drive.google.com/uc?id=1I1FMMaBYsny0cckcCgCh8kRaOcX5UyQU
---

<div class="git-showcase">
  <div>
  <a class="github-button" href="https://github.com/Gogul09" data-show-count="true" aria-label="Follow @Gogul09 on GitHub">Follow @Gogul09</a>
  </div>

  <div>
  <a class="github-button" href="https://github.com/Gogul09/mobile-net-projects/fork" data-icon="octicon-repo-forked" data-show-count="true" aria-label="Fork Gogul09/mobile-net-projects on GitHub">Fork</a>
  </div>

  <div>
  <a class="github-button" href="https://github.com/Gogul09/mobile-net-projects" data-icon="octicon-star" data-show-count="true" aria-label="Star Gogul09/mobile-net-projects on GitHub">Star</a>
  </div>  
</div>

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#basic-mobilenet-in-python">Basic MobileNet in Python</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#convert-keras-model-into-tf-js-layers-format">Convert Keras model into Tf JS layers format</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#keras-model-into-tensorflow-js">Keras model into TensorFlow JS</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#load-keras-model-into-tf-js">1. Load Keras model into TF.js</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#upload-image-from-disk">2. Upload image from disk</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#predict-using-mobilenet-model">3. Predict using MobileNet model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#references">References</a></li>
  </ul>
</div>

<p class="hundred-days"><span>#100DaysOfMLCode</span></p>

**In this blog post, we will understand how to perform image classification using Keras MobileNet, deploy it in Google Chrome using TensorFlow.js and use it to make live predictions in the browser.**

Being a Python developer with knowledge on Web Development is surely going to help you in the field of Artificial Intelligence in the long run. Because we now have the awesome capabilities of Keras and TensorFlow in a web browser using [TensorFlow.js](https://js.tensorflow.org/){:target="_blank"}.

The interactive demo of what we will be making at the end of the tutorial is shown below. You can play around with the buttons provided to make predictions on an image.

<ol>
  <li>Click <b>Load Model</b> to load MobileNet model in your browser.</li>
  <li>Loading image -
    <ul style="margin-bottom: 0px !important;">  
      <li>Click <b>Demo Image</b> to import a random image that belongs to an ImageNet category.</li>
      <li>Click <b>Upload Image</b> if you want to import an image from your disk.</li>
    </ul>
  </li>
  <li>Click <b>Predict</b> to make predictions on the image loaded in the browser.</li>
</ol>

<div class="demo-container">
  <h3>Keras MobileNet + TensorFlow.js Demo</h3>
  <div class="demo-wrapper">
    <div>
      <button id="load-button" class="input-button" onclick="loadModel()">Load Model</button>
      <button id="demo-image-button" class="input-button" onclick="loadDemoImage()">Demo Image</button>
      <label for="select-file-image" class="input-button">
          Upload Image
      </label>
      <input id="select-file-image" type="file" style="display: none;">
      <div id="progress-box" style="display: none; width: 100% !important;">
        <img src="/images/software/mobile-net-tensorflow-js/loading.gif" id="demo-load" width="100px" />
        <p style="color: white;">Loading mobilenet model..</p>
      </div>
    </div>
    <div>
      <button id="predict-button" class="input-button predict-button">Predict</button>
    </div>
  </div>
  <div class="demo-output">
    <div class="out-box" id="select-file-box" style="display: none;">
      <img id='test-image' />
    </div>
    <div class="out-box" id="predict-box" style="display: none;">
      <p id="prediction"></p>
      <br>
      <p><b style="color: #c2c2c2 !important; font-style: italic; font-weight: 100; font-size: 11px;">Top-5 Predictions</b></p>
      <ul id="predict-list">
      </ul>
    </div>
  </div>
</div>

<div class="note">
<p><b>Note:</b> The above demo uses state-of-the-art Keras MobileNet that's trained on ImageNet with <b>1000 categories</b>. If you upload an image that doesn't belong to any of the 1000 ImageNet categories, then the prediction <b>might not</b> be accurate!</p>
</div>

<div class="downloads">
  <span>Downloads</span>
  <div><button title="Download HTML" onclick="window.open('https://github.com/Gogul09/mobile-net-projects/blob/master/index.html', '_blank')">HTML</button></div>
  <div><button title="Download CSS" onclick="window.open('https://github.com/Gogul09/mobile-net-projects/blob/master/app.css', '_blank')">CSS</button></div>
  <div><button title="Download JavaScript" onclick="window.open('https://github.com/Gogul09/mobile-net-projects/blob/master/mobile-net.js', '_blank')">JavaScript</button></div>
  <div><button title="Download Python" onclick="window.open('https://github.com/Gogul09/mobile-net-projects/blob/master/basic-mobile-net.py', '_blank')">Python</button></div>
</div>

### Basic MobileNet in Python

In Keras, MobileNet resides in the <span class="coding">applications</span> module. Keras offers out of the box image classification using MobileNet if the category you want to predict is available in the [ImageNet categories](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a){:target="_blank"}. If the category doesn't exist in ImageNet categories, there is a method called fine-tuning that tunes MobileNet for your dataset and classes which we will discuss in another tutorial.

MobileNet offers tons of advantages than other state-of-the-art convolutional neural networks such as VGG16, VGG19, ResNet50, InceptionV3 and Xception.
* MobileNets are light weight deep neural networks best suited for mobile and embedded vision applications.
* MobileNets are based on a streamlined architecture that uses depthwise separable convolutions.
* MobileNet uses two simple global hyperparameters that efficiently trades off between accuracy and latency.
* MobileNet could be used in object detection, finegrain classification, face recognition, large-scale geo localization etc.

Following are the advantages of using MobileNet over other state-of-the-art deep learning models.
* Reduced network size - **17MB**.
* Reduced number of parameters - **4.2 million**.
* Faster in performance and are useful for mobile applications.
* Small, low-latency convolutional neural network.

Advantages always come up with some disadvantages and with MobileNet, it's the accuracy. Yes! Eventhough MobileNet has reduced size, reduced parameters and performs faster, it is less accurate than other state-of-the-art networks as discussed in [this](https://arxiv.org/pdf/1704.04861.pdf){:target="_blank"} paper. But don't worry. There is only a slight reduction in accuracy when compared to other networks.

In this tutorial, we will follow the steps shown in Figure 1 to make Keras MobileNet available in a web browser using TensorFlow.js.

<figure>
  <img src="/images/software/mobile-net-tensorflow-js/concept.png" class="typical-image">
  <figcaption>Figure 1. Keras MobileNet in Google Chrome using TensorFlow.js</figcaption>
</figure>

First, we will write a simple python script to make predictions on a test image using Keras MobileNet.

Before sending an image into MobileNet, we need to process the image using 4 simple steps. And to do that, you don't need OpenCV. Keras provides all the necessary functions under <span class="coding">keras.preprocessing</span> module, and with some basic numpy functions, you are ready to go!

1. Load the image and convert it to MobileNet's input size (224, 224) using <span class="coding">load_img()</span> function.
2. Convert the image into a numpy array using <span class="coding">img_to_array()</span>.
3. Expand the dimensions of the numpy array using <span class="coding">np.expand_dims()</span>.
4. Preprocess the image by rescaling all the values to the range [-1, 1] using <span class="coding">mobilenet.preprocess_input()</span>.

<div class="code-head">basic-mobile-net.py<span>code</span></div>

```python
# organize imports
import numpy as np
from keras.models import Model
from keras.preprocessing import image
from keras.applications import imagenet_utils, mobilenet
import tensorflowjs as tfjs

# process an image to be mobilenet friendly
def process_image(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  img_array = image.img_to_array(img)
  img_array = np.expand_dims(img_array, axis=0)
  pImg = mobilenet.preprocess_input(img_array)
  return pImg

# main function
if __name__ == '__main__':

  # path to test image
  test_img_path = "G:\\git-repos\\mobile-net-projects\\dataset\\test\\test_image_1.jpg"

  # process the test image
  pImg = process_image(test_img_path)

  # define the mobilenet model
  mobilenet = mobilenet.MobileNet()

  # make predictions on test image using mobilenet
  prediction = mobilenet.predict(pImg)

  # obtain the top-5 predictions
  results = imagenet_utils.decode_predictions(prediction)
  print(results)

  # convert the mobilenet model into tf.js model
  save_path = "output\\mobilenet"
  tfjs.converters.save_keras_model(mobilenet, save_path)
  print("[INFO] saved tf.js mobilenet model to disk..")
```

```
[[('n01806143', 'peacock', 0.9998889), 
  ('n01806567', 'quail', 3.463593e-05), 
  ('n02018795', 'bustard', 2.7573227e-05), 
  ('n01847000', 'drake', 1.1352683e-05), 
  ('n01795545', 'black_grouse', 1.0532762e-05)]]
```
{: .code-output}

* Lines 2-5 imports all the necessary functions to work with.
* Lines 8-13 is the special definition we use to process an image so that it becomes MobileNet friendly.
* Line 19 defines the test image path.
* Line 22 preprocesses the test image.
* Line 25 instantiates the MobileNet model.
* Line 28 makes predictions on the test image using MobileNet model.
* Line 31 gives the top-5 predictions of the test image.
* Line 32 prints out the top-5 predictions of the test image.
* Lines 36-38 converts <a href="#convert-keras-model-into-tf-js-layers-format">keras mobilenet model into tf.js layers format</a> at <span class="coding">save_path</span>.

Please make sure you change the <span class="coding">test_img_path</span> in line 19 to test an image from your disk. Figure 2 (shown below) is the test image that I have chosen and the MobileNet model accurately predicted it as a <span class="coding">peacock</span> with a probability of 99.99%. Pretty cool! 😍 

<figure>
  <img src="/images/software/mobile-net-tensorflow-js/test_image_1.jpg" class="typical-image">
  <figcaption>Figure 2. Input test image for MobileNet</figcaption>
</figure>

Cool! Everything works perfectly in our Python environment. Now, we will use this pretrained mobile net model in a web browser.

<h3 id="convert-keras-model-into-tf-js-layers-format">Convert Keras model into Tf.js layers format</h3>

Before deploying a keras model in web, we need to convert the Keras mobilenet python model into tf.js layers format (which we already did in lines 36-38).

To deploy a Keras model in web, we need a package called <span class="coding">tensorflowjs</span>. Run the below command to get it.

<div class="code-head">install tensorflowjs<span>cmd</span></div>

```python
pip install tensorflowjs
```

After installing it, you can either run the command as a standalone one or you can integrate it in your python script as shown below (which I prefer).

<div class="code-head">1. keras to tf.js layers format<span>cmd</span></div>

```python
tensorflowjs_converter --input_format keras \
                       path_to_keras_model.h5 \
                       path/to/tfjs_target_dir
```

<div class="code-head">2. inside python script<span>code</span></div>

```python
import tensorflowjs as tfjs

def train(...):
    model = keras.models.Sequential() # for example
    ...
    model.compile(...)
    model.fit(...)
    tfjs.converters.save_keras_model(model, tfjs_target_path)
```

The <span class="coding">tfjs_target_path</span> or <span class="coding">save_path</span> (in our case) is a folder that contains <span class="coding">model.json</span> and a set of sharded weight binary files. If you take a look into <span class="coding">model.json</span> file, you will see the model architecture or graph (a description of layers and their connections) plus a manifest of the weight files.

### Keras model into TensorFlow JS

For this tutorial, I used my GitHub pages repo to hold the keras mobilenet model files. I copied the entire folder under <span class="coding">save_path</span> [here](https://gogul09.github.io/models/mobilenet/model.json){:target="_blank"}.

This is crucial for our application to work because if you host these model files in a different server, you might face [CORS issue](https://enable-cors.org/){:target="_blank"} in your web app. Storing your model files in the same domain as your web app is safer and preferred way.

Let's get started with the sweet TensorFlow.js 😘

You need these three javascript libraries loaded in your website. 
1. <span class="coding">IMAGENET_CLASSES</span> variable that has all the ImageNet categories indexed which could easily be loaded from [here](https://github.com/tensorflow/tfjs-examples/blob/master/mobilenet/imagenet_classes.js){:target="_blank"}.
2. TensorFlow.js latest source.
3. jQuery to make JavaScript easier.

<div class="code-head">index.html<span>code</span></div>

```html
<script type="text/javascript" src="/js/imagenet_classes.js"></script> 
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script type="text/javascript" src="https://code.jquery.com/jquery-2.1.1.min.js"></script>  
```

Once you load all the above three scripts, you can open up a new file named <span class="coding">mobile-net.js</span> that will have all the functionality needed to make Keras MobileNet model work in a web browser.

The user interface that I made at the start of the tutorial has HTML, CSS and JavaScript code combined. We will look into model specific part instead of looking into every single line of code.

<h3 id="load-keras-model-into-tf-js">1. Load Keras model into TF.js</h3>

Firstly, you need to load the Keras pretrained model json that you have stored in your web server. To do this, you can use the below code snippet. 

The below code snippet is an <span class="coding">async</span> function that loads a keras model json using <span class="coding">tf.loadModel()</span>. In line 17, <span class="coding">await</span> means without disturbing the UI, you are asking JavaScript to load model behind the scenes. To view the status of model loading, we use a progress bar as well as <span class="coding">console.log()</span>.

<div class="code-head">mobile-net.js<span>code</span></div>

```javascript
let model;
async function loadModel() {
  console.log("model loading..");

  // display model loading progress box
  loader = document.getElementById("progress-box");
  load_button = document.getElementById("load-button");
  loader.style.display = "block";

  // model name is "mobilenet"
  modelName = "mobilenet";
  
  // clear the model variable
  model = undefined;
  
  // load the model using a HTTPS request (where you have stored your model files)
  model = await tf.loadLayersModel('https://gogul09.github.io/models/mobilenet/model.json');
  
  // hide model loading progress box
  loader.style.display = "none";
  load_button.disabled = true;
  load_button.innerHTML = "Loaded Model";
  console.log("model loaded..");
}
```

<h3 id="upload-image-from-disk">2. Upload image from disk</h3>

To upload an image from disk, you can use the below code snippet which makes use of HTML5 File API. I have used a button **Upload Image** which has a <span class="coding">change</span> handler associated with it.

<div class="code-head">index.html<span>code</span></div>

```html
<!-- used to get image from disk -->
<input id="select-file-image" type="file">
```

<div class="code-head">mobile-net.js<span>code</span></div>

```javascript
// if there is a change to "Upload Image" button, 
// load and render the image
$("#select-file-image").change(function() {
  renderImage(this.files[0]);
}

// renders the image which is loaded from disk to the img tag 
function renderImage(file) {
  var reader = new FileReader();
  reader.onload = function(event) {
    img_url = event.target.result;
    document.getElementById("test-image").src = img_url;
  }
  reader.readAsDataURL(file);
}
```

<h3 id="predict-using-mobilenet-model">3. Predict using MobileNet model</h3>

To make predictions using mobilenet that's now loaded into Tf.js environment, we need to perform two steps.
1. Preprocess the input image to be mobilenet friendly.
2. Make predicitons on the input image.

#### 3.1 Preprocess the input image
As I have already mentioned, input image size to mobilenet is [224, 224] as well as the features are scaled between [-1, 1]. You need to perform these two steps before making predictions using the model. To do this, we use <span class="coding">preprocessImage()</span> function that takes in two arguments <span class="coding">image</span> and <span class="coding">modelName</span>.

The input image can easily be loaded using <span class="coding">tf.fromPixels()</span> , resized using <span class="coding">resizeNearestNeighbor()</span> and converting all the values in the image to float using <span class="coding">toFloat()</span>.

After that, we feature scale the values in the image tensor using a scalar value of 127.5 which is the center value of image pixel range [0, 255]. For each pixel value in the image, we subtract this offset value and divide by this offset value to scale between [-1, 1]. We then expand the dimensions using <span class="coding">expandDims()</span>.

<div class="code-head">mobile-net.js<span>code</span></div>

```javascript
// preprocess the image to be mobilenet friendly
function preprocessImage(image, modelName) {

  // resize the input image to mobilenet's target size of (224, 224)
  let tensor = tf.browser.fromPixels(image)
    .resizeNearestNeighbor([224, 224])
    .toFloat();

  // if model is not available, send the tensor with expanded dimensions
  if (modelName === undefined) {
    return tensor.expandDims();
  } 

  // if model is mobilenet, feature scale tensor image to range [-1, 1]
  else if (modelName === "mobilenet") {
    let offset = tf.scalar(127.5);
    return tensor.sub(offset)
      .div(offset)
      .expandDims();
  } 

  // else throw an error
  else {
    alert("Unknown model name..")
  }
}
```

#### 3.2 Predict using Tf.js model

After preprocessing the image, I have made a handler for **Predict** button. Again, this is also an <span class="coding">async</span> function that uses <span class="coding">await</span> till the model make successfull predictions.

Prediction using a Tf.js model is straightforward as Keras which uses <span class="coding">model.predict(tensor)</span>. To get the predictions, we pass it <span class="coding">data()</span> to the former.

Results from the predictions are mapped to an array named <span class="coding">results</span> using <span class="coding">IMAGENET_CLASSES</span> that we loaded at the beginning of this tutorial. We also sort this array based on the probability that is highest using <span class="coding">sort()</span> and take only the top-5 probabilities using <span class="coding">slice()</span>. 

<div class="code-head">mobile-net.js<span>code</span></div>

```javascript
// If "Predict Button" is clicked, preprocess the image and
// make predictions using mobilenet
$("#predict-button").click(async function () {
  // check if model loaded
  if (model == undefined) {
    alert("Please load the model first..")
  }

  // check if image loaded
  if (document.getElementById("predict-box").style.display == "none") {
    alert("Please load an image using 'Demo Image' or 'Upload Image' button..")
  }

  // html-image element can be given to tf.fromPixels
  let image  = document.getElementById("test-image");
  let tensor = preprocessImage(image, modelName);

  // make predictions on the preprocessed image tensor
  let predictions = await model.predict(tensor).data();

  // get the model's prediction results
  let results = Array.from(predictions)
    .map(function (p, i) {
      return {
        probability: p,
        className: IMAGENET_CLASSES[i]
      };
    }).sort(function (a, b) {
      return b.probability - a.probability;
    }).slice(0, 5);

  // display the top-1 prediction of the model
  document.getElementById("results-box").style.display = "block";
  document.getElementById("prediction").innerHTML = "MobileNet prediction - <b>" + results[0].className + "</b>";

  // display top-5 predictions of the model
  var ul = document.getElementById("predict-list");
  ul.innerHTML = "";
  results.forEach(function (p) {
    console.log(p.className + " " + p.probability.toFixed(6));
    var li = document.createElement("LI");
    li.innerHTML = p.className + " " + p.probability.toFixed(6);
    ul.appendChild(li);
  });
});
```

There you go! We now have the power of state-of-the-art Keras pretrained model MobileNet in a client browser that is able to make predictions on images that belong to ImageNet category.

Notice that the mobilenet model loads very quickly in the browser and makes predictions very fast 😎

<script type="text/javascript" src="/js/imagenet_classes.js"></script> 
<script src="https://unpkg.com/@tensorflow/tfjs"></script>
<script type="text/javascript" src="https://code.jquery.com/jquery-2.1.1.min.js"></script>  
<script type="text/javascript" src="/js/mobile-net.js"></script>

### References

1. [TensorFlow.js - Official Documentation](https://js.tensorflow.org/){:target="_blank"}
2. [Keras - Official Documentation](https://keras.io/){:target="_blank"}
3. [Importing a Keras model into TensorFlow.js](https://js.tensorflow.org/tutorials/import-keras.html){:target="_blank"}
4. [Introduction to TensorFlow.js - Intelligence and Learning](https://www.youtube.com/watch?v=Qt3ZABW5lD0){:target="_blank"}
5. [TensorFlow.js: Tensors - Intelligence and Learning](https://www.youtube.com/watch?v=D-XzAeVvMkg){:target="_blank"}
6. [TensorFlow.js Quick Start](https://www.youtube.com/watch?v=Y_XM3Bu-4yc){:target="_blank"}
7. [Session 6 - TensorFlow.js - Intelligence and Learning](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6YIeVA3dNxbR9PYj4wV31oQ){:target="_blank"}
8. [Session 7 - TensorFlow.js Color Classifier - Intelligence and Learning](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6bmMRCIoTi72aNWHo7epX4L){:target="_blank"}
9. [Tensorflow.js Explained](https://www.youtube.com/watch?v=Nc8kZABv-KE){:target="_blank"}
10. [Webcam Tracking with Tensorflow.js](https://www.youtube.com/watch?v=9KqNk5keyCc){:target="_blank"}
11. [Try TensorFlow.js in your browser](https://www.youtube.com/watch?v=pbCExciEbrc){:target="_blank"}
