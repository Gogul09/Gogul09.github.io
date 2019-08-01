---
layout: live-demo
category: software
class: Demo
title: Emotion Recognizer
subheading: An interactive demo that uses deep learning to predict emotion after learning live in the browser using Google's Tensorflow.js
description: An interactive demo that uses deep learning to predict emotion after learning live in the browser using Google's Tensorflow.js
author: Gogul Ilango
permalink: /software/emotion-recognizer
image: https://drive.google.com/uc?id=1XyjJNykHALeJE_tZp81i86BqQgHkm5K_
---

<!--
<div id="mobile-tablet-warning" class="mobile-tablet-warning">
  <h3>Oops!</h3>
  <p>This demo is highly experimental in mobile and tablet devices. If you are still crazy in trying out this demo, I suggest you to use a laptop or a desktop. Thanks for your understanding! You're cute 😊</p>
  <a href="https://www.youtube.com/watch?v=WBi_PZ66z5E&feature=youtu.be" target="_blank">YouTube Video Link</a><br><br>
  <a href="https://github.com/Gogul09/emotion-recognizer" target="_blank">GitHub Code Link</a>
</div>
-->

<div class="emotion-container" id="emotion-container">
  <header class="emotion-header">
    <h3 onclick='showEmotionDescription()'>Emotion Recognizer</h3>
    <p id="emotion-description">This is a simple implementation of emotion recognition using two deep neural networks (extractor and classifier).<br>You can follow the instructions <a href="https://www.youtube.com/watch?v=WBi_PZ66z5E" target="_blank">here</a> to get this working. Made with <a href="https://github.com/Gogul09/emotion-recognizer" target="_blank">TensorFlow.js</a> and ❤️</p>
  </header>
  <div class="emotion-wrapper">
    <div class="emotion-player emotion-flex">  
      <div class="emotion-kit">
        <div id="emotion-laugh"  class="emotion-kit-comps">
          <img id="emoticon-laugh" src="/images/live-demo/emotion/laugh.gif" />
        </div>
        <div id="emotion-excited" class="emotion-kit-comps">
          <img id="emoticon-excited" src="/images/live-demo/emotion/excited.gif" />
        </div>
        <div id="emotion-sad" class="emotion-kit-comps">
          <img id="emoticon-sad" src="/images/live-demo/emotion/sad.gif" />
        </div>
        <div id="emotion-angry" class="emotion-kit-comps">
          <img id="emoticon-angry" src="/images/live-demo/emotion/angry.gif" />
        </div>
        <div id="emotion-sleep" class="emotion-kit-comps">
          <img id="emoticon-sleep" src="/images/live-demo/emotion/sleep.gif" />
        </div>
      </div>
      <div class="emotion-live-cam">
        <div>
          <video id="main-stream-video" width="200px" height="200px" ></video>
        </div>
      </div>
    </div>
    <!--- Controller starts here -->
    <div class="emotion-controller emotion-flex" id="emotion-controller">
      <!--- Tuner starts here -->
      <div class="emotion-tuner" id="emotion-tuner">
        <!--- Learning Rate starts here -->
        <div class="emotion-dropdown">
          <div class="emotion-dropdown-cell">
          <label>Learning Rate</label>
          <select id="emotion-learning-rate">
            <option>0.00001</option>
            <option selected>0.0001</option>
            <option>0.001</option>
            <option>0.003</option>
          </select>
        </div>
        </div>
        <!--- Learning Rate ends here -->
        <!--- Batch Size starts here -->
        <div class="emotion-dropdown">
          <div class="emotion-dropdown-cell">
          <label>Batch Size</label>
          <select id="emotion-batch-size">
            <option>0.05</option>
            <option selected>0.4</option>
            <option>0.1</option>
            <option>1</option>
          </select>
        </div>
        </div>
        <!--- Batch Size ends here -->
        <!--- Epochs starts here -->
        <div class="emotion-dropdown">
          <div class="emotion-dropdown-cell">
          <label>Epochs</label>
          <select id="emotion-epochs">
            <option>10</option>
            <option selected>20</option>
            <option>40</option>
          </select>
        </div>
        </div>
        <!--- Epochs ends here -->
        <!--- Hidden Units starts here -->
        <div class="emotion-dropdown">
          <div class="emotion-dropdown-cell">
          <label>Hidden Units</label>
          <select id="emotion-hidden-units">
            <option>10</option>
            <option selected>100</option>
            <option>200</option>
          </select>
        </div>
        </div>
        <!--- Hidden Units ends here -->
      </div>
      <!--- Tuner ends here -->
      <!--- Buttons starts here -->
      <div class="emotion-buttons">
        <button onclick="startWebcam()">Start</button>
        <button onclick="train()">Train</button>
        <button onclick="predictPlay()">Play</button>
        <button onclick="stopWebcam()">Stop</button>
      </div>
      <!--- Buttons ends here -->
      <!--- User acquire starts here -->
      <div class="emotion-acquire">
        <div class="emotion-acquire-outer">
          <div class="emotion-acquire-inner" id="sample-laugh" onclick="captureSample(this.id, 0)">
            <div class="emotion-acquire-cam">
              <img id="image-laugh" />
            </div>
            <p>Add 🤣</p>
          </div>
          <p id="count-laugh">0 samples</p>
        </div>
        <div class="emotion-acquire-outer">
          <div class="emotion-acquire-inner" id="sample-excited" onclick="captureSample(this.id, 1)">
            <div class="emotion-acquire-cam">
              <img id="image-excited" />
            </div>
            <p>Add 😲</p>
          </div>
          <p id="count-excited">0 samples</p>
        </div>
        <div class="emotion-acquire-outer">
          <div class="emotion-acquire-inner" id="sample-sad" onclick="captureSample(this.id, 2)">
            <div class="emotion-acquire-cam">
              <img id="image-sad" />
            </div>
            <p>Add 😰</p>
          </div>
          <p id="count-sad">0 samples</p>
        </div>
        <div class="emotion-acquire-outer">
          <div class="emotion-acquire-inner" id="sample-angry" onclick="captureSample(this.id, 3)">
            <div class="emotion-acquire-cam">
              <img id="image-angry" />
            </div>
            <p>Add 😡</p>
          </div>
          <p id="count-angry">0 samples</p>
        </div>
        <div class="emotion-acquire-outer">
          <div class="emotion-acquire-inner" id="sample-sleep" onclick="captureSample(this.id, 4)">
            <div class="emotion-acquire-cam">
              <img id="image-sleep" />
            </div>
            <p>Add 😴</p>
          </div>
          <p id="count-sleep">0 samples</p>
        </div>
      </div>
      <!--- User acquire ends here -->
    </div>
    <!--- Controller ends here -->
  </div>
</div>

<script src="https://unpkg.com/@tensorflow/tfjs"></script>
<script type="text/javascript" src="https://code.jquery.com/jquery-2.1.1.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.4.0/Chart.min.js"></script> 
<script type="text/javascript" src="/js/emotion-recognizer.js"></script>