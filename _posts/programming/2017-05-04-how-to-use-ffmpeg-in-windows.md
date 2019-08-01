---
layout: post
category: software
class: Tools
title: How to use ffmpeg in Windows?
description: This tutorial is for beginners to get started with ffmpeg for any video related operation.
author: Gogul Ilango
permalink: /software/ffmpeg-basics
image: https://drive.google.com/uc?id=1NrhTV8rczqhTdD8P_BY8Aj0i1e0EQ0MG
---

<span class="coding">ffmpeg</span> is the coolest application when it comes to playing with a video file. Instead of downloading so many useless converters to convert from one file format to another, <span class="coding">ffmpeg</span> completes any task using simple commands given to it. In this tutorial, I will list out the most useful commands one needs for video conversion and video-specific operations.

### Installation
You can easily install <span class="coding">ffmpeg</span> for Windows from [this](http://ffmpeg.zeranoe.com/builds/) website. 

* Choose the correct architecture (64-bit/32-bit) and click on <span class="coding">Download FFmpeg</span>. 
* After downloading, unzip the file and rename it to <span class="coding">ffmpeg</span>. 
* Go into the folder where you will find a <span class="coding">bin</span> folder. 
* Copy the path of the <span class="coding">bin</span> folder (For me it was, *C:\Users\gogul\Downloads\ffmpeg\bin*). 
* Right-click <span class="coding">Computer</span> and click on <span class="coding">Properties</span>.
* Then, click on <span class="coding">Advanced System Settings</span> and then, click the <span class="coding">Environmental Variables</span> button.
* You will find two sections - User variables and System variables. Under User variables, there will be a variable named <span class="coding">path</span>. Click that and click on the <span class="coding">Edit</span> button. Add the <span class="coding">bin</span> path to the variable value after a ";".
* Now, click <span class="coding">Ok</span> and close all the windows.
* Open up the command prompt and type <span class="coding">FFMPEG</span>. You should get the following output. It means that you have successfully installed <span class="coding">ffmpeg</span>.

### Convert .avi to .mp4

```python
ffmpeg -i input.avi -c:v libx264 -crf 19 -preset slow -c:a libfdk_aac -b:a 192k -ac 2 output.mp4
```

If <span class="coding">libfdk_aac</span> is not available, then use the following.

```python
ffmpeg -i input.avi -c:v libx264 -crf 19 -preset slow -c:a aac -b:a 192k -ac 2 out.mp4
```

### Convert .mp4 to .gif

The below two commands skips first 11 seconds of the video (.mp4) and outputs a 10 second ".gif" which is 640 pixels wide with aspect ratio preserved.

```python
ffmpeg -y -ss 11 -t 10 -i input.mp4 -vf fps=10,scale=640:-1:flags=lanczos,palettegen out.png

ffmpeg -ss 11 -t 10 -i input.mp4 -i out.png -filter_complex "fps=10,scale=640:-1:flags=lanczos[x];[x][1:v]paletteuse" out.gif
```