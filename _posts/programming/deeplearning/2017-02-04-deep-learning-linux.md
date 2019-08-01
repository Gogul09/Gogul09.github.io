---
layout: post
category: software
class: Environment Setup
title: Deep Learning Environment Setup for Linux
description: This tutorial is for beginners who need to setup environment for Deep Learning and Computer Vision in Linux.
author: Gogul Ilango
permalink: /software/deep-learning-linux
image: https://drive.google.com/uc?id=1hgL8nz8un1Tx0HJqTZ018WhTru5sTf24
---

<b>In this tutorial, we will setup the environment for Deep Learning and Computer Vision in Linux. This tutorial assumes that you have knowledge about using Linux Environment such as Ubuntu and basic knowledge of <a href="https://gogul09.github.io/hardware/linux-helpers" target="_blank">Linux commands</a></b>.

<div class="note">
<p>
<b>Update</b>: If you use Windows, I would suggest you to download <a href="https://www.virtualbox.org/">Virtual Box</a> (which is open-source) and install <a href="http://releases.ubuntu.com/14.04/">Ubuntu</a> in it (which is also open-source). But if you really want to work in a Windows machine, then check out this <a href="https://gogul09.github.io/software/deep-learning-windows" target="_blank">post</a> for setting up Deep Learning environment in Windows 10.
</p>
</div>

* Install **python** (high-level programming language) <br>
The first step is to download and install Python from [this](https://www.python.org/downloads/){:target="_blank"} website. Download Python 2.7 version as it is the most stable release of Python. Python 3.0 and above versions are good to download, but there are some backwards-incompatible changes which might produce some errors. So, better use Python 2.7 version.

* Install **pip** (python package manager) <br>

```python
wget https://bootstrap.pypa.io/get-pip.py
sudo python get-pip.py
```

* Install all the other necessary packages. <br>

```python
sudo pip install numpy
sudo pip install scipy
sudo apt-get install python-tk
sudo pip install seaborn
sudo pip install matplotlib
sudo pip install pandas
sudo pip install librosa
sudo pip install scikit-learn
sudo pip install tensorflow
sudo pip install theano
sudo pip install keras
```

## Dependencies for Computer Vision

* Developer tools

```python
sudo apt-get install build-essential cmake git pkg-config
```

* Image I/O packages

```python
sudo apt-get install libjpeg8-dev libtiff4-dev libjasper-dev libpng12-dev
```

* GTK Development Library

```python
sudo apt-get install libgtk2.0-dev
```

* Accessing frames and video processing

```python
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
```

* Optimize routines in OpenCV

```python
sudo apt-get install libatlas-base-dev gfortran
```

* Install OpenCV 3.1

```python
cd ~ 
git clone https://github.com/Itseez/opencv.git 
cd opencv 
git checkout 3.1.0 
```

* Install OpenCV contrib 3.1

```python
cd ~
git clone https://github.com/Itseez/opencv_contrib.git
cd opencv_contrib
git checkout 3.1.0
```

* Build OpenCV

```python
cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=ON -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D BUILD_EXAMPLES=ON ..
```

* Compile OpenCV

```python
make
```

* Install OpenCV

```python
sudo make install 
sudo ldconfig
```

<div class="note">
<p><b>Note</b>: If you install librosa in Windows, you will need FFmpeg which could be downloaded <a href="https://ffmpeg.org/" target="_blank">here</a>. Extract the <span class="coding">.zip</span> file and add <span class="coding">bin</span> folder of FFmpeg to <span class="coding">PATH</span> in system environment variables.</p>
</div>

After installing all these libraries, open up a terminal and type <span class="coding">python</span>. After that, import all the packages using <span class="coding">import (package-name)</span> (for example, <span class="coding">import numpy</span>) and verfiy the version of each package. In case if you get any error, **uninstall** that package using <span class="coding">sudo pip uninstall (package-name)</span>, and then **reinstall** that package again with pip. Sometimes I faced this issue and solved it using this simple method.