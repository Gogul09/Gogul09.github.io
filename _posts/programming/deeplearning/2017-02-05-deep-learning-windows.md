---
layout: post
category: software
class: Environment Setup
title: Deep Learning Environment Setup for Windows
description: This tutorial is for beginners who need to setup environment for Deep Learning in Windows 10.
author: Gogul Ilango
permalink: /software/deep-learning-windows
image: https://drive.google.com/uc?id=1i93OWSHxa497jukB4bE-hMGQj92R2qNS
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#hardware">Hardware</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#directory-structure">Directory structure</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#environment-variables">Environment variables</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#visual-studio-2015-community-edition">Visual Studio 2015 Community Edition</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#anaconda">Anaconda (64-bit)</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#cuda">CUDA 8.0.44 (64-bit)</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#mingw">MinGW-w64 (5.4.0)</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#theano">Theano</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#openblas">OpenBLAS 0.2.14</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#enabling-cpu-or-gpu">Enabling CPU or GPU</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_11" href="#keras">Keras</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_12" href="#intermediate-check">Intermediate check</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_13" href="#opencv">OpenCV 3.1.0</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_14" href="#final-check">Final check</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_15" href="#installing-additional-libraries">Installing additional libraries</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_16" href="#cudnn">Installing cuDNN (optional)</a></li>
  </ul>
</div>

In this blog post, we will setup the environment for Deep Learning in Windows 10. At the end of this post, we will have a machine that is ready to run latest Deep Learning libraries such as Theano, TensorFlow and Keras. We will also learn how to enable GPU to speed up training Deep Neural Networks using CUDA. Let's jump right in!

### Hardware 

All the following steps are performed and checked with my hardware. 
My hardware specs are given below. 

* **CPU**: Intel i7-7500U CPU @ 2.70GHz, 16 GB RAM 
* **OS**:  Windows 10 (64-bit) 
* **GPU**: NVIDIA GeForce 940MX, 4 GB RAM 

### Directory structure 
Let's start with the directory structure which we will follow in this entire post. This is a mandatory step else it becomes more difficult later. 
Create a folder named "deeplearning" in C drive. 

```python
C:\deeplearning 
```
This is the master folder inside which we will keep all the dependencies listed above except Visual Studio 2015 Community Edition.

### Environment variables 

Before getting into installing dependencies, please become familiar with "Environment variables" and "path". If you are not familiar with these two terms in Windows, it is highly recommended to read this. Else go to the next section. 

* Go to "This PC" (or Computer) -> Right click it -> Select "Properties". 
* Select "Advanced System Settings" 
* Select "Environment Variables" 
* After that you will see a window having "User Variables" and "System Variables". 
* We will be highly using the "System Variables" for this post. Before getting into the next step, have a look at this System Variables and find "path" variable in that. 
* We can create a new system variable, edit it, assign a value to it (normally a path) and delete it. 
* We will be using "System Variables" a lot in this post. So, please become familiar with this before getting further. 

The following are the software, libraries and tools needed to setup Deep Learning environment in Windows 10 (64-bit). 

<h3 id="visual-studio-2015-community-edition">Visual Studio 2015 Community Edition </h3>

* Go to [this website](https://www.microsoft.com/en-us/download/details.aspx?id=48146){:target="_blank"} and click on Download". You will be taken to the page as shown below. 

<div class="note">
<p>
  <b>Update: </b>At the time of writing this post, the above URL worked. In case, if it doesn't, check out <a href="https://stackoverflow.com/questions/44290672/how-to-download-visual-studio-community-edition-2015-not-2017" target="_blank">this</a> link.
</p>
</div>

<figure>
  <img src="/images/software/deep-learning-windows/VS1.png" class="typical-image">
</figure>

* Check "vs_community.exe" and choose next. The download will start. After download finishes, run the setup and choose the config settings as shown below. 

<figure>
  <img src="/images/software/deep-learning-windows/VS11.png" class="typical-image">
</figure>

<figure>
  <img src="/images/software/deep-learning-windows/VS2.png" class="typical-image">
</figure>

<figure>
  <img src="/images/software/deep-learning-windows/VS3.png" class="typical-image">
</figure>

<figure>
  <img src="/images/software/deep-learning-windows/VS4.png" class="typical-image">
</figure>

#### Adding environment variables 

* Add "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin" to "path" in System Variables. 
* Create a system variable named "INCLUDE" with the value "C:\Program Files (x86)\Windows Kits\10\Include\10.0.10240.0\ucrt"
* Create a system variable named "LIB" with the value "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.10240.0\um\x64;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.10240.0\ucrt\x64"  

<h3 id="anaconda">Anaconda (64-bit)</h3>

I have installed Anaconda 64-bit installer to get Python 2.7. This worked without any error!

* Go to [this website](https://www.continuum.io/downloads){:target="_blank"} and select "Anaconda 64-bit installer" under Python 2.7. 

<figure>
  <img src="/images/software/deep-learning-windows/Anaconda.png" class="typical-image">
</figure>

<figure>
  <img src="/images/software/deep-learning-windows/Anaconda2.png" class="typical-image"> 
</figure>

<br>
<div class="note"><p>
<b>Update (13/12/2017)</b>: If you need TensorFlow as the backend for Keras in Windows 10, you can now do that by installing <a href="https://www.anaconda.com/download/#windows" target="_blank">Anaconda Python 3.6 (64-bit) installer</a>. Choose Python 3.6 64-bit Graphical Installer from that link and download it.
</p>
</div>

* Run the setup and follow the installation. 
* Choose the installation directory as - 

```python
>>> C:\deeplearning\anaconda 
```

* In the next screen, check the two options to add anaconda and python 2.7 to "path". 

#### Adding environment variables

* Create a system variable named "PYTHON_HOME" with the value "C:\deeplearning\anaconda" 
* Add the following in "path" under System Variables. (Add all these one by one by double clicking the value of "path" and clicking "New")
   * %PYTHON_HOME% 
   * %PYTHON_HOME%\Scripts
   * %PYTHON_HOME%\Library\bin 
* After adding the variables in path, open up a command prompt as administrator and execute the following command. 

```python
>>> conda install libpython 
```

<h3 id="cuda">CUDA 8.0 (64-bit)</h3>

To enable GPU to speed up training neural networks for Deep Learning, we need to install CUDA. 

* Go to [this website](https://developer.nvidia.com/cuda-downloads){:target="_blank"} and choose the options of your need. 

<figure>
  <img src="/images/software/deep-learning-windows/CUDA1.png" class="typical-image"> 
</figure>

* Run the setup and choose the installation directory as - 

```python
>>> C:\deeplearning\cuda 
```

* After completing the installation, the installer automatically creates "CUDA_PATH" in System Variables. Check this, else you need to add it for sure. 
* In case if you don't find CUDA related environment variables, follow the steps below - 

#### Adding environment variables

* Create a system variable named "CUDA_PATH" with the value "C:\deeplearning\cuda"
* Add the following in "path" under System Variables.
    * %CUDA_PATH%\libnvvp
    * %CUDA_PATH%\bin 


<h3 id="mingw">MinGW-w64 (5.4.0)</h3>

* Go to [this website](https://sourceforge.net/projects/mingw-w64/files/Toolchains%20targetting%20Win32/Personal%20Builds/mingw-builds/installer/){:target="_blank"} and download the "mingw-w64-install.exe". 
* After downloading, run the setup and choose the installation directory as - 

```python
>>> C:\deeplearning\mingw-w64-5.4.0 
```

* Choose the config options as shown below - 

<figure>
  <img src="/images/software/deep-learning-windows/mingw.png" width="400px"> 
</figure>

#### Adding environment variables

* Create a system variable named "MINGW_HOME" with the value "C:\deeplearning\mingw-w64-5.4.0"
* Add the following in "path" under System Variables.
  * %MINGW_HOME%\mingw64\bin 

<h3 id="theano">Theano</h3>
We will be installing Theano 0.8.2 using _git_ from our command prompt. 

* Create a folder named "theano-0.8.2" in "C:\deeplearning". 

```python
>>> C:\deeplearning\theano-0.8.2 
```

* Open command prompt as administrator and type the following - 

```python
>>> C:\deeplearning\ 
>>> git clone https://github.com/Theano/Theano.git theano-0.8.2 --branch rel-0.8.2 
```

* Wait for the repository to get cloned and checked out. 
* Enter into the theano folder and install it using the following commands 

```python
>>> cd C:\deeplearning\theano-0.8.2 
>>> python setup.py install --record installed_files.txt 
```

<h3 id="openblas">OpenBLAS 0.2.14</h3>

* Go to [this website](https://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int32.zip/download){:target="_blank"} to download "OpenBLAS" which is needed to perform parallel computation by running both CPU and GPU together. For example, data augmentation could be performed by CPU while the GPU could be used to speed up training the neural network. 
* Download the .zip file and extract the files to the folder - 

```python
>>> C:\deeplearning\openblas-0.2.14-int32 
```

#### Adding environment variables 

* Create a system variable named "OPENBLAS_HOME" with the value "C:\deeplearning\openblas-0.2.14-int32"
* Add the following in "path" under System Variables.
  * %OPENBLAS_HOME%\bin  

<h3 id="enabling-cpu-or-gpu">Enabling CPU or GPU</h3>

To switch between CPU and GPU, we can do a simple trick. Theano uses the processing unit by checking the variable "THEANO_FLAGS" in "path" under System Variables. 

* To use CPU, copy-paste the following as the value to "THEANO_FLAGS" in "path".

```python
floatX=float32,device=cpu,lib.cnmem=0.8,blas.ldflags=-LC:/deeplearning/openblas-0.2.14-int32/bin -lopenblas
```

* To use GPU, copy-paste the following as the value to "THEANO_FLAGS" in "path".

```python
floatX=float32,device=gpu,dnn.enabled=False,lib.cnmem=0.8,blas.ldflags=-LC:/deeplearning/openblas-0.2.14-int32/bin -lopenblas
```


<h3 id="opencv">OpenCV 3.1.0</h3>

<b>For Python 2.7</b><br>
Forget searching Google like "How to install OpenCV in Windows 10". You will find more complicated procedures and techniques to install it, which fails the most. The simplest solution is to use Anaconda. This is the 100% working solution in my case. I tried a lot fighting with CMAKE, git, compiler issues etc.. and found this as the best solution. In case if you find any other method, kindly let me know. 

* Open up a command prompt and enter the following command.

```python
>>> conda install -c https://conda.anaconda.org/menpo opencv3 
```

<b>For Python 3.6</b><br>
* Open up a command prompt and enter the following command.

```python
>>> pip install opencv-python
>>> pip install opencv-contrib-python
```

After installing it, type the following command.

```
>>> python 
>>> import cv2 
>>> cv2.__version__ 
```

You should get something like this.

<figure>
  <img src="/images/software/deep-learning-windows/opencv.png"> 
</figure>

<h3 id="intermediate-check">Intermediate check</h3>

If you followed this post carefully till here, you should end up with your "System Variables" as shown below. 

<figure>
  <img src="/images/software/deep-learning-windows/path.png"> 
</figure>

<h3 id="keras">Keras</h3>

<div class="note">
<p>
<b>Update (13/12/2017):</b> You can now install Keras and TensorFlow using pip in Windows 10, if you have installed <a href="https://www.anaconda.com/download/#windows" target="_blank">Python 3.6</a> from Anaconda website. It is as simple as two pip commands given below.
</p>  
</div>

```python
>>> pip install tensorflow
>>> pip install keras
```

<b>For Python 2.7</b><br>
We will be installing Keras 1.1.0 using git from our command prompt. 

* Create a folder named "keras-1.1.0" in "C:\deeplearning". 

```python
>>> C:\deeplearning\keras-1.1.0 
```

* Open command prompt as administrator and type the following -

```python
>>> C:\deeplearning\ 
>>> git clone https://github.com/fchollet/keras.git keras-1.1.0 --branch 1.1.0 
```

* Wait for the repository to get cloned and checked out. 
* Enter into the keras folder and install it using the following commands. 

```python
>>> cd C:\deeplearning\keras-1.1.0 
>>> python setup.py install --record installed_files.txt 
```

Since, we are using Theano, we don't need TensorFlow as the backend for Keras. But Keras comes in default with TensorFlow as its backend. To change this, navigate to the folder -

```python
>>> C:\Users\username\.keras
```

Here, username is your name. You will find "keras.json" file. Open it and change it like this. 

<figure>
  <img src="/images/software/deep-learning-windows/keras.png"> 
</figure>
<br>
<div class="note"><p>
<b>Update (13/12/2017)</b>: If you want to use TensorFlow as the backend for Keras, you need to change "image_dim_ordering" to "tf" and "backend" to "tensorflow".
</p>
</div>

<h3 id="final-check">Final check</h3>

If you followed all the steps correctly till here, you can execute the following final check to verify the installation of all the dependencies. 

<figure>
  <img src="/images/software/deep-learning-windows/check1.png"> 
</figure>

<div class="note" style="margin-top: 20px"><p>
<b>Note</b>: I have installed CUDA 8.0 in a different folder. But it still worked. Make sure you assign "CUDA_PATH" variable in System Variables to the correct folder where it got installed. If it was automatically mapped to the correct path, leave it as it is. 
</p></div>

<figure>
  <img src="/images/software/deep-learning-windows/check 2.png">
</figure>

* Open up a text editor and copy-paste the following code. 
* Save it in "Desktop" as "test.py".

```python
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time

vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
iters = 1000

rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
    r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
    print('Used the cpu')
else:
    print('Used the gpu')
```
    
If you want to use CPU - 

* Open the "Environment Variables" and go to "System Variables" -> "THEANO_FLAGS" 
* Double click the value of "THEANO_FLAGS" and change it to -

```python
floatX=float32,device=cpu,lib.cnmem=0.8,blas.ldflags=-LC:/deeplearning/openblas-0.2.14-int32/bin -lopenblas
```

If you want to use GPU - 

* Open the "Environment Variables" and go to "System Variables" -> "THEANO_FLAGS" 
* Double click the value of "THEANO_FLAGS" and change it to -

```python
floatX=float32,device=gpu,dnn.enabled=False,lib.cnmem=0.8,blas.ldflags=-LC:/deeplearning/openblas-0.2.14-int32/bin -lopenblas
```

* Open the command prompt and navigate to the "Desktop". * Run the following command.

```python
>>> cd C:\Users\username\Desktop
>>> python test.py
```

* I have enabled GPU and I got the following result. 

<figure>
  <img src="/images/software/deep-learning-windows/out1.png" > 
</figure>

* In case if you meet any error, go to -

```python
>>> cd C:\Users\username\AppData\local\Theano
```

* Select all folders, delete the contents and try again. It will work. Else, restart the system once and try again. 

### 11. Installing additional libraries 
If you installed pip for windows, it is much simpler to install some additional packages needed for development. Some are listed below. Open up command prompt as administrator and enter the commands listed.

```python
>>> pip install scikit-learn
>>> pip install mahotas
>>> pip install jupyter
>>> pip install imutils
>>> pip install librosa
```

<h3 id="cudnn">Installing cuDNN</h3>

* Go to [this](https://developer.nvidia.com/cudnn){:target="_blank"} website, register and download cuDNN for CUDA 8.0 and Windows 10.
* After downloading the file, unzip it.
* Copy the contents from each of the folder (bin, include, lib) inside cuDNN folder and paste it in CUDA path having (bin, include, lib) respectively. That is -
  * Copy the contents inside "C:\Users\<username>\Downloads\cuDNN\bin" folder to "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin"
  * Copy the contents inside "C:\Users\<username>\Downloads\cuDNN\include" folder to "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include"
  * Copy the contents inside "C:\Users\<username>\Downloads\cuDNN\lib\x64" folder to "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64"
* After performing the above steps, if you want to use cuDNN to speed up your Deep Learning process, you must change the THEANO_FLAGS as shown below.

If you want to use cuDNN and GPU- 

* Open the "Environment Variables" and go to "System Variables" -> "THEANO_FLAGS" 
* Double click the value of "THEANO_FLAGS" and change it to -

```python
floatX=float32,device=gpu,optimizer_including=cudnn,lib.cnmem=0.8,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic,blas.ldflags=-LC:/deeplearning/openblas-0.2.14-int32/bin -lopenblas
```

Once you have updated the path, open up command prompt and type the following.

```python
>>> python
>>> import theano
```

You should get something like this. 

<figure>
  <img src="/images/software/deep-learning-windows/cuDNN.png"> 
</figure>

That's it. You are all set to begin your journey in Deep Learning using Windows 10. Feel free to share this link to someone who is struggling to setup environment for Deep Learning and Computer Vision in Windows 10.