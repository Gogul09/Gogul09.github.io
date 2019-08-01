---
layout: post
category: software
class: Environment Setup
title: Python Setup for Windows 10
description: This tutorial is for beginners who need to setup environment for Python and its libraries in Windows 10.
author: Gogul Ilango
permalink: /software/python-windows-10
image: https://drive.google.com/uc?id=1crAJHODHElM6acLc-WKgwU8VFn05lUOT
---

In this blog post, we will setup the environment for Python and its libraries in Windows 10. This is useful for a beginner or a professional who quickly wants to setup Python in a Windows 10 machine.

### Hardware used 
All the following steps are checked and verified with the specs mentioned below.
* **CPU**: Intel i7-7500U CPU @ 2.70GHz, 16 GB RAM
* **OS**: Windows 10 (64-bit)

### Anaconda
There are two versions of Python that you can choose to install and [Anaconda](https://www.anaconda.com/what-is-anaconda/){:target="_blank"} is the easiest way to do it. 
1. [Python 2.7](https://www.python.org/download/releases/2.7/){:target="_blank"}
2. [Python 3.6](https://www.python.org/downloads/release/python-360/){:target="_blank"}

I prefer to use Python 2.7 because it is more stable than Python 3.6. But if you prefer installing Python 3.6, go ahead and let me know whether you installed all dependencies successfully in the comments.

#### Python 2.7
* Head over to [Anaconda's Downloads](https://www.anaconda.com/download/){:target="_blank"} section. You will see a webpage as shown below. Click on Python 2.7 version download button. You can choose between 64-bit and 32-bit according to your OS.

<figure>
  <img src="/images/software/python-windows-10/anaconda-download.png" class="typical-image">
  <figcaption>Figure 1. Anaconda Download page</figcaption>
</figure>

* Get into your local disk C. Create a folder named <span class="coding">software</span> or <span class="coding">machine_learning</span> (my choice!). Get into the folder that you created and create another folder named <span class="coding">anaconda</span>. Below is the folder structure.

```
C:\machine_learning\anaconda
```

<figure>
  <img src="/images/software/python-windows-10/disk-info.png" class="typical-image">
  <figcaption>Figure 2. Disk info</figcaption>
</figure>

* Click "Next" and check the two options provided. This will take care of path settings in Windows 10. Click "Install" and it will take some time (around 10-15 mins) to install.

<figure>
  <img src="/images/software/python-windows-10/path-settings.png" class="typical-image">
  <figcaption>Figure 3. Path Settings</figcaption>
</figure>

* After installing, open up a command prompt and type <span class="coding">python</span>. You should get into Python shell. 

---

You can install or uninstall or update python packages using Anaconda commands. To learn more about Anaconda commands, please check [this](https://conda.io/docs/_downloads/conda-cheatsheet.pdf){:target="_blank"} cheat sheet.