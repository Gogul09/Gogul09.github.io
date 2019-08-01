---
layout: post
category: hardware
class: PD Concepts
title: Python for Hardware Design
description: How python programming language could be used in hardware design such as ASIC logic design or physical design. Come let's explore.
author: Gogul Ilango
permalink: /hardware/python-for-hardware-design
image: https://drive.google.com/uc?id=1XZNZ6xPx0UUCbnncmz_GBWpX7xhHJ6lo
cardimage: https://drive.google.com/uc?id=1XZNZ6xPx0UUCbnncmz_GBWpX7xhHJ6lo
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
  	<li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#why-python-is-so-popular">Why Python is so popular?</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#how-to-learn-python-for-free">How to learn Python for free?</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#python-for-automation">Python for Automation</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#python-for-text-processing">Python for Text Processing</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#python-for-logic-design">Python for Logic Design</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#python-for-ml-dl-in-hardware">Python for ML/DL in Hardware</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#python-for-eda">Python for EDA</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#python-for-web-dashboards">Python for Web Dashboards</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#conclusion">Conclusion</a></li>
  </ul>
</div>

<div class="objectives">
<p>I always get questions from my coworkers as well as my readers like the ones below.</p>
<ul>
	<li>Where do I start learning python?</li> 
	<li>Why/How should a hardware engineer learn python in 2019?</li> 
	<li>How python could be used to eliminate manual tasks?</li>
	<li>Why python is preferred over other scripting languages?</li>
</ul>
</div>

If you check out the link [here](https://octoverse.github.com/2017/){:target="_blank"}, its pretty much obvious that <span class="coding">javascript</span> and <span class="coding">python</span> leads the programming languages race in 2017. Python has its root in almost any major tech domain such as web development, data analytics, machine learning, deep learning, computer vision, natural language processing, audio processing etc., as shown [here](https://gogul09.github.io/software/python-programming){:target="_blank"}.

But, what if we use python to design, simulate and implement hardware circuits in silicon? Is it possible to take the amazing advantages that python offer over other hardware related programming languages such as HDL (verilog or vhdl), Tcl, Perl or Shell? Come let's explore! 

<figure>
  <img src="https://drive.google.com/uc?id=1z2Y6YEAYahJA_7TgcMESLuJIqtsApkal" />
</figure>

<h3 id="why-python-is-so-popular">Why Python is so popular?</h3>

If you are a beginner to the programming world, I would highly suggest you to learn <span class="coding">python</span> first. This is because of the following major reasons.
* **Simple**: Python is simple to learn because its very similar to how we humans think.
* **More Productive**: Python is highly productive when compared to other languages such as <span class="coding">c</span>, <span class="coding">c++</span> or <span class="coding">java</span> as it is much more readable, concise, expressive and takes lesser time and effort to write code.
* **One-liners**: Python has so many one-liners and english-like commands (keywords) that boosts programmers productivity a lot.
* **Community**: Python has a very big developer friendly community and its very easy to find python developers around the world in platforms such as GitHub or Stack Overflow.
* **Libraries**: Python has rich set of well documented libraries and frameworks for different tech domains as shown [here](https://gogul09.github.io/software/python-programming){:target="_blank"}.
* **Open-Source**: Python ecosystem is so popular because most of the libraries and frameworks available online are open-source (meaning anyone can use it for their development purposes adhering to the licenses provided). 

Despite its advantages, python is much slower compared to languages such as <span class="coding">c++</span>. But wait! It's not a big disadvantage. You can still use python for most of the tasks that require minimal execution time (not speed-intensive applications such as games). That's why data-intensive domains such as deep learning libraries use python as a high-level wrapper for a human to code and beneath that wrapper, they use C++ for faster execution.

---

<h3 id="how-to-learn-python-for-free">How to learn python for free?</h3>

When I started learning python three years back, I used the following resources. I guess, these resources are more than enough to get you comfortable with python.

* [HackerRank](https://www.hackerrank.com/domains/python){:target="_blank"} is the best learning platform for python. You have to learn and solve programs based on levels of difficulty. Once you solve programs here, you will get that confidence in using the language for your own purposes.
* [Learn Python - Socratica](https://www.youtube.com/playlist?list=PLi01XoE8jYohWFPpC17Z-wWhPOSuh8Er-){:target="_blank"} has an excellent playlist full of neatly made python tutorials in YouTube for free. If you love learning by videos, this is the best ever python tutorial playlist out there.
* Corey Schafer is another awesome YouTube channel that delivers no-bullshit python tutorials.
  * [Python - Setting up a Python Environment](https://www.youtube.com/playlist?list=PL-osiE80TeTt66h8cVpmbayBKlMTuS55y){:target="_blank"}
  * [Python Programming Beginner Tutorials](https://www.youtube.com/playlist?list=PL-osiE80TeTskrapNbzXhwoFUiLCjGgY7){:target="_blank"}
  * [Python Tutorials](https://www.youtube.com/playlist?list=PL-osiE80TeTt2d9bfVyTiXJA-UTHn6WwU){:target="_blank"}
  * [Python OOP Tutorials](https://www.youtube.com/playlist?list=PL-osiE80TeTsqhIuOqKhwlXsIBIdSeYtc){:target="_blank"}

Other than these, you can check out some extensive list of learning resources that I collected for you [here](https://gogul09.github.io/software/python-programming){:target="_blank"}.

---

<h3 id="python-for-automation">Python for Automation</h3>

The main use of python lies in automating repeated manual tasks that we perform daily. These manual tasks might include opening a terminal, going to a path, finding some file, parsing some values in that file, putting that parsed value in a csv file and sending a mail with that csv file attached. This is one of the classic examples where you can write a single python script to do all the steps that are mentioned.

Other examples where python scripts can be used are organizing files in a particular path, manipulating strings, csv files, excel spreadsheets etc., sending email and text messages and much more. [Automate the boring stuff with python](https://automatetheboringstuff.com/){:target="_blank"} is a great learning resource that you can read on how to do these.

---

<h3 id="python-for-text-processing">Python for Text Processing</h3>

Manipulating text files is a common task in any domain whether you are in VLSI, computer networking, image processing, signal processing etc., Python offers rich set of libraries and modules to do this. 

Some of the most commonly used text processing modules in python using which you can manipulate text files (such as CSV files, JSON files, YAML file, HTML files etc.,), perform shell operations, parse large text files, load/store very large text files and manipulate file formats are
* <span class="coding">string</span> module
* <span class="coding">os</span> module
* <span class="coding">sys</span> module
* <span class="coding">re</span> module
* <span class="coding">csv</span> module
* <span class="coding">json</span> module
* <span class="coding">yaml</span> module
* <span class="coding">h5py</span> module
* <span class="coding">bs4</span> module

---

<h3 id="python-for-logic-design">Python for Logic Design</h3>

Ok cool! Now you understood, python could be used to create automation utilities that involve text processing which reduces time involved in solving repeated manual tasks in your day-to-day work life.

#### MyHDL

Can python be used in designing hardware circuits which typically need a HDL for writing code? Of course, there is a great open-source python project called [MyHDL](http://www.myhdl.org/){:target="_blank"} that turns python into a hardware description and verification language, providing hardware engineers with the power of the python ecosystem.

Moreover, MyHDL designs can be converted to verilog or VHDL automatically and implemented using a standard tool flow. Before getting too much excited about this library, please read [What MyHDL is not?](http://www.myhdl.org/start/whatitisnot.html){:target="_blank"}

You can check out FPGA designs using MyHDL [here](https://buildmedia.readthedocs.org/media/pdf/fpga-designs-with-myhdl/latest/fpga-designs-with-myhdl.pdf){:target="_blank"}. You can check out MyHDL manual [here](https://buildmedia.readthedocs.org/media/pdf/myhdl/stable/myhdl.pdf){:target="_blank"}. Also, do read [this](https://www.quora.com/How-can-Python-be-used-in-hardware-designing){:target="_blank"} quora answer.

#### PyMTL

Another python based hardware modeling framework is [PyMTL](https://github.com/cornell-brg/pymtl){:target="_blank}. Following links provide training resources on using this library.
* [Verilog Hardware Description Language](http://www.csl.cornell.edu/courses/ece4750/handouts/ece4750-tut4-verilog.pdf){:target="_blank"}
* [PyMTL Hardware Modeling Framework Tutorial](http://www.csl.cornell.edu/courses/ece4750/handouts/ece4750-tut3-pymtl.pdf){:target="_blank"}
* [PyMTL CL Modeling Tutorial](https://github.com/cornell-ece5745/ece5745-sec-pymtl-cl/blob/master/README.md){:target="_blank"}
* [PyMTL/HLS Tutorial](https://github.com/cornell-brg/pymtl-tut-hls/blob/master/README.md){:target="_blank"}
* [PyMTL-Based ASIC Toolflow Tutorial](http://www.csl.cornell.edu/courses/ece5745/handouts/ece5745-tut-asic-new.pdf){:target="_blank"}


---

<h3 id="python-for-ml-dl-in-hardware">Python for ML/DL in Hardware</h3>

Apart from the hypes that machine learning (ML) and deep learning (DL) have created in the recent years, still VLSI domain isn't deeply affected by ML or DL. This is because, VLSI industry is so complex because of conflicting goals to optimize hardware designs for timing, power and area with lesser time to market.

This is definitely the time to use ML or DL in hardware design, particularly creating ML models for smaller tasks in a bigger design flow. I have documented some of the current research papers and articles related to using ML or DL in VLSI chip design [here](https://gogul09.github.io/hardware/research-papers-vlsi-ml){:target="_blank"}.

But, for a beginner who is interested to use ML or DL in hardware design, below are the python libraries needed to get started.

**Scientific Computing Stack**
* [NumPy](https://www.numpy.org/){:target="_blank"}
* [SciPy](https://www.scipy.org/){:target="_blank"}
* [Matplotlib](https://matplotlib.org/){:target="_blank"}
* [Seaborn](https://seaborn.pydata.org/){:target="_blank"}
* [Pandas](https://pandas.pydata.org/){:target="_blank"}

**Machine Learning**
* [Scikit-learn](https://scikit-learn.org/){:target="_blank"}

**Deep Learning**
* [Keras](https://keras.io/){:target="_blank"}
* [TensorFlow](https://www.tensorflow.org/){:target="_blank"}

---

<h3 id="python-for-eda">Python for EDA</h3>

As far as technology node reduces, the need for electronic design automation tools increases. You can't design a multi-million instances based design manually. You need EDA tools that helps in implementation.

One particular python project that I found related to EDA was [PyEDA](https://pyeda.readthedocs.io/en/latest/){:target="_blank"}. You can watch [this](https://www.youtube.com/watch?v=cljDuK0ouRs){:target="_blank"} YouTube video to learn more about this hobby project by [Chris Drake](https://github.com/cjdrake){:target="_blank"}.

---

<h3 id="python-for-web-dashboards">Python for Web Dashboards</h3>

To analyze your design data which has lots of files with millions of lines, you need a simpler way to look at important metrics, reports and status of your design in a nice looking web interface. Python provides cool libraries to create web dashboards that speeds up your productivity as well as reducing your analysis time. 

Using the python libraries given below, and with some HTML, CSS and JavaScript knowledge, you can create beautiful web dashboards to analyze your design data and reduce time to market a lot.

* [Django](https://www.djangoproject.com/){:target="_blank"}
* [Flask](http://flask.pocoo.org/){:target="_blank"}
* [PyMongo](https://api.mongodb.com/python/current/){:target="_blank"}

---

<h3 id="conclusion">Conclusion</h3>

As you can see, <span class="coding">python</span> can be used in multiple areas in hardware design. According to me, python has just started to find its place in hardware design. Also, I feel that it will be used by engineers around the world very soon due to its simpler nature. If you have found anything related to python that could be used for hardware design, please leave that in the comments below. Peace!