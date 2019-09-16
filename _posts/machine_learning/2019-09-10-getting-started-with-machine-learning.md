---
layout: post
category: software
class: Machine Learning
title: Getting Started with Machine Learning
description: Understand how to get started with Machine Learning which has turned to be the hottest topic of 21st century.
author: Gogul Ilango
permalink: software/getting-started-machine-learning
image: https://drive.google.com/uc?id=1b5sC_T-CysvP_toHuolmcrI_i_UpN7R3
--- 

<div class="sidebar_tracker" id="sidebar_tracker">
   <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
   <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
   <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#demystifying-ml">Demystifying ML</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#math-is-god">Math is God</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#coursera-is-our-teacher">Coursera is our teacher</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#learn-from-scratch">Learn from scratch</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#applying-everything">Applying everything</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#follow-blogs">Follow blogs</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#make-use-of-time">Make use of time</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#we-need-to-go-deep">We need to go deep</a></li>
  </ul>
</div>

If you are an absolute beginner with no idea about machine learning, but wish to learn this technology, this page is dedicated for you. Machine learning is one of the hottest topic in 21st century. And no wonder why every tech company in the world is hunting for engineers with machine learning skillset. Most of my friends ask me this very important and confusing question of our time.

> Where and how should I start learning machine learning?

This is because the number of tutorials, videos, lectures, infographics, slides etc., related to this topic has outnumbered recently with plethora of freely available learning materials for anyone willing to learn. Nevertheless, I want to breakdown this topic into pieces that I followed/following to learn more in this domain.

> I want to promise you that you can learn and master machine learning and deep learning without spending any money (provided you have an internet connection). Yes. It's possible if you have the interest and will to learn this technology.

<figure>
    <img src="/images/infographics/ai-basics.png" />
    <figcaption>Figure 1. Artificial Intelligence Basics</figcaption>
</figure>

<h3 id="demystifying-ml">Demystifying ML</h3>

Before I get into any jargon, let me say that machine learning is all about **data** and **computational power**. The whole idea behind this learning thing is that we make use of past data and our algorithmic thinking to solve problems around our world.

Easiest example to think is **weather prediction**. How do we know tomorrow's weather in Tanjore? Only because of past recorded (historic) data collected over years combined with algorithmic thinking to make future predictions. 

Key take away from these two points is that, without data, machine learning fails and without algorithms + compute, machine learning fails.

<h3 id="math-is-god">Math is God</h3>

You must have learnt mathematics during your school/college days. As we are dealing with data, before making a machine learn, we need to learn the algorithms first. Hence, pre-requisite for any ML practitioner would be math!

* [Linear Algebra](https://www.edx.org/learn/linear-algebra){:target="_blank"}
* [Probability](https://www.edx.org/learn/probability){:target="_blank"}
* [Statistics](https://www.edx.org/learn/calculus){:target="_blank"}
* [Calculus](https://www.edx.org/learn/calculus){:target="_blank"}
* [Algorithms and Data Structures](https://www.edx.org/course?search_query=algorithms){:target="_blank"}

And you need to trust me here because, how do you think your machine learning algorithm predicts tomorrow's weather? Code? Nope. Its math translated to code! We need a firm grip on fundamental concepts in math to move forward in this fast distruptive technology. 

We can take the courses mentioned above to familiarize ourselves with math needed for machine learning. If you skip this step and think that you can write code without math, you will definitely revisit this line at some point in time.


<h3 id="programming">Programming</h3>

Python programming language is a no-brainer choice when it comes to machine learning. This is because it is easy to learn, highly readable by others and has larger community for help around the world. We need to learn python using freely available learning materials given below.

* [HackerRank](https://www.hackerrank.com/){:target="_blank"}
* [Python Programming Resources](https://gogul.dev/software/python-programming){:target="_blank"}

It will take some time to get a good grip on Python (say 1-2 months). After that, we could start learning python libraries that are used for machine learning such as 

* [NumPy](https://numpy.org/){:target="_blank"}
* [SciPy](https://www.scipy.org/){:target="_blank"}
* [Pandas](https://pandas.pydata.org/){:target="_blank"}
* [Matplotlib](https://matplotlib.org/){:target="_blank"}

<h3 id="coursera-is-our-teacher">Coursera is our teacher</h3>

The best place online to learn anything related to Artificial Intelligence is [Coursera](http://coursera.org){:target="_blank}. Once we get comfortable with math, we can take free (audit) coursera courses to learn about machine learning. Some of the useful courses that I completed are given below. I highly recommend you to take these courses instead of searching for some other content in the internet as these are structured courses taught by highly experienced people.

* [Machine Learning](https://www.coursera.org/learn/machine-learning){:target="_blank"} by Stanford University - This is the defacto standard ML course taught by everyone's favorite Andrew NG sir. This course takes you from an ultimate beginner to a novice in ML.
* [Applied Machine Learning in Python](https://www.coursera.org/learn/python-machine-learning){:target="_blank"} by University of Michigan -  This amazing course teaches us to apply ML using python and sciki-learn. This course is taught by Professor Kevyn Collins-Thompson and the assignments are pretty challenging.

Before taking the second course, we must be aware of [scikit-learn](https://scikit-learn.org/){:target="_blank"} library which is used by majority of ML practitioners around the world. This library is written in python and has abstracted all the ML algorithms for us, so that we could concentrate on problem solving. Which means, making use of logistic regression algorithm for a problem would be as simple as the below two lines of code.

<div class="code-head">log_reg.py<span>code</span></div>

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=9)
```

<figure>
    <img src="/images/infographics/supervised-learning.png" />
    <figcaption>Figure 2. Supervised Machine Learning in a Nutshell</figcaption>
</figure>

As you see from the above figure, as a beginner, we will mostly focus on supervised machine learning which means we have *labeled data* using which we train an algorithm to predict label of unseen new data. Currently, majority of the applications fall under this category of machine learning. 

There are two other types of machine learning such as *Unsupervised Learning* and *Reinforcement Learning* which we will be learning once we get a good grip on *Supervised Learning*.

<h3 id="learn-from-scratch">Learn from scratch</h3>

Once we become familiar with python and its libraries + completing the above two coursera courses, we need to understand how to implement each machine learning algorithm from scratch using python without any library.

This is very much needed to highlight us from others. Learning how each algorithm really works and trying to implement them from scratch requires patience and dedication. Below I have listed the majority of ML algorithms we need to learn implementing from scratch.

#### Classification

* Linear Discriminant Analaysis
* Logistic Regression
* K-Nearest Neighbors
* Support Vector Machine
* Gaussian Naive Bayes
* Decision Trees
* Random Forests
* Gradient Boosted Decision Trees
* Neural Nets Classifier

#### Regression

* Linear regression
* Lasso regression
* K-Neighbors regression
* Decision tree regression
* Elastic net regression
* Support vector regression
* Ada boost regression
* Extra trees regression
* Gradient boosting regression
* Neural Nets regressor

<h3 id="applying-everything">Applying everything</h3>

Learning and understanding is just half the part. We need to apply what we have learnt so far. Solving ML problems is highly helpful for you as well as the community. That's why, we have a dedicated place called [Kaggle](http://kaggle.com){:target="_blank"}. 

Using Kaggle, we can compete with ML practitioners worldwide and win prizes! Kaggle has integrated Jupyter notebook environment for us, so that we can work on a problem, code it, solve it and share it with the community online.

<h3 id="follow-blogs">Follow Blogs</h3>

As machine learning research is moving very fast, to catch up with the latest tools, techniques, algorithms and concepts, we need to follow blogs around the internet. Some of the useful blogs that I follow to learn ML are given below.

* [Machine Learning Mastery](https://machinelearningmastery.com/){:target="_blank"} - This blog by Jason Brownlee is dedicated to developers who wish to write code and see results. I have been following this blog for the past four years and its one of the highly useful blog for ML practitioners.
* [Google AI Blog](https://ai.googleblog.com/){:target="_blank"} - This blog is from the AI giant Google and is aimed at state-of-the-art research that is happening there.
* [DeepMind Blog](https://deepmind.com/blog){:target="_blank"} - AlphaGo is the first computer program to defeat a professional human Go player, the first to defeat a Go world champion, and is arguably the strongest Go player in history. Blog of the people behind this amazingness.
* [OpenAI Blog](https://blog.openai.com/){:target="_blank"} - A nonprofit dedicated to AI research that is sponsored by Elon Musk and Peter Thiel. The focus of its blog is to communicate the state-of-the-art research to the public.

<h3 id="make-use-of-time">Make use of time</h3>

Time is valuable. Once its lost, its lost forever. Hence, we need to utilize the plethora of learning resources freely available for us to learn. Some of the resources that I personally found highly useful are given below.

* [Introduction to Machine Learning NPTEL Prof.S.Sarkar IIT Kharagpur](https://www.youtube.com/playlist?list=PLYihddLF-CgYuWNL55Wg8ALkm6u8U7gps){:target="_blank"}
* [Machine Learning Tutorial in Python - Edureka](https://www.youtube.com/playlist?list=PL9ooVrP1hQOHUfd-g8GUpKI3hHOwM_9Dn){:target="_blank"}
* [Machine Learning Fundamentals](https://www.youtube.com/playlist?list=PL_onPhFCkVQhUzcTVgQiC8W2ShZKWlm0s){:target="_blank"}
* [Machine Learning Tutorial Videos - Simplilearn](https://www.youtube.com/playlist?list=PLEiEAq2VkUULYYgj13YHUWmRePqiu8Ddy){:target="_blank"}
* [Machine Learning Tutorial Python](https://www.youtube.com/playlist?list=PLeo1K3hjS3uvCeTYTeyfe0-rN5r8zn9rw){:target="_blank"}

<h3 id="we-need-to-go-deep">We need to go deep</h3>

Once we get comfortable with machine learning, we can move to deep learning. Deep Learning is a branch of machine learning where the primary algorithm used is a deep neural network. State-of-the-art problems such as image classification, speech recognition, machine translation, style transfer etc., are solved using flavors of deep neural networks such as Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Long-Short Term Memory Networks (LSTM), Generative Adversarial Networks (GAN), Gated Recurrent Units (GRU) etc.,

I followed the below courses to learn about these amazing concepts.

* [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning){:target="_blank"}
* [TensorFlow in Practice Specialization](https://www.coursera.org/specializations/tensorflow-in-practice){:target="_blank"}

I still learn about everything discussed above and I find this field very interesting. Hope this article inspires you to get started with machine learning.