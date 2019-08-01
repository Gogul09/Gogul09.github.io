---
layout: post
category: software
class: Physics
title: Deep Learning Meets Physics
description: Let's understand what our universe is made of using Deep Learning and Machine Learning algorithms.
permalink: software/deep-learning-meets-physics
image: https://drive.google.com/uc?id=1Dh5DXRQH0HJCZgSJ_3K-3sYT4jHRpo32
cardimage: https://drive.google.com/uc?id=1Dh5DXRQH0HJCZgSJ_3K-3sYT4jHRpo32
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#cern">CERN</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#the-standard-model">The Standard Model</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#research-papers">Research Papers</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#ml-challenge">ML Challenge</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#tools-used-for-dl">Tools used for DL</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#references">References</a></li>
  </ul>
</div>

I'm always fascinated by the matter around us which make my neurons think what's inside matter? How matter formed at the first place? Who created or what created matter that exist now in our universe? Why there isn't much anti-matter as there is matter in this universe? These fundamental questions are still being researched by the greatest human beings of our time across the world. 

> Space and time are modes by which we think, not conditions under which we live - Albert Einstein

<h3 id="cern">CERN</h3>

When you seek answers for fundamental questions like these over the internet, you will find [CERN](https://home.cern/){:target="_blank"} in many of the links that Google provides. If you love science, then you must know the composition of our universe is found to be 73% dark energy, 23% dark matter, 3.6% intergalactic gas and the rest, visible matter that we see around us.

<figure>
  <canvas id="universe-pie-chart"></canvas>
  <figcaption>Predicted Composition of our Universe [<a href="https://en.wikipedia.org/wiki/Universe" target="_blank">source</a>]</figcaption>
</figure>

Scientists are CERN are researching about this composition of our universe and still they couldn't understand what constitutes [dark matter](https://en.wikipedia.org/wiki/Dark_matter){:target="_blank"} or [dark energy](https://en.wikipedia.org/wiki/Dark_energy){:target="_blank}. Before we talk about how deep learning fits in here, lets see some swashbuckling facts about CERN.

* International organization straddling Swiss-French border, founded 1954 having world's largest facilities for fundamental research in particle physics.
* 23 members states - 1.1 B CHF budget (~1.1B USD)
* 3000 members of personnel + 15,000 associated members from 90 countries.
* Has [Large Hadron Collider (LHC)](https://en.wikipedia.org/wiki/Large_Hadron_Collider){:target="_blank"} - Largest machine in the world which is 27km long.
* Fastest racetrack on Earth where protons travel at 99.9999991% of the speed of light.
* Emptiest place in the solar system where particules circulate in the highest vacuum.
* Hottest spot in the galaxy where lead ion collisions create temperatures 100,000x hotter than the hearth of the sun.
* In 1989, Tim Berners-Lee proposed the creation of a distributed information system which is evolved into what we call the World Wide Web. 
* The World's first [web page](http://info.cern.ch/){:target="_blank"} was originated from CERN.

<div class="youtube-video-container">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/h2MlS09KJP4" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

There are four major experiments done using the Large Hadron Collider (LHC) at CERN as shown in Figure 2. These are detectors with 100 million sensors that gather information about particles trajectory, electrical charge and energy.

* [ATLAS](https://en.wikipedia.org/wiki/ATLAS_experiment){:target="_blank"} - A Toroidal LHC Apparatus
* [CMS](https://en.wikipedia.org/wiki/Compact_Muon_Solenoid){:target="_blank"} - Compact Muon Solenoid
* [ALICE](https://en.wikipedia.org/wiki/ALICE_experiment){:target="_blank"} - A Large Ion Collider Experiment
* [LHCb](https://en.wikipedia.org/wiki/LHCb_experiment){:target="_blank"} - LHC-beauty

Other three experiments at LHC are 

* [TOTEM](https://en.wikipedia.org/wiki/TOTEM_experiment){:target="_blank"} - Total Cross Section, Elastic Scattering and Diffraction Dissociation
* [LHCf](https://en.wikipedia.org/wiki/LHCf_experiment){:target="_blank"} - LHC-forward
* [MoEDAL](https://en.wikipedia.org/wiki/MoEDAL_experiment){:target="_blank"} - Monopole and Exotics Detector At the LHC


Kindly visit [this](https://home.cern/science/accelerators/large-hadron-collider){:target="_blank"} to know more about LHC.
<figure>
  <img src="https://drive.google.com/uc?id=1CKH6Sm8AMJPBR0nz2HIidI20B6piX8EX">
  <figcaption>Figure 2. Experiments at CERN - CMS, ALICE, ATLAS, LHCb</figcaption>
</figure>

Ok, why do we need such a big instrumental setup to explore what's inside an atom? Let's understand why we need to figure out what's inside an atom at the first place? 

It's assumed that everything that we see around us (sun, earth, moon, stars, trees, humans etc.,) began with the [Big Bang](https://en.wikipedia.org/wiki/Big_Bang){:target="_blank"}. Before learning about the fundamental particles and forces of nature, please read [Chronology of the Universe](https://en.wikipedia.org/wiki/Chronology_of_the_universe){:target="_blank"} to understand the history, present and future of our universe so that you will come to a conclusion that finding the most fundamental particle (that makes up matter) and how forces of nature interact with that particle is what we need to find out.

<figure>
  <img src="https://drive.google.com/uc?id=1eJG249efsG0_UapFZNUfX9KxKxj8oOZs">
  <figcaption>Figure 3. Diagram of evolution of the (observable part) of the universe from the Big Bang (left) to the present <br> [<a href="https://en.wikipedia.org/wiki/Chronology_of_the_universe">source</a>]</figcaption>
</figure>

Above picture tells us, Big Bang happened at a specific point in time which converted energy into matter (made of particles). Using LHC, if we make proton beams to collide with each other near to speed of light, then we can discover what's inside proton. The study of such sub-atomic particles is called [particle physics](https://en.wikipedia.org/wiki/Particle_physics){:target="_blank"} which allow humans to seek answers to science's most fundamental questions.

<h3 id="the-standard-model">The Standard Model</h3>

Matter that we see around us is found to be made of few basic building blocks united by four fundamental forces in nature. You can read more about the standard model [here](https://home.cern/science/physics/standard-model){:target="_blank"}. 

<figure>
  <img src="https://drive.google.com/uc?id=17dGzpVj42v1fYvYXe7HHO2F7e1xYinkh">
  <figcaption>Figure 4. Standard Model of Elementary Particles</figcaption>
</figure>

It turns out that to find out these sub-atomic particles, LHC experiments generates petabyte of data per second, which means its an ultimate place to use Deep Learning algorithms. For example, an experiment at [CMS](https://en.wikipedia.org/wiki/Compact_Muon_Solenoid){:target="_blank"} (Compact Muon Solenoid) generates 40 million collisions per second (PB/s) which is filtered in real-time to 100,000 selections per second (TB/s) and 1,000 selections per second (GB/s), selecting potentially interesting events or triggers.

<div class="youtube-video-container">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/ehHoOYqAT_U" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

The complexity in these experiments is so huge that if you are a data-lover, it seems like a place for you to explore, analyze, visualize and find meaning out of these. Take a look at the data center numbers of CERN shown below.

<figure>
  <img src="https://drive.google.com/uc?id=1SNS96iE_Wjo0hp6tpnfqWEVuyJw4ABiF">
  <figcaption>Figure 5. CERN Data Center in Numbers</figcaption>
</figure>

<h3 id="research-papers">Research Papers</h3>

Some research work that I follow and read are collected below on application of deep learning algorithms for high-energy physics.

* [Searching for Exotic Particles in High-Energy Physics with Deep Learning](https://arxiv.org/pdf/1402.4735.pdf){:target="_blank"}
* [Deep Learning and Its Application to LHC Physics](https://arxiv.org/pdf/1806.11484.pdf){:target="_blank"}
* [TensorNetwork: A Library for Physics and Machine Learning](https://arxiv.org/pdf/1905.01330.pdf){:target="_blank"}
* [Topology classification with deep learning to improve real-time event selection at the LHC](https://arxiv.org/pdf/1807.00083.pdf){:target="_blank"}
* [Accelerating Science with Generative Adversarial Networks: An Application to 3D Particle Showers in Multi-Layer Calorimeters](https://arxiv.org/pdf/1705.02355.pdf){:target="_blank"}
* [Bi-directional RNNs and CNNs for impact parameter based tagging](https://en.wikipedia.org/wiki/B-tagging){:target="_blank"}

#### A jet multiclass classifier

A [jet](https://en.wikipedia.org/wiki/Jet_(particle_physics)){:target="_blank"} is a narrow cone of hadrons and other particles produced by the hadronization of a quark or gluon. Simple Deep Neural Nets on high-level features (jet masses, multiplicities, energy correlation functions) etc., can be used to create a jet multiclass classifier.

It's very interesting to know Deep Learning algorithms such as Convolutional Neural Network (CNN), Recurrent Neural Network (RNN), Bi-directional RNN, Long-Short Term Memory (LSTM) network, Gated Recurrent Units (GRU) and Generative Adversarial Network (GAN) are used in LHC experiments to analyze and get insights about the nature of our universe.

---

<h3 id="ml-challenge">ML Challenge</h3>

On 4 July 2012, CMS and ATLAS experiments at LHC confirmed the discovery of Higgs Boson. How a particle decays into other particles is the key factor to understand and measure the characteristics of the particle. It's confirmed that Higgs Boson decays into two tau particles which is a small signal buried in background noise.

Recently, I became aware of Higgs Boson Machine Learning Challenge conducted by CERN and Kaggle on 2014. Here is the link to the challenge - [Higgs boson machine-learning challenge](https://home.cern/news/news/computing/higgs-boson-machine-learning-challenge){:target="_blank"}. 

If you know machine learning, it's enough to participate in this competition as the task is to classify ATLAS events as **tau tau decay of a higgs boson** or **background noise** using the features characterizing the events.

For a beginner like me who is interested in applying ML/DL for high-energy physics, I found this as the perfect start!

---

<h3 id="tools-used-for-dl">Tools used for DL</h3>

It's amazing to hear that python and its ecosystem is used for data analytics in CERN LHC. Similar to how the image of black hole was created using [Python and its ecosystem](http://www.blog.pythonlibrary.org/2019/04/11/python-used-to-take-photo-of-black-hole/){:target="_blank"}, we could use the same to understand more about our universe.

Some of the tools used at CERN to do data analytics are

* Jupyter Notebook
* Apache Spark
* Apache Kafka
* Analytics Zoo
* BigDL
* HDFS
* TensorFlow
* Keras
* NumPy
* SciPy
* Pandas
* Matplotlib

---


<h3 id="references">References</h3>

* [Deep Learning on Apache Spark at CERN’s Large Hadron Collider with Intel Technologies](https://www.youtube.com/watch?v=bMU5Luyuk1Q){:target="_blank"}
* [The Rise of Deep Learning](https://cerncourier.com/the-rise-of-deep-learning/){:target="_blank"}
* [Reshaping Particle Physics Experiments with Deep Learning - Maurizio Pierini](https://www.youtube.com/watch?v=5z0Zp530Tms){:target="_blank"}
* [What is Dark Matter and Dark Energy?](https://www.youtube.com/watch?v=QAa2O_8wBUQ){:target="_blank"}
* [How is CERN using deep learning?](https://www.quora.com/How-is-CERN-using-deep-learning){:target="_blank"}
* [Dark Energy, Dark Matter](https://science.nasa.gov/astrophysics/focus-areas/what-is-dark-energy){:target="_blank"}
* [Dark Matter and Dark Energy](https://www.nationalgeographic.com/science/space/dark-matter/){:target="_blank"}
* [Dark-matter made of ‘Dark Atoms’?](http://iitk.ac.in/snt/blog/2013/07/24/dark-matter.htm){:target="_blank"}
* [What does CERN mean for the future of the universe](https://science.howstuffworks.com/science-vs-myth/everyday-myths/what-does-cern-mean-for-future-of-universe.htm){:target="_blank"}
* [Invited Talk - Deep Learning meets Physics](https://cds.cern.ch/record/2621728?ln=en){:target="_blank"}
* [How Does the Large Hadron Collider Work?](https://www.youtube.com/watch?v=328pw5Taeg0){:target="_blank"}
* [Chronology of the universe](https://en.wikipedia.org/wiki/Chronology_of_the_universe){:target="_blank"}
* [Big Bang](https://en.wikipedia.org/wiki/Big_Bang){:target="_blank"}
* [Time](https://en.wikipedia.org/wiki/Time){:target="_blank"}
* [Universe](https://en.wikipedia.org/wiki/Universe){:target="_blank"}
* [Jet (particle physics)](https://en.wikipedia.org/wiki/Jet_(particle_physics)){:target="_blank"}

<script src="https://cdn.jsdelivr.net/npm/chart.js@2.8.0"></script>

<script type="text/javascript">
var ctx = document.getElementById('universe-pie-chart').getContext('2d');
var uChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
        datasets: [{
            data: [73, 23, 3.6, 0.4],
            backgroundColor: ['#333333', 'rgb(77,65,103)', 'rgb(255,204,0)', 'rgb(240,51,93)'],
        }],

        labels: [
            'Dark Energy',
            'Dark Matter',
            'Intergalactic Gas',
            'Stars, etc.,'
        ]
    }
});
</script>