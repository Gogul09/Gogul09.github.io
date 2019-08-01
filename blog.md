---
layout: page-default
heading: blog
title: Blog
subheading: my journey as an engineer
description: This blog on Deep Learning, VLSI Design and STA is written by Gogul Ilango. Master VLSI Design, Physical Design, Static Timing Analysis, Deep Learning through my articles, tutorials and resources.
color: grad-blog
image: images/gi-logo.jpg
permalink: /blog
---

{% include colorful-header.html %}

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      
      <div class="gem-box">
        <div class="carbon_advertisement">
          <script async type="text/javascript" src="//cdn.carbonads.com/carbon.js?serve=CK7I623I&placement=gogul09githubio" id="_carbonads_js"></script>
        </div>
        <div class="asic-design" onclick="location.href='{{ site.baseurl }}/asic-design';">
          <img src="https://drive.google.com/uc?id=1Zte3kdk2gCs6-euVEE4UrYzHn__s19WG" />
          <h4>ASIC Design</h4>
          <p>Learn how to design a chip using which you could create electronics applications.</p>
          {% for post in site.posts %}
            {% if post.categories contains 'hardware' %}
              {% if post.class contains 'PD Concepts' or post.class contains 'STA Concepts' or post.class contains 'PDN Concepts' or post.class contains 'ML for PD' %}
                {% capture asic_design_count %} {{ asic_design_count | plus: 1 }} {% endcapture %}
              {% endif %}
            {% endif %}
          {% endfor %}
          <p class="no_of_posts">{{ asic_design_count }} posts</p>
        </div>
        <div class="deep-learning" onclick="location.href='{{ site.baseurl }}/deep-learning';">
            <img src="https://drive.google.com/uc?id=1pWZbfob5C4N06StzEhgQ1loPAo982w4N" />
            <h4>Deep Learning</h4>
            <p>Learn how to create deep neural networks to solve challenging problems.</p>
            {% for post in site.posts %}
            {% if post.categories contains 'software' %}
              {% if post.class contains 'TensorFlow.js' or post.class contains 'Keras' or post.class contains 'Deep Learning' %}
                {% capture deep_learning_count %} {{ deep_learning_count | plus: 1 }} {% endcapture %}
              {% endif %}
            {% endif %}
          {% endfor %}
          <p class="no_of_posts">{{ deep_learning_count }} posts</p>
        </div>
        <div class="machine-learning" onclick="location.href='{{ site.baseurl }}/machine-learning';">
            <img src="https://drive.google.com/uc?id=1NUs6UggicDf0XqPj6PavX2C1ZmoMJJzG" />
            <h4>Machine Learning</h4>
            <p>Learn how to create algorithms that don't require you to write rules.</p>
            {% for post in site.posts %}
            {% if post.categories contains 'software' %}
              {% if post.class contains 'Machine Learning' %}
                {% capture machine_learning_count %} {{ machine_learning_count | plus: 1 }} {% endcapture %}
              {% endif %}
            {% endif %}
          {% endfor %}
          <p class="no_of_posts">{{ machine_learning_count }} posts</p>
        </div>
        <div class="computer-vision" onclick="location.href='{{ site.baseurl }}/computer-vision';">
          <img src="https://drive.google.com/uc?id=1f08__sg5JJCHdoeq6WB0Iq68WZCTiMao" />
          <h4>Computer Vision</h4>
          <p>Learn how to make your USB webcam or camera to understand world's information.</p>
          {% for post in site.posts %}
            {% if post.categories contains 'software' %}
              {% if post.class contains 'Computer Vision' %}
                {% capture computer_vision_count %} {{ computer_vision_count | plus: 1 }} {% endcapture %}
              {% endif %}
            {% endif %}
          {% endfor %}
          <p class="no_of_posts">{{ computer_vision_count }} posts</p>
        </div>
        <div class="programming" onclick="location.href='{{ site.baseurl }}/programming';">
          <img src="https://drive.google.com/uc?id=1GQpDt79Jb27N66IDU-V-83_rdNUJjDzp" />
          <h4>Programming</h4>
          <p>Learn how to make use of your brain to write code.</p>
          {% for post in site.posts %}
            {% if post.categories contains 'software' %}
              {% if post.class contains 'Front-End Web' or post.class contains 'Programming Languages' or post.class contains 'Tools' or post.class contains 'Environment Setup' %}
                {% capture programming_count %} {{ programming_count | plus: 1 }} {% endcapture %}
              {% endif %}
            {% endif %}
          {% endfor %}
          <p class="no_of_posts">{{ programming_count }} posts</p>
        </div>
        <div class="resources" onclick="location.href='{{ site.baseurl }}/resources';">
          <img src="https://drive.google.com/uc?id=1WbtqR0nNiS9plUrOWTq2tgQOqFrq0e3j" />
          <h4>Resources</h4>
          <p>Learn how to make use of internet to learn anything free.</p>
          {% for post in site.posts %}
            {% if post.categories contains 'hardware' or post.categories contains 'software' %}
              {% if post.class contains 'Resources' %}
                {% capture resources_count %} {{ resources_count | plus: 1 }} {% endcapture %}
              {% endif %}
            {% endif %}
          {% endfor %}
          <p class="no_of_posts">{{ resources_count }} posts</p>
        </div>
      </div>
    </div>
  </div>
</div>