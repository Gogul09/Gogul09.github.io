---
layout: page-doc
title: Deep Learning
subheading: Learn how to create deep neural networks to solve challenging problems.
description: Learn how to create deep neural networks to solve challenging problems.
color: grad-blog
image: https://drive.google.com/uc?id=1pWZbfob5C4N06StzEhgQ1loPAo982w4N
permalink: /deep-learning
---

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      <div class="page-holder">
        <h3>physics</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'Physics' %}
                <li>
                  <a class="post-link" href="{{ site.baseurl }}{{ post.url }}">
                    <div class="page-treasure">
                      <h2>{{ post.title }}</h2>
                      <p>{{ post.description }}</p>
                    </div>
                  </a>
                </li>
              {% endif %}
            {% endif %}
        {% endfor %}
        </ul>
        <h3>music</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'Music' %}
                <li>
                  <a class="post-link" href="{{ site.baseurl }}{{ post.url }}">
                    <div class="page-treasure">
                      <h2>{{ post.title }}</h2>
                      <p>{{ post.description }}</p>
                    </div>
                  </a>
                </li>
              {% endif %}
            {% endif %}
        {% endfor %}
        </ul>
        <h3>tensorflow.js</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'TensorFlow.js' %}
                <li>
                  <a class="post-link" href="{{ site.baseurl }}{{ post.url }}">
                    <div class="page-treasure">
                      <h2>{{ post.title }}</h2>
                      <p>{{ post.description }}</p>
                    </div>
                  </a>
                </li>
              {% endif %}
            {% endif %}
        {% endfor %}
        </ul>
        <h3>keras</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'Keras' %}
                <li>
                  <a class="post-link" href="{{ site.baseurl }}{{ post.url }}">
                    <div class="page-treasure">
                      <h2>{{ post.title }}</h2>
                      <p>{{ post.description }}</p>
                    </div>
                  </a>
                </li>
              {% endif %}
            {% endif %}
        {% endfor %}
        </ul>
        <h3>deep learning</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'Deep Learning' %}
                <li>
                  <a class="post-link" href="{{ site.baseurl }}{{ post.url }}">
                    <div class="page-treasure">
                      <h2>{{ post.title }}</h2>
                      <p>{{ post.description }}</p>
                    </div>
                  </a>
                </li>
              {% endif %}
            {% endif %}
        {% endfor %}
        </ul>
      </div>
    </div>
  </div>
</div>