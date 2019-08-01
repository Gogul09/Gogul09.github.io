---
layout: page-doc
title: Machine Learning
subheading: Learn how to create algorithms that don't require you to write logic.
description: Learn how to create algorithms that don't require you to write logic.
color: grad-blog
image: https://drive.google.com/uc?id=1NUs6UggicDf0XqPj6PavX2C1ZmoMJJzG
permalink: /machine-learning
---

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      <div class="page-holder">
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'Machine Learning' %}
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