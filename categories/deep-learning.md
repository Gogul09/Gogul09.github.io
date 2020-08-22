---
layout: page-doc
title: Deep Learning
subheading: Learn how to create deep neural networks to solve challenging problems.
description: Learn how to create deep neural networks to solve challenging problems.
color: grad-blog
image: https://drive.google.com/uc?id=1MeJ0SD7Ci8TxQPOXHuEU1YvGqU03mhlq
permalink: /deep-learning
---

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      <div class="page-holder">
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'Deep Learning' %}
                <li>
                  <a class="post-link" href="{{ site.baseurl }}{{ post.url }}">
                    <div class="page-treasure-wrapper">
                      <div class="page-treasure-image" >
                        <div style="background-image: url('{{ post.image }}')"></div>
                      </div>
                      <div class="page-treasure">
                        <h3>{{ post.title }}</h3>
                        <p>{{ post.description }}</p>
                      </div>
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