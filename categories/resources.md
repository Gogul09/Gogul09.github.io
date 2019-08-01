---
layout: page-doc
title: Resources
subheading: Learn how to make use of internet to learn anything free.
description: Learn how to make use of internet to learn anything free.
color: grad-blog
image: https://drive.google.com/uc?id=1WbtqR0nNiS9plUrOWTq2tgQOqFrq0e3j
permalink: /resources
---

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      <div class="page-holder">
        <ul>
        {% for post in site.posts %}
          {% if post.category contains 'software' or post.category contains 'hardware' %}
            {% if post.class contains 'Resources' %}
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