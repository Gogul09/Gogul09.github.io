---
layout: page-doc
title: Computer Vision
subheading: Learn how to make your USB webcam or camera to understand world's information.
description: Learn how to make your USB webcam or camera to understand world's information.
color: grad-blog
image: https://drive.google.com/uc?id=1f08__sg5JJCHdoeq6WB0Iq68WZCTiMao
permalink: /computer-vision
---

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      <div class="page-holder">
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'Computer Vision' %}
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