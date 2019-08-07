---
layout: page-doc
title: Travel
subheading: Learn how traveling creates peace within you.
description: Learn how traveling creates peace within you.
color: grad-blog
image: https://drive.google.com/uc?id=13ujc6Gh87Kd0sBZqNYltN19yJQ3ioLZz
permalink: /travel
---

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      <div class="page-holder">
        <ul>
        {% for post in site.posts %}
          {% if post.category contains 'travel' %}
              <li>
                <a class="post-link" href="{{ site.baseurl }}{{ post.url }}">
                  <div class="page-treasure">
                    <h2>{{ post.title }}</h2>
                    <p>{{ post.description }}</p>
                  </div>
                </a>
              </li>
            {% endif %}
        {% endfor %}
        </ul>
      </div>
    </div>
  </div>
</div>