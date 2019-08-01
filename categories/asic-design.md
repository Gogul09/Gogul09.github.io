---
layout: page-doc
title: ASIC Design
subheading: Learn how to design a chip using which you could create electronics applications.
description: Learn how to design a chip using which you could create electronics applications.
color: grad-blog
image: https://drive.google.com/uc?id=1Zte3kdk2gCs6-euVEE4UrYzHn__s19WG
permalink: /asic-design
---

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      <div class="page-holder">
        <h3>physical design</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'hardware' %}
            {% if post.class contains 'PD Concepts' %}
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
        <h3>timing analysis</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'hardware' %}
            {% if post.class contains 'STA Concepts' %}
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
        <h3>power analysis</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'hardware' %}
            {% if post.class contains 'PDN Concepts' %}
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
        <h3>intelligence in chip design</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'hardware' %}
            {% if post.class contains 'ML for PD' %}
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