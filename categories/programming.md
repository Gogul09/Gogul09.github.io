---
layout: page-doc
title: Programming
subheading: Learn how to make use of your brain to write code.
description: Learn how to make use of your brain to write code.
color: grad-blog
image: https://drive.google.com/uc?id=1GQpDt79Jb27N66IDU-V-83_rdNUJjDzp
permalink: /programming
---

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      <div class="page-holder">
        <h3>front-end web</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'Front-End Web' %}
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
        <h3>programming languages</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'Programming Languages' %}
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
        <h3>tools</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'Tools' %}
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
        <h3>environment setup</h3>
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'software' %}
            {% if post.class contains 'Environment Setup' %}
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