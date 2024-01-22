---
layout: page-doc
title: ASIC Design
subheading: Learn how to design a chip using which you could create electronics applications.
description: Learn how to design a chip using which you could create electronics applications.
color: grad-blog
image: https://drive.google.com/uc?id=1CxWWpzbp529wp0BqXOmc5NwOimC6j9nS
permalink: /asic-design
---

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      <div class="page-holder">
        <ul>
        {% for post in site.posts %}
          {% if post.categories contains 'hardware' %}
          		{% if post.class contains 'ASIC Design' %}
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