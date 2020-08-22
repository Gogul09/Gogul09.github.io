---
layout: page-doc
title: Music Production
subheading: Music is the universal language that unites every human being. 
description: Music is the universal language that unites every human being. 
color: grad-blog
image: https://drive.google.com/uc?id=1U-j90IU_ElPxAaQChQrQv1wBLo9JpcFk
permalink: /music-production
---

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      <div class="page-holder">
        <ul>
        {% for post in site.posts %}
          {% if post.category contains 'music-production' %}
              <li>
                  <a class="post-link" href="{{ site.baseurl }}{{ post.url }}">
                    <div class="page-treasure-wrapper">
                      <div class="music-treasure-image page-treasure-image" >
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
        {% endfor %}
        </ul>
      </div>
    </div>
  </div>
</div>