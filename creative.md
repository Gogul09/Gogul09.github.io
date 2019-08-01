---
layout: page-default
heading: creations
title: Creations
subheading: i used to create stuff that are visually neat and pleasing to our eyes, and you might guessed it right, i love pixel perfecting.
description: This page contains all the creative stuff such as interactive demos, infographics and my tutorial slides related to technology.
color: grad-creations
permalink: /creations/
---

<div class="blog-intro {{ page.color }}">
  <div>
    <h1>{{ page.heading }}</h1>
    <p>{{ page.subheading }}</p>
  </div>
</div>

<div class="home-container">
  <div class="home-articles">
    <div class="home-wrapper">
      <!--Demo STARTS-->
      <div style="display: block !important;">
        <div class="category-box">
          <h4 class="diamond-header">javascript</h4>
          <ul>
            {% for post in site.posts %}
              {% if post.category contains 'software' %}
                {% if post.class contains 'Demo' %}
                  <li>
                    <a class="post-link" href="{{ site.baseurl }}{{ post.url }}">
                        <div class='demo_box'>
                          <img src="{{ post.image }}" />
                          <h4>{{ post.title }}</h4>
                          <p>{{ post.description }}</p>
                        </div>
                     </a>
                  </li>
                {% endif %}
              {% endif %}
            {% endfor %}
          </ul>
          <h4 class="diamond-header">android</h4>
          <ul>
            <li>
              <div class='demo_box'>
                <img id="me-project-1" alt="alive-me-project.png" src="https://drive.google.com/uc?id=1EsPF_GM2XsW-bsSCKjuRJ1EWcxCY5bD4" onclick="showHideModal(this.id);" />
                <h4>A.L.I.V.E</h4>
                <p>Smart Autonomous Gardening Rover with Plant Recognition using Neural Networks</p>
                <a class="btn-links" href="https://sciencedirect.com/science/article/pii/S1877050916315356" target="_blank">paper</a>
              </div>
            </li>
            <li>
              <div class='demo_box'>
                <img id="me-project-2" alt="pocket-counselor-me-project.png" src="https://drive.google.com/uc?id=1bzqx2gwog3B_q8W86QJNgrPy7M0vM246" onclick="showHideModal(this.id);" />
                <h4>Pocket Counselor<br>MS in US</h4>
                <p>The one stop shop for all MS in US aspirants</p>
                <a class="btn-links" href="https://play.google.com/store/apps/details?id=com.dwappfactory.pocketcounselorlite&hl=en_IN" target="_blank">app</a>
              </div>
            </li>
          </ul>
          <h4 class="diamond-header">teachings</h4>
          <ul>
            <li>
              <div class='demo_box'>
                <a href="https://docs.google.com/presentation/d/e/2PACX-1vR2c4s31uAiZpRumnZfXwZVC1WK-0WtOhatyQ44JhhZo3MdqByqzHkL37t92_thzUW2tOo_gVsRStbY/pub?start=false&loop=false&delayms=3000" target="_blank"><img alt="first-meetup.png" src="/images/school-of-ai/first-meetup.png" /></a>
                <p>Chennai School of AI - First Meetup</p>
              </div>
            </li>
            <li>
              <div class='demo_box'>
                <img id="infographics-1" alt="ai-basics.png" src="/images/infographics/ai-basics.png" onclick="showHideModal(this.id);" />
                <p>Artificial Intelligence - Basics</p>
              </div>
            </li>
            <li>
              <div class='demo_box'>
                <img id="infographics-2" alt="supervised-learning.png" src="/images/infographics/supervised-learning.png" onclick="showHideModal(this.id);" />
                <p>Supervised Machine Learning</p>
              </div>
            </li>
          </ul>
        </div>
      </div>
      <!--Demo ENDS-->
    </div>
  </div>
</div>

<div id="creative_modal" class="modal">
  <button class="modal_download" id="modal_download" onclick="downloadImage()">Download</button>
  <span class="close">&times;</span>
  <img class="modal-content" id="modal_image">
</div>