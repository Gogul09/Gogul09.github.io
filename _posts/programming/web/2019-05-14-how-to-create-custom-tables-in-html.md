---
layout: post
category: software
class: Front-End Web
title: How to create custom tables in HTML?
description: Learn how to code custom table in HTML using rowspan and colspan attributes.
author: Gogul Ilango
permalink: /software/how-to-create-custom-tables-in-html
---

I always wondered is there a way to create tables like the one shown below without using nested tables (i.e., table inside a table by hacking <span class="coding">border</span> property in CSS). 

<table>
  <tr>
    <td rowspan="2">Gender</td>
    <td colspan="2">Average</td>
    <td rowspan="2">Red Eyes</td>
  </tr>
  <tr>
    <td>Height</td>
    <td>Weight</td>
  </tr>
  <tr>
    <td>Males</td>
    <td>165</td>
    <td>65</td>
    <td>30%</td>
  </tr>
  <tr>
    <td>Females</td>
    <td>150</td>
    <td>50</td>
    <td>45%</td>
  </tr>
</table>

If you want to create the above table using nested tables concept, you would give up on the <span class="coding">width</span> of your table's <span class="coding">td</span> elements. Based on your <span class="coding">td</span> content, the <span class="coding">width</span> property will vary, and eventually you mess up with the borders and finally, you have a messy table with misaligned borders!

One hack is to use a fixed width for your <span class="coding">td</span> elements and fill up the content based on that with <span class="coding">overflow: auto</span> enabled in CSS. But, that's not the best solution to this problem.

---

To solve this problem, we need to preserve the <span class="coding">width</span> of the <span class="coding">td</span> elements somehow. And the best solution is to take a pen and paper!

1. Draw the table that you expect using a pen and paper to clearly understand the layout that you're going to code.

2. After figuring out the layout, use <span class="coding">rowspan</span> and <span class="coding">colspan</span> attributes for your <span class="coding">td</span> elements to bring that customized table (without any nested tables concept). It's that simple 😊

You can look at the below code and output to understand <span class="coding">rowspan</span> and <span class="coding">colspan</span> better.

<div class="code-head">index.html<span>code</span></div>

```html
<table>
  <tr>
    <td rowspan="3">Country</td>
  </tr>
  <tr>
    <td rowspan="2">Region</td>
  </tr>
  <tr>
    <td>Product</td>
    <td>Profit</td>
  </tr>
  <tr>
    <td rowspan="6">India</td>
    <td rowspan="2">Western</td>
    <td>Monitor</td>
    <td>10</td>
  </tr>
  <tr>
    <td>Desk Lamp</td>
    <td>20</td>
  </tr>
  <tr>
    <td rowspan="2">Central</td>
    <td>Monitor</td>
    <td>30</td>
  </tr>
  <tr>
    <td>Desk Lamp</td>
    <td>25</td>
  </tr>
  <tr>
    <td rowspan="2">Eastern</td>
    <td>Monitor</td>
    <td>31</td>
  </tr>
  <tr>
    <td>Desk Lamp</td>
    <td>17</td>
  </tr>
</table>
```

<table>
  <tr>
    <td rowspan="3">Country</td>
  </tr>
  <tr>
    <td rowspan="2">Region</td>
  </tr>
  <tr>
    <td>Product</td>
    <td>Profit</td>
  </tr>
  <tr>
    <td rowspan="6">India</td>
    <td rowspan="2">Western</td>
    <td>Monitor</td>
    <td>10</td>
  </tr>
  <tr>
    <td>Desk Lamp</td>
    <td>20</td>
  </tr>
  <tr>
    <td rowspan="2">Central</td>
    <td>Monitor</td>
    <td>30</td>
  </tr>
  <tr>
    <td>Desk Lamp</td>
    <td>25</td>
  </tr>
  <tr>
    <td rowspan="2">Eastern</td>
    <td>Monitor</td>
    <td>31</td>
  </tr>
  <tr>
    <td>Desk Lamp</td>
    <td>17</td>
  </tr>
</table>