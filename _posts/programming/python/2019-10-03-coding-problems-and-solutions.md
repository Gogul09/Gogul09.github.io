---
layout: post
category: software
class: Programming
title: Coding Problems and Solutions
description: Understand how to use python and logical thinking to solve coding problems asked by tech companies around the world.
author: Gogul Ilango
permalink: /software/coding-problems-and-solutions
image: https://drive.google.com/uc?id=1VcOjbVA3yzQqEHMnFKpd5mHKTiAdWcsY
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#easy">Easy</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#medium">Medium</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#hard">Hard</a></li>
  </ul>
</div>

<h3 id="easy">Easy</h3>


<h5 class="coding-problem-head">problem 001</h5>
Given a list of numbers and a number k, return whether any two numbers from the list add up to k. For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17. Can you do this in one pass?

<div class="code-head">problem_e001.py<span>code</span></div>

```python
def prob_e001(arr, k):
  s = set()
  for i in range(len(arr)):
    temp = k - arr[i]
    if temp in s:
      print("Pair with sum "+ str(k) + " is (" + str(arr[i]) + ", " + str(temp) + ")")
    s.add(arr[i])

import time
start_time = time.time()
prob_e001([1, -1, 5, 6], 7)
print("--- %s micro seconds ---" % ((time.time() - start_time)*1000000))
```

```
Pair with sum 7 is (6, 1)
--- 314.95094299316406 micro seconds ---
```
{: .code-out}

<h3 id="medium">Medium</h3>

<h5 class="coding-problem-head">problem 001</h5>

Given an array of integers, return a new array such that each element at index i of the new array is the product of all the numbers in the original array except the one at i.

For example, 
* If the input is [1, 2, 3, 4, 5], the expected output is [120, 60, 40, 30, 24]
* If the input is [3, 2, 1], the expected output is [2, 3, 6]

<div class="code-head">problem_m001.py<span>code</span></div>

```python
# best approach without using division
# time complexity  : O(n)
# space complexity : O(n)
def prob_m001(arr):
  left  = [None] * len(arr)
  right = [None] * len(arr)
  res   = [None] * len(arr)

  left[0], right[(len(arr))-1] = 1, 1

  for i in range(1, len(arr)):
    left[i] = arr[i-1] * left[i-1]

  for i in range((len(arr)-2), -1, -1):
    right[i] = arr[i+1] * right[i+1]

  for i in range(len(arr)):
    res[i] = left[i] * right[i]

  return res

import time
start_time = time.time()
print(prob_m001([1, 2, 3, 4, 5]))
print("--- %s micro seconds ---" % ((time.time() - start_time)*1000000))
```

```
[120, 60, 40, 30, 24]
--- 437.73651123046875 micro seconds ---
```
{: .code-out}

<h3 id="hard">Hard</h3>