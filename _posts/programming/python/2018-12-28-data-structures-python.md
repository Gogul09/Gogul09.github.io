---
layout: post
category: software
class: Programming Languages
title: Data Structures & Algorithms in Python
description: Understand Data Structures and Algorithms using Python Programming Language.
author: Gogul Ilango
permalink: /software/data-structures-and-algorithms-in-python
image: https://drive.google.com/uc?id=1BmxJGJpdBOkiDG5cNaWfzFwA155PLRtp
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#time-complexity">Time Complexity</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#complexity-classes">Complexity Classes</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#linked-list">Linked List</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#bubble-sort">Bubble Sort</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#merge-sort">Merge Sort</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#quick-sort">Quick Sort</a></li>
  </ul>
</div>

<h3 id="time-complexity">Time Complexity</h3>

Instead of asking, *how much time does it take to run a function*, in time complexity's language, we ask *how does the runtime of a function grow?* To learn more about Big O notation and Time Complexity, please watch [this](https://www.youtube.com/watch?v=D6xkbGLQesk){:target="_blank"} excellent video.

<div class="note">
    <p><b>Finding Time Complexity</b></p>
    <ul>
        <li>Find the fastest growing term</li>
        <li>Take out the coefficient</li>
    </ul>
</div>

* \\(O(1)\\) - Swap two numbers.
* \\(O(logn)\\) - Search in a sorted array with binary search.
* \\(O(n)\\) - Search for a maximum element in an unsorted array.
* \\(O(n*logn)\\) - Merge Sort, Quick Sort, Heap Sort.
* \\(O(n^2)\\) - Bubble Sort.
* \\(O(2^n)\\) - Travelling Salesman Problem with Dynamic Programming.
* \\(O(n!)\\) - Travelling Salesman Problem with Brute Force Search.

<figure>
    <img src="https://drive.google.com/uc?id=1oIw8LnBDfmWtKmDsbAdXvRSxmHNOQXSr" class="typical-image" />
    <figcaption>Big O & Time Complexity</figcaption>
</figure>

<h3 id="complexity-classes">Complexity Classes</h3>

* \\(\text{P}\\) - Polynomial
  * One of the most fundamental complexity classes.
  * Contains all decision problems that can be solved by a deterministic Turing machine.
  * \\(\text{P}\\) is the class of computational problems that are efficiently solvable.
  * Ex: sorting algorithms.
* \\(\text{NP}\\) - Non-deterministic Polynomial
  * If we have a solution to a problem, we can verify this solution in polynomial time (by a deterministic Turing machine).
  * For instance where the answer in Yes, have efficiently verifiable proofs of the fact that the answer is indeed yes.
  * The complexity class \\(\text{P}\\) is contained in \\(\text{NP}\\).
  * Most important question is \\(\text{N}\\) = \\(\text{NP}\\) is it true?
  * Ex: Integer Factorization, Travelling Salesman Problem. 
* \\(\text{NP complete}\\)
  * A decision problem is \\(\text{NP complete}\\) when it is both in \\(\text{NP}\\) and \\(\text{NP hard}\\).
  * Although any given solution to an \\(\text{NP complete}\\) problem can be verified in polynomial time, there is no known efficient way to locate a solution in the first place.
  * We ususually just look for an approximate solution.
  * Ex: Chinese Postman Problem, Graph Coloring, Hamiltonian Cycle.
* \\(\text{NP hard}\\)
  * This is a class of problems that are at least as hard as the hardest problems in \\(\text{NP}\\).
  * A problem H is \\(\text{NP hard}\\) when every problem L in \\(\text{NP}\\) can be reduced in polynomial time to H.
  * As a consequence, finding a polynomial algorithm to solve any \\(\text{NP hard}\\) problem would give polynomial algorithms for all the problems in \\(\text{NP}\\).
  * Ex: Halting problem.

<h3 id="linked-list">Linked List</h3>

<div class="code-head"><span>code</span>linked_list.py</div>

```python
# class to create a node that has data and pointer
class node: 
    def __init__(self, data=None):
        self.data = data
        self.next = None

# class to create a linked list of nodes
class linked_list:
    def __init__(self):
        self.head = node() 

    def append(self, data):
        new_node = node(data)
        cur = self.head
        while cur.next != None:
            cur = cur.next
        cur.next = new_node

    def length(self):
        cur = self.head
        total = 0
        while cur.next != None:
            total += 1
            cur = cur.next
        return total

    def display(self):
        elems = []
        cur_node = self.head
        while cur_node.next != None:
            cur_node = cur_node.next
            elems.append(cur_node.data)
        print(elems)

    def get(self, index):
        if index >= self.length():
            print("ERROR: index out of range!")
            return None
        cur_idx = 0
        cur_node = self.head
        while True:
            cur_node = cur_node.next
            if cur_idx == index:
                return cur_node.data
            cur_idx += 1

    def erase(self, index):
        if index >= self.length():
            print("ERROR: index out of range!")
            return None
        cur_idx = 0
        cur_node = self.head
        while True:
            last_node = cur_node
            cur_node = cur_node.next
            if cur_idx == index:
                last_node.next = cur_node.next
                return
            cur_idx += 1 

if __name__ == '__main__':
    my_list = linked_list()
    my_list.append(1)
    my_list.append(2)
    my_list.append(3)
    my_list.append(4)
    my_list.display()
    print("Element at 2nd index: {}".format(my_list.get(2)))
    my_list.erase(2)
    print("Elements after erasing element at index 2")
    my_list.display()
```

```
[1, 2, 3, 4]
Element at 2nd index: 3
Elements after erasing element at index 2
[1, 2, 4]
```
{: .code-output}

<h3 id="bubble-sort">Bubble Sort</h3>

<figure>
    <img src="https://drive.google.com/uc?id=12AOX_kQQ9hypZxMf7ISz7yNcCmL2SrzH" class="typical-image" />
    <figcaption>Bubble Sort</figcaption>
</figure>

<div class="code-head"><span>code</span>bubble_sort.py</div>

```python
from random import randint

# create randomized array of length "length"
# array integers are of range 0, maxint
def create_array(length=10, maxint=50):
    new_arr = [randint(0, maxint) for _ in range(length)]
    return new_arr

#-------------------------------------
# bubble sort algorithm to input array
#-------------------------------------
def bubble_sort(arr):
    swapped = True
    while swapped:
        swapped = False
        for i in range(1, len(arr)):
            if arr[i-1] > arr[i]:
                arr[i], arr[i-1] = arr[i-1], arr[i]
                swapped = True
    return arr

if __name__ == '__main__':
    a = create_array()
    print(a)
    a = bubble_sort(a)
    print(a)
```

```
[37, 36, 13, 12, 43, 4, 32, 14, 32, 4]
[4, 4, 12, 13, 14, 32, 32, 36, 37, 43]
```
{: .code-output}

<h3 id="merge-sort">Merge Sort</h3>

<figure>
    <img src="https://drive.google.com/uc?id=13bP9IE_XhDD6dympWB5DqGIhKfgefEsT" class="typical-image" />
    <figcaption>Merge Sort</figcaption>
</figure>

<div class="code-head"><span>code</span>merge_sort.py</div>

```python
from random import randint

# create randomized array of length "length"m
# array integers are of range 0, maxint
def create_array(length=10, maxint=50):
    new_arr = [randint(0, maxint) for _ in range(length)]
    return new_arr

#-------------------------------------
# merge sort to combine two arrays
#-------------------------------------
def merge(a,b):
    # final output array
    c = []

    a_idx, b_idx = 0, 0
    while a_idx<len(a) and b_idx<len(b):
        if a[a_idx]<b[b_idx]:
            c.append(a[a_idx])
            a_idx += 1
        else:
            c.append(b[b_idx])
            b_idx += 1

    if a_idx == len(a): c.extend(b[b_idx:])
    else:               c.extend(a[a_idx:])

    return c

#-------------------------------------
# merge sort algorithm to input array
#-------------------------------------
def merge_sort(a):

    # a list of zero or one elements is sorted, by definition
    if len(a) <= 1: return a

    # split the list in half and call merge sort recursively on each half
    mid = int(len(a)/2)
    left, right = merge_sort(a[:mid]), merge_sort(a[mid:])

    # merge the now-sorted sublists
    return merge(left,right)

if __name__ == '__main__':
    a = create_array()
    print(a)
    s = merge_sort(a)
    print(s)
```

```
[45, 8, 25, 1, 32, 37, 34, 3, 4, 3]
[1, 3, 3, 4, 8, 25, 32, 34, 37, 45]
```
{: .code-output}

<h3 id="quick-sort">Quick Sort</h3>

<figure>
    <img src="https://drive.google.com/uc?id=15Gj4lbSkyka2zTIz2EwSkELfTWjOj-k9" class="typical-image" />
    <figcaption>Quick Sort</figcaption>
</figure>

<div class="code-head"><span>code</span>quick_sort.py</div>

```python
from random import randint

# create randomized array of length "length"m
# array integers are of range 0, maxint
def create_array(length=10, maxint=50):
    new_arr = [randint(0, maxint) for _ in range(length)]
    return new_arr

# quick sort algorithm to input array
def quick_sort(a):

    # a list of zero or one elements is sorted, by definition
    if len(a) <= 1: return a

    # list to hold values based on pivot
    smaller, equal, larger = [], [], []

    # choose a random pivot element
    pivot = a[randint(0,len(a)-1)]

    # iterate over each element and compare with pivot  
    for x in a:
        if x<pivot:     smaller.append(x)
        elif x==pivot:  equal.append(x)
        else:           larger.append(x)

    # recursively quick sort sub list and concatenate
    return quick_sort(smaller) + equal + quick_sort(larger) 

if __name__ == '__main__':
    a = create_array()
    print(a)
    s = quick_sort(a)
    print(s)
```

```
[3, 27, 12, 8, 12, 39, 1, 2, 23, 8]
[1, 2, 3, 8, 8, 12, 12, 23, 27, 39]
```
{: .code-output}