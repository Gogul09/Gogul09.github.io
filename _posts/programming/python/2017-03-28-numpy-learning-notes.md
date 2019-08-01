---
layout: post
category: software
class: Programming Languages
title: NumPy Learning Notes
description: Understand the syntax and how to's of NumPy python package which is highly used for scientific computing and data manipulation.
author: Gogul Ilango
permalink: /software/numpy-learning-notes
image: https://drive.google.com/uc?id=1kpEvCKb6ETgknBQzIqPu3ihThz_FZ9uK
---

[NumPy](http://www.numpy.org/){:target="_blank"} is a scientific computing package for Python programming language which is highly used when **matrices** and **vectors** are involved for computations. Some of the domains where NumPy is used are Deep Learning, Computer Vision, Machine Learning, Image Processing, Data Analytics etc.

In this page, you will find the **syntax** and **most common how to's** of NumPy in Python. This might be useful for python beginners to find syntax and code quickly online in a single page.

<div class="math-cover">
    <h3>Contents</h3>
    <div class="toc-box">
        <ul>
            <li><a href="#check-numpy-version">Check numpy version</a></li>
            <li><a href="#basic-array-representation">Basic Array Representation</a></li>
            <li><a href="#how-to-create-a-numpy-array-of-ones">How to create a numpy array of ones?</a></li>
            <li><a href="#how-to-create-a-numpy-array-of-zeros">How to create a numpy array of zeros?</a></li>
            <li><a href="#how-to-create-a-identity-matrix-in-numpy">How to create a identity matrix in numpy?</a></li>
            <li><a href="#math-operations-in-numpy">Math operations in numpy</a></li>
            <li><a href="#how-to-perform-dot-product-in-numpy">How to perform dot product in numpy?</a></li>
            <li><a href="#how-to-concatenate-numpy-arrays">How to concatenate numpy arrays?</a></li>
            <li><a href="#how-to-concatenate-numpy-arrays-along-particular-axis">How to concatenate numpy arrays along particular axis?</a></li>
            <li><a href="#how-to-compute-sum-of-a-matrix">How to compute sum of a matrix?</a></li>
            <li><a href="#how-to-reshape-a-numpy-array">How to reshape a numpy array?</a></li>
            <li><a href="#how-to-flatten-a-numpy-array">How to flatten a numpy array?</a></li>
            <li><a href="#how-to-expand-the-shape-of-a-numpy-array">How to expand the shape of a numpy array?</a></li>
            <li><a href="#how-to-find-unique-items-in-a-numpy-array">How to find unique items in a numpy array?</a></li>
        </ul>
    </div>
</div>

> **Update**: As Python2 faces [end of life](https://pythonclock.org/), the below code only supports **Python3**.

<h3 class="code-head" id="check-numpy-version">Check numpy version<span>code</span></h3>

```python
import numpy as np
print(np.__version__)  # prints "1.11.3" for me
```

<h3 class="code-head" id="basic-array-representation">Basic Array Representation<span>code</span></h3>

```python
# create a 3x1 array (datatype="int" chosen by numpy)
a = np.array([1,2,3])

print(a)       # prints "[1,2,3]"
print(a.shape) # prints "[3L,]"
print(type(a)) # prints "<type 'numpy.ndarray'>"

# create a 3x1 array with datatype "float"
b = np.array([1.1,2.2,3.3], dtype="float")

print(b)       # prints "[1.1,2.2,3.3]"
print(b.shape) # prints "[3L,]"
print(type(b)) # prints "<type 'numpy.ndarray'>"

# accessing elements in the numpy array
print(a[0])       # prints "1"
print(b[2])       # prints "3.3"
print(type(a[2])) # prints "<type 'numpy.int32'>"
print(type(b[1])) # prints "<type 'numpy.float64'>"

# create a 2x2 array
c = np.array([[1,2],[3,4]])

print(c)         # prints "[[1,2],[3,4]]"
print(c.shape)   # prints "(2L, 2L)"
```

<h3 class="code-head" id="how-to-create-a-numpy-array-of-ones">How to create a numpy array of ones?<span>code</span></h3>

```python
# create a 2x2 array full of ones (dtype=float64 chosen by numpy)
d = np.ones((2,2))

print(d)             # prints "[[ 1.  1.]
                    #          [ 1.  1.]]"
print(d.shape)       # prints "(2L, 2L)"
print(type(d))       # prints "<type 'numpy.ndarray'>"
print(type(d[0][0])) # prints "<type 'numpy.float64'>"
```

<h3 class="code-head" id="how-to-create-a-numpy-array-of-zeros">How to create a numpy array of zeros?<span>code</span></h3>

```python
# create a 3x3 array full of zeros (dtype=float64 by default)
e = np.zeros((3,3))

print(e)             # prints "[[ 0.  0.  0.]
                     #          [ 0.  0.  0.]
                     #          [ 0.  0.  0.]]"
print(e.shape)       # prints "(3L, 3L)"
print(type(e))       # prints "<type 'numpy.ndarray'>"
print(type(e[0][0])) # prints "<type 'numpy.float64'>"
```

<h3 class="code-head">How to create a numpy array of random values?<span>code</span></h3>

```python
# create a 2x2 matrix with random values
f = np.random.random((2,2))

print(f)        # prints "[[ 0.65155439  0.39628659]
                #          [ 0.33349215  0.03323669]]" for me
print(f.shape)  # prints (2L, 2L)
```

<h3 class="code-head" id="how-to-create-a-identity-matrix-in-numpy">How to create a identity matrix in numpy?<span>code</span></h3>

```python
# create an identity matrix of order 3
g = np.eye(3)

print(g)             # prints "[[ 100.  100.]
                     #          [ 100.  100.]]"
print(g.shape)       # prints (2L, 2L)
```

<h3 class="code-head">How to create a numpy array with a identical values?<span>code</span></h3>

```python
# create a 2x2 constant matrix of value "100" in each cell (dtype=float64 by default)
h = np.full((2,2), 100)

print(h)             # prints "[[ 100.  100.]
                     #          [ 100.  100.]]"
print(h.shape)       # prints (2L, 2L)
```

<h3 class="code-head" id="math-operations-in-numpy">Math operations in numpy<span>code</span></h3>

```python
# create two 2x2 matrices
a = np.array([[2,2],[2,2]])
b = np.array([[2,2],[2,2]])

# element-wise addition
add = a + b
print(add)            # prints "[[4 4]
print(np.add(a,b))    #          [4 4]]"

# element-wise subtraction
sub = a - b
print(sub)               # prints "[[0 0]
print(np.subtract(a,b))  #          [0 0]]"

# element-wise multiplication
mul = a * b
print(mul)                  # prints "[[4 4]
print(np.multiply(a,b))     #          [4 4]]"

# element-wise division
div = a / b
print(div)                # prints "[[1 1]
print(np.divide(a,b))     #          [1 1]]"
```

<h3 class="code-head" id="how-to-perform-dot-product-in-numpy">How to perform dot product in numpy?<span>code</span></h3>

```python
# create a matrix and a vector
W = np.random.random((3,3))
x = np.array([1,2,3])

print(W)               # prints "[[ 0.3342051   0.87642564  0.35777489]
                       #          [ 0.24531674  0.36355452  0.39563227]
                       #          [ 0.83769694  0.7987359   0.97012682]]"     
print(x)               # print("[1 2 3]"

print(W.shape)         # prints "(3L, 3L)"
print(x.shape)         # prints "(3L,)"

# take dot product between matrix and vector
inner = np.dot(W,x)

print(inner)           # prints "[ 3.16038104  2.15932259  5.34554919]"
print(inner.shape)     # prints "(3L,)"
```

<h3 class="code-head" id="how-to-concatenate-numpy-arrays">How to concatenate numpy arrays?<span>code</span></h3>

```python
# create two arrays
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])

print(v1.shape)      # prints "(3L,)"
print(v2.shape)      # prints "(3L,)"

# concatenate arrays
concat = np.concatenate((v1, v2))  # note that arrays must be passed as tuples

print(concat)        # prints "[1 2 3 4 5 6]"
print(concat.shape)  # prints "(6L,)"
```

<h3 class="code-head" id="how-to-concatenate-numpy-arrays-along-particular-axis">How to concatenate numpy arrays along particular axis?<span>code</span></h3>

```python
# create two matrices
v3 = np.array([[1,2,3],[4,5,6]])
v4 = np.array([[40,40,40],[50,50,50]])

print(v3.shape)      # prints "(2L,3L)"
print(v4.shape)      # prints "(2L,3L)"

# concatenate arrays along different axis
concat_0 = np.concatenate((v3, v4), axis=0)  # concatenate along axis=1
concat_1 = np.concatenate((v3, v4), axis=1)  # concatenate along axis=1

# vertical stack : same as np.concatenate with axis=0
v_stack  = np.vstack((v3,v4))

# horizontal stack : same as np.concatenate with axis=1
h_stack  = np.hstack((v3,v4))

print(concat_0)        # prints "[[ 1  2  3]
print(v_stack)         #          [ 4  5  6]
                       #          [40 40 40]
                       #          [50 50 50]]"
print(concat_0.shape)  # prints "(4L, 3L)"

print(concat_1)        # prints "[[ 1  2  3 40 40 40]
print(h_stack)         #          [ 4  5  6 50 50 50]]"
print(concat_1.shape)  # prints "(2L, 6L)"
```

<h3 class="code-head">How to find transpose of a matrix in numpy?</h3>

```python
# create a 2x3 matrix
m = np.random.random((2,3))

# take the transpose of matrix
mT = m.T

print(m)          # prints "[[ 0.91066068  0.43176095  0.3599931 ]
                  #          [ 0.3701366   0.26224812  0.76553986]]"
print(mT)         # prints "[[ 0.91066068  0.3701366 ]
                  #         [ 0.43176095  0.26224812]
                  #         [ 0.3599931   0.76553986]]"

print(m.shape)    # prints "(2L, 3L)"
print(mT.shape)   # prints "(3L, 2L)"
```

<h3 class="code-head" id="how-to-compute-sum-of-a-matrix">How to compute sum of a matrix?<span>code</span></h3>

```python
# create a 2x3 matrix
p = np.array([[10,20,30],[40,50,60]])

print(p.shape)           # prints "(2L, 3L)"
print(np.sum(p))         # prints "210"
print(np.sum(p, axis=0)) # sum of each column: prints "[50 70 90]"
print(np.sum(p, axis=1)) # sum of each row   : prints "[60 150]"
```


<h3 class="code-head" id="how-to-reshape-a-numpy-array">How to reshape a numpy array?<span>code</span></h3>

```python
import numpy as np

# create a 2x3 array
a = np.array([[1,1,1],[2,2,2]])

print(a)              # prints "[[1 1 1]
                      #         [2 2 2]]"
print(a.shape)        # prints "(2, 3)"

# make 2x3 array to 6x1 array
b = np.reshape(a, 6)

print(b)              # prints "[1 1 1 2 2 2]"
print(b.shape)        # prints "(6,)"

# make 6x1 array to 3x2 array
c = np.reshape(b, (3,2))

print(c)              # prints "[[1 1]
                      # [1 2]
                      # [2 2]]"
print(c.shape)        # "(3, 2)"
```

<h3 class="code-head" id="how-to-flatten-a-numpy-array">How to flatten a numpy array?<span>code</span></h3>

```python
# create a 3x3 array
c = np.array([[1,2,3],[4,5,6],[7,8,9]])

print(c)          # prints "[[1 2 3]
                  #          [4 5 6]
                  #          [7 8 9]]"
print(c.shape)    # prints "(3, 3)"

# flatten the input array
d = np.ravel(c)
e = c.flatten()

print(d)          # prints "[1 2 3 4 5 6 7 8 9]"
print(e)          # prints "[1 2 3 4 5 6 7 8 9]"

# you can use order to specify how the array needs to be flattened
# order='F' --> To index elements in column major
# order='C' --> To index elements in row major
print(np.ravel(c, order='F'))  # prints "[1 4 7 2 5 8 3 6 9]"
print(np.ravel(c, order='C'))  # prints "[1 2 3 4 5 6 7 8 9]"
```

<h3 class="code-head" id="how-to-expand-the-shape-of-a-numpy-array">How to expand the shape of a numpy array?<span>code</span></h3>

```python
# create an array with 3 elements
a = np.array([500,200,100])

print(a)             # print("[500 200 100]"
print(a.shape)       # print("(3,)"

# insert a new axis along the row
b = np.expand_dims(a, axis=0)  

# insert a new axis along the column
c = np.expand_dims(a, axis=1)

print(b)             # prints "[[500 200 100]]"
print(b.shape)       # prints "(1, 3)"

print(c)             # prints "[[500]
                    #          [200]
                    #          [100]]"
print(c.shape)       # prints "(3, 1)"
```

<h3 class="code-head" id="how-to-find-unique-items-in-a-numpy-array">How to find unique items in a numpy array?<span>code</span></h3>

```python
# create an array with 5 elements
m = np.array([1,2,1,1,4])

print(m)         # prints "[1 2 1 1 4]"  
print(m.shape)   # prints "(5,)"

# get the unique elements and thier indexes
n, indices = np.unique(m, return_index=True)

print(n)         # prints "[1 2 4]"
print(indices)   # prints "[0 1 4]"
```