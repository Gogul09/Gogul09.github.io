---
layout: post-doc
category: software
class: Programming Languages
title: Python Learning Notes
description: Understand the syntax, modules and how to's of Python programming language which is heavily used in today's technology industry.
author: Gogul Ilango
permalink: /software/python-learning-notes
image: https://drive.google.com/uc?id=1-OEoUjX1r1V-Nx6kJIwUmEVt44rZnJxk
---

<div id="awesomeSideNav" class="sidenav">
    <a href="javascript:void(0)" class="closebtn" onclick="closeSideNav()">&times;</a>
    <h3><a href="#basics">Basics</a></h3>
    <ul>
        <li><a href="#hello-world">Hello World</a></li>
        <li><a href="#data-types">Data types</a></li>
        <li><a href="#multiple-variable-assignments">Multiple variable assignments</a></li>
        <li><a href="#math-operations">Math operations</a></li>
        <li><a href="#logical-operations">Logical operations</a></li>
        <li><a href="#conditions">Conditions</a></li>
        <li><a href="#loops">Loops</a></li>
        <li><a href="#strings">Strings</a></li>
        <li><a href="#lists">List</a></li>
        <li><a href="#tuples">Tuple</a></li>
        <li><a href="#set">Set</a></li>
        <li><a href="#dictionaries">Dictionary</a></li>
        <li><a href="#exception-handling">Exception Handling</a></li>
        <li><a href="#functions_1">Functions</a></li>
    </ul>
    <h3><a href="#intermediate">Intermediate</a></h3>
    <ul>
        <li><a href="#list-comprehensions">List Comprehensions</a></li>
        <li><a href="#dict-comprehensions">Dict Comprehensions</a></li>
        <li><a href="#set-comprehensions">Set Comprehensions</a></li>
        <li><a href="#lambda">Lambda</a></li>
        <li><a href="#generators">Generators</a></li>
        <li><a href="#args-and-kwargs">args & kwargs</a></li>
        <li><a href="#generator-expressions">Generator Expressions</a></li>
    </ul>
    <h3><a href="#modules">Modules</a></h3>
    <ul>
        <li><a href="#re-module">re module</a></li>
        <li><a href="#os-module">os module</a></li>
        <li><a href="#sys-module">sys module</a></li>
        <li><a href="#sys-module">shutil module</a></li>
    </ul>
    <h3><a href="#oop">OOP</a></h3>
    <ul>
        <li><a href="#classes">Classes</a></li>
        <li><a href="#class-variables">Class Variables</a></li>
        <li><a href="#inheritance">Inheritance</a></li>
    </ul>
    <h3><a href="#how-tos">How to's</a></h3>
    <ul>
        <li><a href="#how-to-handle-files">How to handle files?</a></li>
        <li><a href="#how-to-read-file-line-by-line">How to read file line-by-line?</a></li>
        <li><a href="#how-to-write-file-line-by-line">How to write file line-by-line?</a></li>
        <li><a href="#how-to-load-json-file">How to load json file?</a></li>
        <li><a href="#how-to-check-if-list-is-empty">How to check if list is empty?</a></li>
        <li><a href="#how-to-access-index-in-for-loop">How to access index in for loop?</a></li>
        <li><a href="#how-to-sort-a-dictionary-by-key-or-value-alphabetically">How to sort a dictionary by key or value alphabetically?</a></li>
        <li><a href="#how-to-call-tcl-procedure-in-python">How to call tcl procedure in python?</a></li>
    </ul>
</div>

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#basics">Basics</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#intermediate">Intermediate</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#modules">Modules</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#oop">OOP</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#how-tos">How To's</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#resources">Resources</a></li>
  </ul>
</div>
 
In this page, you will find the **syntax**, **modules** and most common **how to's** of Python programming language. This might be helpful for you to find python syntax and code quickly in a single web page. Click on any of the contents in the sidebar to view the python code.

<div class="note">
<p><b>Note:</b> If you are new to programming, learning Python is the best way to start. Because, Python is easier to learn than any other programming language out there as it is concise, expressive, productive, has larger community, has rich set of libraries, frameworks and requires less time to write code.</p>
<ul>
    <li><a href="https://medium.com/@trungluongquang/why-python-is-popular-despite-being-super-slow-83a8320412a9" target="_blank">Why Python is Popular Despite Being (Super) Slow</a></li>
    <li><a href="https://www.kdnuggets.com/2017/07/6-reasons-python-suddenly-super-popular.html" target="_blank">6 Reasons Why Python Is Suddenly Super Popular</a></li>
    <li><a href="https://www.zarantech.com/blog/surprising-facts-python-gaining-popularity-among-developers/" target="_blank">Surprising Facts Why Python is gaining more popularity among Developers</a></li>
</ul>
</div>

> **Update**: As Python2 faces [end of life](https://pythonclock.org/), the below code only supports **Python3**.

<h3 class="centered-heading" id="basics"><span>Basics</span></h3>

<h3 class="code-head" id="hello-world">Hello World<span>code</span></h3>

```python
print("Hello World")
# prints "Hello World"
# above line is a comment
'''
This is a multi-line
comment in python
'''
```

<h3 class="code-head" id="data-types">Data types<span>code</span></h3>

```python
# declare an integer
a = 12
print(a)       # prints "12"
print(type(a)) # prints <type 'int'>

# declare a float
b = 1.7
print(b)       # prints "1.7"
print(type(b)) # prints <type 'float'>

# declare a string
c = "Python"
print(c)       # prints "Python"
print(type(c)) # prints <type 'str'>

# declare a boolean
d = True
print(d)       # prints "True"
print(type(d)) # prints <type 'bool'>
```

<h3 class="code-head" id="multiple-variable-assignments">Multiple variable assignments<span>code</span></h3>

```python
# assign values to multiple variables in a single line
a, b, c = 1, 2, 3

# assign values with different data types to multiple variables in a single line
a, b, c = 1, 3.5, "hello"

print(type(a)) # prints <type 'int'>
print(type(b)) # prints <type 'float'>
print(type(c)) # prints <type 'str'>
```

<h3 class="code-head" id="math-operations">Math operations<span>code</span></h3>

```python
a = 10

print(a + 1)  # Addition: prints "11"
print(a - 1)  # Subtraction: prints "9"
print(a * 2)  # Multiplication: prints "20"
print(a / 2)  # Division: prints "5"
print(a ** 2) # Exponentiation: prints "100"
```

<h3 class="code-head" id="logical-operations">Logical operations<span>code</span></h3>

```python
# declare some boolean variables
x = True
y = False

print(x and y)  # prints "False"
print(x or y)   # prints "True"
print(not x)    # prints "False"
print(x & y)    # prints "False"
```

<h3 class="code-head" id="conditions">Conditions<span>code</span></h3>

```python
# if-elif-else statements

import random
a = [50, 100, 200, 300]

# pick a random number from the list "a"
b = random.choice(a)

# the conditionals
if (b < 100):
    print("Number - " + str(b) + " is less than 100")
elif (b >= 100 and b < 200):
    print("Number - " + str(b) + " is greater than or equal to 100 but less than 200")
else:
    print("Number -" + str(b) + " is greater than or equal to 200")

# for me it prints
# Number - 100 is greater than or equal to 100 but less than 200
```

<h3 class="code-head" id="loops">Loops<span>code</span></h3>

```python
# while loop
c = 0
while (c < 10):
    print(c,  end='')
    c += 1

# prints 0123456789

# for loop
numbers = [1, 2, 4]
for x in numbers:
    print(x)

# prints 1
#        2
#        4

x = 5
for c in range(x):
    print(c)

# prints 0
#        1
#        2
#        3
#        4
```

<h3 class="code-head" id="strings">Strings<span>code</span></h3>

```python
# declare two strings
a = "Python"
b = " is awesome!"

print(len(a))                        # Length of the string: prints "6"
print(len(b))                        # prints "12"
print(a + b)                         # String concatenation: prints "Python is awesome!"
print(a, b)                          # prints "Python  is awesome!"
print("{}{}".format(a, b))          # prints "Python is awesome!"
print("%s%s" % (a, b))               # sprintf style formatting: prints "Python is awesome!"
print(a.upper())                    # converts all characters to uppercase: prints "PYTHON"
print(a.lower())                     # converts all characters to lowercase: prints "python"
print(b.strip())                     # removes trailing and leading whitespaces: prints "is awesome!"
print(b.replace("awesome", "great")) # replace a substring with a new string: prints " is great!"
```

<h3 class="code-head" id="lists">Lists<span>code</span></h3>

```python
# declare a list
l = [1,2,3,4,5]

# length of list
print(len(l))      # prints "5"

# indexing
print(l[0])        # prints "1"
print(l[1])        # prints "2"
print(l[len(l)-1]) # prints "5"
print(l[-1])       # negative indexing: prints "5"

# insert and remove
l.append(6)        # inserts "6" at last
print(l)            # prints "[1,2,3,4,5,6]"
item = l.pop()     # removes last element and returns that element
print(item)         # prints "6"
l.append("string") # adds different data type too
print(l)            # prints "[1,2,3,4,5,'string']"
l.pop()            # removes last string element

# slicing list
print(l[1:2])       # prints "2"
print(l[1:3])       # prints "2,3"
print(l[0:])        # prints "[1,2,3,4,5,'string']"
print(l[0:-1])      # prints "[1,2,3,4,5]"
print(l[:])         # prints "[1,2,3,4,5,'string']"

# loop over the list
for item in l:
    print(item)     # prints each item in list one by one

# enumerate over the list
for i, item in enumerate(l):
    print("{}-{}".format(i, item)) # prints each item with its index


# squaring elements in a list
for item in l:
    if item%2 == 0:
        print(item2)      # square each even number in the list

# above can be achieved using a list comprehension too! (one-line)
print([x2 for x in l if x%2==0])

# sort the list
b = [5, 7, 2, 4, 9]

# ascending order
b.sort()
print(b)  # prints [2, 4, 5, 7, 9]

# descending order
b.sort(reverse=True)
print(b) # prints [9, 7, 5, 4, 2]

# reverse the list (notice this is not descending order sort)
a = ["dhoni", "sachin", "warner", "abd"]
a.reverse()
print(a) # prints ['abd', 'warner', 'sachin', 'dhoni']

# count of object in list
a = [66, 55, 44, 22, 11, 55, 22] 
print(a.count(22)) # prints 2
```

<h3 class="code-head" id="tuples">Tuples<span>code</span></h3>

```python
# declare a tuple
t = (500, 200)

print(type(t))    # prints "<type 'tuple'>"
print(t[1])       # prints 200

# tuple of tuples
tt = ((200,100), t)

print(tt)         # prints "((200, 100), (500, 200))"
print(tt[1])      # prints "(500, 200)"

# loop over tuple
for item in t:
    print(item)   # prints each item in the tuple


# built-in tuple commands
print(len(t)) # prints the length of tuple which is 2
print(max(t)) # prints the max-valued element which is 500
print(min(t)) # prints the min-valued element which is 200

# convert list to tuple
l = [400, 800, 1200]
l_to_t = tuple(l)

print(type(l_to_t)) # prints <class 'tuple'>
```

<h3 class="code-head" id="set">Set<span>code</span></h3>

```python
# set is a collection of unordered and unindexed data which is written with curly brackets.
s = {"ironman", "hulk", "thor", "thanos"}

for x in s:
    print(x)

'''
prints 
ironman
thor
hulk
thanos
'''

# check if value exist in set
if "thanos" in s:
    print("endgame") # prints 'endgame'

# add a single item to a set using 'add'
s.add("rocket")

# add multiple items to a set using 'update'
s.update(["blackhood", "blackwidow"])

# get length of a set
print(len(s)) # prints 7

# 'remove' or 'discard' an item from the set
# 'remove' raise an error if item to remove does not exist
# 'discard' will not raise any error if item to remove does not exist
s.remove("thanos")
s.discard("blackwidow")

# clear the set
s.clear()

# delete the set
del s
```

<h3 class="code-head" id="dictionaries">Dictionaries<span>code</span></h3>

```python
# declare a dictionary
d = { "1" : "Ironman", 
      "2" : "Captain America", 
      "3" : "Thor"
    }

print(type(d))    # prints "<type 'dict'>"
print(d["1"])     # prints "Ironman"

# loop over dictionary
for key in d:
    print(key)    # prints each key in d
    print(d[key]) # prints value of each key in d (unsorted)

# change values in the dictionary
d["2"] = "Hulk"
for key, value in d.items():
    print(key + " - " + value)

'''
prints
1 - Ironman
2 - Hulk
3 - Thor
'''

# check if key exists in a dictionary
if "3" in d:
    print("Yes! 3 is " + d["3"])
# prints 'Yes! 3 is Thor'

# get length of the dictionary
print(len(d)) # prints 3

# insert a key-value pair to a dictionary
d["4"] = "Thanos"

# remove a key-value pair from the dictionary
d.pop("4")

# same thing using 'del' keyword
del d["2"]

# clear a dictionary
d.clear()
```

<h3 class="code-head" id="exception-handling">Exception Handling<span>code</span></h3>

```python
# try-except-finally
# try: test a block of code for errors.
# except: allows handling of errors.
# finally: execute code, regardless of the result of try and except blocks.
try:
    print(x)
except:
    print("Something is wrong!")

# prints 'Something is wrong!' as x is not defined

try:
    print(x)
except: 
    print("Something is wrong!")
finally:
    print("Finally always execute after try-except.")

# prints
# Something is wrong!
# Finally always execute after try-except.
```

<h3 class="code-head" id="functions_1">Functions<span>code</span></h3>


```python
def squared(x):
    return x*x

print(squared(2))   # prints "4"
```

<!-- --------------------------------------------------------------------- -->

<h3 class="centered-heading" id="intermediate"><span>Intermediate</span></h3>

<h3 class="code-head" id="list-comprehensions">List Comprehensions<span>code</span></h3>

```python
nums = [1, 2, 3, 4, 5, 6, 7]

# traditional for loop
l = []
for n in nums:
    l.append(n)
print(l) # prints '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]'

# meet list comprehension
l = [n for n in nums]
print(l) # prints '[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]'

# get square of each number
l = [n*n for n in nums]
print(l) # prints '[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]'

# same thing achieved using 'map' + 'lambda'
# 'map' means running everything in the list for a certain function
# 'lambda' means an anonymous function
l = map(lambda n: n*n, nums)
for x in l:
    print(x) 

# prints 
# 1
# 4
# 9
# 16
# 25
# 36
# 49
# 64
# 81
# 100

# using 'if' in list comprehension
l = [n for n in nums if n%2 == 0]
print(l) # prints '[2, 4, 6, 8, 10]'

# returning tuples with two for loops in list comprehension
l = []
for letter in "ab":
    for num in range(2):
        l.append((letter, num))
print(l) # prints '[('a', 0), ('a', 1), ('b', 0), ('b', 1)]'

# same thing using list comp
l = [(letter, num) for letter in "ab" for num in range(2)]
print(l) # prints '[('a', 0), ('a', 1), ('b', 0), ('b', 1)]'
```

<h3 class="code-head" id="dict-comprehensions">Dict Comprehensions<span>code</span></h3>

```python
names = ["Robert Downey Jr", "Chris Evans", "Chris Hemsworth", "Mark Ruffalo"]
heros = ["Ironman", "Captain America", "Thor", "Hulk"]

# traditional dictionary using zip()
d = {}
for name, hero in zip(names, heros):
    d[name] = hero
for name in d:
    print(name + " - " + d[name])

'''
prints 
Mark Ruffalo - Hulk
Chris Hemsworth - Thor
Robert Downey Jr - Ironman
Chris Evans - Captain America
'''

# meet dict comprehension
d = {name: hero for name, hero in zip(names, heros)}
for name in d:
    print(name + " - " + d[name])

'''
prints 
Mark Ruffalo - Hulk
Chris Hemsworth - Thor
Robert Downey Jr - Ironman
Chris Evans - Captain America
'''

# dict comprehension with condition
d = {name: hero for name, hero in zip(names, heros) if name != "Mark Ruffalo"}
for name in d:
    print(name + " - " + d[name])

'''
prints 
Chris Hemsworth - Thor
Robert Downey Jr - Ironman
Chris Evans - Captain America
'''
```

<h3 class="code-head" id="set-comprehensions">Set Comprehensions<span>code</span></h3>

```python
nums = [1, 1, 2, 1, 3, 4, 4, 5, 5, 6, 7, 8, 8, 9]

# traditional set (list of unique elements)
s = set()
for n in nums:
    s.add(n)
print(s) # prints {1, 2, 3, 4, 5, 6, 7, 8, 9}

# meet set comprehension
s = {n for n in nums}
print(s) # prints {1, 2, 3, 4, 5, 6, 7, 8, 9}
```

<h3 class="code-head" id="generator-expressions">Generator Expressions<span>code</span></h3>

```python
# I need to yield 'n*n' for each 'n' in nums
nums = [1,2,3,4,5,6,7,8,9]

# traditional generator function
def gen_func(nums):
    for n in nums:
        yield n*n

m = gen_func(nums)
for i in m:
    print(i)

# generator expression
m = (n*n for n in nums)
for i in m:
    print(i)

'''
both prints 
1 
4 
9 
16
25
36
49
64
81
'''
```

<h3 class="code-head" id="lambda">Lambda<span>code</span></h3>

```python
# a lambda function = a small anonymous function
# takes any number of arguments, but can have only one expression
# lambda arguments : expression

# lambda function with one argument
add_hundred = lambda x : x + 100
print(add_hundred(5)) # prints 105

# lambda function with multiple arguments
multiply = lambda a, b, c : a*b*c
print(multiply(10,5,10)) # prints 500
```

<h3 class="code-head" id="generators">Generators<span>code</span></h3>

```python
# generators uses the keyword 'yield' to take one element at a time
# it helps increase performance and make code more readable
def square_numbers(nums):
    for i in nums:
        yield (i*i)

nums = square_numbers([1,2,3,4,5])
print(type(nums)) # <class 'generator'>
print(nums)       # <generator object square_numbers at 0x0000018525078B48>
for n in nums:
    print(n)
# prints 
# 1
# 4
# 9
# 16
# 25


# another example
def people_list(num_people):
    result = []
    for i in xrange(num_people):
        person = {
                   "id"    : i,
                   "name"  : random.choice(names),
                   "major" : random.choice(majors) 
                }
        result.append(person)
    return result

def people_generator(num_people):
    for i in xrange(num_people):
        person = {
                    "id"    : i,
                    "name"  : random.choice(names),
                    "major" : random.choice(majors)
                  }
        yield person

t1 = time.clock()
people = people_list(1000000)
t2 = time.clock()

print("[INFO] Took {} seconds".format(t2-t1))
# prints
# [INFO] Took 1.2467856325541558 seconds

t1 = time.clock()
people = people_generator(1000000)
t2 = time.clock()

print("[INFO] Took {} seconds".format(t2-t1))
# prints
# [INFO] Took 0.12330942238179077 seconds
```

<h3 class="code-head" id="args-and-kwargs">args & kwargs<span>code</span></h3>

```python
# *args used to pass non-keyworded variable-length arguments to a function
def my_nums(*args):
    for n in args:
        print(n, end=" ")
    print(type(args))

# prints
my_nums(1,2,4,5,6)
# prints 
# '1 2 4 5 6'
# <class 'tuple'>

# **kwargs used to pass keyworded variable-length arguments to a function
def my_fullname(**kwargs):
    for key, value in kwargs.items():
        print(key + " - " + value)
    print(type(kwargs))

# prints
my_fullname(firstname="Gogul", lastname="Ilango") 
# prints 
# lastname - Ilango
# firstname - Gogul
# <class 'dict'>
```

<!-- --------------------------------------------------------------------- -->


<h3 class="centered-heading" id="modules"><span>Modules</span></h3>

<div class="code-head" id="re-module">
    <span title="click to see re rules" id="btn-re-python" onclick="boxHandler(this.id)">re rules    
    </span>
    <div class="rule-box" id="box-re-python">
        <p><b>Identifiers</b></p>
        <div style="text-align: center; width: 100%">
            <table>
                <tr>
                    <td>\d</td>
                    <td>any number</td>
                </tr>
                <tr>
                    <td>\D</td>
                    <td>anything but a number</td>
                </tr>
                <tr>
                    <td>\s</td>
                    <td>space</td>
                </tr>
                <tr>
                    <td>\S</td>
                    <td>anything but a space</td>
                </tr>
                <tr>
                    <td>\w</td>
                    <td>any character</td>
                </tr>
                <tr>
                    <td>\W</td>
                    <td>anything but a character</td>
                </tr>
                <tr>
                    <td>.</td>
                    <td>any character, except for a newline</td>
                </tr>
                <tr>
                    <td>\b</td>
                    <td>the whitespace around words</td>
                </tr>
                <tr>
                    <td>\.</td>
                    <td>a period</td>
                </tr>
            </table>
        </div>
        <p><b>Modifiers</b></p>
        <div style="text-align: center; width: 100%">
            <table>
                <tr>
                    <td>{1,3}</td>
                    <td>expecting 1-3 digits ~ \d{1,3}</td>
                </tr>
                <tr>
                    <td>+</td>
                    <td>match 1 or more</td>
                </tr>
                <tr>
                    <td>?</td>
                    <td>match 0 or 1</td>
                </tr>
                <tr>
                    <td>*</td>
                    <td>match 0 or more</td>
                </tr>
                <tr>
                    <td>$</td>
                    <td>match the end of a string</td>
                </tr>
                <tr>
                    <td>^</td>
                    <td>match the start of a string</td>
                </tr>
                <tr>
                    <td>|</td>
                    <td>either or (\d{1-3} | \w{5-6})</td>
                </tr>
                <tr>
                    <td>[]</td>
                    <td>range or variance</td>
                </tr>
                <tr>
                    <td>{x}</td>
                    <td>expecting "x" amount (of digits)</td>
                </tr>
            </table>
        </div>
        <p><b>White Space Characters</b></p>
        <div style="text-align: center; width: 100%">
            <table>
                <tr>
                    <td>\n</td>
                    <td>new line</td>
                </tr>
                <tr>
                    <td>\s</td>
                    <td>space</td>
                </tr>
                <tr>
                    <td>\t</td>
                    <td>tab</td>
                </tr>
                <tr>
                    <td>\e</td>
                    <td>escape</td>
                </tr>
                <tr>
                    <td>\f</td>
                    <td>form feed</td>
                </tr>
                <tr>
                    <td>\r</td>
                    <td>return</td>
                </tr>
            </table>
        </div>
    </div>
Regular Expressions</div>


```python
import re 

# multi-line string example
str = '''
Rahul is 19 years old, and Ashok is 24 years old.
Murali is 65, and his grandfather, Karthik, is 77.
'''

# findall()
ages  = re.findall(r'\d{1,3}', str)
names = re.findall(r'[A-Z][a-z]*', str) 
print(ages)  # prints ['19', '24', '65', '77']
print(names) # prints ['Rahul', 'Ashok', 'Murali', 'Karthik']

# finditer()
ages = re.finditer(r'\d{1,3}', str)
for m in ages:
    print(m.group())

# prints 
# 19
# 24
# 65
# 77

# split()
str = "This is an example string"
splitted = re.split(r'\s*', str)
print(splitted) # prints ['This', 'is', 'an', 'example', 'string']

# match()
str = "Dogs are braver than Cats"
matches = re.match(r'[A-Z][a-z]*', str)
print(matches.group()) # prints "Dogs"

# search()
str = "For data science help, reach support@datacamp.com"
searches = re.search(r'([\w]+)@([\w\.]+)', str)
print(searches.group())  # prints support@datacamp.com
print(searches.group(1)) # prints support
print(searches.group(2)) # prints datacamp.com

# sub()
str = "Ironman is the strongest avenger!"
result = re.sub('Ironman',  'Thor', str)
print(result)
```

<h3 class="code-head" id="os-module">os module<span>code</span></h3>

```python
# os module is a powerful module in python
import os

# get current working directory
print(os.getcwd()) # prints 'G:\\workspace\\Python'

# change current working directory
os.chdir("G:\\workspace\\python\\learning")
print(os.getcwd()) # prints 'G:\\workspace\\python\\learning'

# list directories in the current working directory
print(os.listdir()) # prints ['built-ins', 'Lists', 'numpy', 'strings']

# list only files in the current working directory
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print(f)

# create a directory in the current working directory
os.mkdir("dicts")
os.makedirs("dicts/nested-dicts")

# remove a directory in the current working directory
os.rmdir("dicts")
os.removedirs("dicts/nested-dicts")

# rename a file or directory
os.rename("Lists", "lists")

# stats of a file or directory
os.stat("lists")
# prints 'os.stat_result(st_mode=16895, st_ino=281474977861215, st_dev=4143122855, st_nlink=1, st_uid=0, st_gid=0, st_size=0, st_atime=1480252546, st_mtime=1480252546, st_ctime=1480252090)'

# traverse a directory tree
for dirpath, dirnames, filenames in os.walk("G:\\workspace\\python\\learning"):
    print("Current Path: ", dirpath)
    print("Directories: ", dirnames)
    print("Files: ", filenames)
    print()

'''
prints 
Current Path:  G:\workspace\python\learning
Directories:  ['Built-ins', 'lists', 'NumPy', 'Strings']
Files:  []

Current Path:  G:\workspace\python\learning\Built-ins
Directories:  []
Files:  ['evalu.py', 'input.py', 'zipped.py']

Current Path:  G:\workspace\python\learning\lists
Directories:  []
Files:  ['list_01.py', 'tuple_01.py']

Current Path:  G:\workspace\python\learning\NumPy
Directories:  []
Files:  ['ceilr.py', 'concatenate.py', 'eye_identity.py', 'flatten.py', 'math.py', 'mean_var_std.py', 'min_max.py', 'rev_array.py', 'sum_prod.py', 'zeros_ones.py']

Current Path:  G:\workspace\python\learning\Strings
Directories:  []
Files:  ['formatting.py']
'''

# check if a file exist
print(os.path.isfile("G:\\workspace\\python\\learning\\Strings\\formatting.py"))
# prints 'True'

# check if a directory exist
print(os.path.exists("G:\\workspace\\python\\learning\\Strings"))
print(os.path.isdir("G:\\workspace\\python\\learning\\Strings"))
# both prints 'True'

# accessing environment variable
print(os.environ.get("HOME"))
# prints 'C:\\Users\\Gogul Ilango'
```

<h3 class="code-head" id="sys-module">sys module<span>code</span></h3>

```python
# sys module is used to parse input arguments given to a python file.
# this is used if you call a python script with arguments in command line.
import sys

firstarg  = ""
secondarg = ""
try:
    firstarg  = sys.argv[1]
    secondarg = sys.argv[2]
except:
    if (firstarg == ""):
        print("No first argument!")
    if (secondarg == ""):
        print("No second argument!")

# error text
sys.stderr.write("This is stderr text\n")
sys.stderr.flush()
sys.stdout.write("This is stdout text\n")
```

<h3 class="code-head" id="shutil-module">shutil module<span>code</span></h3>

```python
# shutil module is used for copying, moving, removing directory trees.
import shutil 

# copy file from one location to another
shutil.copyfile("/path/to/file_source", "path/to/file_destination")

# recursively copy entire directory tree from source to destination
shutil.copytree("source_dir", "destination_dir")

# recursively delete a directory tree
shutil.rmtree("/one/two/three")
```

<!-- --------------------------------------------------------------------- -->

<h3 class="centered-heading" id="oop"><span>OOP</span></h3>

Model real-world entities as software *objects* (having *class* as a blueprint) which have some data associated with them (*variables*) and can perform certain functions (*methods*).

Examples are
* A <span class="coding">person</span> class with
  * variables: <span class="coding">name</span>, <span class="coding">property</span>, <span class="coding">age</span>, <span class="coding">address</span>
  * methods: <span class="coding">walking</span>, <span class="coding">talking</span>, <span class="coding">running</span>, <span class="coding">swimming</span>
* An <span class="coding">email</span> class with
  * variables: <span class="coding">recipients</span>, <span class="coding">subject</span>, <span class="coding">body</span>
  * methods: <span class="coding">add attachment</span>, <span class="coding">send</span>, <span class="coding">discard</span>

<h3 class="code-head" id="classes">Classes<span>code</span></h3>

```python
# create a class
class Customer(object):

    # init method is a must
    def __init__(self, name, age):
        self.name = name
        self.age  = age

    # a simple print method
    def print_customer(self):
        print("Customer: {}, Age: {}".format(self.name, self.age))

# define an instance
a = Customer("Gogul", "24")
a.print_customer() # prints "Customer: Gogul, Age: 24"
```

<h3 class="code-head" id="class-variables">Class Variables<span>code</span></h3>

```python
class Customer(object):

    # this is a class variable
    raise_amount = 1.05
    num_of_custs = 0

    def __init__(self, name, age, pay):
        self.name = name
        self.age  = age
        self.pay  = pay

        Customer.num_of_custs += 1

    def apply_raise(self):
        print("Customer {} new pay is {}".format(self.name, float(self.pay) * self.raise_amount))

if __name__ == '__main__':
    # class variable not updated
    a = Customer("Gogul", "24", "5000")
    a.apply_raise() # prints Customer Gogul new pay is 5250.0

    # class variable updated
    b = Customer("Mahadevan", "25", "6000")
    b.raise_amount = 2.05
    b.apply_raise() # Customer Mahadevan new pay is 12299.999999999998

    # print dict of an instance
    print(a.__dict__) # prints {'name': 'Gogul', 'age': '24', 'pay': '5000'}

    # There are 2 customers
    print("There are {} customers".format(Customer.num_of_custs))
```

### Inheritance

It is the process by which one class takes on the attributes and methods of another class. This inherited class is called <span class="coding">child</span> class. The class from which <span class="coding">child</span> class was inherited is called <span class="coding">parent</span> class.

<div class="note">
<p><b>Note:</b> Child classes <span class="coding">override</span> or <span class="coding">extend</span> the attributes and behaviors of parent class i.e., child classes inherit all of the parent's attributes and behaviors but <i>can also specify different functionality</i> (attributes or behaviors) to follow.</p>
</div>

<h3 class="code-head" id="inheritance">Inheritance<span>code</span></h3>

```python
# parent class
class Avenger:
    # initializer with instance attributes
    def __init__(self, name):
        self.name = name

    # instance method
    def slogan():
        print("We are here to protect the universe!")

# Child class inherited from Avenger class
class Ironman(Avenger):
    def change_suit(self, suit):
        self.suit = suit

    def fly(self, speed):
        print("Sir! You are now flying at {} km/h with my favorite {} suit!".format(speed, self.suit))

# Child class inherited from Avenger class
class Thor(Avenger):
    def change_weapon(self, weapon):
        self.weapon = weapon

    def slam(self, message):
        print("{} now use {}. {}".format(self.name, self.weapon, message))

# first object
ironman = Ironman("Robert Downey Jr")
ironman.change_suit("MARK XLVII")
ironman.fly(300)
# prints "Sir! You are now flying at 300 km/h with my favorite MARK XLVII suit!"

# second object
thor = Thor("Chris Hemsworth")
thor.change_weapon("StormBreaker")
thor.slam("Bring me Thanos!!!!!!!!!!")
# print "Chris Hemsworth now use StormBreaker. Bring me Thanos!!!!!!!!!!"
```

<!-- --------------------------------------------------------------------- -->

<h3 class="centered-heading" id="how-tos"><span>How to's</span></h3>

<h3 class="code-head" id="how-to-handle-files">How to handle files?<span>code</span></h3>

```python
'''
four different methods (modes) to open a file
"r" - read; default mode; opens a file for reading, error if the file does not exist.
"a" - append; opens a file for appending, creates the file if it does not exist.
"w" - write; opens a file for writing, creates the file if it does not exist.
"x" - create; creates the file, returns an error if the file exist.

"t" - text; default/text mode
"b" - binary; binary mode
'''
```

<h3 class="code-head" id="how-to-read-file-line-by-line">How to read file line-by-line?<span>code</span></h3>

```python
# not the memory efficient way
filename = "entry_1.txt"
with open(filename) as f:
    data = f.readlines()
# remove whitespaces at end of each line
data = [x.strip() for x in data] 

# memory efficient way
filename = "entry_1.txt"
data = []
with open(filename) as f:
    for line in f:
        data.append(line)
# remove whitespaces at end of each line
data = [x.strip() for x in data]
```

<h3 class="code-head" id="how-to-write-file-line-by-line">How to write file line-by-line?<span>code</span></h3>

```python
l = ["pikachu", "charmander", "pidgeotto"]
fout = open("entry2.txt", "w")
for x in l:
    fout.write(x)
fout.close()
```

<h3 class="code-head" id="how-to-load-json-file">How to load json file?<span>code</span></h3>

```python
import json

file_input = "data.json"
with open(file_input) as data_file:    
    data = json.load(datafile)
```

<h3 class="code-head" id="how-to-check-if-list-is-empty">How to check if list is empty?<span>code</span></h3>

```python
# method 1 
if not myList:
    print("list is empty")

# method 2
if len(myList) == 0:
    print("list is empty")
```

<h3 class="code-head" id="how-to-access-index-in-for-loop">How to access index in for loop?<span>code</span></h3>

```python
myList = ["a", "b", "c"]

# method 1
for idx, l in enumerate(myList):
    print(str(idx) + "-" + l)

# method 2
idx = 0
for l in myList:
    print(str(idx) + "-" + l)
    idx += 1

# both methods print
# 0-a 
# 1-b
# 2-c
```

<h3 class="code-head" id="how-to-sort-a-dictionary-by-key-or-value-alphabetically">How to sort a dictionary by key or value alphabetically?<span>code</span></h3>

```python
a = {}
a["India"] = "Dhoni"
a["SouthAfrica"] = "ABD"
a["Australia"] = "Smith"

# sort dictionary based on 'key' alphabetically
for key in sorted(a.keys(), key=lambda x:x.lower()):
    print("{0} - {1}".format(key, a[key]))

# prints 
# Australia - Smith
# India - Dhoni
# SouthAfrica - ABD

# sort dictionary based on 'value' alphabetically
for key in sorted(a.keys(), key=lambda x:a[x]):
    print("{0} - {1}".format(key, a[key]))

# prints 
# SouthAfrica - ABD
# India - Dhoni
# Australia - Smith
```

<h3 class="code-head" id="how-to-call-tcl-procedure-in-python">How to call tcl procedure in python?<span>code</span></h3>

```tcl
# let's say you have a tcl file named 'test.tcl' with contents as below.
puts "Hello"
proc sum {a b} {
    set c [expr $a + $b]
    puts "Addition of $a and $b is $c"
}
```

```python
# to call a tcl proc in python, we need 'tkinter' library which comes with python usually.
import tkinter
r = tkinter.Tk()
r.tk.eval("source test.tcl")
r.tk.eval("sum 10 20")

# prints 
# Hello
# Addition of 10 and 20 is 30
```

<h3 class="centered-heading" id="resources"><span>Resources</span></h3>

* [Python Programming](https://gogul09.github.io/software/python-programming){:target="_blank"}