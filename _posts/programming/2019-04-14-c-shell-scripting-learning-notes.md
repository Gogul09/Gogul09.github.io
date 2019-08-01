---
layout: post-doc
category: software
class: Programming Languages
title: C Shell Scripting Learning Notes
description: Learn the syntax and how to's of C Shell Scripting which is used to automate repeated manual tasks in most of the industries.
author: Gogul Ilango
permalink: /software/c-shell-scripting-learning-notes
---

<div id="awesomeSideNav" class="sidenav">
    <a href="javascript:void(0)" class="closebtn" onclick="closeSideNav()">&times;</a>
    <h3><a href="#basics">Basics</a></h3>
    <ul>
        <li><a href="#expressions">Expressions</a></li>
        <li><a href="#control-structures">Control Structures</a></li>
        <li><a href="#loops">Loops</a></li>
        <li><a href="#break">break</a></li>
        <li><a href="#continue">continue</a></li>
        <li><a href="#goto">goto</a></li>
        <li><a href="#switch">switch</a></li>
        <li><a href="#interrupt-handling">Interrupt Handling</a></li>
    </ul>
    <h3><a href="#how-tos">How to's</a></h3>
    <ul>
        <li><a href="#how-to-store-the-output-of-a-command">How to store the output of a command?</a></li>
        <li><a href="#how-to-read-user-input">How to read user input?</a></li>
        <li><a href="#how-to-use-single-and-double-quotes">How to use single and double quotes?</a></li>
    </ul>
</div>

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#basics">Basics</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#how-tos">How To's</a></li>
  </ul>
</div>

In this page, you will find the **syntax** and **most common how to's** of C Shell Scripting. This might be useful for beginners in programming or professionals in automation industries such as VLSI or Telecomm to find <span class="coding">csh</span> syntax and code quickly using this single page online reference material. Click on any of the contents in the sidebar to view the code.

<h3 class="centered-heading" id="basics"><span>Basics</span></h3>

<h3 class="code-head" id="expressions">Expressions<span>code</span></h3>

```shell
#!/bin/csh

# declaring variables
set a = 2
set b = 6

# math operations (white space must taken care)
set c = `expr $a+$b`
echo $c # prints 2+6

# add
set c = `expr $a + $b`
echo $c # prints 8

# subtract
set c = `expr $b - $a`
echo $c # prints 4

# divide
set c = `expr $b / $a`
echo $c # prints 12

# remainder
set c = `expr $b % $a`
echo $c # prints 12
```

<h3 class="code-head" id="control-structures">Control Structures<span>code</span></h3>

```shell
# check if file exist
if (-e filename) echo "File exist"

#-------------------------
# file status expressions
#-------------------------
# d - file is a directory
# e - file exists
# f - file is an ordinary file
# o - user owns the file
# r - user has read access to the file
# w - user has write access to the file
# x - user has execute access to the file
# z - file is zero bytes long

# if/then/else
set a = 5
set b = 10

if ($a > $b) then
    echo "a is greater than b"
else if ($a < $b) then
    echo "a is lesser than b"
else
    echo "a is equal to b"
endif

# prints "a is lesser than b"
```

<h3 class="code-head" id="loops">loops<span>code</span></h3>

```shell
#----------------
# foreach loop
#----------------
# declare a word list
set colors = "white red black green blue"

# iterate over the word list (paranthesis is important)
foreach c ($colors)
    echo $c
end

#----------------
# while loop
#----------------
set a = 0
while ($a < 4)
    echo "a is $a"
    @ a++
end

# prints
# a is 0
# a is 1
# a is 2
# a is 3
```

<h3 class="code-head" id="break">break<span>code</span></h3>

```shell
set a = 0
while (1 > 0)
    echo "a is $a"
    @ a++
    if ($a == 4) break
end
echo "breaked from while loop after a is $a"

# prints
# a is 1
# a is 2
# a is 3
# breaked from while loop after a is 4
```

<h3 class="code-head" id="continue">continue<span>code</span></h3>

```shell
set colors = "white red black green blue"
foreach c ($colors)
    if ($c == black) then
        echo "as color is $c, continuing.."
        continue
    endif
    echo "color is $c"
end

# prints
# color is white
# color is red
# as color is black, continuing..
# color is green
# color is blue
```

<h3 class="code-head" id="goto">goto<span>code</span></h3>

```shell
set a = 20

if ($a == 10) then
    goto csk
else if ($a == 20) then
    goto rcb
else
    goto kkr

csk:
    echo "this is csk"
    exit 1

rcb:
    echo "this is rcb"
    exit 1

kkr:
    echo "this is kkr"
    exit 1

# prints 
# this is rcb
```

<h3 class="code-head" id="switch">switch<span>code</span></h3>

```shell
if ($#argv == 0) then
    echo "No arguments provided.."
    exit 1
else
    switch($argv[1])
    case [yY][eE][sS]:
        echo "Input is YES"
        breaksw
    case [nN][oO]:
        echo "Input is NO"
    default:
        echo "Input is not YES/NO"
        breaksw
    endsw
endif

# source test.csh yes
# print "Input is YES"
```

<h3 class="code-head" id="interrupt-handling">Interrupt Handling<span>code</span></h3>

```shell
# used to transfer control to onintr statement once CTRL+C is done to kill the script.
onintr close
while (1 > 0)
    echo "Loading avengers...."
    sleep 2
end

close:
echo "Avengers killed.."
echo "Yet Ironman is safe!"
```

<h3 class="centered-heading" id="how-tos"><span>How to's</span></h3>

<h3 class="code-head" id="how-to-store-the-output-of-a-command">How to store the output of a command?<span>code</span></h3>

```shell
# backquotes is used to store command's output to a variable for further processing
set a = `pwd`
echo $a # prints /usr2/gilango/study

set b = `date`

echo $b    # Sun Apr 14 13:45:25 IST 2019
echo $b[1] # Sun 

foreach f ($b)
    echo $f
end

# prints
# Sun
# Apr
# 14
# 13:45:25
# IST
# 2019
```

<h3 class="code-head" id="how-to-read-user-input">How to read user input?<span>code</span></h3>

```shell
# two ways to read user input


# method 1: set a = $<
echo -n "Input a value for a: "
set a = $<
echo "You entered a as: $a"

# prints 
# Input a value for a: 9
# You entered a as: 9

# method 2: set a = `head -1`
echo -n "Input a value for a: "
set a = `head -1`
echo "You entered a as: $a"

# prints 
# Input a value for a: 21
# You entered a as: 21
```

<h3 class="code-head" id="how-to-use-single-and-double-quotes">How to use single and double quotes?<span>code</span></h3>

```shell
# single quotes
#   * allow inclusion of spaces
#   * prevent variable substitution
#   * allow filename generation

# double quotes
#   * allow inclusion of spaces
#   * allow variable substitution
#   * allow filename generation

set a = -l
echo 'ls $a' # prints ls $a
echo "ls $a" # prints ls -l
```