---
layout: post
category: software
class: Programming Languages
title: Tcl Learning Notes
description: Understand the syntax and how to's of Tcl programming language which is highly used in VLSI and Networking companies.
author: Gogul Ilango
permalink: /software/tcl-learning-notes
image: https://drive.google.com/uc?id=1cKgd7bFYwAyFIznfmvj6uNLbd2W8U0G7
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#standard-io">Standard IO</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#math-operations">Math operations</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#looping-statements">Looping statements</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#conditional-statements">Conditional statements</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#logical-operations">Logical operations</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#switch">Switch</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#lists">Lists</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#dictionaries">Dictionaries</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#strings">Strings</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#regular-expressions">Regular expressions</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_11" href="#procs">Procs</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_12" href="#file-handling">File handling</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_13" href="#utils">Utils</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_14" href="#others">Others</a></li>
  </ul>
</div>

In this page, you will find the commonly used **syntax** and **how to's** that is specific to Tcl programming language. If you know what Tcl is, but wanted to quickly refer to Tcl syntax, then you might find this page useful. I will keep on updating this article. So, kindly bookmark this page if you find it useful :)

<h3 id="standard-io" class="code-head">Standard IO<span>code</span></h3>

```tcl
# get user input
set a [gets stdin]

# print output to screen
puts $a
```

<h3 id="math-operations" class="code-head">Math operations<span>code</span></h3>

```tcl
# basic math operations
set add [expr $a + $b]
set sub [expr $a - $b]
set mul [expr $a * $b]
set div [expr $a / $b]
set mod [expr $a % $b]

# some more math operations
puts "[expr sqrt(4)]"
puts "[expr sin(4)]"
puts "[expr cos(4)]"
puts "[expr tan(4)]"
puts "[expr sinh(4)]"
puts "[expr cosh(4)]"
puts "[expr tanh(4)]"
puts "[expr log(1)]"
puts "[expr ceil(4.6)]"
puts "[expr floor(4.6)]"
puts "[expr srand(4.6)]"
```

<h3 id="looping-statements" class="code-head">Looping statements<span>code</span></h3>

```tcl
# for loop
for {set i 0} {$i < 50} {incr i} {
	puts $i
}

# while loop
set i 0
while {$i < 50} {
	puts $i
	incr i
} 
```

<h3 id="conditional-statements" class="code-head">Conditional statements<span>code</span></h3>

```tcl
set i 50 
if {$i > 100} {
	puts "Ferrari"
} elseif {$i > 50} {
	puts "Mclaren"
} else {
	puts "Pikachu"
}
```

<h3 id="logical-operations" class="code-head">Logical operations<span>code</span></h3>

```tcl
set i "Batman"
set j 2

# logical AND & Logical OR
if {$i == "Ironman" && $j == 1} {
	puts "I love Ironman"
} elseif {$i == "Batman" || $j == 2} {
	puts "I love Batman"
}

# boolean
set a true
if $a {
	puts "Yes! It is true"
}
```

<h3 id="switch" class="code-head">Switch<span>code</span></h3>

```tcl
set x 30

# the best syntax to use "switch" in Tcl
# although there are multiple ways
# such as writing in a single-line
switch $x {
  "10" {puts "this is 10"}
  "20" {puts "this is 20"}
  "default" {puts "this is default"}
}

# prints "this is default"
```

<h3 id="lists" class="code-head">Lists<span>code</span></h3>

```tcl
# Three ways to create a list
set l {"Dhoni" "Kohli" "Dravid"}
set l [list Dhoni Kohli Dravid]
set l [split "Dhoni Kohli Dravid"]

# length of a list
puts "[llength $l]"

# get the item at a specific index
puts "[lindex $l 2]"

# retrieve several items within a range
puts "[lrange $l 0 2]"

# concat two lists
set l2 {Ganguly Sehwag}
set l [concat $l $l2]

# append an item to list
lappend l Rohit

# insert an item to a particular index
set l [linsert $l 2 "Dhawan"]

# replacing an item in the list 
# Note that we need to give here a range
set l [lreplace $l 2 3 "Laxman"]

# search for an item in the list
# returns index of the item if there is match
# else returns -1
set check [lsearch $l "Dhoni"]

# sorting a list (increasing order)
puts "[lsort $l]"

# sorting a list (decreasing order)
puts "[lsort -decreasing $l]"
```

<h3 id="dictionaries" class="code-head">Dictionaries<span>code</span></h3>

```tcl
# create a dictionary with key-value pair
set d [dict create 1 Dhoni 2 Kohli 3 Rohit]

# get the unique keys in the dictionary
set keys [dict keys $d]

# loop over each key to get its corresponding value
foreach key $keys {
	# get the value of a key
	set value [dict get $d $key]
	puts "$key -- $value"
}

# prints the size of dictionary
puts "[dict size $d]"

# prints the values in the dictionary
puts "[dict values $d]"
```

<h3 id="strings" class="code-head">Strings<span>code</span></h3>

```tcl
set s "/home/gogul/software/sublime/packages"

# get the length of the string
puts "[string length $s]"

# get the character at a specified index in the string
puts "[string index $s 2]"

# get the characters within a range in the string (sub-string)
puts "[string range $s 0 10]"

# split the string and make it a list 
# observe the delimiter here
set l [split $s /]

# compare two strings (not numerically)
# -1 --> if string1 is less than string2
# 0  --> if string1 is equal to string2
# 1  --> if string1 is greater than string2
set s1 "Dhoni"
set s2 "Kohli"
puts "[string compare $s1 $s2]"

# match a pattern in the string
puts "[string match "*gogul*" $s]"

# convert all letters to upper-case
puts "[string toupper $s]"

# convert all letters to lower-case
puts "[string tolower $s]"

# convert all letters to title 
# first letter upper-case followed by lower-case
puts "[string totitle $s]"
```

<h3 id="regular-expressions" class="code-head">Regular expressions<span>code</span></h3>

```tcl
set s "This is a string with a number 123456789"

# regex to match a number
set result [regexp {\d.*\d} $s match]

# returns 1 if there is a match
puts $result

# returns the matched string
puts $match

# search for a pattern and if present,
# substitute it with a string
# and return the result
set s "Dhoni is an awesome player in cricket."

regsub Dhoni $s Kohli result

# prints Kohli is an awesome player in cricket.
puts $result
```

<h3 id="procs" class="code-head">Procs<span>code</span></h3>

```tcl
# a simple proc with two arguments and a return statement
proc addition {a b} {
	set c [expr $a + $b]
	return $c
}

puts "[addition 5 4]"

# proc with multiple arguments
# "args" as the third argument holds all additional values supplied
proc more_args {a b args} {
	puts $a 
	puts $b
	puts $args
}

# prints 
# 5
# 6
# 90 60 10
more_args 5 6 90 60 10
```

<h3 id="file-handling" class="code-head">File handling<span>code</span></h3>

```tcl
# opens a file to write
set fp [open "sample_text_file" w]

# writes data to the file
puts $fp "Hello World"

# close the file after writing (else data won't be saved)
close $fp


# open a file to read
set fp [open "sample_text_file" r]

# read the content of the file
set fp_data [read $fp]


# split by newline
set lines [split $fp_data "\n"]

# loop through each line in the file
foreach line $lines {
	# do some processing
}
```

<h3 id="utils" class="code-head">Utils<span>code</span></h3>

```tcl
# execute shell commands inside Tcl
# prints the present working directory
exec pwd

# prints the current datetime
exec date

# creates a new directory in the present working directory
exec mkdir new_folder

# save the pwd in a variable 
set p [pwd]

# change directory
cd another_folder
```

<h3 id="others" class="code-head">Others<span>code</span></h3>

```tcl
# execute a file within a file
# so that procs/variables used in another file
# are available in the current file
source filename
```
<br>