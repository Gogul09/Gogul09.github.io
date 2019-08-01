---
layout: post
category: software
class: Programming Languages
title: JavaScript Learning Notes
description: Understand the syntax and how to's of JavaScript programming language which is highly used in Front-end Web Development and Machine Learning.
author: Gogul Ilango
permalink: /software/javascript-learning-notes
image: https://drive.google.com/uc?id=1sL3r7BTCF3V2IHj5ftlB1cfX9MTZMoJX
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#console">Console</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#variables">Variables</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#math-operations">Math Operations</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#strings">Strings</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#conditionals">Conditionals</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#loops">Loops</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#arrays">Arrays</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#objects">Objects</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#functions">Functions</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#try-catch">Try Catch</a></li>
  </ul>
</div>

In this page, you will find the commonly used **syntax** and **how to's** that are specific to JavaScript programming language. If you know what JavaScript (JS) is, but wanted to quickly refer to JS syntax, then you might find this page useful. I will keep on updating this article. So, kindly bookmark this page if you find it useful 😉

<h3 id="console">Console</h3>

For any type of JavaScript developer, <span class="coding">console</span> is the thing that makes them alive 😅. So, in this page, you will find <span class="coding">console.log()</span> mostly to display the output of a specific operation. 

If you're new to <span class="coding">console</span> in JavaScript, please open a browser (such as Google Chrome), right-click, inspect and click on <span class="coding">console</span> tab. From here, you can start executing the below code.

<h3 id="variables" class="code-head">Variables<span>code</span></h3>

```javascript
// define a variable
var name = prompt("What's your name?");

if(name != null) {
  console.log(name);
} else {
  alert("Please enter a valid name!");
}
```

<h3 id="math-operations" class="code-head">Math Operations<span>code</span></h3>

```javascript
// add
console.log("5 + 4 = ", 5 + 4); // prints 5 + 4 = 9

// sub
console.log("5 - 4 = ", 5 - 4); // prints 5 - 4 = 1

// mul
console.log("5 * 4 = ", 5 * 4); // prints 5 * 4 = 20

// div 
console.log("5 / 4 = ", 5 / 4); // prints 5 / 4 = 1.25

// mod
console.log("5 % 4 = ", 5 % 4); // prints 5 % 4 = 1

// max number
console.log(Number.MAX_VALUE); // prints 1.7976931348623157e+308

// min number
console.log(Number.MIN_VALUE); // prints 5e-324

// round off
a = 2.2555;
console.log(a.toFixed(2)); // prints 2.26

// pre increment
b = 1;
console.log(++b); // prints 2

// post increment
c = 1;
console.log(c++); // prints 1
console.log(c); // prints 2

// math functions
console.log(Math.abs(-8)); // prints 8
console.log(Math.cbrt(1000)); // prints 10
console.log(Math.ceil(6.45)); // prints 7
console.log(Math.floor(6.45)); // prints 6
console.log(Math.log(10)); // prints 2.302585092994046
console.log(Math.log10(10)); // prints 1
console.log(Math.max(10, 5)); // prints 10
console.log(Math.min(10, 5)); // prints 5
console.log(Math.pow(4, 2)); // prints 16
console.log(Math.sqrt(1000)); // prints 31.622776601683793
console.log(Math.floor(Math.random() * 10)) // prints 2

// conversions
a = "3.14"; // note this is a string
console.log(parseInt(a)); // prints 3
console.log(parseFloat(a)); // prints 3.14
```

<h3 id="strings" class="code-head">Strings<span>code</span></h3>

```javascript
var str = "This is a classic example for strings";

// length of string
console.log(str.length); // prints 37

// index of a substring
console.log(str.indexOf("example")); // prints 18

// slice a string based on start index and end index
console.log(str.slice(10, 17)); // prints classic

// slice a string based on start index
console.log(str.slice(10)); // prints classic example for strings

// substring using start index and length
console.log(str.substr(10, 7)); // prints classic

// substring replace
console.log(str.replace("classic", "awesome")) // prints This is a awesome example for strings

// get character at an index
console.log(str.charAt(10)); // prints c

// split string into an array 
console.log(str.split(" ")); // prints ["This", "is", "a", "classic", "example", "for", "strings"]

// upper case and lower case
console.log(str.toUpperCase()); // prints THIS IS A CLASSIC EXAMPLE FOR STRINGS
console.log(str.toLowerCase()); // prints this is a classic example for strings
``` 

<h3 id="conditionals" class="code-head">Conditionals<span>code</span></h3>

```javascript
// relational operators : === != > < >= <=
// logical operators    : && || !


// normal conditionals
var num = 2;
if (num < 4) {
  console.log("Pikachu!"); // prints Pikachu! 
} else if (num >= 4 && num < 10) {
  console.log("Charmander!");
} else {
  console.log("Pokemon!");
}

// switch statement
var age = 25;
switch(age) {
  case 21: 
      console.log("Awesome!");
  case 22: 
      console.log("Amazing!");
  case 23: 
      console.log("Astounding!");
  case 24: 
      console.log("Fantastic!");
  case 25: 
      console.log("Brilliant!");
  default:
      console.log("State of mind :)");
}
// prints Brilliant! State of mind :)

// ternary operator
var canVote = (age >= 18) ? true : false;
console.log(canVote); // prints true

```

<h3 id="loops" class="code-head">Loops<span>code</span></h3>

```javascript
//------------
// while loop
//------------
var i = 1;
while(i <= 5) {
  console.log("i = " + i);
  i += 1;
}
// prints 
// i = 1
// i = 2
// i = 3
// i = 4
// i = 5

//---------------
// do while loop
//---------------
var j = 5;
do {
  console.log("came here once!");
} while(i == 10);
// prints came here once!

//----------
// for loop
//----------
for(var k=0; k<4; k++) {
  console.log("k = " + k);
}
// prints
// k = 0
// k = 1
// k = 2
// k = 3

//-----------------------------------------------
// for loop - enumerable properties of an object
//-----------------------------------------------
var team = {captain: "dhoni", batsman: "kohli", bowler: "nehra"};
for(t in team) {
  console.log(t + " : " + team[t]);
}

// prints
// captain : dhoni
// batsman : kohli
// bowler  : nehra
```

<h3 id="arrays" class="code-head">Arrays<span>code</span></h3>

```javascript
var arr = ["pikachu", "charmander", "bulbasaur", "squirtle"];

// accessing array items
console.log(arr[0]) // prints pikachu

// delete the 2nd item
console.log(arr.splice(2, 1)); // prints ["bulbasaur"]
console.log(arr); // prints ["pikachu", "charmander", "squirtle"]

// array to string
console.log(arr.toString()); // prints "pikachu,charmander,squirtle"

// array to string using join
console.log(arr.join(", ")) // prints "pikachu, charmander, squirtle"

// sort ascending
console.log(arr.sort()); // prints ["charmander", "pikachu", "squirtle"]

// sort descending
console.log(arr.reverse()); // prints ["squirtle", "pikachu", "charmander"]

// add item to the end
arr.push("mewtwo");
console.log(arr); // prints ["squirtle", "pikachu", "charmander", "mewtwo"]

// remove item from the end
arr.pop();
console.log(arr); // prints ["squirtle", "pikachu", "charmander"]
```

<h3 id="objects" class="code-head">Objects<span>code</span></h3>

```javascript

// object similar to dictionary in python
var team = {1: "gogul", 2: "deepan", 3: "mohan", 4: "sibi", 5: "sathiesh"}

// accessing keys and values
for(t in team) {
  console.log(t + " : " + team[t]);
}

// prints
// 1 : gogul
// 2 : deepan
// 3 : mohan
// 4 : sibi
// 5 : sathiesh

// get the keys in object
console.log(Object.keys(team)); 
// prints ["1", "2", "3", "4", "5"]

// get the values in object
console.log(Object.values(team)); 
// ["gogul", "deepan", "mohan", "sibi", "sathiesh"]

// accessing values using keys (this is not index)
console.log(team["2"]); // prints "deepan"

// object of object
var globalTeam = {main: "Not yet filled", sub: team};
console.log(globalTeam["main"]); // prints "Not yet filled"
console.log(globalTeam["sub"]);  // prints {1: "gogul", 2: "deepan", 3: "mohan", 4: "sibi", 5: "sathiesh"}
```

<h3 id="functions" class="code-head">Functions<span>code</span></h3>

```javascript
function multiply(a, b) {
  return (a*b);
}
console.log(multiply(5,4)); // prints 20
```

<h3 id="try-catch" class="code-head">Try Catch<span>code</span></h3>

```javascript
var a = 10; 
try {
  console.log("Hello World" + a + b);
} catch(err) {
  console.log(err.message); // prints "b is not defined"
}
```

<br>