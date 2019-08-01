---
layout: post
category: software
class: Tools
title: How to install SASS in Linux?
description: Learn how to install SASS in linux so that you could use it to write neat and clean CSS.
author: Gogul Ilango
permalink: /software/sass-install-linux
image: https://drive.google.com/uc?id=1EgCue2ZX4VCVZzVrLLgUMeiBchtIVH9X
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#download-sass-from-github">1. Download SASS from GitHub</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#unzip-and-untar">2. Unzip and Untar</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#add-it-to-path">3. Add it to PATH</a></li>
  </ul>
</div>

[SASS](https://sass-lang.com/){:target="_blank"} is the best CSS preprocessor out there. Using a preprocessor like SASS makes your CSS journey more easier, efficient, readable, maintainable and organized. SASS has amazing capabilities which could be learnt quickly [here](https://sass-lang.com/guide){:target="_blank"} and [here](https://tutorialzine.com/2016/01/learn-sass-in-15-minutes){:target="_blank"}.

In this page, you will learn how to install SASS in Linux easily without using any package manager. In case if you wish to install in any other OS or using a package manager, you can visit [SASS official installation](https://sass-lang.com/install){:target="_blank"} instructions.

<h3 id="download-sass-from-github">1. Download SASS from GitHub</h3>

Head over to the [SASS GitHub page](https://github.com/sass/dart-sass/releases/tag/1.9.0){:target="_blank"} and download the <span class="coding">*.tar.gz</span> file based on your linux os version. To check your linux architecture, use the below command in your linux terminal.

<div class="code-head">Get Linux OS version<span>cmd</span></div>

```
uname -m
```

<h3 id="unzip-and-untar">2. Unzip and Untar</h3>

After downloading the <span class="coding">.tar.gz</span> file, unzip it and untar it. You will see a <span class="coding">dart-sass</span> directory.

<h3 id="add-it-to-path">3. Add it to PATH</h3>

Copy the path of the <span class="coding">dart-sass</span> directory and add it to <span class="coding">PATH</span>. To view <span class="coding">PATH</span> variables, you can use the below command.

<div class="code-head">View PATH variables<span>cmd</span></div>
```
echo $PATH
```

If you have <span class="coding">.cshrc</span> file in your OS, add the below line to your <span class="coding">.cshrc</span> file and source it.

<div class="code-head">Add to PATH<span>cmd</span></div>

```
setenv PATH "$PATH\:path_to_dart-sass"
```

Make sure you add the absolute path of <span class="coding">dart-sass</span> directory in the above command. Also, remember to include :\ between <span class="coding">$PATH</span> and <span class="coding">path_to_dart-sass</span>.

After adding and saving this line in <span class="coding">.cshrc</span> file, you can source it in your terminal. Now, SASS is installed and you are ready to use it!

<div class="note">
<p>
	<b>Note:</b> If you have <span class="coding">.bashrc</span> file in your linux environment, you can follow the guide <a href="https://katiek2.github.io/path-doc/" target="_blank">here</a> on adding <span class="coding">dart-sass</span> directory to the PATH.
</p>
</div>

<div class="code-head">Basic SASS usage<span>cmd</span></div>
```
sass input.scss output.css
```