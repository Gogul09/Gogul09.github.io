---
layout: post
category: software
class: Tools
title: How to install R and Shiny in Ubuntu?
description: This tutorial is for beginners who need to install R, RStudio and Shiny in Ubuntu.
author: Gogul Ilango
permalink: /software/how-to-install-r-shiny-in-windows
image: https://drive.google.com/uc?id=1e9sx0mm7b763Zr6uM_16FgqjYWGDPYG3
---

R is a great programming language to do Data Analysis and Statistical Operations on data. It has a easy-to-follow readable syntax (somewhat similar to Python) and provides rich source of libraries/packages to perform anything from Data Manipulation to Data Visualization. YOu can install these packages from CRAN (Comprehensive R Archive Network) which is a place where all the R packages/distributions exist. In this tutorial, we will learn how to install R, RStudio and Shiny (an R package to create interactive web apps) in Ubuntu 14.04.

### Installing R

Update the list of packages in the sources list

```python
sudo apt-get update
sudo apt-get upgrade
```

Install R using <span class="coding">apt-get</span>

```python
sudo apt-get install r-base
```

Check whether installation was successful.

```python
R
```

It must display something like this -

```python
R version 3.4.0 (2017-04-21) -- "You Stupid Darkness"
Copyright (C) 2017 The R Foundation for Statistical Computing
Platform: x86_64-pc-linux-gnu (64-bit)

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

```

Now, you are inside the R interactive shell and you can easily quit the shell using - 

```python
q(save="no")
```

After setting up R, you need to install packages (which are similar to libraries). The common format to install any R package from CRAN is -

```python
install.packages("the_name_of_your_package")
```

### Installing Shiny 

Let's install one of my favorite R packages, Shiny. Using this package, you can create an interactive webpages with zero HTML/CSS/Javascript knowledge.

```python
install.packages("shiny")
```

### Installing RStudio
R provides an elegant IDE to work with called RStudio. It is easy to work with the IDE rather than using the interactive shell. You can download RStudio [here](https://www.rstudio.com/products/rstudio/download2/). An example of how RStudio looks is shown below. <em>(source: http://rprogramming.net/wp-content/uploads/2012/10/RStudio-Screenshot.png)</em>

<img src="http://rprogramming.net/wp-content/uploads/2012/10/RStudio-Screenshot.png" alt="RStudio" />
