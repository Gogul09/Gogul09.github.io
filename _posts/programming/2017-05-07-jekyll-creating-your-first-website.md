---
layout: post
category: software
class: Tools
title: How to create a website in 15 minutes using Jekyll?
description: Learn how to create your first website using Jekyll and host it for free using GitHub pages.
author: Gogul Ilango
permalink: /software/jekyll-create-your-first-website
image: https://drive.google.com/uc?id=1ws8RxgmdhQ8Gn_CYviaSQQY9wx6XeI12
---

<p class="intro-para"><a href="https://jekyllrb.com/" target="_blank">Jekyll</a> is a static site generator used to design and develop static website that is simple, secure, flexible and powerful.</p>

Static sites offer tons of advantages than dynamic websites. Static site generators such as Jekyll offers way more features than WordPress. Some of the cool advantages are 
* **Simple**: There is no need of server-side programming. No backend. No server. Basic knowledge on front-end components such as HTML, CSS and JavaScript is enough to build a website.
* **Security**: As there is no server or database interaction involved, there is no possibility of attacks. In fact, you can see the entire code of your website rendered through Jekyll. 
* **Flexible**: As you create your own website, you will have full control over it. Unlike Wordpress which has limited customization options, Jekyll offers full customization. You can add unlimited CSS classes or JavaScript files and even use a CSS preprocessor such as Sass which is highly helpful!
* **Powerful**: Your website will behave much faster as there is no server interaction involved.
* **Hosting**: After creating your Jekyll website, [GitHub pages](https://pages.github.com/){:target="_blank"} offer free hosting of Jekyll sites using GitHub. You just need a GitHub account and a repository to make your site live!

In this tutorial, we will create our first awesome website using Jekyll, host it in GitHub for free and get our free domain name similar to **https://gogul09.github.io** in 15 minutes.

### Install Ruby in Windows
First, you need to install Ruby which is available [here](https://rubyinstaller.org/){:target="_blank"} for Windows. Simply download the installer and complete the setup. After that, open up a command prompt and type <span class="coding">gem -v</span>. You should get something like this.

<div class="code-head"><span>cmd</span> gem -v</div>

```python
2.5.2
```

Also, you can type <span class="coding">ruby -v</span> and should get something like this to proceed.

<div class="code-head"><span>cmd</span> ruby -v</div>

```python
ruby 2.3.3p222 (2016-11-21 revision 56859) [i386-mingw32]
```

Now, install bundler which provides a consistent environment for Ruby projects by tracking and installing the exact gems and versions that are needed.

<div class="code-head"><span>cmd</span></div>

```python
gem install bundler
```

### Install Github pages gem
After installing ruby, you need to install a gem called <span class="coding">github-pages</span> that includes all the dependencies and jekyll too.

<div class="code-head"><span>cmd</span></div>

```python
gem install github-pages
```

Verify you have installed Jekyll by typing <span class="coding">jekyll -v</span> in the command prompt.

<div class="code-head"><span>cmd</span> jekyll -v</div>

```python
jekyll 3.4.3
```

That's it! You have installed all the dependencies to create your first awesome website.

### Creating a new jekyll site
Before creating your first website, you need to follow a folder structure to keep everything organized. In case if you wish to create multiple sites, proper folder structure reduces so much stress in future while maintaining it. I follow the one below inside which all my websites are kept inside its own folder.

<div class="code-head"><span>path</span> folder structure</div>

```python
g:\workspace\web

g:\workspace\web (root-folder)
  -- first_awesome_website (sub-folder)
  -- second_cool_website (sub-folder)
  -- third_amazing_website (sub-folder)
```

To create your first website, open up a command prompt, get into the above folder structure and type in the first Jekyll command <span class="coding">jekyll new site_name</span>. For this tutorial, I will keep the name of the website as **first_awesome_website**. You can keep any name as you like.

<div class="code-head"><span>cmd</span> jekyll new first_awesome_website</div>

```
Running bundle install in g:/workspace/web/first_awesome_website...
Bundler: Fetching gem metadata from https://rubygems.org/...........
Bundler: Fetching version metadata from https://rubygems.org/..
Bundler: Fetching dependency metadata from https://rubygems.org/.
Bundler: Resolving dependencies...
Bundler: Installing public_suffix 3.0.2
Bundler: Using colorator 1.1.0
Bundler: Installing ffi 1.9.25 (x86-mingw32)
Bundler: Using forwardable-extended 2.6.0
Bundler: Installing rb-fsevent 0.10.3
Bundler: Installing ruby_dep 1.5.0
Bundler: Installing kramdown 1.17.0
Bundler: Using liquid 3.0.6
Bundler: Using mercenary 0.3.6
Bundler: Using rouge 1.11.1
Bundler: Using safe_yaml 1.0.4
Bundler: Installing thread_safe 0.3.6
Bundler: Using bundler 1.14.0
Bundler: Installing addressable 2.5.2
Bundler: Installing rb-inotify 0.9.10
Bundler: Installing pathutil 0.16.1
Bundler: Installing tzinfo 1.2.5
Bundler: Installing sass-listen 4.0.0
Bundler: Installing listen 3.1.5
Bundler: Installing tzinfo-data 1.2018.5
Bundler: Installing sass 3.5.6
Bundler: Installing jekyll-watch 1.5.1
Bundler: Installing jekyll-sass-converter 1.5.2
Bundler: Using jekyll 3.4.3
Bundler: Installing jekyll-feed 0.10.0
Bundler: Using minima 2.1.1
Bundler: Bundle complete! 4 Gemfile dependencies, 26 gems now installed.
Bundler: Use `bundle show [gemname]` to see where a bundled gem is installed.
New jekyll site installed in g:/workspace/web/first_awesome_website.
```

After executing the above command, Jekyll creates your website with default settings. You can go into the below path to check the contents that Jekyll has created.

<div class="code-head"><span>path</span>g:\workspace\web\first_awesome_website</div>

```
_posts
_config.yml
about.md
Gemfile
Gemfile.lock
index.md
```

We will look into all the above contents in a while. But before that, let's see your website locally.

Get into the folder that we created **first_awesome_website** and execute the following two commands.

<div class="code-head"><span>cmd</span>generate & run locally</div>

```
jekyll build
jekyll serve
```

Above two commands should return results similar to the one shown below.

<div class="code-head"><span>output</span>jekyll build</div>

```
WARN: Unresolved specs during Gem::Specification.reset:
      rb-fsevent (>= 0.9.4, ~> 0.9)
      rb-inotify (>= 0.9.7, ~> 0.9)
WARN: Clearing out unresolved specs.
Please report a bug if this causes problems.
Configuration file: g:/workspace/web/first_awesome_website/_config.yml
            Source: g:/workspace/web/first_awesome_website
       Destination: g:/workspace/web/first_awesome_website/_site
 Incremental build: disabled. Enable with --incremental
      Generating...
                    done in 0.588 seconds.
 Auto-regeneration: disabled. Use --watch to enable.
```

<div class="code-head"><span>output</span>jekyll serve</div>

```
WARN: Unresolved specs during Gem::Specification.reset:
      rb-fsevent (>= 0.9.4, ~> 0.9)
      rb-inotify (>= 0.9.7, ~> 0.9)
WARN: Clearing out unresolved specs.
Please report a bug if this causes problems.
Configuration file: g:/workspace/web/first_awesome_website/_config.yml
Configuration file: g:/workspace/web/first_awesome_website/_config.yml
            Source: g:/workspace/web/first_awesome_website
       Destination: g:/workspace/web/first_awesome_website/_site
 Incremental build: disabled. Enable with --incremental
      Generating...
                    done in 0.378 seconds.
  Please add the following to your Gemfile to avoid polling for changes:
    gem 'wdm', '>= 0.1.0' if Gem.win_platform?
 Auto-regeneration: enabled for 'g:/workspace/web/first_awesome_website'
Configuration file: g:/workspace/web/first_awesome_website/_config.yml
    Server address: http://127.0.0.1:4000/
  Server running... press ctrl-c to stop.
```

Now, you can visit our website locally at **http://localhost:4000/**. You should get something similar to the one shown below.

<figure>
  <img src="/images/software/jekyll-create-your-first-website/demo_1.png" class="typical-image">
  <figcaption>Figure 1. Our first awesome Jekyll website.</figcaption>
</figure>

Awesome! You have made your own website using Jekyll in minutes. To quit the running procees, you can press <span class="coding">CTRL + C</span> in the command prompt. To view the site again locally, just type the command <span class="coding">jekyll serve</span> and view it in the browser at the same address **http://localhost:4000/**. 

Keep in mind that you haven't yet hosted your website. Hosting the website means you need a dedicated space in the internet for people to view your website if given a URL. [GitHub Pages](https://pages.github.com/){:target="_blank"} provides free web hosting for sites made from Jekyll. We will discuss it soon.

Running a jekyll site locally provides the following advantages. 
* You can tweak/edit the site multiple times before making changes to the live website.
* You can use your favorite text editor to write your posts in Markdown and view the changes instantly in the browser locally. 
* Only after making all the necessary changes, you can push it to GitHub and see it live.

### Basic files & folders

Now, we will look into all the default files and folders created by Jekyll inside your master folder. 
* **_posts**: This is the folder inside which all your blog posts will be kept. Whatever you write in this folder will be converted to html and corresponding styles will be applied automatically. All files must strictly follow the naming convention for proper parsing and display.

<div class="code-head"><span>rule</span>default naming convention for a blog post</div>
```
yyyy-mm-dd-<post_name>.md
```

* **_site**: This is the folder generated by Jekyll which will be used for distributing your website. All content inside this folder is what makes your site live. This folder gets updated automatically as you make changes. Do not update or delete or add anything inside this folder.
* **config.yml**: This is the YAML configuration file or shortly, the settings file for your site. It looks like the one shown below.

<div class="code-head"><span>file</span>config.yml</div>

```yaml
# Welcome to Jekyll!
#
# This config file is meant for settings that affect your whole blog, values
# which you are expected to set up once and rarely edit after that. If you find
# yourself editing this file very often, consider using Jekyll's data files
# feature for the data you need to update frequently.
#
# For technical reasons, this file is *NOT* reloaded automatically when you use
# 'bundle exec jekyll serve'. If you change this file, please restart the server process.

# Site settings
# These are used to personalize your new site. If you look in the HTML files,
# you will see them accessed via {{ site.title }}, {{ site.email }}, and so on.
# You can create any custom variable you would like, and they will be accessible
# in the templates via {{ site.myvariable }}.
title: Your awesome title
email: your-email@domain.com
description: > # this means to ignore newlines until "baseurl:"
  Write an awesome description for your new site here. You can edit this
  line in _config.yml. It will appear in your document head meta (for
  Google search results) and in your feed.xml site description.
baseurl: "" # the subpath of your site, e.g. /blog
url: "" # the base hostname & protocol for your site, e.g. http://example.com
twitter_username: jekyllrb
github_username:  jekyll

# Build settings
markdown: kramdown
theme: minima
gems:
  - jekyll-feed
exclude:
  - Gemfile
  - Gemfile.lock

```

This is basically a key-value file which is used to store important configurations for your website. Any changes made to this file requires terminating your local server process and restarting it. You can now modify the <span class="coding">keys</span> present in this file with your own <span class="coding">values</span>.

### Hosting in GitHub Pages

Before jumping into the concepts of Jekyll, let's first host your website in [GitHub pages](https://pages.github.com/){:target="_blank"} for free so that you will get your unique URL with which you can see your website live in the internet.

To host a Jekyll site in GitHub, you need a [GitHub account](https://services.github.com/on-demand/intro-to-github/create-github-account){:target="_blank"}. Make sure you follow that link to create a github account.

Now, you need to create a repository that acts as a container or master folder for your website. The name of your repository must be unique and this name is what you will be using to generate your unique GitHub URL. GitHub offers a free domain name for your website which comes like 

<div class="code-head"><span>link</span>GitHub pages link format</div>

```python
https://yourusername.github.io
```

Follow [this](https://help.github.com/articles/create-a-repo/){:target="_blank"} link to create a repository with the **exact name** as your **GitHub account name** with **.github.io** appended. 

> For example, my github account name is **Gogul09** and so, I created a repository with the name **Gogul09.github.io**.

Execute the following git commands one by one to take your website to the GitHub repository that you created. Make sure you are still in the master folder **first_awesome_website** that you created in the command prompt at the beginning of this post and then execute these commands.

<div class="code-head"><span>cmd</span>git commands</div>

```python
git init
git add . 
git commit -m "first commit"
git remote add origin https://github.com/yourusername/yourrepo.git
git push origin master -f
```

After 5-10 minutes, you will be seeing your website using the below link. 

<div class="code-head"><span>link</span></div>

```python
https://yourusername.github.io
```

Congratulations! You have created your first website using Jekyll and hosted it for free using GitHub pages.

### Powerful Sass Integration

If you don't know the capabilities of a CSS preprocessor such as [Sass](https://sass-lang.com/){:target="_blank"}, I highly encourage you to learn it. Using sass, you can make your CSS more structured, elegant and easier to modify. 

When you develop websites, there comes a time when your single CSS file goes out of hands and its difficult to update a CSS class or modify a layout. In such situations, Sass is highly useful.

Jekyll has powerful integration for Sass. In fact, Jekyll comes with its own Sass compiler. Here are the steps to make use of Sass for your website.
* Create a folder named <span class="coding">_sass</span> in the root directory which is **first_awesome_website** for this tutorial.
* Create a file named <span class="coding">style.scss</span> in the root directory. 
* Append the below lines in your <span class="coding">_config.yml</span> file.

<div class="code-head"><span>code</span>_config.yml</div>

```yaml
sass:
  style: compressed
```

* First two lines of your <span class="coding">style.scss</span> should include **triple dashes** like the one shown below. After that you can use this file as the master Sass file for your website. You can also import Sass files residing inside <span class="coding">_sass</span> folder as shown below. 

<div class="code-head"><span>code</span>_config.yml</div>

```sass
---
---

@import "typography";
@import "colors";

// start writing after this
div {
  width: 100%;
}
```

<div class="note">
<p>
  <b>Note:</b> Any file inside <span class="coding">_sass</span> folder must have <b>_</b> appended to its name. For example, this is a Sass partial file - <span class="coding">_typography.scss</span>.
</p>
</div>


### Customizing layout and style

The default theme provided by Jekyll is awesome and responsive. If you wish to override the default theme, carry on. For the rest of this tutorial, follow the below steps to modify the layout of your website and include custom styling.

* Create **two** folders in the root directory. <span class="coding">_includes</span> and <span class="coding">_layouts</span>.
  * <span class="coding">_includes</span> folder will include all the pieces that comes repeatedly in your website such as header, footer, meta etc. We will create three files inside this folder for now.
    1. <span class="coding">meta.html</span> will hold all the meta tags inside head tag.
    2. <span class="coding">header.html</span> will hold all the header related elements such as logo, navigation bar etc.,
    3. <span class="coding">footer.html</span> will hold all the footer related elements such as copyrights etc.,

<div class="code-head"><span>code</span>meta.html</div>

```html
<meta charset="utf-8">
<meta http-equiv="x-ua-compatible" content="ie=edge">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>{% raw  %} "{% if page.title %}{{ page.title }}{% else %}{{ site.title }}{% endif %} {% endraw  %}</title>

<link rel="stylesheet" type="text/css" href="{{ site.baseurl }}/style.css" />
<link href='https://fonts.googleapis.com/css?family=Nunito' rel='stylesheet' type='text/css'>
```

<div class="code-head"><span>code</span>header.html</div>

```html
<header>
  <h2>{{ "{{ site.name " }}}}</h2>
</header>
```

<div class="code-head"><span>code</span>footer.html</div>

```html
<footer>
  <p>Made with the awesome Jekyll</p>
</footer>
```

Now, create a file named <span class="coding">default.html</span> inside <span class="coding">_layouts</span> directory. This will act as the blueprint for your website.

<div class="code-head"><span>code</span>default.html</div>

```html
<!DOCTYPE html>
<html>
  <head>
    {% raw  %} {% include meta.html %} {% endraw  %}
  </head>

  <body>
    {% raw  %} {% include header.html %} {% endraw  %}
    
    <div class="container">
      <div></div>
      <article>
        {% raw  %} {{ content }} {% endraw  %}
      </article>
      <div></div>
    </div>

    {% raw  %} {% include footer.html %} {% endraw  %}
  </body>

</html>
```

Cool! Open up <span class="coding">index.md</span> and replace it with the following content.

```html
---
layout: default
---

<h3>Welcome to my Blog</h3>

<div>
  {% raw  %}{% for post in site.posts %}{% endraw  %}
    <ul>
      <li><a href="{% raw  %}{{ site.baseurl }}{{ post.url }}{% endraw  %}">{% raw  %}{{ post.title }}{% endraw  %}</a></li>
    </ul>
  {% raw  %}{% endfor %}{% endraw  %}
</div>
```

Awesome! Open up <span class="coding">style.scss</span> and replace it with the following content.

```sass
---
---

body {
  background-color: #ffffff !important;
  margin: 0px;
  padding: 0px;
  font-family: "Nunito", sans-serif;
}

header {
  padding: 20px;
  background-color: rebeccapurple;
  text-align: center;

  h2 {
    color: white;
    font-weight: 100;
  }
}

footer {
  position: fixed;
  bottom: 0;
  right: 0;
  height: 60px;
  background-color: black;
  text-align: center;
  width: 100%;
  
  p {
    color: white;
    height: 60px;
    margin-top: 20px;
      font-size: 13px;
  }
}

.container {
  display: flex;
  margin-bottom: 100px;
  div {
    flex: 1;
  }
  article {
    flex: 1;
  }
}

article {
  background-color: #eaeaea;
  height: 100%;
  padding: 20px;
  h3 {
    text-align: center;
  }
}

.post-header {
  background-color: #252525 !important;
    border-radius: 10px;
    box-shadow: inset 0 3px 30px black;
  h1, p {
    color: white;
    font-weight: 100;
  }
}
```

Now, if you see your website locally **http://localhost:4000**, it will look like the one shown below. 


<figure>
  <img src="/images/software/jekyll-create-your-first-website/demo_2.png" class="typical-image">
  <figcaption>Figure 2. Front page.</figcaption>
</figure>

<figure>
  <img src="/images/software/jekyll-create-your-first-website/demo_3.png" class="typical-image">
  <figcaption>Figure 3. Post page.</figcaption>
</figure>

Start exploring what we did above and your imagination is all that is needed to beautify layouts and styles.

### References

* [Jekyll](https://jekyllrb.com/){:target="_blank"}
* [Awesome Jekyll Tutorial](https://www.youtube.com/playlist?list=PLLAZ4kZ9dFpOPV5C5Ay0pHaa0RJFhcmcB){:target="_blank"}
* [Override Theme Defaults in Jekyll](https://jekyllrb.com/docs/themes/#overriding-theme-defaults){:target="_blank"}
* [Make a Static Website with Jekyll](https://www.taniarascia.com/make-a-static-website-with-jekyll/){:target="_blank"}
* [Using Sass with Jekyll](http://markdotto.com/2014/09/25/sass-and-jekyll/){:target="_blank"}