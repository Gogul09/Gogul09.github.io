---
layout: post
category: software
class: Programming Languages
title: Linux Learning Notes
description: Understand the syntax, commands and how to's of Linux which is highly used in tech companies.
author: Gogul Ilango
permalink: /software/linux-learning-notes
image: https://drive.google.com/uc?id=1Oaa17RG4fx6PmOIHbijQLMkGDsYuplW2
---

In this page, you will learn some of the useful Linux commands along with its descriptions. This might be used as a handy reference to quickly know about the syntax of a command and its usefulness.

<style type="text/css">
.tg  {
	border-collapse: collapse;
	border-spacing: 0;
	border-color: #ccc;
  margin-top: 10px;
}

.tg td {
	font-size: 15px;
	padding: 10px 5px;
	border-style: solid;
	border-width: 1px;
	overflow: hidden;
	word-break: normal;
	border-color: #ccc;
	color: #333;
	background-color: #fff;
	text-align: left;
}

.tg th {
	font-size: 15px;
	font-weight: bold;
	padding: 10px 5px;
	border-style: solid;
	border-width: 1px;
	overflow: hidden;
	word-break: normal;
	border-color: #ccc;
	color: #333;
	background-color: #f0f0f0;
	text-align: center;
}

.tg .tg-yw4l {
	vertical-align: top;
}

</style>
<table class="tg">
  <tr>
    <th class="tg-yw4l" style="font-family: 'Open Sans', sans-serif;">Command</th>
    <th class="tg-yw4l" style="font-family: 'Open Sans', sans-serif;">Description</th>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">wget URL</span></td>
    <td class="tg-yw4l"><p>Downloads the file specified by the URL.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">pwd</span></td>
    <td class="tg-yw4l"><p>Displays the current directory you are in.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">sudo</span></td>
    <td class="tg-yw4l"><p>Allows the user to act like a superuser.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">sudo -i</span></td>
    <td class="tg-yw4l"><p>Allows the user to get root access.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">sudo apt-get install package_name</span></td>
    <td class="tg-yw4l"><p>Allows the user to act like a superuser and install packages.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">cd directory_name</span></td>
    <td class="tg-yw4l"><p>Changes from current directory to the mentioned directory.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">cd ..</span></td>
    <td class="tg-yw4l"><p>Moves back one directory.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">ls</span></td>
    <td class="tg-yw4l"><p>To view the contents in a directory including files and sub-directories.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">ls -a</span></td>
    <td class="tg-yw4l"><p>To view the contents in a directory including hidden files.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">man command</span></td>
    <td class="tg-yw4l"><p>Displays the information about the command specified.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">whereis file/directory</span></td>
    <td class="tg-yw4l"><p>Shows where the specified file/directory is.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">mkdir directory_name</span></td>
    <td class="tg-yw4l"><p>Creates a directory with the given name.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">mv old_filename new_filename</span></td>
    <td class="tg-yw4l"><p>Renames the file. <br> Moves file from one location to another.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">cp old_filename new_filename</span></td>
    <td class="tg-yw4l"><p>Copies file from one location to another.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">rm filename</span></td>
    <td class="tg-yw4l"><p>Removes the specified filename.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">rmdir directoryname</span></td>
    <td class="tg-yw4l"><p>Removes the specified empty directory.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">rm -rf directoryname</span></td>
    <td class="tg-yw4l"><p>Removes files and sub-directories in the specified directory.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">touch filename</span></td>
    <td class="tg-yw4l"><p>Creates a new file. <br> Modifies the datetime of the already created filename to current datetime.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">ifconfig &amp; iwconfig</span></td>
    <td class="tg-yw4l"><p>Allows the user to look at the network configuration.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">ping URL</span></td>
    <td class="tg-yw4l"><p>Allows the user to test connectivity issues.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">vi filename</span></td>
    <td class="tg-yw4l"><p>Opens the specified file in the vi editor to view/make changes.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">telnet ip_address</span></td>
    <td class="tg-yw4l"><p>Connects to the specified IP address.</p></td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">chmod 777 file_name</span></td>
    <td class="tg-yw4l"><p>Modifies the permissions of the specified file.</p>
    	<ul>
    	 <li><b>4</b> - Read</li>
    	 <li><b>2</b> - Write</li>
    	 <li><b>1</b> - Execute</li>
    	 <li><b>0</b> - No permissions</li>
    	</ul>
      <p>The three digits in 777 represents <b>users</b>, <b>groups</b> and <b>others</b>.<br><br>
      <b>7</b> means - 4 + 2 + 1, meaning <b>users</b> can <b>read</b>, <b>write</b> and <b>execute.</b></p>
   </td>
  </tr>
  <tr>
    <td class="tg-yw4l"><span class="coding">chmod -R 777 directory</span></td>
    <td class="tg-yw4l"><p>Modifies the permissions of the specified directory <b>recursively</b>. Meaning it applies the changes for all the files and sub-directories.</p></td>
  </tr>
	<tr>
    <td class="tg-yw4l"><span class="coding">du -sh filename</span></td>
    <td class="tg-yw4l"><p>Prints the file size</p></td>
  </tr>
</table>
