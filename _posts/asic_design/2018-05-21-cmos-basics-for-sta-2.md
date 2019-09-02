---
layout: post
category: hardware
class: ASIC Design
title: CMOS Basics - Part 2
description: Understand the fundamental concepts of Static Timing Analysis in VLSI (ASIC design) such as Switching Activity, Propagation Delay, Slew and Skew.
author: Gogul Ilango
permalink: /hardware/cmos-basics-for-sta-2
image: https://drive.google.com/uc?id=1Bk2p60RLiz3AOcDDS0-11FfeJFC17Xjq
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#switching-activity">Switching Activity</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#propagation-delay">Propagation Delay</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#slew">Slew</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#timing-arcs-and-unateness">Timing Arcs and Unateness</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#references">References</a></li>
  </ul>
</div>

In the <a href="https://gogul09.github.io/hardware/cmos-basics-for-sta-1" target="_blank">previous blog post</a>, we learnt about the basics of CMOS from a timing perspective and understood what standard cells, logic levels and time constant are, and how resistances and capacitances affects timing of a design.

In this blog post, we will utilize that knowledge and look into some more concepts such as switching activity, propagation delay, slew and timing arcs.

<div class="objectives">
  <h3>Objectives</h3>
  <p>After reading this tutorial, we will understand</p>
  <ul>
    <li>What is switching activity in a standard cell?</li>
    <li>What is propagation delay and why it occurs?</li>
    <li>Why slew exists for a waveform?</li>
    <li>What is Intrinsic Delay?</li>
    <li>What are Timing Arcs and Unateness?</li>
  </ul>
</div>

### Switching Activity
Let's consider the CMOS inverter example for this blog post and the same is applicable for larger circuits as we build up.

As we know that both PMOS and NMOS act like switches that either allow VDD or VSS at a particular time instant, we could model a simple CMOS inverter analogous to an RC network as shown in Figure 1.

<figure>
  <img src="https://drive.google.com/uc?id=1JBkfQvOEoSVROErqERkVwQEv6AC49QcF" />
  <figcaption>Figure 1. Switching Activity in CMOS inverter + Charging and Discharging waveforms</figcaption>
</figure>

* **Pull-up**: When the input is logic 0, SW0 (PMOS) is turned ON and SW1 (NMOS) is turned OFF. Output is pulled to VDD and becomes logic 1. Before the activation of SW0, if output is at logic 0, then the voltage transition (charging) at the output is given by the equation 

<div class="math-cover">
$$
V = Vdd * [1 - e^{(\frac{-t}{Rdh*Cload})}]
$$
</div> 

* **Pull-down**: When the input is logic 1, SW0 (PMOS) is turned OFF and SW1 (NMOS) is turned ON. Output is pulled to VSS and becomes logic 0. Before the activation of SW1, if output is at logic 1, then the voltage transition (discharging) at the output is given by the equation 

<div class="math-cover">
$$
V = Vdd * e^{(\frac{-t}{Rdl*Cload})}
$$
</div>

The above two equations are valid for the RC network we considered. When it comes to standard cells (CMOS cells), during the transition from one logic state to another (i.e charging or discharging of capacitor), for a definite amount of time, current flows from VDD to VSS which give rise to <a href="http://nptel.ac.in/courses/117101058/downloads/Lec-26.pdf" target="_blank">short-circuit power dissipation</a>. 

<div class="note">
  <p><b>Concept 1:</b> The time taken at the output when there is a signal transition from one logic state to another (due to which both pull-up and pull-down structures are turned ON) is called as the Transition Time.</p>
</div>

It is important to analyze this transition time because it is responsible for the propagation delay and slew of a standard cell.

### Propagation Delay
Due to the transition time that arises during the switching activity in a standard cell, we will not be able to get the output <i>instantly</i> as soon as the input changes. There is a <i>finite delay</i> involved during this transition called as the Propagation Delay. Let's quickly understand what propagation delay mean.

<figure>
  <img src="https://drive.google.com/uc?id=1mPghFF9RutGLCXFSGo3CPutuw2x-gvwa" />
  <figcaption>Figure 2. Propagation Delay of a CMOS inverter</figcaption>
</figure>

Look at the waveform in Figure 2[a] which shows propagation delays of an ideal inverter. In the ideal case, there is no transition time (zero short-circuit current) and hence the delay corresponds to the intrinsic delay associated with the CMOS cell only.

#### Quick facts about Intrinsic Delay
* It is the internal delay associated within a standard cell.
* It is the delay from input pin of the cell to the output pin of the cell.
* This is due to the internal capacitance associated with the CMOS transistors present inside the cell.
* It is highly dependent on the size of the CMOS transistors as capacitance increases if size increases.

Now look at the waveform in Figure 2[b] which shows propagation delay of a real inverter. Two types of edges that appear here are 

* **Rising edge**: Signal transition from logic 0 to logic 1.
* **Falling edge**: Signal transition from logic 1 to logic 0.

Due to these two edges, there are typically four propagation delay threshold points that are being measured (shown as red circle in Figure 2[b]) which are in terms of the percentage of VDD.

```python
# threshold of an input rising edge
input_threshold_pct_rise : 50.0;

# threshold of an input falling edge
input_threshold_pct_fall : 50.0;

# threshold of an output rising edge
output_threshold_pct_rise : 50.0;

# threshold of an output falling edge
output_threshold_pct_fall : 50.0;
```

Mostly, 50% threshold (percent of VDD) is taken for delay measurement in standard cell libraries.

<div class="note">
  <p><b>Concept 2:</b> Due to signal transition from one logic level to another, there comes a finite amount of delay (apart from intrinsic delay) at the output of a standard cell which corresponds to the propagation delay of the standard cell.</p>
</div>

### Slew
Another important factor to be considered in delay calculation is the slew. Slew rate corresponds to the rate of change of voltage of a rising or falling edge of a waveform (which is measured with respect to the transition time).

Notice that, transition time is just the inverse of slew rate. Thus, larger the transition time, smaller is the slew rate and vice-versa. 

Similar to propagation delay threshold points, there are four slew threshold points (percent of VDD) for a single waveform as shown in Figure 3.

```python
# rising edge thresholds
slew_lower_threshold_pct_rise : 30.0;
slew_upper_threshold_pct_rise : 70.0;

# falling edge thresholds
slew_lower_threshold_pct_fall : 30.0;
slew_upper_threshold_pct_fall : 70.0;
```

<figure>
  <img src="https://drive.google.com/uc?id=1Z-ydhpgxeL-UrbXfMCsAmqR1d8mUxo-o" />
  <figcaption>Figure 3. Slew rate of a waveform with 4 slew threshold points</figcaption>
</figure>

* **Rise slew** is the time difference between the rising edge reaching 70% and 30% of VDD.
* **Fall slew** is the time difference between the falling edge reaching 70% and 30% of VDD.

> Did you know that slew is the reason behind <b>portamento</b> (glide or lag) feature in a music-synthesizer? - <a href="https://en.wikipedia.org/wiki/Slew_rate" target="_blank">Read more</a>

### Timing Arcs and Unateness
Static Timing Analysis (STA) for the entire chip is performed by evaluating timing paths in the design. As STA is purely based on timing paths (not functionality), timing arc is one of the very important components of a timing path.

Timing arc is described by something called as "Unateness". Unateness is an important property of standard cells. Every standard cell has its associated multiple timing arcs that are specified in the cell library. Each timing arc specifies the unateness between an input pin and the output pin of the cell. 

<div class="note">
  <p><b>Concept 3</b>: Timing arc describes the relationship between an input pin and an output pin of a cell by means of unateness. This is specified as <span class="coding">timing_sense</span> in the cell library.</p>
</div>

Typically, there are three types of timing arcs for a standard cell.
1. Positive Unate.
2. Negative Unate.
3. Non-Unate.

#### 1. Positive Unate
If there is a rising transition at the input (source) that is causing a rising transition at the output (sink), or if there is a falling transition at the input (source) that is causing a falling transition at the output (sink), then it is called Positive Unate. Ex: OR, AND.

#### 2. Negative Unate
If there is a rising transition at the input (source) that is causing a falling transition at the output (sink), or if there is a falling transition at the input (source) that is causing a rising transition at the output (sink), then it is called Negative Unate. Ex: NOR, NAND.

#### 3. Non-Unate
If the output transition cannot be found from a single input's transition and depends on multiple input transitions, then it is called Non-Unate. Ex: XOR.

<figure>
  <img src="https://drive.google.com/uc?id=14ZxXRmdEv8TYhrz88UEQie1u-v3AW7QH" />
  <figcaption>Figure 4. Timing Arcs and Unateness</figcaption>
</figure>

In addition to the timing arcs of a cell, there exists timing arcs for nets in the design as shown in Figure 4(d). Generally, such net timing arc is from output pin of a cell (driver or source) to the input pin of the cell (load or sink). Same unateness concept is applied here as well. 

For net timing arcs, the existence of such arcs is taken from the netlist and are calculated using the parasitic values given in SPEF (Standard Parasitics Exchange Format) file.

In the next post, we will look into extending all these timing concepts for a larger circuit and understand various delays involved.

### References 

1. Static Timing Analysis for Nanometer Designs: A Practical Approach, Jayaram Bhasker and Rakesh Chadha.
2. Digital Integrated Circuits: A Design Perspective, Jan Rabaey.