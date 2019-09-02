---
layout: post
category: hardware
class: ASIC Design
title: Static Timing Analysis - Timing Paths and Delays
author: Gogul Ilango
permalink: /hardware/sta-timing-paths-and-delays
description: Understand the basic concepts behind Static Timing Analysis in VLSI (ASIC design) such as Timing Paths and Delays.
image: https://drive.google.com/uc?id=1G5IufAyiH6rg0KGrmvTkDvjYBEwcLKV1
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#static-timing-analysis">Static Timing Analysis</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#characteristics-of-sta">Characteristics of STA</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#timing-paths">Timing Paths</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#delays">Delays</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#min-and-max-timing-paths">Min and Max Timing Paths</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#references">References</a></li>
  </ul>
</div>

**In this blog post, we will learn the basic concepts involved in Static Timing Analysis and understand why it is highly preferred in practice to meet timing.** 

When designing a chip, performance is the number one concern for chip designers. The chip needs to meet timing to work properly else user experience suffers. To verify whether a design meets timing, there are different techniques such as Timing Simulation, Static Timing Analysis (STA) and Dynamic Timing Analysis (DTA). 

<div class="objectives">
  <h3>Objectives</h3>
  <p>After reading this tutorial, we will understand</p>
  <ul>
    <li>What is Static Timing Analysis (STA)?</li>
    <li>Why STA is used?</li>
    <li>What are timing paths?</li>
    <li>What are the different types of delays?</li>
  </ul>
</div>

<h3>Prerequisites</h3>
<ul>
	<li><a href="https://gogul09.github.io/hardware/cmos-basics-for-sta-1" target="_blank">CMOS basics - Part 1</a></li>
	<li><a href="https://gogul09.github.io/hardware/cmos-basics-for-sta-2" target="_blank">CMOS basics - Part 2</a></li>
</ul>

### Static Timing Analysis
According to [Wikipedia](https://en.wikipedia.org/wiki/Static_timing_analysis){:target="_blank"}, Static timing analysis (STA) is a simulation method of computing the expected timing of a digital circuit without requiring a simulation of the full circuit.

In other words, STA is the method of summing up cell delays and net delays in a design (which equals path delays) and comparing the path delays to the constraints (timing specifications). 

STA is done at many stages in a typical ASIC design flow (Figure 1). Performing STA before detail routing only provides the approximations based on different factors. But the real picture of STA is obtained only after detail routing the entire design in the layout phase (physical design). This is because, only after detail routing the entire design, parasitics are extracted (Resistances and Capacitances) based on detailed routes which is an important input to perform STA. 

<figure>
  <img src="https://drive.google.com/uc?id=1lCtMbPcCUDYtUCeNOpJoHueuurS8_CwJ" class="typical-image" />
  <figcaption>Figure 1. How STA fits in the ASIC design flow?</figcaption>
</figure>

As you might know, a smartphone or a laptop or a tablet runs by the clock frequency (which is given by the specifications). Due to this predefined clock frequency, it is necessary for the design to meet this frequency under different external environmental conditions which includes Process, Voltage and Temperature (PVT). Thus, the purpose of STA is to check if the design can operate at the desired speed (frequency) under different external environmental conditions.

### Characteristics of STA
* STA is a complete and exhaustive technique of verifying the timing of a chip.
* STA is fast and accurate measurement of circuit timing.
* STA does not check for logical functionality in the design.
* STA uses simplified timing models to check for violations in the design.
* STA is all about analyzing the cell delays and net delays over millions of paths in a design and fixing if any violation arises in those paths by comparing with the timing constraints.
* STA analyzes entire design once and the required timing checks are carried out for all possible paths and scenarios of the design.
* STA is the mainstay of design over the last few decades.
* <span class='inline-note'>STA = Delay calculation + Timing Checks</span>
* Types of timing checks performed in STA are 
  * Setup check
  * Hold check
  * Recovery check
  * Removal check
  * Data-to-Data user-specified timing check
  * Clock-gating check
  * Minimum period check
  * Minimum pulse width check
  * Design rule checks (min/max transition time, capacitance, fanout)

<div class="note">
  <p><b>Did you know:</b> The word <span class="coding">Static</span> in STA is due to the fact that it does not depend on the data values being applied at the inputs and the whole timing analysis is based on simplified timing models and delays.</p>
</div>

When a <span class="coding">gate-level</span> netlist is available, STA is done based on 
* **Interconnect modelling** - ideal, wireload model, global routes (approximate RC values), detail routes (accurate RCs).
* **Clock modelling** - ideal clocks (zero delay) or propagated clocks (real delays).
* **Crosstalk & Noise** - Coupling between signals (metal traces).

### Timing Paths 
For a typical design which includes 10-100 million gates, there exists huge number of timing paths. These timing paths are defined by timing arcs which we discussed in the [previous](https://gogul09.github.io/hardware/cmos-basics-for-sta-2){:target="_blank"} tutorial. 

There are two types of timing arcs in a timing path. 
* **Cell timing arc** - Timing arc between an input pin and the output pin of a cell. 
* **Net timing arc** - Timing arc of a net (wire) that is between a driver (output pin) and a load (input pin).

The basic measures of the above imaginary timing arcs are 
* **Delay** - Provided in the cell library (for cell) and SPEF (for net).
* **Unateness** - Provided in the cell library (for cell) and always positive unate (for net).
* **Transition time** - Provided as slew thresholds in the cell library (for cell).

A timing path has a start point and an end point. 
* **Start point** - All input ports/pins or clock ports/pins of sequential cells are considered as start points.
* **End points** - All output ports/pins or D pin of sequential cells are considered as end points.

<figure>
  <img src="https://drive.google.com/uc?id=1cqatynNzlnR7sHOVeBRv4MQ29nl37IOO" />
  <figcaption>Figure 2. Timing Paths and its types.</figcaption>
</figure>

Based on the above mentioned start point and end point, there are four types of timing paths (Figure 2) based on <span class="coding">direction</span>. 
1. **Input to Register** - Start point is an input pin/port and end point is the D pin/port of a register (flipflop). It might include both combinational and sequential cells. 
2. **Register to Register** - Start point is the CLK pin/port of a  register (flipflop) and end point is the D pin/port of next register (flipflop). Also, this type of path might include combinational and sequential cells.
3. **Register to Output** - Start point is the CLK pin/port of a register (flipflop) and end point is an output pin/port. Both sequential and combinational cells are included here.
4. **Input to Output** - Start point is an input pin/port and end point is an output pin/port. It includes combinational cells only.

In addition to the above timing paths, based on <span class="coding">signal type</span>, there are two additional types of timing path categorization in a design.

1. **Clock path** - The timing path which is fully traversed by clock signals is called as Clock path. In a clock path, there could only be clock inverters or clock buffers. Additionally, to save power, there could be presence of clock gating cells in this clock path (ex: AND gate). Such paths are called as "Gated clock paths". Clock paths are further categorized into two types.
* **Launch path** - The timing path which is traversed by the clock signal from source pin to launch register (flipflop) CLK pin (FF1 in Figure 2).
* **Capture path** - The timing path which is traversed by the clock signal from source pin to capture register (flipflop) CLK pin (FF2 in Figure 2).
2. **Data path** - The timing path which is fully traversed by data signals is called as Data path. In a data path, there could be combinational cells, data buffers etc.,

<figure>
  <img src="https://drive.google.com/uc?id=11NegqqiFsnVqWgKhCCeUivqUTtK5tOUD" class="typical-image"  />
  <figcaption>Figure 3. Timing Paths Categories.</figcaption>
</figure>

### Delays 
In the real world, delays are what makes STA interesting! Due to delays, there arises timing violations and the job of a chip design engineer is to analyze and understand these delays, and fix the timing violations. There are two types of delays in STA.

#### 1. Cell delay

CMOS transistors inside a standard cell takes finite amount of time to switch from one logic state to another. This time taken is called as the cell delay or the propagation delay of a cell which is typically specified in the cell timing library (.lib). 

The propagation delay of a standard cell depends on three factors (Figure 4).
1. **Input Slew** - The transition time at the input i.e. the time it takes for an input pin (input capacitance) to switch between logic states (low-high or high-low).
2. **Output Capacitance** - The capacitance that needs to get charged/discharged at the output of a cell driving single or multiple loads. This capacitance introduces a finite amount of delay.
3. **Intrinsic Delay** - The internal delay of a cell when a signal with zero transition time is applied to the input pin and no output load is present.

<figure>
  <img src="https://drive.google.com/uc?id=1Qnu-S2z1W59UOLeshycEYnP1PABmJd9-" class="typical-image" />
  <figcaption>Figure 4. Cell Delay and its components.</figcaption>
</figure>

#### 2. Net delay

Interconnect delay or net delay is due to the physical wire (metal traces having resistances and capacitances) of a logical net (i.e. between a driver and load/loads). All net timing arcs are positive unate. This is because there can be either rise-rise transition from driver to load or fall-fall transition from driver to load. 

Net delay is calculated using two methods depending on the design phase.
* **Wire-load models** (approximate - used in logic design) - Based on statistics and poisson distribution to calculate the resistance and capacitance values of the nets based on length, fanout and area.
* **Actual physical layout estimate** (accurate - used in physical design) - Uses backannotation i.e. the process of extracting the resistances and capacitances after detail routing all the nets in a design (also called as [parasitic extraction](https://en.wikipedia.org/wiki/Parasitic_extraction){:target="_blank"}). There are three different file formats for representing parasitics in a design for timing analysis.
1. Detailed Standard Parasitic Format (DSPF) 
2. Reduced Standard Parasitic Format (RSPF)
3. Standard Parasitic Exchange Format (SPEF)

<div class="note">
<p><b>Did you know:</b> <a href="https://en.wikipedia.org/wiki/Standard_Parasitic_Exchange_Format" target="_blank">SPEF</a> is the industry standard file format of choice to represent parasitic values (resistances and capacitances) of a design.</p>
</div>

### Min and Max Timing Paths

Apart from the functionality based timing paths, there exist two types of timing paths based on path delay. <span class="inline-note">Path delay = Cell delay + Net delay</span>. As there are multiple ways for the logic to propagate from one start point to an end point (Figure 5), the maximum and minimum timing paths can be calculated easily based on cell delay and net delay. 
1. **Min path**: The path between two points with shortest delay.
2. **Max path**: The path between two points with largest delay.

<figure>
  <img src="https://drive.google.com/uc?id=1KiMCUIwJkAurL7EH6YYRpHbQ8pxl_OZ2" />
  <figcaption>Figure 5. Min and Max Timing Paths.</figcaption>
</figure>

In the next post, we will understand various concepts involved in STA such as setup time, hold time, max delay, min delay etc.,

### References 

1. Static Timing Analysis for Nanometer Designs: A Practical Approach, Jayaram Bhasker and Rakesh Chadha.
2. Digital Integrated Circuits: A Design Perspective, Jan Rabaey.