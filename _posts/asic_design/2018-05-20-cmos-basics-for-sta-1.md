---
layout: post
category: hardware
class: ASIC Design
title: CMOS Basics - Part 1
description: Understand the basics of CMOS and terminologies used to perform Static Timing Analysis in VLSI (ASIC design)
author: Gogul Ilango
permalink: /hardware/cmos-basics-for-sta-1
image: https://drive.google.com/uc?id=1cTetisIy4dJrwezQbVUyZOhslpVbWAI3
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#structure-of-cmos">Structure of CMOS</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#cmos-inverter">CMOS Inverter</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#standard-cells">Standard Cells</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#logic-levels">Logic Levels</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#modelling-of-standard-cells">Modelling of standard cells</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#references">References</a></li>
  </ul>
</div>

To understand Static Timing Analysis (STA) which is responsible for the performance of state-of-the-art Integrated Circuts (ICs), we need to become stronger in CMOS fundamentals.

<div class="objectives">
  <h3>Objectives</h3>
  <p>After reading this tutorial, we will understand</p>
  <ul>
    <li>What is the structure of CMOS?</li>
    <li>What is a technology node?</li>
    <li>What is a CMOS inverter?</li>
    <li>What are standard cells?</li>
    <li>What are logic levels?</li>
    <li>How standard cells are modelled?</li>
  </ul>
</div>

### Structure of CMOS
CMOS, abbreviated as Complementary Metal Oxide Semiconductor is the reason behind the electronic gadgets that we use in our day-to-day lives. CMOS is a combination of PMOS and NMOS. To learn about MOSFET in general, please read [this](https://gogul09.github.io/hardware/mosfet-fundamentals){:target="_blank"} post and come back.

All integrated circuits (ICs) are built based on something called as "technology node" (45nm, 32nm, 28nm, 14nm, 7nm etc). This technology node is purely realized from a physical design point of view rather than logic design (which mostly depends on functionality). For example, if an IC is designed with 14nm technology, then it means that it uses 14nm cell libraries and must be designed with 14nm technology rules given by the semiconductor foundry. 

<a href="https://en.wikichip.org/wiki/technology_node" target="_blank">Technology node</a> is the measure of feature size which is further a measure of channel length between drain and source in a MOS transistor (see Figure 1). This is the smallest length used to build a MOS transistor. So, for a 28nm technology node, it means the smallest channel length between drain and source is 28nm. As we go further down in technology node, more transistors can be packed in a single chip allowing us to build designs that operates much faster.

<figure>
  <img src="https://drive.google.com/uc?id=1au9a7dY1kzNgfzpjBSFlXnwlWEFgC4PZ" />
  <figcaption>Figure 1. Structure of CMOS</figcaption>
</figure>

### CMOS Inverter
Understanding the characteristics of a single <a href="https://en.wikipedia.org/wiki/CMOS" target="_blank">CMOS inverter</a> provides the necessary knowledge to build more complex stuff on top of it using any CMOS logic gate. Figure 2 shows a typical CMOS inverter circuit. This inverter circuit transforms an input signal into an inverted output signal. 

<p style="margin-bottom: 0px !important;">There are two stable states for a CMOS inverter (Figure 2).</p>
* **Pull-up** - If input A is low (\\(V_{ss}\\) or logic 0), NMOS is off and PMOS is on, making the output Z to be pulled to \\(V_{dd}\\). 
* **Pull-down** - If input A is high (\\(V_{dd}\\) or logic 1), NMOS is on and PMOS is off, making the output Z to be pulled to \\(V_{ss}\\).

<figure>
  <img src="https://drive.google.com/uc?id=11nlWRKKUil_mS4F_oTRbzAv8wIkCkVp1" />
  <figcaption>Figure 2. CMOS Inverter (Pull-up and Pull-down structures)</figcaption>
</figure>

<p style="margin-bottom: 0px !important;">Similar to the CMOS inverter, any CMOS logic gate can be realized using a pull-up structure (made of PMOS transistors) and pull-down structure (made of NMOS transistors) connected to the output Z.</p>
* Pull-up and pull-down structures are complementary to each other, meaning when pull-up is turned ON, pull-down is turned OFF and vice-versa.
* The inputs (A and B) control this behaviour and the output (Z) purely depends on the functionality of the logic gate.
* In both these cases, there is no current passing from the inputs or from the power supply VDD as only one structure (pull-up or pull-down) is turned on for a particular set of inputs, which provides high noise immunity and low static power consumption.

### Standard Cells
Any complex chip is built using basic building blocks in digital design such as <span class="coding">and</span>, <span class="coding">or</span>, <span class="coding">not</span>, <span class="coding">nor</span>, <span class="coding">nand</span>, <span class="coding">flipflop</span>, <span class="coding">buffer</span> etc. Based on a technology node, these building blocks (called <a href="https://en.wikipedia.org/wiki/Standard_cell" target="_blank">standard cells</a>) are predesigned (i.e will have logical, symbol, abstract, layout and schematic views plus timing information for different scenarios) and given to the IC designer.

Standard cells have fixed height and variable width. Height of these cells are same as the height of a standard cell row. In some cases, there might be double-height or triple-height cells that occupies two or three rows respectively.

As chip design is all about trade-off between area, power and speed, the designer is given freedom to pick standard cells that have same logic function differing in speed and area. 

Some of the characteristics of a typical standard cell library are as follows.

* Large varieties of drive strengths for all logic cells and storage cells.
* Large varieties of drive strengths for buffers and inverters.
* Varities of cells with respect to Threshold voltage (LVT, SVT, HVT).
* Cells with balanced rise and fall delays (clock buffers).
* Varieties of delay cells which are useful to fix hold-time violations.
* Physical cells (Tap cells, Tie cells, Endcap cells, Filler cells, Spare cells etc.,)
* Power related cells (Level shifters, Clock gating cells etc.,)

> In ASIC design, standard cells are the bricks used to build a house, the chip. 

### Logic Levels
In real-world, if we define \\(V_{dd}\\) as 5V and \\(V_{ss}\\) as 0V, due to noise and electro-magnetic stuff happening inside a chip, a signal can never be at 5V nor be at 0V everytime. Hence, we propose something called as "logic-levels".

Two values \\(V_{IHmin}\\) and \\(V_{ILmax}\\) are defined based on \\(V_{dd}\\) and \\(V_{ss}\\) as shown in Figure 3. Any voltage value above \\(V_{IHmin}\\) is logic 1 and any voltage value below \\(V_{ILmax}\\) is logic 0. These values (also called as <a href="https://en.wikipedia.org/wiki/Noise_margin" target="_blank">noise margin</a>) are known from the DC characteristics of the standard cell considered. 

For example, in a given technology node, if the signal is intended to swing between <span class="coding">0.0V</span> and <span class="coding">1.2V</span>, then any value below <span class="coding">0.2V</span> (\\(V_{ILmax}\\)) becomes a logic 0 and any value above <span class="coding">1.0V</span> (\\(V_{IHmin}\\)) becomes a logic 1.

<figure>
  <img src="https://drive.google.com/uc?id=19d4yl9b_6ygY4MybuKr2TPs8IWRF0mk-" />
  <figcaption>Figure 3. Logic levels of CMOS (Noise margin)</figcaption>
</figure>

### Modelling of standard cells
Before proceeding further, we need to familiarize ourselves with the below terminologies.
* **Instance** - A single occurance or a single object of a standard cell. (ex: a 2-input AND gate could be instantiated as <span class="coding">CELL_AND_1</span>)
* **Pins** - Collection of pins of an instance which includes both inputs and outputs. (ex: For a 2-input AND gate, <span class="coding">CELL_AND_1/a</span>, <span class="coding">CELL_AND_1/b</span> and <span class="coding">CELL_AND_1/z</span> are the pins)
* **Net** - A logical connection between two or more pins of different instances.
* **Wire** - A physically realized connection of a net using different metal layers and vias.
* **Fanout** - The maximum number of inputs (loads) that the output of a single standard cell (driver) can drive.
* **Fanin** - The maximum number of inputs that a standard cell (or a logic gate) can accept. 
* **Drive strength** - Capacity of a standard cell's output (driver) to drive a value to the cell (load) connected.

Consider the following multi-fanout example (Figure-4) to understand how resistances and capacitances affects the delay involved in a standard cell.
* \\(G1\\), \\(G2\\), \\(G3\\), \\(G4\\), \\(G5\\) are the standard cell instances (or logic gates).
* \\(Cs1\\), \\(Cs2\\), \\(Cs3\\), \\(Cs4\\), \\(Cs5\\) are the capacitances of wires.
* \\(Cout(n)\\) is the output pin capacitance of the cell \\(n\\).
* \\(Cin(n)\\) is the input pin capacitance of the cell \\(n\\).

<figure>
  <img src="https://drive.google.com/uc?id=1ZmN6GiAwks6VMuAk8PhQDzltqGfAS2bq" />
  <figcaption>Figure 4. Modelling of standard cells</figcaption>
</figure>

Recall that <a href="https://en.wikipedia.org/wiki/RC_time_constant" target="_blank">RC time constant</a> \\(\tau\\) of a wire is the product of resistance \\(R\\) and capacitance \\(C\\).

<div class="math-cover">
$$
\tau = R * C
$$
</div>

In a standard cell, the inputs to the cell represents a capacitive load only. So, the input pins of a standard cell have input capacitances given by the foundry for a particular technology node. The wires (which represents the physical connections between different pins) also possess capacitive load. 

We are looking at capacitance between pins and nets because this is how an input or an output changes state. Normally, there is a definite time taken for this charging and discharging mechanism happening based on functionality.

For the multi-fanout structure shown in Figure 4, the total capacitance on the output pin (driver) of the cell G1 is the sum of output capacitance of the driving cell + sum of all the capacitance of the wires between + sum of all the input capacitances of the cells (load) it is driving.

<div class="math-cover">
$$
Total\;cap\;(Output\;G1) = \begin{align} Cin(G2) + Cin(G3) + Cin(G4) + Cin(G5) + \\ Cs1 + Cs2 + Cs3 +Cs4 + Cs5 + \\ Cout(G1) \end{align}
$$
</div>

From a timing perspective, the time taken for the driver pin (G1) to switch from one logic state to another depends on this \\(Total\;cap\;(Output\;G1)\\).

<div class="note">
  <p><b>Concept 1:</b> Total output capacitance of a standard cell controls the time taken for that cell to switch from one logic state to another. When a cell's output switches state, the speed of switching is determined by how fast the capacitance on the output net can be charged or discharged.</p>
</div>

In addition to capacitances, we need to be aware of the pull-up and pull-down equivalent resistances that are present inside the standard cell. 

Drive strength of a standard cell is an important factor that affects overall system's timing. This drive strength of a cell depends on how large/small the pull-up and pull-down structures are (in terms of size).

* The inverse of the **output pull-up resistance** is called the **output high-drive** of the cell.
* The inverse of the **output pull-down resistance** is called the **output low-drive** of the cell.

| Structure         | Resistance | Drive Strength            |
|-------------------|------------|---------------------------|
| Larger pull-up    | Smaller    | Larger output high-drive  |
| Smaller pull-up   | Larger     | Smaller output high-drive |
| Larger pull-down  | Smaller    | Larger output low-drive   |
| Smaller pull-down | Larger     | Smaller output low-drive  |

Normally, the standard cells from the foundry are designed to have equal or similar drive strengths (both high and low) for both these structures.

<div class="note">
  <p><b>Concept 2:</b> Output pull-up and pull-down equivalent resistances of a cell are responsible for the driving strength or speed of that cell.</p>
</div>

This is how both resistances and capacitances (parasitics) play a huge role in meeting the timing of an IC. Figure 5 shows the electrically equivalent model of Figure 4 with resistances and capacitances shown.

<figure>
  <img src="https://drive.google.com/uc?id=15cC-avnLbjjGABZYXlKs61KTT6IP3nCf" />
  <figcaption>Figure 5. Electrically equivalent model of Figure 4.</figcaption>
</figure>

We can finally say that the output charging delay of a standard cell depends on \\(R_{out}\\) and \\(C_{out}\\). Depending on the switching activity (high or low), \\(R_{out}\\) = \\(Rdh\\) or \\(Rdl\\) respectively and \\(C_{out}\\) = \\(Total\;cap\;(Output\;G1)\\).

Thus, the output charging delay at driver pin G1 (for high or low) = \\(R_{out}\\) * \\(C_{out}\\)

In the <a href="https://gogul09.github.io/hardware/cmos-basics-for-sta-2" target="_blank">next blog post</a>, we will understand how this delay impacts timing by introducing propagation delay and slew.

### References 

1. Static Timing Analysis for Nanometer Designs: A Practical Approach, Jayaram Bhasker and Rakesh Chadha.
2. Digital Integrated Circuits: A Design Perspective, Jan Rabaey.