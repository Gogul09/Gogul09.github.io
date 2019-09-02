---
layout: post
category: hardware
class: ASIC Design
title: MOSFET fundamentals
description: Learn the fundamentals of Metal Oxide Semiconductor Field Effect Transistor (MOSFET)
author: Gogul Ilango
permalink: /hardware/mosfet-fundamentals
image: https://drive.google.com/uc?id=1EnLLTeC5eWSy-TPQye-sg9aMndUISJyT
---

<div class="sidebar_tracker" id="sidebar_tracker">
  <button onclick="closeSidebar('sidebar_tracker_content')">X</button>
  <p onclick="showSidebar('sidebar_tracker_content')">Contents</p>
  <ul id="sidebar_tracker_content">
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_1" href="#mosfet-basic-structure">MOSFET-Basic structure</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_2" href="#p-substrate">P-substrate</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_3" href="#isolation-region">Isolation region</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_4" href="#n-diffusion-region">n+ Diffusion region</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_5" href="#sio2-or-metal-oxide">SiO2 or Metal Oxide</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_6" href="#poly-silicon-or-metal-electrode">Poly-silicon or Metal Electrode</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_7" href="#mosfet-working">MOSFET-Working</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_8" href="#mosfet-regions-of-operation">MOSFET-Regions of operation</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_9" href="#1-cutoff-region">1. Cutoff region</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_10" href="#2-linear-or-triode-region">2. Linear or Triode region</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_11" href="#3-saturation-region">3. Saturation region</a></li>
    <li><a class="sidebar_links" onclick="handleSideBarLinks(this.id)" id="link_12" href="#summary">Summary</a></li>
  </ul>
</div>

Smartphone that you hold in your hand has billions of transistors embedded in a single tiny chip. Can you imagine how these tiny transistors are built and packed within a single chip? It is all because of a special component in Electronics called MOSFET (Metal Oxide Semiconductor Field Effect Transistor). What happens inside this tiny little device must be studied to understand how a chip (ASIC) work.

<div class="objectives">
  <h3>Objectives</h3>
  <p>After reading this tutorial, we will understand</p>
  <ul>
    <li>The basic structure of a MOSFET.</li>
    <li>The three main operating regions of a MOSFET.</li>
    <li>How MOSFET behaves as a Voltage-Controlled Current Source?</li>
  </ul>
</div>

### MOSFET-Basic structure

MOSFET is an active electronic device which has the ability to produce current provided a voltage. Think of voltage and current analogous to a water pump.

If water molecules are considered as electrons and the flow of water is considered as current, then how much you open the water pump is considered as voltage, so that the water molecules (coming out of the pump) flows faster or slower. 

> The force or the push that makes electrons flow in a circuit is voltage. 

MOSFET is an active device because it can control the flow of electrons i.e. current. Passive devices such as resistors, capacitors or inductors normally won't do this.

A typical MOSFET that is hidden inside your smartphone processor has a structure shown in Fig 1. Basically, this representation is for our understanding purposes only. In real-world, the fabrication process will look a lot different than this.

<figure>
  <img src="https://drive.google.com/uc?id=1dEYy-z34LpBBa5FPlkqDdGmfv5xwQ9bA"/>
  <figcaption>Figure 1. MOSFET - Basic Structure</figcaption>
</figure>

<br>
As you can see from the above figure, MOSFET is a four terminal device. The four terminals are - 

* **Source**
* **Gate**
* **Drain**
* **Bulk** - Normally, Bulk and Source are connected to each other, and so, in many text books, MOSFET is shown as a **three** terminal device.

Additionally, there are five important parts that combine along with these four terminals in a MOSFET. They are - 

* P-substrate
* Isolation region
* n+ Diffusion region
* Poly-silicon or Metal Electrode
* SiO2 or Metal Oxide

### P-substrate
Every chip is made of Silicon as it is the most abundant material available in this planet. And undoubtedly, p-substrate is nothing but Silicon doped with p-type material such as Boron. Doping means to add impurities to a semi-conductor so that its conducting behaviour is changed. This is mainly done to increase the conductivity of a semi-conductor material such as Silicon or Germanium. One key point to remember here is that, p-substrate mainly have holes as the majority carriers.

<figure>
  <img src="https://drive.google.com/uc?id=1x3gOZlKWb3phTzsCaaLfh3n7ZoZOYMI9"/>
  <figcaption>Figure 2. MOSFET - p-substrate</figcaption>
</figure>

### Isolation region
As your smartphone chip contains millions of transistors sitting next to each other, we need isolation to protect each of these transistors so that there isn't any damage due to short-circuit or EMI. Because of this, we typically have isolation regions in every MOSFET sitting close to each other. Commonly used isolation material is Silicon dioxide (SiO2) which you find in Glass.

<figure>
  <img src="https://drive.google.com/uc?id=1_MGHP9VCHA2Rsfg9jEy9JMQ6kX65g86s" />
  <figcaption>Figure 3. MOSFET - Isolation region</figcaption>
</figure>

### n+ Diffusion region
After providing isolation, we diffuse/add n+ impurities inside the p-type substrate, mainly along the top-left and top-right regions, leaving a defined amount of space in the middle. This is done with the help of lithographic techniques that are specific to the fabrication industry. Mainly, the Source and Drain terminals of a MOSFET are taken from these n+ diffusion regions.

<figure>
  <img src="https://drive.google.com/uc?id=1mW9pAxyHNnwrwfuiC55ZCaAiYy-P5g2-" />
  <figcaption>Figure 4. MOSFET - n+ Diffusion region</figcaption>
</figure>

### SiO2 or Metal Oxide
Above the space left by n+ diffusion regions, we keep a metal-oxide layer normally built using SiO2. This acts as an insulator that does not allow current to pass through. The main reason behind adding this metal-oxide layer will be explained shortly. But, the key take away is that because of this layer, we add Oxide in a MOSFET.

<figure>
  <img src="https://drive.google.com/uc?id=1aIbg4HuJnsUQnirlqT_9AXs3OaGB1vXT" />
  <figcaption>Figure 5. MOSFET - SiO2 or Metal Oxide</figcaption>
</figure>

### Poly-silicon or Metal Electrode
Above the metal-oxide layer, we introduce a metal-layer that will be used to take the **Gate** terminal.

<figure>
  <img src="https://drive.google.com/uc?id=1nAyeTRc8Re_MY_NRqbb4yN2ezTrpXGTx" />
  <figcaption>Figure 6. MOSFET - Poly-silicon or Metal Electrode</figcaption>
</figure>

Thus, in a MOSFET -

* **M** stands for **Metal**, which refers to the **Metal electrode**. 
* **O** stands for **Oxide**, which refers to the **Metal Oxide (SiO2)**.
* **S** stands for **Semi-conductor**, which refers to the **p-substrate**.

### MOSFET-Working

There are two modes of operation in a MOSFET namely -

1. **Depletion** mode - Channel **present** between drain and source.
2. **Enhancement** mode - Channel **absent** between drain and source.

In this tutorial, we will strictly focus on Enhancement mode MOSFET. We will also consider n-channel enhancement mode MOSFET which means we have p-substrate as the body. In contrast, a p-channel enhancement mode MOSFET will have n-substrate as the body.

A n-channel enhancement mode MOSFET acts as a voltage-controlled current source - meaning if you provide voltage at the gate-source terminal, you create a channel between source and drain region that allows current to pass through. The symbols of enhancement mode MOSFET used while working with circuit diagrams is shown in Fig 7.

<figure>
  <img src="https://drive.google.com/uc?id=1HfoBVoYHjx-W2W7zmDM-HKuvsRJEjmeY" />
  <figcaption>Figure 7. MOSFET - Enhancement mode symbols without bulk substrate</figcaption>
</figure>

In short, this type of MOSFET typically works like this - 

> You give me some voltage and I will produce current.

### MOSFET-Regions of operation
Based on the amount of gate-source voltage \\( V_{GS} \\) and drain-source voltage \\( V_{DS} \\) applied, there are three regions of operation in a MOSFET namely -

* Cutoff region
* Linear or Triode region 
* Saturation region

### 1. Cutoff region
If you don't apply any voltage to the gate-source terminal, you don't create any channel from drain to source, and there is no flow of current. This is what cutoff region mean. If \\( V_{GS} \\) (gate-to-source voltage) is 0, then \\( i_D \\) (current flowing from drain to source) is 0. 

<div class="math-cover">
$$
i_D = 0 ; V_{GS} = 0
$$
</div>

In addition, to turn on the MOSFET, we need to cross a minimal amount of voltage. This is typically in the range 0.7 - 1.1 V. 

> The minimum voltage required to turn on the MOSFET is called Threshold Voltage \\(V_T \\)

Thus, in cutoff region of a MOSFET, the voltage between gate and source must be in between 0 and the threshold voltage (\\(V_T \\)).

<div class="math-cover">
$$
0 < V_{GS} < V_T
$$
</div>

### 2. Linear or Triode region
In linear or triode region, the MOSFET creates a channel in between drain to source. So, you must be asking yourself, how this channel is created?

The answer is CAPACITOR. Look at the figure given below.

<figure>
  <img src="https://drive.google.com/uc?id=1HGY2NOrUFJd3vo5uvFGQ0y3tc6sQvWM1" />
  <figcaption>Figure 8. Parallel plate capacitor</figcaption>
</figure>

If you could recall the working of a capacitor (from your high school), you should know that when two parallel metal plates with a dielectric in between are kept in contact with a voltage source, the plates gets charged. If these two plates have same charge (instead of different charge as shown in the figure), then they typically act as magnet, causing repulsion of charges. This is ultimately the principle behind working of a MOSFET.

> The metal electrode and p-substrate are the parallel plates and SiO2 oxide layer is the dielectric which together makes a MOS capacitor. 

If you slightly increase the gate-source voltage \\( V_{GS} \\), the metal electrode gets charged positively. The creation of positive charges in the metal electrode causes the holes (positive charges) present in the space between n+ diffusion regions to repel. This repulsion of positive charges accumulates negative ions towards the semi-conductor surface. If \\( V_{GS} \\) is increased to a point at which the semi-conductor surface becomes negatively charged, then that \\( V_{GS} \\) voltage is called as the threshold voltage. The overall phenomenon is referred as Strong Inversion. 

Thus, by controlling the amount of push that we provide through \\( V_{GS} \\), we essentially control the width of the channel that is created between drain and source. Figure 9 shows the application of gate-source voltage and the corresponding channel creation between drain and source. Note that, we make the source and drain terminals grounded so that the resistance between them is high initially (meaning zero current).

<figure>
  <img src="https://drive.google.com/uc?id=1WXw8P0_yulW0ZW99jTEJIj-FzQYuQoyB" />
  <figcaption>Figure 9. MOSFET - Linear or Triode region</figcaption>
</figure>

The channel created between drain and source allows current to pass through which is governed by the equation shown below in linear region. Note that, we have included the effect of channel length modulation here, which is a second-order effect.

<div class="math-cover">
$$\ i_D = \frac{1}{2} u_n C_{OX} \frac{W}{L} ((V_{GS} - V_T)V_{DS}- \frac{V_{DS}^2}{2})(1 + \lambda V_{DS})$$
</div>

Here, 

* \\(i_D \\) - is the drain current flowing through the channel.
* \\(u_n \\) - is the mobility of electrons in the inversion layer.
* \\(C_{OX} \\) - is the gate-oxide capacitance per unit area.
* \\(\frac{W}{L} \\) - is the ratio of width and length of the semi-conductor.
* \\(V_{GS} \\) - is the gate-source voltage or input voltage applied to the MOSFET.
* \\(V_T \\) - is the threshold voltage of the MOSFET.

Thus, the three most important equations to keep in mind for an n-channel Enhancement mode MOSFET in linear or triode region are -

<div class="math-cover">
$$\ V_{GS} > V_T$$

<br>

$$\ V_{DS} < (V_{GS} - V_T)$$

<br>

$$\ i_D = \frac{1}{2} u_n C_{OX} \frac{W}{L} ((V_{GS} - V_T)V_{DS}- \frac{V_{DS}^2}{2})(1 + \lambda V_{DS})$$
</div>

You might be wondered to look at the second equation. The voltage between drain and source must be lesser than the difference in voltage between source and gate, and the threshold voltage. If not, the MOSFET enters into the saturation region, which means the drain current won't increase further (saturates) even if you apply more \\(V_{GS} \\). This is because of \\(V_{DS} \\) (drain-source voltage) which increases the resistance between drain and source. If this voltage exceeds \\(V_{GS} - V_T \\), then the MOSFET enters the saturation region.

### 3. Saturation region
If we keep on increasing the drain-source voltage \\(V_{DS} \\), the resistance between source and drain gets increased and the channel is pinched off. As a result, the drain current begins to saturate at this point. 

### Why does \\(i_{D} \\) gets saturated if \\(V_{DS} \\) increases?

#### Case 1

When \\(V_{DS} \\) is small, there is presence of electric field \\(E \\) in the channel which gets increased as - 

<div class="math-cover">
$$ E = \frac{V_{DS}}{d} $$
</div>

This increase in electric field increases the drift velocity \\(V_{drift} \\) and current density \\(J \\) of the charge carriers, thereby resulting in rise of drain current \\(i_{D} \\). 

<div class="math-cover">
$$ i_{D} = A n e V_{drift} $$
</div>

where -

* \\(A \\) is the area of the conductor.
* \\(n \\) is the free electron density.
* \\(e \\) is the charge of the electron.
* \\(V_{drift} \\) is the drift velocity.

<div class="math-cover">
$$ \frac{i_{D}}{A} = n e V_{drift} = J $$
</div>

where \\(J \\) is the current density. Also, 

<div class="math-cover">
$$ \sigma = \frac{J}{E} $$
</div>

Thus, we can bring the relation between electric field and drift velocity as - 

<div class="math-cover">
$$ \sigma E = n e V_{drift} $$
</div>

and 

<div class="math-cover">
$$ E = \frac{n e V_{drift}}{\sigma} $$
</div>
The above equations are valid only if mobility of charge carriers (\\(\mu \\)) is maintained constant. This is due to the relation -

<div class="math-cover">
$$ V_{drift} = \mu E $$ 
</div>

Thus, if \\(\mu \\) is constant, then \\(V_{drift} \\) increases with \\(E \\) as they are directly proportional.

#### Case 2

If \\(\mu \\) is not constant, then there are three cases to be considered - 

<div class="math-cover">
$$
\mu =
\begin{cases}
constant,  & \text{if $E$ < $10^3$ } \frac{V}{cm} \\
\text{proportional to} \frac{1}{\sqrt{E}},  & \text{if $10^3$ < $E$ < $10^4$ } \frac{V}{cm} \\
\text{proportional to} \frac{1}{E},  & \text{if $E$ > $10^4$ } \frac{V}{cm}
\end{cases} 
$$
</div>

If \\(V_{DS} \\) **increases so much**, then electric field \\(E \\) **increases** beyond \\(10^4 \frac{V}{cm} \\). Thus, due to the relation - \\(V_{drift} = \mu E \\) and as \\(\mu = \frac{1}{E}\\), even if a **larger** electric field is produced, the mobility of charge carriers gets **decreased**, making the drift velocity **constant**. As drift velocity is **constant**, current density \\(J \\) remain **constant** and thus, the drain current \\(i_D \\) remain **constant**. 

In the saturation region, the equations that run behind the scenes are shown below - 

<div class="math-cover">
$$\ V_{GS} > V_T$$

<br>

$$\ V_{DS} > (V_{GS} - V_T)$$

<br>

$$\ i_D = \frac{1}{2} u_n C_{OX} \frac{W}{L} ((V_{GS} - V_T)^2)(1 + \lambda V_{DS})$$
</div>

All the three operating regions of MOSFET are shown below.

<figure>
  <img src="https://drive.google.com/uc?id=1rgxlmNoz0_xlKPBHLi0vzfQmUX3UUUDd" />
  <figcaption>Figure 10. MOSFET - Regions of operation</figcaption>
</figure>

### Summary

In this tutorial, we have learnt the internal structure of an n-channel enhancement mode MOSFET and understood the different parts of a MOSFET. Then, we saw the working of a MOSFET and the regions of operations of a MOSFET. Thus, we have learnt the fundamentals of a MOSFET.