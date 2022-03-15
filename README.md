# Design of an optimal controller for a Multi-Input Multi-output non-linear dynamic system based on LQR, MPC and Minimum Phase implementation.

## Introduction
The project focusses on study of a Multi-Input Multi-output system and design of controller for
set point tracking based on several different control strategies[1]. A four-tank interacting system
with input water provided through two separate pumps has been chosen for the study. The
challenge is to design a controller based on LQR, MPC and minimum phase implementation to
control the water level of the tanks at a designated level.
## System Description
The system consists of four tanks, two pumps, 2 control valves and a reservoir as shown in
Figure 1 and is based on the system introduced by Johanson (2000)[2]. The inputs to the pumps
are the voltages which in conjugation with pump constants define the flow of water supplied by
the respective pumps. The valves provide a mechanism to set a relative flow of water supply to
lower tanks and the upper tanks. The output from the bottom holes of upper tanks also serves
as input to the lower tanks in addition to direct input from pump flow. The bottom tanks drain
into the reservoir. The flow of water through the bottom holes of the tanks is dependent on the
area of the hole and height of water in the respective tanks
![image](https://user-images.githubusercontent.com/25398418/158436108-7b276503-e29d-45b3-b813-0589550935f1.png)

For modelling purposes, the inputs voltage to the pump is described as system input and height

of the tanks is described as system states.

**Inputs**: Pump voltage V1 and V2.

  For modeling purpose: u1= V1 & u2= V2
  
**States:** Height of tanks - h1, h2, h3 and h4

For modeling purpose: x1= h1, x2= h2, x3= h3 & x4= h4

Parameter’s description:

Vi = Voltage input to ith pump

ki = Pump flow constant of ith pump

ri = Valve setting of ith valves

hi = Height of water in ith tank

ai = Area of orifice of ith tank

Ai = Area of ith tank

Based on the principles of mass conservation, a non-linear dynamic model of the system has
been obtained as shown below:
![image](https://user-images.githubusercontent.com/25398418/158437266-1fdb843b-2e3c-4878-94a4-d32ae5c0c82e.png)

The values of the parameters used for the model under study have been tabulated below[2]:

![image](https://user-images.githubusercontent.com/25398418/158437530-6d3ff604-1761-46a7-a502-b729f060dab9.png)

## Tank Jacabian.py manual



