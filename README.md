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

## Python files explained.

**1. Tank jacobian.py:** To define model parameters, model function, perform steady state analysis, Calcualte Jacobian and export to excel files.

**2. Tank-LQR.py:** Import Jacobian outputs, define State equation, Define LQR parameters (Use const function parameters to tune the controller response with trial and error), Solve Riccati equation to calculate gain K, Silmulate Linear and Non linear LQR controller.

**3. Tank-MPCnew.py:** Import Jacobian outputs, define State equation, Convert to Discrete time model (Use const function parameters to tune the controller response with trial and error), Define MPC function, perform closed loop simulation using MPC controller.

**4. Tank_MPC_NL.py:** Define Non-linear Function (Use const function parameters to tune the controller response with trial and error), Define MPC function, perform closed loop simulation using Non linear MPC controller.

**5. Tank_MinP.py:** Define process parameters, define control parameters (Use const function parameters to tune the controller response with trial and error), use symbolic calculation to solve Minimum phase equations and get State and costate equation, define function for Boundary value problem and Boundary conditions, Solve boundary Value problem to get optimal controller solutions with Minimum Phase implementation.


