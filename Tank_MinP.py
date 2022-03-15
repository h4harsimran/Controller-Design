# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:31:26 2021

@author: harsi
"""

import numpy as np
from numpy import sqrt
from scipy.integrate import solve_bvp
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sympy as sp
from IPython.display import display

#%% Process parameters
r1,r2=[0.7,0.6]
k1,k2=[3.33,3.35]
a1,a2,a3,a4=[0.071,0.057,0.071,0.057]
A1,A2,A3,A4=[28,32,28,32]
x10,x20,x30,x40=[12.4,12.7,1.8,1.4]
u10,u20=[3,3]
g=981
X0=[x10,x20,x30,x40]

#%% Control parameters

C=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
Qc1=0.001
Qc2=1
Rc1=0.01
Q=(np.transpose(C)@C+Qc1*np.eye(4))*Qc2
Q[0,0]=10
Q[1,1]=10

R=Rc1*np.eye(2)
#%% 
U0=np.array([u10,u20]).reshape((2,1)) #initial input
Xs=np.array([13,13.5,1.72,1.5]).reshape((4,1)) #set point

#%%% symbolic calculations
t=sp.symbols('t', real=True)
x1,x2,x3,x4,u1,u2,p1,p2,p3,p4=sp.symbols('x1,x2,x3,x4,u1,u2,p1,p2,p3,p4')

dx1=r1*k1*u1/A1+a3*(sp.sqrt(2*g*x3))/A1-a1*(sp.sqrt(2*g*x1))/A1
dx2=r2*k2*u2/A2+a4*(sp.sqrt(2*g*x4))/A2-a2*(sp.sqrt(2*g*x2))/A2
dx3=(1-r2)*k2*u2/A3-a3*(sp.sqrt(2*g*x3))/A3 
dx4=(1-r1)*k1*u1/A4-a4*(sp.sqrt(2*g*x4))/A4
f_m = sp.Matrix([dx1,dx2,dx3,dx4])
x=sp.Matrix([x1,x2,x3,x4])
u=sp.Matrix([u1,u2])
#%%% Hamiltonian calculation
g=np.transpose(x-Xs)@Q@(x-Xs)+np.transpose(u-U0)@R@(u-U0)
p_m = sp.Matrix([p1,p2,p3,p4])
H=g+np.transpose(p_m)@f_m

#%% state and costate equations
p1_s=-sp.diff(H,x1)[0]
p2_s=-sp.diff(H,x2)[0]
p3_s=-sp.diff(H,x3)[0]
p4_s=-sp.diff(H,x4)[0]
u1_s=sp.diff(H,u1)[0]
u2_s=sp.diff(H,u2)[0]

sol_u=sp.solve((u1_s,u2_s),(u1,u2)) #solving for u

#%% subsituting u in state and costate

# dx1=dx1.subs([(u1,sol_u[u1]),(u2,sol_u[u2])])
# dx2=dx2.subs([(u1,sol_u[u1]),(u2,sol_u[u2])])
# dx3=dx3.subs([(u1,sol_u[u1]),(u2,sol_u[u2])])
# dx4=dx4.subs([(u1,sol_u[u1]),(u2,sol_u[u2])])
# p1_s=p1_s.subs([(u1,sol_u[u1]),(u2,sol_u[u2])])
# p2_s=p2_s.subs([(u1,sol_u[u1]),(u2,sol_u[u2])])
# p3_s=p3_s.subs([(u1,sol_u[u1]),(u2,sol_u[u2])])
# p4_s=p4_s.subs([(u1,sol_u[u1]),(u2,sol_u[u2])])

#%% saving subsituted equations for bvp
# df=pd.DataFrame([dx1,dx2,dx3,dx4,p1_s,p2_s,p3_s,p4_s])
df=pd.DataFrame([dx1,dx2,dx3,dx4,p1_s,p2_s,p3_s,p4_s,sol_u[u1],sol_u[u2]])
df.columns=['Func']
df=df['Func']
df=df.astype('str')
#%% define function for BVP 

def f(t,X): 
        
    x1=X[0]
    x2=X[1]
    x3=X[2]
    x4=X[3]
    p1=X[4]
    p2=X[5]
    p3=X[6]
    p4=X[7]
    
    u1=eval(df.iloc[8])
    u2=eval(df.iloc[9])    
    f1=eval(df.iloc[0])
    f2=eval(df.iloc[1])
    f3=eval(df.iloc[2])
    f4=eval(df.iloc[3])
    fp1=eval(df.iloc[4])
    fp2=eval(df.iloc[5])
    fp3=eval(df.iloc[6])
    fp4=eval(df.iloc[7])
    
    
    return np.vstack((f1,f2,f3,f4,fp1,fp2,fp3,fp4))
#%% Boundary conditionbs
def bc(ya,yb):
    return np.array([ya[0]-x10,ya[1]-x20,ya[2]-x30,ya[3]-x40,yb[4],yb[5],yb[6],yb[7]])

#%% BVP solution
tl=300
tlk=10
t=np.linspace(0,tl,tl*tlk+1) #intial mesh
Xa=np.ones((8,t.size)) #initial guess

sol=solve_bvp(f,bc,t,Xa) #BVP solver

#%%optimal input
uf1=sol_u[u1]
uf2=sol_u[u2]
#%%
uopt1=np.zeros(t.size)
uopt1[0]=u10
uopt2=np.zeros(t.size)
uopt2[0]=u20
for k in range(0,t.size-1):
    uopt1[k+1]=uf1.subs([(x1,sol.y[0,k]),(x2,sol.y[1,k]),(x3,sol.y[2,k]),(x4,sol.y[3,k]),(p1,sol.y[4,k]),(p2,sol.y[5,k]),(p3,sol.y[6,k]),(p4,sol.y[7,k])])
    uopt2[k+1]=uf2.subs([(x1,sol.y[0,k]),(x2,sol.y[1,k]),(x3,sol.y[2,k]),(x4,sol.y[3,k]),(p1,sol.y[4,k]),(p2,sol.y[5,k]),(p3,sol.y[6,k]),(p4,sol.y[7,k])])


#%% plot
Xs_plot=np.array([[Xs[0]*np.ones(t.size)],[Xs[1]*np.ones(t.size)],[Xs[2]*np.ones(t.size)],[Xs[3]*np.ones(t.size)]])
plt.figure(1)
plt.subplot(211)
plt.plot(sol.x,sol.y[0,:],t,Xs_plot[0,:].T,'--')
plt.legend(['x1','Setpoint'])
plt.title('Minimum Phase: States')
plt.ylabel('Height (cm)')
plt.subplot(212)
plt.xlabel('Time (sec)')
plt.ylabel('Height (cm)')
plt.plot(sol.x,sol.y[1,:],t,Xs_plot[1,:].T,'--')
plt.legend(['x2','Setpoint'])

plt.figure(2)
plt.subplot(211)
plt.title('Minimum Phase: States')
plt.ylabel('Height (cm)')
plt.plot(sol.x,sol.y[2,:],t,Xs_plot[2,:].T,'--')
plt.legend(['x3','Setpoint'])
plt.subplot(212)
plt.xlabel('Time (sec)')
plt.ylabel('Height (cm)')
plt.plot(sol.x,sol.y[3,:],t,Xs_plot[3,:].T,'--')
plt.legend(['x4','Setpoint'])

plt.figure(3)
plt.subplot(211)
plt.title('Minimum Phase: Co-States')
plt.plot(sol.x,sol.y[4,:])
plt.legend(['p1'])
plt.subplot(212)
plt.xlabel('Time (sec)')
plt.plot(sol.x,sol.y[5,:])
plt.legend(['p2'])

plt.figure(4)
plt.subplot(211)
plt.title('Minimum Phase: Co-States')
plt.plot(sol.x,sol.y[6,:])
plt.legend(['p3'])
plt.subplot(212)
plt.xlabel('Time (sec)')
plt.plot(sol.x,sol.y[7,:])
plt.legend(['p4'])

plt.figure(5)
plt.subplot(211)
plt.title('Minimum Phase: Inputs')
plt.plot(sol.x,uopt1)
plt.legend(['u1'])
plt.subplot(212)
plt.xlabel('Time (sec)')
plt.plot(sol.x,uopt2)
plt.legend(['u2'])
#%%
display(sol.y[:,-1])
