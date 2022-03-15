import numpy as np
import pandas as pd
from scipy.integrate import odeint
import math
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sympy as sp
from IPython.display import display

#%% Setting Constant values and intial values
r1,r2=[0.7,0.6]
k1,k2=[3.33,3.35]
a1,a2,a3,a4=[0.071,0.057,0.071,0.057]
A1,A2,A3,A4=[28,32,28,32]
x10,x20,x30,x40=[12.4,12.7,1.8,1.4]
u10,u20=[3,3]
g=981
X0=[x10,x20,x30,x40]


#%% define differential equation model
def f(X0,t):
    x1,x2,x3,x4=X0
    dx1=r1*k1*u10/A1+a3*(math.sqrt(2*g*x3))/A1-a1*(math.sqrt(2*g*x1))/A1
    dx2=r2*k2*u20/A2+a4*(math.sqrt(2*g*x4))/A2-a2*(math.sqrt(2*g*x2))/A2
    dx3=(1-r2)*k2*u20/A3-a3*(math.sqrt(2*g*x3))/A3 
    dx4=(1-r1)*k1*u10/A4-a4*(math.sqrt(2*g*x4))/A4
    return [dx1,dx2,dx3,dx4]

#%% create plot values and integrate
tl=500
tlk=100
t=np.linspace(0,tl,tl*tlk+1)
X=odeint(f,X0,t)

#%% for steady state
st=fsolve(f,X0,args=(0))
st_df=pd.DataFrame(st)
st_df.to_excel('st.xlsx', index=False)

#%% plot
plt.figure(1)
plt.subplot(211)
plt.title('Steady state')
plt.plot(t,X[:,0],label='x1')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Height')
plt.subplot(212)
plt.plot(t,X[:,1],label='x2')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Height')


plt.figure(2)
plt.subplot(211)
plt.title('Steady state')
plt.plot(t,X[:,2],label='x3')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Height')
plt.subplot(212)
plt.plot(t,X[:,3],label='x4')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Height')


#%%% Jacobian Calculation
u1,u2=u10,u20
x1,x2,x3,x4=sp.symbols('x1,x2,x3,x4', real=True)
dx1=r1*k1*u1/A1+a3*(sp.sqrt(2*g*x3))/A1-a1*(sp.sqrt(2*g*x1))/A1
dx2=r2*k2*u2/A2+a4*(sp.sqrt(2*g*x4))/A2-a2*(sp.sqrt(2*g*x2))/A2
dx3=(1-r2)*k2*u2/A3-a3*(sp.sqrt(2*g*x3))/A3 
dx4=(1-r1)*k1*u1/A4-a4*(sp.sqrt(2*g*x4))/A4
function_matrix = sp.Matrix([dx1,dx2,dx3,dx4])
J_x=function_matrix.jacobian([x1,x2,x3,x4])
J_xval=np.array(J_x.subs([(x1,st[0]),(x2,st[1]),(x3,st[2]),(x4,st[3])]),np.float)

x1,x2,x3,x4=st
u1,u2=sp.symbols('u1,u2', real=True)
dx1=r1*k1*u1/A1+a3*(sp.sqrt(2*g*x3))/A1-a1*(sp.sqrt(2*g*x1))/A1
dx2=r2*k2*u2/A2+a4*(sp.sqrt(2*g*x4))/A2-a2*(sp.sqrt(2*g*x2))/A2
dx3=(1-r2)*k2*u2/A3-a3*(sp.sqrt(2*g*x3))/A3 
dx4=(1-r1)*k1*u1/A4-a4*(sp.sqrt(2*g*x4))/A4
function_matrix = sp.Matrix([dx1,dx2,dx3,dx4])
J_u=function_matrix.jacobian([u1,u2])
J_uval=np.array(J_u.subs([(u1,u10),(u2,u20)]),np.float)


#%%% export to excel
df1=pd.DataFrame(J_xval)
df1.to_excel('J_xval.xlsx', index=False)
df2=pd.DataFrame(J_uval)
df2.to_excel('J_uval.xlsx', index=False)
#display(J_uval)




