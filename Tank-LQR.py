import numpy as np
import pandas as pd
from scipy.linalg import solve_continuous_are
import math
import matplotlib.pyplot as plt

df1=pd.read_excel('J_xval.xlsx')

df2=pd.read_excel('J_uval.xlsx')


#%% state equation

A=np.array(df1)
B=np.array(df2)
C=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
D=np.array([[0,0],[0,0]])

#%% LQR weight parameters

Qc1=0.001
Qc2=1
Rc1=0.01
Q=(np.transpose(C)@C+Qc1*np.eye(A.shape[1]))*Qc2
Q[0,0]=10
Q[1,1]=10

R=Rc1*np.eye(B.shape[1])

#%% riccati equation

P=solve_continuous_are(A,B,Q,R)
eig_P=np.linalg.eig(P)

#%% gain K
Klqr=-np.linalg.inv(R)@np.transpose(B)@P

#%% Process parameters

r1,r2=[0.7,0.6]
k1,k2=[3.33,3.35]
a1,a2,a3,a4=[0.071,0.057,0.071,0.057]
A1,A2,A3,A4=[28,32,28,32]
x10,x20,x30,x40=[12.4,12.7,1.8,1.4]
u10,u20=[3,3]
g=981
X0=[x10,x20,x30,x40]

#%% Simulation intialization

tl=300
tlk=100
t=np.linspace(0,tl,tl*tlk+1) 
X=np.zeros((len(t),A.shape[1]))
X[0,:]=X0
Xs=np.array([13,13.5,1.72,1.5]) #set point


Us=np.array([u10,u20])
U=np.zeros((len(t),2))
U[0,:]=[u10,u20]

x1=np.zeros(len(t))
x2=np.zeros(len(t))
x3=np.zeros(len(t))
x4=np.zeros(len(t))
u1=np.zeros(len(t)-1)
u2=np.zeros(len(t)-1)
deltaT=1/tlk

#%% Non linear LQR Simulation


def f(X0):
    x1,x2,x3,x4=X0
    dx1=r1*k1*u10/A1+a3*(math.sqrt(2*g*x3))/A1-a1*(math.sqrt(2*g*x1))/A1
    dx2=r2*k2*u20/A2+a4*(math.sqrt(2*g*x4))/A2-a2*(math.sqrt(2*g*x2))/A2
    dx3=(1-r2)*k2*u20/A3-a3*(math.sqrt(2*g*x3))/A3 
    dx4=(1-r1)*k1*u10/A4-a4*(math.sqrt(2*g*x4))/A4
    return np.array([dx1,dx2,dx3,dx4])
    
for i in range(len(t)-1):
    u10,u20=U[i,:]
    X[i+1,:]=X[i,:]+deltaT*f(X[i,:])   
    U[i+1,:]=Us+Klqr@(X[i+1,:]-Xs)  

Xs_plot=np.array([[Xs[0]*np.ones(t.size)],[Xs[1]*np.ones(t.size)],[Xs[2]*np.ones(t.size)],[Xs[3]*np.ones(t.size)]])


#%% Linear LQR simulation
Alqr=A+B@Klqr

Xl=np.zeros((len(t),4))
Xl[0,:]=X0
for k in range(len(t)-1):
    Xl[k+1]=Xl[k]+(Alqr@(Xl[k,:]-Xs))*deltaT
    
#%% Plots  
plt.figure(1)
plt.subplot(211)
plt.title('LQR')
plt.plot(t,X[:,0],t,Xl[:,0],t,Xs_plot[0,:].T,'--')
plt.legend(['x1-Non_Linear','x1-Linear','Setpoint'],loc='lower right',framealpha=0,fontsize='small')
plt.ylabel('Height (cm)')
plt.subplot(212)
plt.plot(t,X[:,1],t,Xl[:,1],t,Xs_plot[1,:].T,'--')
plt.legend(['x2-Non_Linear','x2-Linear','Setpoint'],loc='lower right',framealpha=0,fontsize='small')
plt.xlabel('Time (sec)')
plt.ylabel('Height (cm)')

plt.figure(2)
plt.subplot(211)
plt.title('LQR')
plt.plot(t,X[:,2],t,Xl[:,2],t,Xs_plot[2,:].T,'--')
plt.legend(['x3-Non_Linear','x3-Linear','Setpoint'],loc='upper right',framealpha=0,fontsize='small')
plt.ylabel('Height (cm)')
plt.subplot(212)
plt.plot(t,X[:,3],t,Xl[:,3],t,Xs_plot[3,:].T,'--')
plt.legend(['x4-Non_Linear','x4-Linear','Setpoint'],loc='upper right',framealpha=0,fontsize='small')
plt.xlabel('Time (sec)')
plt.ylabel('Height (cm)')

