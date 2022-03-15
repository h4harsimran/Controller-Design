import numpy as np
import pandas as pd
import math
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

#%% Define Non-linear state equation
def f(X0,U):
    x1,x2,x3,x4=X0
    u10,u20=U
    dx1=r1*k1*u10/A1+a3*(math.sqrt(2*g*x3))/A1-a1*(math.sqrt(2*g*x1))/A1
    dx2=r2*k2*u20/A2+a4*(math.sqrt(2*g*x4))/A2-a2*(math.sqrt(2*g*x2))/A2
    dx3=(1-r2)*k2*u20/A3-a3*(math.sqrt(2*g*x3))/A3 
    dx4=(1-r1)*k1*u10/A4-a4*(math.sqrt(2*g*x4))/A4
    return np.array([dx1,dx2,dx3,dx4])

#%% define euler solver
def sol_NLF(x0,us):
    x0= x0.reshape(4,)
    us = us.reshape(2,)
    x0=x0+dt*f(x0,us)
    return x0

#%% defie inputs and outputs sizes

A=np.ones((4,4))
Ad=A
B=np.ones((4,2))
Bd=B
C=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
dt=0.1


#%% Cost function parameters
Qc1=0.001
Qc2=1
Rc1=0.01
Q=(np.transpose(C)@C+Qc1*np.eye(A.shape[1]))*Qc2
Q[0,0]=10
Q[1,1]=10

R=Rc1*np.eye(B.shape[1])

#%% Define MPC fucntions
def Uconv(U,N,k):
    Unew=np.zeros((N,k))
    for i in range(k):
        Unew[:,i]=U[N*i:N*(i+1)]
    return Unew

def mpc(Q,R,xk,N,umax):
    u_init=0.1*np.zeros((Bd.shape[1]*N)) # intial guess for input trajectory
    u_bounds = ((Us[0]-umax,Us[1]+umax),)*Bd.shape[1]*N  # input bounds
    sol=minimize(mpc_obj,u_init,args=(Q,R,xk,N), method='SLSQP', bounds=u_bounds, options={'eps': 1e-6, 'disp':True})
    U=sol.x
    U=Uconv(U,N,Bd.shape[1])
    uk=U[0,:]
    return uk

def mpc_obj(U, Q, R, xk, N):
    
    U=Uconv(U,N,Bd.shape[1])
    xe=xk-Xs
    J=0.5*np.transpose(xe)@Q@xe
    
    for k in range(N):
        uk = U[k,:].reshape((Bd.shape[1],1))
        xk = sol_NLF(xk,uk)
        xk= np.reshape(xk,(4,1))
        xe=xk-Xs
        ue=uk-Us
        J += 0.5*np.transpose(ue)@R@ue + 0.5*(np.transpose(xe))@Q@xe 
    return J.flatten() 

        
#%% MPC parameters
umax = 3 #input bounds
N = 3 #prediction horizon, change it to get better results with trial and error
## Closed-loop simulation & predictions
tk = np.arange(0,300/dt) # simulate for 35 steps
# each state vector is saved as a 2 by 1 vector - corresponds to the last two indices in x
x = np.zeros((len(tk),Ad.shape[1],1))
#y = np.zeros((len(tk),Cd.shape[0]))
u = np.zeros((len(tk)-1,Bd.shape[1])) # store actually implemented control inputs


#%% Process parameters
r1,r2=[0.7,0.6]
k1,k2=[3.33,3.35]
a1,a2,a3,a4=[0.071,0.057,0.071,0.057]
A1,A2,A3,A4=[28,32,28,32]
x10,x20,x30,x40=[12.4,12.7,1.8,1.4]
u10,u20=[3,3]
Us=np.array([u10,u20]).reshape((2,1))
g=981
X0=np.array([x10,x20,x30,x40])
#%%
Xs=np.array([13,13.5,1.72,1.5]).reshape((4,1)) #set point
x00=X0
x[0,:,:] =x00.reshape((Ad.shape[1],1))  # initial condition
# x[0,:,:] = np.array([[-1],[-0.1]])

#%% MPC solved
for k in range(0,tk.size-1):
    # only the first u is actually implemented
    #u[k,:] = 0 # set u = 0 to generate open-loop trajectory
    xk = x[k,:,:] # current x
    uk = mpc(Q,R,xk,N,umax) # calls MPC function to calculate the optimal uk
    u[k,:] = uk
    xsol = sol_NLF(xk, uk)
    x[k+1,:,:]=xsol.reshape((4,1)) # the state at the next step

#%% Plots
from matplotlib.pyplot import *
figure(1)
plot(tk*dt,x[:,0,:], tk*dt, x[:,1,:])
title('MPC-Non_Linear (States)')
xlabel('Time (sec)')
ylabel('Height (cm)')
legend(['x1', 'x2'],loc='best')

figure(2)
plot(tk*dt,x[:,2,:],tk*dt, x[:,3,:])
title('MPC-Non_Linear (States)')
xlabel('Time (sec)')
ylabel('Height (cm)')
legend(['x3', 'x4'],loc='best')

figure(3)
uforplotting = np.zeros(shape=(tk.size,2))
title('MPC-Non_Linear (Inputs)')
uforplotting[0,:] = u[0,:]
for k in range(1,tk.size):
    uforplotting[k,:] = u[k-1,:]
step(tk*dt,uforplotting)
xlabel('Time (sec)')
ylabel('Pump Voltage (V)')
legend(['u1','u2'])
