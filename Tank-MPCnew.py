import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import cont2discrete

#%% Import Jacobian outputs
df1=pd.read_excel('J_xval.xlsx')

df2=pd.read_excel('J_uval.xlsx')


#%% state equation

A=np.array(df1)
B=np.array(df2)
C=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
D=np.array([[0,0],[0,0]])

#%% coverting to Discrete time model

sysc=(A,B,C,D)
dt=0.1
sysd=cont2discrete(sysc,dt)
Ad,Bd,Cd,Dd=sysd[0:4]


#%% Cost function parameters
Qc1=0.001
Qc2=1
Rc1=0.01
Q=(np.transpose(C)@C+Qc1*np.eye(A.shape[1]))*Qc2
Q[0,0]=10
Q[1,1]=10

R=Rc1*np.eye(B.shape[1])

#%% define MPC functions

def f(Ad,Bd,xk,uk): 
    x = np.zeros(shape=(xk.size,1))
    x=Ad@xk+Bd@uk #state equation in discrete time
    return x

def Uconv(U,N,k):
    Unew=np.zeros((N,k))
    for i in range(k):
        Unew[:,i]=U[N*i:N*(i+1)]
    return Unew

def mpc(Ad,Bd,Q,R,xk,N,umax):
    u_init=0.1*np.zeros((Bd.shape[1]*N)) # intial guess for input trajectory
    u_bounds = ((-umax, umax),)*Bd.shape[1]*N # input bounds
    sol=minimize(mpc_obj,u_init,args=(Ad,Bd,Q,R,xk,N), method='SLSQP', bounds=u_bounds, options={'eps': 1e-6, 'disp':True})
    U=sol.x
    U=Uconv(U,N,Bd.shape[1])
    uk=U[0,:]
    return uk

def mpc_obj(U, Ad, Bd, Q, R, xk, N):
    
    U=Uconv(U,N,Bd.shape[1])

    J=0.5*np.transpose(xk)@Q@xk
    
    for k in range(N):
        uk = U[k,:].reshape((Bd.shape[1],1))
        
        xk = f(Ad,Bd,xk,uk)
        J += 0.5*np.transpose(uk)@R@uk + 0.5*(np.transpose(xk))@Q@xk 
    return J.flatten() 

#%% MPC parameters
umax = 3 #bounds on inputs
N = 8 #Prediction horizon, change it to get better results by trail and error
## Closed-loop simulation & predictions
tk = np.arange(0,300/dt) # simulate for 35 steps
# each state vector is saved as a 2 by 1 vector - corresponds to the last two indices in x
x = np.zeros((len(tk),Ad.shape[1],1))
u = np.zeros((len(tk)-1,Bd.shape[1])) # store actually implemented control inputs

#%% Define process parameters
r1,r2=[0.7,0.6]
k1,k2=[3.33,3.35]
a1,a2,a3,a4=[0.071,0.057,0.071,0.057]
A1,A2,A3,A4=[28,32,28,32]
x10,x20,x30,x40=[12.4,12.7,1.8,1.4]
u10,u20=[3,3]
g=981
X0=[x10,x20,x30,x40]
#%%
#Xs=np.array([13,13.5,1.9,1.5]) 
Xs=np.array([13,13.5,1.72,1.5])#set point
x00=X0-Xs
x[0,:,:] =x00.reshape((Ad.shape[1],1))  # initial condition

#%% solve MPC

for k in range(0,tk.size-1):
    # only the first u is actually implemented
    #u[k,:] = 0 # set u = 0 to generate open-loop trajectory
    xk = x[k,:,:] # current x
    uk = mpc(Ad,Bd,Q,R,xk,N,umax) # calls MPC function to calculate the optimal uk
    
    u[k,:] = uk
    xsol = f(Ad, Bd, xk, uk.reshape((Bd.shape[1],1)))
    x[k+1,:,:]=xsol # the state at the next step
    
#%%% Plots
from matplotlib.pyplot import *
figure(1)
plot(tk*dt,x[:,0,:]+Xs[0], tk*dt, x[:,1,:]+Xs[1])
title('MPC-Linear (States)')
xlabel('Time (sec)')
ylabel('Height (cm)')
legend(['x1', 'x2'],loc='upper right')

figure(2)
plot(tk*dt,x[:,2,:]+Xs[2],tk*dt, x[:,3,:]+Xs[3])
title('MPC-Linear (States)')
xlabel('Time (sec)')
ylabel('Height (cm)')
legend(['x3', 'x4'],loc='lower right')

figure(3)
uforplotting = np.zeros(shape=(tk.size,2))
title('MPC Linear (Inputs)')
uforplotting[0,:] = u[0,:]
for k in range(1,tk.size):
    uforplotting[k,:] = u[k-1,:]+[u10,u20]
step(tk*dt,uforplotting)
xlabel('Time (sec)')
ylabel('Pump Voltage (V)')
legend(['u1','u2'])
