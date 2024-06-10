# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:24:51 2024

@author: juanc
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

Nsteps = 100000
Tmax = 200
mu = 0.2

def g(r0, s, i):
    return -r0*mu*s*i

def f(r0, s, i):
    return mu*i*(r0*s - 1)

def Ruggen(R0, S, I):
    h = Tmax/Nsteps
    S[0]=0.999999999
    I[0] = 1 - S[0]
    for j in range(Nsteps-1):
        k0= h*g(R0, S[j], I[j])
        l0 = h*f(R0, S[j], I[j])
        
        k1 = h*g(R0, S[j] + 0.5*k0, I[j] + 0.5*l0)
        l1 = h*f(R0, S[j] + 0.5*k0, I[j] + 0.5*l0)
        
        k2 = h*g(R0, S[j] + 0.5*k1, I[j] + 0.5*l1)
        l2 = h*f(R0, S[j] + 0.5*k1, I[j] + 0.5*l1)
        
        k3 = h*g(R0, S[j] + k2, I[j] + l2)
        l3 = h*f(R0, S[j] + k2, I[j] + l2)
        
        S[j+1]= S[j] + (k0 + 2*k1 + 2*k2 + k3)/6
        I[j+1]= I[j] + (l0 + 2*l1 + 2*l2 + l3)/6
        
S2 = [None]*Nsteps
I2 = [None]*Nsteps

S3 = [None]*Nsteps
I3 = [None]*Nsteps

S5 = [None]*Nsteps
I5 = [None]*Nsteps

Ruggen(2, S2, I2)

Ruggen(3, S3, I3)

Ruggen(5, S5, I5)

#En esta celda se hacen las gráficas.
fig, ax = plt.subplots()
ax.set_xlabel('S', weight='bold')
ax.set_ylabel('I', weight='bold')
plt.grid(True, alpha=0.5)
plt.plot(S2, I2, linestyle='solid',label='R_0 = 2', color = 'black')
plt.plot(S3, I3, linestyle='dashed', label='R_0 = 3', color = 'black')
plt.plot(S5, I5,linestyle='dashdot', label='R_0 = 5', color = 'black')
ax.legend()

t = np.linspace(0, Tmax, Nsteps)
    

fig, ax = plt.subplots()
ax.set_xlabel('t', weight='bold')
ax.set_ylabel('I', weight='bold')
plt.grid(True, alpha=0.5)
plt.plot(t, I2, linestyle='solid',label='R_0 = 2', color = 'black')
plt.plot(t, I3, linestyle='dashed', label='R_0 = 3', color = 'black')
plt.plot(t, I5,linestyle='dashdot', label='R_0 = 5', color = 'black')
ax.legend()


I = np.ones(Nsteps)
fig, ax = plt.subplots()
ax.set_xlabel('t', weight='bold')
ax.set_ylabel('Fración', weight='bold')
plt.grid(True, alpha=0.5)
plt.plot(t, I5, linestyle='dashed',label='Infectados', color = 'green')
plt.plot(t, S5, linestyle='dashed', label='Susceptibles', color = 'red')
plt.plot(t, I - S5 - I5,linestyle='dashed', label='Recuperados', color = 'blue')
ax.legend()


#%%
#En esta sección del código se explora la transición en función de R0.
lamb = np.linspace(0.7, 1.7,21)
R = [None]*21

Tmax = 10000

S= [None]*Nsteps
I=[None]*Nsteps

for j in range(21):
    Ruggen(lamb[j], S, I)
    R[j] = 1 - S[Nsteps-1] - I[Nsteps-1]

fig, ax = plt.subplots()
texto_x='$R_0$'
texto_y = '$R_{\infty}$'
ax.set_xlabel(texto_x,fontsize=18,weight='bold')
ax.set_ylabel(texto_y,fontsize=18, weight='bold')
plt.xlim(0.69, 1.71)  
plt.ylim(-0.01, 0.71)   
plt.plot(lamb, R, linestyle='dashed', color = 'black')

plt.grid(True)

# Mostrar la gráfica
plt.show()