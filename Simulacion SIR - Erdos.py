# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:47:12 2024

@author: juanc
"""

import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import math

#%%
#Parametros de la red
n0 = 1000 #Número de nodos
k0=6 #Aleatoria - Conectividad promedio

#Parametros de la enfermedad
mu = 0.2

#Parametros de la simulacion
Nsim = 200

R0Max=2.5
R0Min=0.5
N=int((R0Max-R0Min)/0.1)+1
R0 = np.linspace(R0Min, R0Max, N)


#%%
#En esta sección del código se crea la red Erdos-Renyi

def GrafoER(k, n0):
    graph = nx.erdos_renyi_graph(n0,k/(n0-1))
    
    largest_cc=max(nx.connected_components(graph))
    graph=graph.subgraph(largest_cc).copy()
    
    return graph

def Averagedegree(graph):
    grados = dict(nx.degree(graph))

    ImpD =  list(grados.values())
    return np.average(ImpD)

#%%
#Esta función simula el sistema sin vacunados
def Simu(g, t, Var, lamb, n):  #g es el grafo, #Var guarda el error
    
    R = [None]*Nsim #Guarda el número de recuperados al final de la simulación
    
    for i in range(Nsim):
        
        #Inicializo el estado de cada nodo.
        nodos = np.zeros(n0)
        
        #indices_ceros me da todos los nodos del grafo que no están vacunados
        indices_ceros = [l for l, valor in enumerate(nodos) if valor == 0 and g.has_node(l)]
        
        #De los nodos sanos en el grafo, escojo algunos para infectarlos
        aleatorio = np.random.choice(indices_ceros, size=int(n0*0.005), replace=False)
        nodos[aleatorio]=1
        
        #El valor inicial de I no importa.
        I=np.sum(nodos==1)
        
        #Mientras haya 1 o más personas infectadas, se realiza la dinámica del SIR
        while I > 0:
            
            nodos1 = np.copy(nodos)
            
            for j in g.nodes:
                if nodos[j]==0:
                    vecinosI = [l for l in g.neighbors(j) if nodos[l] == 1]
                    for l in vecinosI:
                        if random.random() < lamb:
                            nodos1[j] = 1
                            break
                elif nodos[j]==1:
                    if random.random()< mu:
                        nodos1[j] = 2
                        
            nodos = np.copy(nodos1)
            
            #Cuento el número de infectados tras el paso realizado
            I=np.sum(nodos==1)
        
        R[i] = np.sum(nodos==2)
    
    #Guardo el error en el array Var y devuelvo la media de Infectados total
    Var[t] = np.std(R)/n/math.sqrt(Nsim)
    return np.average(R)/n

#%%
R1 = [None]*N
R2 = [None]*N
R4 = [None]*N


VarR1 = [None]*N
VarR2 = [None]*N
VarR4 = [None]*N


n0=1000
k=6
g=GrafoER(k0, n0)
k=Averagedegree(g)
n=g.number_of_nodes()

for t in range(N):
    lamb=R0[t]*mu/k
    R1[t]=Simu(g, t, VarR1, lamb, n)
    
print('He acabado la de 1000')
    
n0=2000
k=6
g=GrafoER(k0, n0)
k=Averagedegree(g)
n=g.number_of_nodes()

for t in range(N):
    lamb = R0[t]*mu/k
    R2[t] = Simu(g, t, VarR2, lamb, n)
    
print('He acabado la de 2000')    
    
n0=5000
k=6
g=GrafoER(k0, n0)
k=Averagedegree(g)
n=g.number_of_nodes()

for t in range(N):
    lamb = R0[t]*mu/k
    R4[t] = Simu(g, t, VarR4, lamb, n)
    
print('He acabado la de 5000')

R3 = [None]*N
VarR3 = [None]*N
n0=10000
k=6
g=GrafoER(k0, n0)
k=Averagedegree(g)
n=g.number_of_nodes()

for t in range(N):
    lamb=R0[t]*mu/k
    R3[t]=Simu(g, t, VarR3, lamb, n)
    
print('He acabado la de 10000')


#%%
#En esta celda se hacen las gráficas.
fig, ax = plt.subplots()

ax.set_xlabel('$R_0$',fontsize=18, weight='bold')
ax.set_ylabel('$R_{\infty}$',fontsize=18, weight='bold')

plt.xlim(0.49, 2.51)
plt.ylim(-0.01, 0.81)

ax.errorbar(R0, R1, yerr=VarR1,label='N=1000', fmt='o', color='yellow')
ax.errorbar(R0, R2, yerr=VarR2, label = 'N=2000', fmt='o', color='orange')
ax.errorbar(R0, R4, yerr=VarR4, label = 'N=5000', fmt='o', color='blue')
ax.errorbar(R0, R3, yerr=VarR3, label = 'N=10000', fmt='o', color='purple')

ax.legend()

plt.grid(True)

# Mostrar la gráfica
plt.show()
