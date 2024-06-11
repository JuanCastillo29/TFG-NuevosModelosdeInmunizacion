# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:48:24 2024

@author: juanc
"""
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import random
import math

#%%

n0 = 5000 #Número de nodos inicial
k=6 # Conectividad media deseada


#Parametros de la enfermedad
mu = 0.2

#Parametros de la simulacion
Nsim = 200 #Numero de veces que se realiza cada simulación


#%%

def Domirank(g, p):
    t = 1 #Valor de theta (es irrelevante)
    
    A = nx.to_numpy_array(g) #A es la matriz de adyacencia
    Eigenvalues = LA.eigvals(A)
    s = -p/min(Eigenvalues) #Valor de sigma

    B = LA.inv(s*A + np.identity(n))
    DomiRank = t*s*np.matmul(np.matmul(B, A), np.ones((n, 1)))
    return np.real(np.array(DomiRank))

#%%

#Esta función simula el sistema sin vacunados
def Simu(g, t, Var, N):  
    
    R = [None]*Nsim #Guarda el número de recuperados al final de la simulación
   
    #Valor de lambda. Esta puesto de esta forma por comodidad para la gráfica de evolución de infectados frente a R_0
    lamb=t/N 
    
    for i in range(Nsim):
        
        #Inicializo el estado de cada nodo.
        nodos = np.zeros(n0)
        
        #indices_ceros me da todos los nodos del grafo que no están vacunados
        indices_ceros = [l for l, valor in enumerate(nodos) if valor == 0 and g.has_node(l)]
        
        #De los nodos sanos en el grafo, escojo algunos para infectarlos
        aleatorio = np.random.choice(indices_ceros, size=n0*0.005, replace=False)
        nodos[aleatorio]=1
        
        #El valor inicial de I no importa.
        I=1
        
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


#Esta función simula el sistema con vacunados
def SimuVac(g, Importancia, p, t, IVar, n, lamb):   
    
    R = [None]*Nsim
    
    for i in range(Nsim):
        
        #Inicializo el estado de cada nodo.
        nodos = np.zeros(n0)
        
        #Si hay que vacunar algún nodo:
        if p != 0:
            #Escojo los nodos a vacunar
            vacunados = Vacunar(g, p, Importancia)
            for j in vacunados:
                nodos[j] = 3 #El estado 3 indica que el nodo está vacunado
        
        indices_ceros = [l for l, valor in enumerate(nodos) if valor == 0 and g.has_node(l)]
       
        #De los nodos sanos en el grafo, escojo algunos para infectarlos 
        aleatorio = np.random.choice(indices_ceros, size=int(n0*0.005), replace=False)
        nodos[aleatorio]=1
        
        #El valor de I inicial no importa
        I=1
        
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
            I=np.sum(nodos==1)
        
        R[i] = np.sum(nodos==2)
    

    IVar[t] = np.std(R)/math.sqrt(Nsim)/n
    return np.average(R)/n

#Esta función me da los índices correspondientes a los valores más altos de un array
def p_valores_mas_altos(array, p):
    ind = np.argpartition(array, -p)[-p:]
    return ind

#Esta función me dice que nodos tengo que vacunar según la métrica dada
def Vacunar (g, p, Importancia):
    Modificar = np.zeros(n0)
    i=0
    for j in range(0,n0):
        if j not in g.nodes:
            i = i+1
        Modificar[j] = Modificar[j]+i
        
    nodos = p_valores_mas_altos(Importancia, p)
    for j in range(0,p):
        nodos[j] = nodos[j] + Modificar[int(nodos[j])]
        
    return nodos



#%%
#En esta sección del código analizo la ev del número de infectados al cambiar lambda
def Explore(graph, I, k, n, R0, N):
    Infectados = [None]*N
    InfectadosVar = [None]*N
    for t in range(0, N):
        lamb = R0[t]*mu/k
        Infectados[t] = SimuVac(graph, I, int(0.05*n), t, InfectadosVar, n, lamb)
    
    return Infectados
 

#%%

graph= nx.erdos_renyi_graph(n0,k/(n0-1))
largest_cc=max(nx.connected_components(graph))
graph=graph.subgraph(largest_cc).copy()
grados = dict(nx.degree(graph))

# N=int((2.5-0.5)/0.1)+1
# R0 = np.linspace(0.5, 2.5, N)

ImpD =  list(grados.values())
k = np.average(ImpD) #Guardo la conectividad media real del grafo
n = graph.number_of_nodes()

N=20

# R1 = Explore(graph, ImpD, k, n, R0, N)

# s=0.90
# Imp = np.squeeze(Domirank(graph, s))
# R3 = Explore(graph, Imp, k, n, R0, N)

ImpR = np.random.rand(n)
# R = Explore(graph, ImpR, k, n, R0, N)

# s=0.05
# Imp = np.squeeze(Domirank(graph, s))
# R2 =  Explore(graph, Imp, k, n, R0, N)
R1 = [None]*N
R = [None]*N #Este es degree
R2 = [None]*N
R3 = [None]*N
R1Var = [None]*N
RVar = [None]*N #Este es degree
R2Var = [None]*N
R3Var = [None]*N

# fig, ax = plt.subplots()
# plt.xlim(0.49, 2.51)  
# plt.ylim(-0.01, 0.7)    
# ax.set_xlabel('$R_0$', fontsize=18,weight='bold')
# ax.set_ylabel('$R_{\infty}$', fontsize=18, weight='bold')
# plt.grid(True, alpha=0.5)
# # Añadir un recuadro con texto
# texto_recuadro = 'Red Erdös-Renyi \nNº de nodos: {}\nConectividad media: {:.1f}\n 5% vacunados'.format( n, k)
# ax.text(1, 0.35, texto_recuadro, ha='center', va='center', fontsize=11, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))



# ax.errorbar(R0, R, marker='o', linestyle='None',label='Aleatorio', color='red')
# ax.errorbar(R0, R1, marker='o', linestyle='None',label='Grado', color='orange')
# ax.errorbar(R0, R2, marker='o', linestyle='None', label='DomiRank - $\sigma$ bajo', color='blue')
# ax.errorbar(R0, R3, marker='o', linestyle='None', label='DomiRank - $\sigma$ alto', color='purple')

# ax.legend()
# plt.show()  



lamb= 3*mu/k





#%%
#Simulación de vacunados con degree
p = np.linspace(0, n-n/N, N).astype(int)
t=0
for l in p:
    R[t]=SimuVac(graph, ImpD, l, t, RVar, n, lamb)
    print(l)
    t=t+1

#%%
#Simulación de vacunados con DomiRank alto
s=0.90
Imp = np.squeeze(Domirank(graph, s))
t=0
for l in p:
    R1[t]=SimuVac(graph, Imp, l, t, R1Var, n, lamb)
    print(l)
    t=t+1
#%%
#Simulación vacunados aleatorios
ImpR = np.random.rand(n)
t=0
for l in p:
    R2[t]=SimuVac(graph, ImpR, l, t, R2Var,n, lamb)
    print(l)
    t=t+1

#%%
#Simulación de vacunados con DomiRank
s=0.05
Imp = np.squeeze(Domirank(graph, s))
t=0
for l in p:
    R3[t]=SimuVac(graph, Imp, l, t, R3Var, n, lamb)
    print(l)
    t=t+1

#%%
fig, ax = plt.subplots()
ax.set_xlabel('V', fontsize=18, weight='bold')
ax.set_ylabel('$R_{\infty}(V)$/$R_{\infty}(0)$',fontsize=14, weight='bold')

plt.xlim(-0.01, 1.01)
plt.ylim(-0.01, 1.02)

ax.errorbar(p/n, R2/R2[0], yerr=R2Var, fmt='o', label='Aleatorio', color='red')
ax.errorbar(p/n, R/R[0], yerr=RVar, fmt='o', label='Grado', color='orange')


ax.errorbar(p/n, R3/R3[0], yerr=R1Var, fmt='o', label='DomiRank - $\sigma$ bajo', color='blue')
ax.errorbar(p/n, R1/R1[0], yerr=R3Var, fmt='o', label='DomiRank - $\sigma$ alto', color='purple')

ax.legend()
plt.grid(True)
# Añadir un recuadro con texto
texto_recuadro = 'Red Erdös-Renyi\nConectividad media: {:.1f}\nNº de nodos: {}\n$\lambda/\mu$: {:.2f} '.format(k, n, lamb/mu)
ax.text(0.75, 0.45, texto_recuadro, ha='center', va='center', fontsize=12, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

# Mostrar la gráfica
plt.show()

