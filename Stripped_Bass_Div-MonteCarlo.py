# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:28:06 2019

@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd


from Stripped_Bass_Models import Nodes_ID
from Stripped_Bass_Models import Group_FCM
from Stripped_Bass_Models import All_agents
from Stripped_Bass_Models import Group_agents
from Stripped_Bass_Models import groups
from Stripped_Bass_Models import list_NonZero_Nodes
from Stripped_Bass_Models import N_nodes


    


# In[]

#from itertools import permutations
from itertools import combinations


# In[]
def similarity (g1,g2):
    ''' how similar the FCM is to the FCM Reference'''   
    def select_k(spectrum, minimum_energy = 0.9):
        running_total = 0.0
        total = sum(spectrum)
        if total == 0.0:
            return len(spectrum)
        for i in range(len(spectrum)):
            running_total += spectrum[i]
            if running_total / total >= minimum_energy:
                return i + 1
        return len(spectrum)
    
    laplacian1 = nx.spectrum.laplacian_spectrum(g1.to_undirected(),weight=None)
    laplacian2 = nx.spectrum.laplacian_spectrum(g2.to_undirected(),weight=None)
    k1 = select_k(laplacian1)
    k2 = select_k(laplacian2)
    k = min(k1, k2)
    similarity = sum((laplacian1[:k] - laplacian2[:k])**2)
            
    return 1/similarity*100

# In[]
def similarity_edges (g1,g2):
    ''' how similar the FCM is to the FCM Reference in terms of edge_list'''   
    
    s1 = set(list(g1.edges()))
    s2 = set(list(g2.edges()))  
    repeated_edges = set.intersection(s1,s2)
    uni_edges = set.union(s1,s2)
    C = len(repeated_edges)/len(uni_edges)
    
            
    return C

# In[]
Rec= Group_agents["Recreational"]
Comm = Group_agents["Commercial"]
Mgr = Group_agents["Manager"]

ECOSYM = Group_FCM["ECOSYSTEM-BASED-MODEL"]
# In[]
dic_rec ={}
for i in Nodes_ID:
    for j in Nodes_ID:
        e=(i,j)
        f = 0
        w = []
        for ag in Rec:
            if e in ag.FCM.edges():
                f += 1
                w.append(ag.FCM[i][j]['weight'])
            
        dic_rec[e]=(f/len(Rec),w)

dic_comm ={}
for i in Nodes_ID:
    for j in Nodes_ID:
        e=(i,j)
        f = 0
        w = []
        for ag in Comm:
            if e in ag.FCM.edges():
                f += 1
                w.append(ag.FCM[i][j]['weight'])
            
        dic_comm[e]=(f/len(Comm),w)

dic_mgr ={}
for i in Nodes_ID:
    for j in Nodes_ID:
        e=(i,j)
        f = 0
        w = []
        for ag in Mgr:
            if e in ag.FCM.edges():
                f += 1
                w.append(ag.FCM[i][j]['weight'])
            
        dic_mgr[e]=(f/len(Mgr),w)
# In[]
import random
def RandomGraph(g):
    if g == "Rec":
        G = nx.DiGraph()
        for e in dic_rec:
            if random.uniform(0, 1) <= dic_rec[e][0]:
                mu = np.array(dic_rec[e][1]).mean()
                sigma =  np.array(dic_rec[e][1]).std()
                w = np.random.normal(mu, sigma, 1)
                G.add_edge(*e, weight=float(w))
            else:
                G.add_edge(*e, weight=0)
    
    if g == "Comm":
        G = nx.DiGraph()
        for e in dic_comm:
            if random.uniform(0, 1) <= dic_comm[e][0]:
                mu = np.array(dic_comm[e][1]).mean()
                sigma =  np.array(dic_comm[e][1]).std()
                w = np.random.normal(mu, sigma, 1)
                G.add_edge(*e, weight=float(w))
            else:
                G.add_edge(*e, weight=0)
    
    if g == "Mgr":
        G = nx.DiGraph()
        for e in dic_mgr:
            if random.uniform(0, 1) <= dic_mgr[e][0]:
                mu = np.array(dic_mgr[e][1]).mean()
                sigma =  np.array(dic_mgr[e][1]).std()
                w = np.random.normal(mu, sigma, 1)
                G.add_edge(*e, weight=float(w))
            else:
                G.add_edge(*e, weight=0)
    return G
# In[]
def groupFormation(N_rec, N_comm, N_mgr):
    group_rec_fcms=[]
    group_comm_fcms=[]
    group_mgr_fcms=[]
    
    for iter in range(N_rec):
        group_rec_fcms.append(RandomGraph("Rec"))
    for iter in range(N_comm):
        group_comm_fcms.append(RandomGraph("Comm"))
    for iter in range(N_rec):
        group_mgr_fcms.append(RandomGraph("Mgr"))
        
    return (group_rec_fcms,group_comm_fcms,group_mgr_fcms)

# In[]:
n_concepts = N_nodes
def aggregation(fcm_list):
       
    adj_ag=np.zeros((n_concepts,n_concepts))
    All_ADJs = []
    for fcm in fcm_list:
        All_ADJs.append(nx.to_numpy_matrix(fcm))
    from statistics import mean
    for i in range (n_concepts):
        for j in range (n_concepts):
            a = [ind[i,j] for ind in All_ADJs if ind[i,j]!=0]
            if len(a)!=0:
                adj_ag[i,j] = mean(a)
    aggregated_FCM = adj_ag   
 
    return aggregated_FCM

# In[]
results = {}
results["Div"]=[]
results["Crowd"]=[]
results["Crowd_edge"]=[]
results["Size"]=[]

for iter in range(10000):
    
    N = np.random.randint(5,100,1)[0]
    R = np.random.randint(1,N,1)[0]
    C = np.random.randint(1,N+1-R,1)[0]
    M = N-R-C
    
    r = -(R/N)*np.log(R/N)
    c = -(C/N)*np.log(C/N)
    m = -(M/N)*np.log(M/N)
    
    d = r+c+m
    
    group_rec_fcms,group_comm_fcms,group_mgr_fcms = groupFormation(R,C,M)
    Aggregated_Rec = aggregation(group_rec_fcms)
    Aggregated_Comm = aggregation(group_comm_fcms)
    Aggregated_Mgr = aggregation(group_mgr_fcms)
    
    from statistics import median as med
    from statistics import mean as mean
    Aggregated_Crowd= np.zeros((N_nodes,N_nodes)) 
    for i in range(N_nodes):
        for j in range(N_nodes):
            Aggregated_Crowd[i,j]= med([Aggregated_Rec[i,j], Aggregated_Comm[i,j], Aggregated_Mgr[i,j]])
            #Aggregated_Crowd[i,j] = mean([Aggregated_Rec[i,j], Aggregated_Comm[i,j], Aggregated_Mgr[i,j]])

    FCM_Crowd = nx.DiGraph(Aggregated_Crowd)
    

    results["Div"].append(d)
    results["Size"].append(N)
    results["Crowd_edge"].append(similarity_edges(FCM_Crowd,ECOSYM))
    results["Crowd"].append(similarity(FCM_Crowd,ECOSYM))

# In[]    

df= pd.DataFrame(results)
df=df.dropna()




# In[]
import seaborn as sns

x = df['Div']/1.5  # for normalization 
#x = df['Size']  # to see the effect of group size

y = df['Crowd']*df['Crowd_edge']


sns.set_style("white")
fig, ax = plt.subplots(figsize=(3,3))
sns.regplot(x, y, scatter=True, fit_reg=False, ci=99.99, order=1, x_jitter=1,
            color='maroon', scatter_kws={"s": 15,"alpha":0.05,"edgecolors":"none"}, line_kws= {"lw":2,"ls":"-","c":"g"}, marker='o', ax=ax)


ax.tick_params(direction='out', length=3, width=1, colors='k',labelsize=9)

#plt.xlim(0,100)
plt.ylim(-0.1,1.1)
plt.xlabel("Diversity",fontsize=10)
#plt.xlabel("Crowd Size",fontsize=10)
plt.ylabel('Performance',fontsize=10)

plt.show()
#plt.savefig("Size4(N-equal)(JustEdges Mean)-MCA_Results.pdf")