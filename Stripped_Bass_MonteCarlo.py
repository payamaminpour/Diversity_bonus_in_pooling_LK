# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 09:28:06 2019

@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""
import matplotlib.pyplot as plt; plt.rcdefaults()
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
            
    return similarity

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
results["Rec"]=[]
results["Comm"]=[]
results["Mgr"]=[]
results["Crowd"]=[]

for iter in range(30):
    group_rec_fcms,group_comm_fcms,group_mgr_fcms = groupFormation(10,10,10)
    Aggregated_Rec = aggregation(group_rec_fcms)
    Aggregated_Comm = aggregation(group_comm_fcms)
    Aggregated_Mgr = aggregation(group_mgr_fcms)
    
    from statistics import median as med
    Aggregated_Crowd= np.zeros((N_nodes,N_nodes)) 
    for i in range(N_nodes):
        for j in range(N_nodes):
            Aggregated_Crowd[i,j]= med([Aggregated_Rec[i,j], Aggregated_Comm[i,j], Aggregated_Mgr[i,j]])
    
    FCM_Rec = nx.DiGraph(Aggregated_Rec)
    FCM_Comm = nx.DiGraph(Aggregated_Comm)
    FCM_Mgr = nx.DiGraph(Aggregated_Mgr)
    FCM_Crowd = nx.DiGraph(Aggregated_Crowd)
    
    results["Rec"].append(similarity(FCM_Rec,ECOSYM))
    results["Comm"].append(similarity(FCM_Comm,ECOSYM))
    results["Mgr"].append(similarity(FCM_Mgr,ECOSYM))
    results["Crowd"].append(similarity(FCM_Crowd,ECOSYM))

# In[]    
df= pd.DataFrame(results)
df = df[['Comm', 'Rec', 'Mgr', 'Crowd']]
dft=df.transpose()
plt.figure(figsize=(4,4))
dft.plot(kind='line',color='k',alpha=0.02,linewidth=0.01, legend=False, figsize=(4,4))
plt.show()
#plt.savefig("MCA_Results2.pdf")
                 