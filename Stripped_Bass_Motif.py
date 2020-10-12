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


from Stripped_Bass_Models import Category
from Stripped_Bass_Models import Nodes_ID as Nodes_dic_id_name
from Stripped_Bass_Models import Group_FCM
from Stripped_Bass_Models import list_NonZero_Nodes


# In[]

#Counting loops
def dfs(graph, start, end):
    fringe = [(start, [])]
    while fringe:
        state, path = fringe.pop()
        if path and state == end:
            yield path
            continue
        for next_state in graph[state]:
            if next_state in path:
                continue
            fringe.append((next_state, path+[next_state]))

Loop_group_dic = {}

for g in Group_FCM:
    Loop_group_dic[g] = list(nx.simple_cycles(Group_FCM[g]))
    #graph = nx.to_dict_of_lists(Group_FCM[g],nodelist=None)
    #Loop_group_dic[g] = [[node]+path  for node in graph for path in dfs(graph, node, node)]
    

# In[]

Loop_group_dic3 = {}
Loop_group_dic2 = {}
SES_Loop_group_dic3 = {}
SES_Loop_group_dic2 = {}

for g in Loop_group_dic:
    L3 = [L for L in Loop_group_dic[g] if len (L) ==3]
    L2 = [L for L in Loop_group_dic[g] if len (L) ==2]

    Loop_group_dic3[g] = L3
    Loop_group_dic2[g] = L2
        
for g in Loop_group_dic:
    SES_L3 = []
    SES_L2 = []
    for loop in Loop_group_dic3[g]:
        Lcat = [Category[Nodes_dic_id_name[n]] for n in loop]
        if 'Social' in Lcat and 'Biological' in Lcat:
            SES_L3.append(loop)
    SES_Loop_group_dic3[g] = SES_L3
    
    for loop in Loop_group_dic2[g]:
        Lcat = [Category[Nodes_dic_id_name[n]] for n in loop]
        if 'Social' in Lcat and 'Biological' in Lcat:
            SES_L2.append(loop)
    SES_Loop_group_dic2[g] = SES_L2
    
# In[]

d = {}
N = {}
C = {}
for g in list_NonZero_Nodes:
    N[g] = len(list_NonZero_Nodes[g])
    C[g] = len(Group_FCM[g].edges())
    d[g] = C[g]/(N[g]*(N[g]-1))

# In[]

#Counting Micro Motif



from itertools import permutations
from itertools import combinations

def count_loops(G,nodes,L_len):
    
    
    All_Loops3 = []
    comb3nodes = list(combinations(nodes,L_len))
    
    for c3 in comb3nodes:
    
        perm = list(permutations(c3,2))
        
        bb = [e for e in perm if e in G.edges()]
        G_b = nx.DiGraph(bb)
        Loops= list(nx.simple_cycles(G_b))
        L3 = [L for L in Loops if len (L) ==L_len]
        
        All_Loops3.extend(L3)
        
    return All_Loops3
    

#__________________________________________________________


def count_multiple_effects (G,nodes,L_len):
    
    All_multEff = 0
    comb3nodes = list(combinations(nodes,L_len))
    
    for c3 in comb3nodes:
    
        perm = list(permutations(c3,2))
        
        bb = [e for e in perm if e in G.edges()]
        G_b = nx.DiGraph(bb)
        if len(G_b.nodes()) >= 3:
            for n in [0,1,2]:
                if (list(G_b.nodes())[n],list(G_b.nodes())[n+1-3]) in G_b.edges()\
                and (list(G_b.nodes())[n],list(G_b.nodes())[n+2-3]) in G_b.edges()\
                and (list(G_b.nodes())[n+2-3],list(G_b.nodes())[n+1-3]) not in G_b.edges()\
                and (list(G_b.nodes())[n+1-3],list(G_b.nodes())[n+2-3]) not in G_b.edges():
                
                    All_multEff+=1
        
    return All_multEff

#__________________________________________________________


def count_Indirect_effects (G,nodes,L_len):
    
    All_indEff = 0
    comb3nodes = list(combinations(nodes,L_len))
    
    for c3 in comb3nodes:
    
        perm = list(permutations(c3,2))
        
        bb = [e for e in perm if e in G.edges()]
        G_b = nx.DiGraph(bb)
        if len(G_b.nodes()) >= 3:
            for n in [0,1,2]:
                if (list(G_b.nodes())[n],list(G_b.nodes())[n+1-3]) in G_b.edges()\
                and (list(G_b.nodes())[n+1-3],list(G_b.nodes())[n+2-3]) in G_b.edges()\
                and (list(G_b.nodes())[n+2-3],list(G_b.nodes())[n]) not in G_b.edges()\
                and (list(G_b.nodes())[n],list(G_b.nodes())[n+2-3]) not in G_b.edges():
                
                    All_indEff+=1
        
    return All_indEff
# In[]   

 
# Raandom Graph
for g in ['crowd_MEAN_MED','Manager','Commercial','Recreational']:
    c3 = []
    c2 = []
    #c = []
    for iteration in range (1):
        
        G_rand = nx.gnp_random_graph(N[g], d[g], seed=None, directed=True)
        
#        adj_matrix = np.random.rand(N[g],N[g])
#        adj_matrix[adj_matrix >= (1-d[g])] = 1 
#        adj_matrix[adj_matrix < (1-d[g])] = 0
#        #for i in range(N[g]):
#        #    for j in range(N[g]):
#         #       if i==j:
#          #          adj_matrix[i,j] = 0
        
#        G_rand =nx.DiGraph(adj_matrix) 
        #cycles = list(nx.simple_cycles(G_rand))
        #cycles3 = [L for L in cycles if len(L)==3]
        cycles3 = count_loops(G_rand,G_rand.nodes(),3)
        #cycles2 = [L for L in cycles if len(L)==2]
        cycles2 = count_loops(G_rand,G_rand.nodes(),2)
        #print (len(SES_Loop_group_dic3[g])/len(cycles3))
        c3.append(len(cycles3))
        c2.append(len(cycles2))
        #c.append(len(cycles))
        
    df = pd.DataFrame(data={"N_Loops_3": c3, "N_Loops_2": c2})
    df.to_csv("./file_{}.csv".format(g), sep=',',index=False)
    
    


# In[]

# Raandom Graph
for g in ['crowd_MEAN_MED','Manager','Commercial','Recreational']:
    InF = []
    MF = []
    for iteration in range (1000):
        
        G_rand = nx.gnp_random_graph(N[g], 0.1, seed=None, directed=True)

        InF3 = count_Indirect_effects(G_rand,G_rand.nodes(),3)
        MF3 = count_multiple_effects(G_rand,G_rand.nodes(),3)
        InF.append(InF3)
        MF.append(MF3)

        
    df = pd.DataFrame(data={"N_InF_3": InF})
    df2 = pd.DataFrame(data={"N_MF_3": MF})
    df.to_csv("./file_InF_{}.csv".format(g), sep=',',index=False)
    df2.to_csv("./file_MF_{}.csv".format(g), sep=',',index=False)
    
