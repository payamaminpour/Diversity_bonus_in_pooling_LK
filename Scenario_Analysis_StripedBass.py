# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:23:02 2018

@author: Payam Aminpour
         Michigan State University
         aminpour@msu.edu
"""


import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import xlrd


from Stripped_Bass_Models import Aggregated_group_maps
Aggregated_group_maps.pop('ECOSYSTEM-BASED-MODEL')
from Stripped_Bass_Models import Nodes_ID

file_location = "C:/Paym Computer/Baqir Project/nodes.xlsx"
workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)
n_concepts = len(Nodes_ID)
Concepts_matrix = [Nodes_ID[i] for i in range(n_concepts)]
node_name = Nodes_ID
activation_vec = np.ones(n_concepts)

# In[]

def TransformFunc (x, n, f_type,Lambda=0.44):
    
    if f_type == "sig":
        x_new = np.zeros(n)
        for i in range (n):
            x_new[i]= 1/(1+math.exp(-Lambda*x[i]))
            
        return x_new    

    
    if f_type == "tanh":
        x_new = np.zeros(n)
        for i in range (n):
            x_new[i]= math.tanh(Lambda*x[i])
        
        return x_new
    
#______________________________________________________________________________

def infer_steady (AdjmT, init_vec = activation_vec  \
                  , n =n_concepts , f_type="sig", infer_rule ="mk"):
        
    act_vec_old= init_vec
    
    resid = 1
    while resid > 0.001:  # here you have to define the stoping rule for steady state calculation
        act_vec_new = np.zeros(n)
        x = np.zeros(n)
                

        if infer_rule == "k":
            x = np.matmul(AdjmT, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(AdjmT, act_vec_old)
        if infer_rule == "r":
            x = (2*act_vec_old-np.ones(n)) + np.matmul(AdjmT, (2*act_vec_old-np.ones(n)))
            
        
        x = np.array(x).reshape(-1,)
        act_vec_new = TransformFunc (x ,n, f_type)
        resid = max(abs(act_vec_new - act_vec_old))

        
        act_vec_old = act_vec_new
    return act_vec_new
#______________________________________________________________________________
def infer_scenario (AdjmT, Scenario_concepts,change_level, init_vec = activation_vec \
                    , n =n_concepts , f_type="sig", infer_rule ="mk" ):
    act_vec_old= init_vec
    
    resid = 1
    while resid > 0.001:
        act_vec_new = np.zeros(n)
        x = np.zeros(n)
        
        if infer_rule == "k":
            x = np.matmul(AdjmT, act_vec_old)
        if infer_rule == "mk":
            x = act_vec_old + np.matmul(AdjmT, act_vec_old)
        if infer_rule == "r":
            x = (2*act_vec_old-np.ones(n)) + np.matmul(AdjmT, (2*act_vec_old-np.ones(n)))
            
        x = np.array(x).reshape(-1,)
        act_vec_new = TransformFunc (x ,n, f_type)
        
        for c in  Scenario_concepts:
            
            act_vec_new[c] = change_level[c]
        
            
        resid = max(abs(act_vec_new - act_vec_old))
        
        act_vec_old = act_vec_new
    return act_vec_new

# In[]


Principles=["Striped Bass Population",
            "Commercial Fishing for Striped Bass",
            "Recreational Fishing for Striped Bass",
            "Prey Abundance",
            "Average Size of Striped Bass",
            "Fish Health",
            "Predator Abundance",
            "Habitat",
            "Spawning"]


prin_concepts_index = []
for name in Principles:
    pi = list(node_name.keys())[list(node_name.values()).index(name)]
    prin_concepts_index.append(pi)


#________________________________Scenarios_____________________________________

list_of_consepts_to_run = {}

list_of_consepts_to_run['S1'] = ["Poaching and illegal activity"] 
list_of_consepts_to_run['S2'] = ["Water temperature"] 
list_of_consepts_to_run['S3'] = ["Inclement Weather"] 
list_of_consepts_to_run['S4'] = ["Water Pollution","Water Quality"] 
list_of_consepts_to_run['S5'] = ["Demand"]                      
list_of_consepts_to_run['S6'] = ["Price"] 

# In[]

def Scenario_Func(Adj_matrix,list_of_consepts_to_run):
    infer_rule = 'k'
    function_type = 'tanh'
    
    change_level = {}
    for c in list_of_consepts_to_run :
        change_level[c] = 1
        if c == 'Water Quality':
            change_level[c] = -1

    
    change_level_by_index = {} 
    for name in change_level.keys():
        change_level_by_index[Concepts_matrix.index(name)] = change_level[name]
    
    Scenario_concepts = [] 
    for name in list_of_consepts_to_run:
        Sce_Con_name =name
        Scenario_concepts.append(Concepts_matrix.index(Sce_Con_name))
    
        
    change_IN_PRINCIPLES = []
        
    
    SteadyState = infer_steady (Adj_matrix.T, f_type=function_type, infer_rule = infer_rule)
    ScenarioState = infer_scenario (Adj_matrix.T, Scenario_concepts,change_level_by_index ,f_type=function_type, infer_rule = infer_rule)
    change_IN_ALL = ScenarioState - SteadyState
    
    for c in Scenario_concepts:
        change_IN_ALL[c] = 0
    
    for i in range (len(prin_concepts_index)): 
        change_IN_PRINCIPLES.append(change_IN_ALL[prin_concepts_index[i]])
        
    return change_IN_PRINCIPLES


# In[]

S = 'S4'
results = {}

results["g"] = [g for g in Aggregated_group_maps]

for p in Principles:
    results[p] = [Scenario_Func(Aggregated_group_maps[g],list_of_consepts_to_run[S])[Principles.index(p)]\
           for g in Aggregated_group_maps]
    
# __________________________________________________________________    

df= pd.DataFrame(results)
df=df.dropna()
df_new = pd.melt(df,id_vars=['g'],value_vars=Principles)
# __________________________________________________________________    


    
plt.figure(figsize=(3,5))   


N = len(Principles)

ind = np.arange(N) 
width = 0.2
i=0

grps = ['Commercial', 'Recreational','Manager','crowd_MEAN_MED']
colors = ['gold','tomato','cyan','dimgrey']

for g, color in zip(grps,colors):  
    plt.barh(ind+ width*i, list(df_new[df_new['g']== g]['value']), width, color = color, label= g)
    i+=1


plt.yticks(ind + width*i / 2, Principles, rotation=0)
#plt.legend(loc='best',fontsize = 8)
plt.title(list_of_consepts_to_run[S][0],fontdict={'fontsize':12})
plt.xlabel('Perceived change', fontsize = 10)
plt.xlim(-1,1)

plt.show()
