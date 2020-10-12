# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:19:56 2019

@author: Payam Aminpour
"""


# coding: utf-8

# In[1]:

import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
import xlrd
#import pandas as pd
import numpy as np
import networkx as nx
#import math
#import random




file_location = "./All_Participants.xlsx"
workbook = xlrd.open_workbook(file_location)
N_participants = workbook.nsheets


# Creat a dictionary keys = name of participants;  values = Adj Matrix 

Allparticipants_old ={} #to keep track of freq

Allparticipants={}

IDs = []  # each participant has a unique name or ID
All_Nodes = []

for i in range(0,N_participants):
    sheet = workbook.sheet_by_index(i)
    n_concepts = sheet.nrows-1
    
    Allparticipants_old[sheet.cell_value(0,0)] = [] #to keep track of freq
    
    
    for row in range (1,n_concepts+1):
        
        Allparticipants_old[sheet.cell_value(0,0)].append(sheet.cell_value(row,0)) #to keep track of freq
        
        
        if sheet.cell_value(row,0) not in All_Nodes:
            All_Nodes.append(sheet.cell_value(row,0))
    
N_nodes = len(All_Nodes)
n=0
Nodes_ID = {}
for nod in All_Nodes:
    Nodes_ID[n]=nod
    n+=1
    
for i in range(0,N_participants):
    sheet = workbook.sheet_by_index(i)
    #print (sheet.name)
    n_concepts = sheet.nrows-1
    Adj_matrix = np.zeros((N_nodes,N_nodes))
    for row in range (1,n_concepts+1):
        for col in range (1,n_concepts+1):
            row_new = list(Nodes_ID.keys())[list(Nodes_ID.values()).index(sheet.cell_value(row,0))]
            col_new = list(Nodes_ID.keys())[list(Nodes_ID.values()).index(sheet.cell_value(0,col))]
            #print (sheet.cell_value(row,col))
            Adj_matrix[row_new,col_new]=sheet.cell_value(row,col)
            
    IDs.append(sheet.cell_value(0,0))

    Allparticipants[sheet.cell_value(0,0)]=Adj_matrix
# In[2]:
file_location = "./categories.xlsx"
workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(0)

Category = {} 
Sub_category = {}

for row in range(1,sheet.nrows):
    Category[sheet.cell_value(row,0)]=sheet.cell_value(row,2)
    Sub_category[sheet.cell_value(row,0)]=sheet.cell_value(row,1)



# In[6]:

Frequence ={}   

for nod in All_Nodes:
    nod_fq = len([indv for indv in Allparticipants_old if nod in Allparticipants_old[indv]])
    Frequence[nod]=nod_fq

# In[6]:

def FCM(ID):
    '''Generate an FCM'''
    
    adj = Allparticipants[ID]
    FCM = nx.DiGraph(adj)
         
    return FCM 

# In[10]:

class Agents (object):
    
    def __init__ (self,ID):
        self.ID = ID
        self.FCM = FCM(self.ID)

    
    def centrality (self):
        '''calculate centrality from FCM theory'''  

        ################MentalModeler Centrality#########################
        cent ={}
#        for nod in self.FCM.nodes():
#            sum_w_out = sum(np.absolute(self.FCM[nod][v]['weight']) for v in self.FCM.successors(nod))
#            sum_w_in  = sum(np.absolute(self.FCM[v][nod]['weight']) for v in self.FCM.predecessors(nod))
#            cent[nod] = sum_w_out + sum_w_in
        
        ###################################################################
        
        #cent= nx.degree_centrality(self.FCM)
        #cent = nx.betweenness_centrality(self.FCM,normalized=True, weight='weight')
        #cent= nx.closeness_centrality(self.FCM)
        cent = nx.katz_centrality(self.FCM, alpha=0.1, beta=0.5, max_iter=1000, tol=1e-06, normalized=False, weight='weight')
        return cent
    


# In[11]:
    
########### Aggregation of Maps #########################
    
n_concepts = N_nodes
def aggregation(agent_list,How):
    
    if How == "AMI":
        adj_ag=np.zeros((n_concepts,n_concepts))
        All_ADJs = []
        for agent in agent_list:
            All_ADJs.append(nx.to_numpy_matrix(agent.FCM))
        from statistics import mean
        for i in range (n_concepts):
            for j in range (n_concepts):
                a = [ind[i,j] for ind in All_ADJs]
                adj_ag[i,j] = mean(a)
        aggregated_FCM = adj_ag   
        
#--------------------------------------------------------------------        
    if How == "AMX":
        adj_ag=np.zeros((n_concepts,n_concepts))
        All_ADJs = []
        for agent in agent_list:
            All_ADJs.append(nx.to_numpy_matrix(agent.FCM))
        from statistics import mean
        for i in range (n_concepts):
            for j in range (n_concepts):
                a = [ind[i,j] for ind in All_ADJs if ind[i,j]!=0]
                if len(a)!=0:
                    adj_ag[i,j] = mean(a)
        aggregated_FCM = adj_ag   
        
#----------------------------------------------------------------------
    if How == "MED":
        adj_ag=np.zeros((n_concepts,n_concepts))
        All_ADJs = []
        for agent in agent_list:
            All_ADJs.append(nx.to_numpy_matrix(agent.FCM))
        from statistics import median as med
        for i in range (n_concepts):
            for j in range (n_concepts):
                a = [ind[i,j] for ind in All_ADJs if ind[i,j]!=0]
                if len(a)!=0:
                    adj_ag[i,j] = med(a)
        aggregated_FCM = adj_ag   
        
#----------------------------------------------------------------------
    
    return aggregated_FCM

# In[12]:

# generating agents
All_agents=[]

for i in IDs:
    a = Agents(ID=i)
    All_agents.append(a)


# In[13]:   
    
file_location = "./Fishermen Database.xlsx"
workbook = xlrd.open_workbook(file_location)
sheet = workbook.sheet_by_index(2)

groups ={}
Group_agents ={}

list_all = []
n = sheet.nrows
for i in range(1,n):
    list_all.append(sheet.cell_value(i,1))

for g in set(list_all):
    groups[g] = []
    
for i in range (1,n):
    groups[sheet.cell_value(i,1)].append(sheet.cell_value(i,0))
    

    
for g in groups:        
    Group_agents[g] = [ag for ag in All_agents if ag.ID in groups[g]]


# In[40]:

Aggregated_group_maps = {}
for g in Group_agents:
    Aggregated_group_maps[g] = aggregation(Group_agents[g],"AMX")
#    pd.DataFrame(Aggregated_group_maps[g]).to_csv("./{}.csv".format(g))

# In[41]:
#------------Median 3 group--------------#

from statistics import median as med

ADJ_med= np.zeros((N_nodes,N_nodes)) 
for i in range(N_nodes):
    for j in range(N_nodes):
        ADJ_med[i,j]= med([Aggregated_group_maps[g][i,j] for g in Aggregated_group_maps])

ADJ_mean = sum([Aggregated_group_maps[g] for g in Aggregated_group_maps])/(len(Aggregated_group_maps))
 
for nod in All_Nodes:
    if Frequence[nod] <= 0:
        nod_id = list(Nodes_ID.keys())[list(Nodes_ID.values()).index(nod)]
        for i in range(N_nodes):
            ADJ_mean[i,nod_id] = 0
            ADJ_mean[nod_id,i] = 0
        
# In[42]
Aggregated_group_maps["crowd_MEAN_MED"] = ADJ_med
#Aggregated_group_maps["crowd_MEAN_MEAN"] = ADJ_mean

for ag in All_agents:
    if ag.ID == "ECOSYSTEM-BASED-MODEL":
        Aggregated_group_maps["ECOSYSTEM-BASED-MODEL"] = nx.to_numpy_matrix(ag.FCM)

Group_FCM ={}

for g in Aggregated_group_maps:
    Group_FCM[g] = nx.DiGraph(Aggregated_group_maps[g])
    #nx.write_edgelist(Group_FCM[g], "C:/Paym Computer/Striped-bass fishery Project/CSV/New/GROUP_EDGES_{}.csv".format(g))




# In[43]
list_NonZero_Nodes = {}

for g in Aggregated_group_maps:
    list_NonZero_Nodes[g] = []
    for nod in Group_FCM[g].nodes():
        if Group_FCM[g].in_degree(nbunch=None, weight=None)[nod] > 0 or Group_FCM[g].out_degree(nbunch=None, weight=None)[nod] > 0:
            list_NonZero_Nodes[g].append(nod)   
            
list_NonZero_Nodes_Names ={}

for g in Aggregated_group_maps:
    list_NonZero_Nodes_Names[g] = {}
    for nod in list_NonZero_Nodes[g]:
        list_NonZero_Nodes_Names[g][nod] = Nodes_ID[nod]

# In[43]

#import csv
#import os
#
#def WriteDictToCSV(csv_file,csv_columns,dict_data):
#    with open(csv_file, 'w') as csvfile:
#        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
#        writer.writeheader()
#        for data in dict_data:
#            writer.writerow(data)
#    return    
#
#
#csv_columns = ['Id','Label','Cat','SubCat']
#
#for g in Aggregated_group_maps:
#    dict_data = []
#    for nod in list_NonZero_Nodes_Names[g]:
#        dict_data.append({'Id':nod,'Label':list_NonZero_Nodes_Names[g][nod],
#                          'Cat':Category[list_NonZero_Nodes_Names[g][nod]],
#                          'SubCat':Sub_category[list_NonZero_Nodes_Names[g][nod]]})
    
    #currentPath = os.getcwd()
    #csv_file = "C:/Paym Computer/Striped-bass fishery Project/CSV/New/GROUP_NODES_{}.csv".format(g)
    
    #WriteDictToCSV(csv_file,csv_columns,dict_data)

# In[44]

sets = []
for g in groups:
    s = set(list_NonZero_Nodes[g]) 
    sets.append(s)
    
repeated_nodes = set.intersection(*sets)

# In[44]

#count = {}
#category = Category
#for g in Aggregated_group_maps:
#    count[g] = {}
#    for subcat in set(category.values()):
#        nodes = list_NonZero_Nodes_Names[g] 
#        a = [n for n in nodes if category[nodes[n]]==subcat]
#        count[g][subcat] = len(a)
#        