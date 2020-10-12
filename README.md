# Diversity_bonus_in_pooling_LK
Replication materials for "The diversity bonus in pooling local knowledge about complex problems"

This repository contains data and Python code for replicating the figures, analyses, and simulations described in "The diversity bonus in pooling local knowledge about complex problems" by Payam Aminpour, Steven A. Gray, Alison Singer, Steven B. Scyphers, Antonie J. Jetter, Rebecca Jordan, Robert Murphy Jr., and Jonathan H. Grabowski


## Download
Clone/Download this repository to your local computer. You will simply be downloading the git repository.

Go to the GitHub page for https://github.com/payamaminpour/Diversity_bonus_in_pooling_LK
Click on the button labeled "Clone or download" on the right side of the screen.
A drop-down menu will appear-- click on "Download ZIP"
This automatically downloads a compressed version of all of the files.
When the download is complete, open it.
Once you open the download, you should see a single folder called "Diversity_bonus_in_pooling_LK-master". Drag this folder to your your working directory.
un-zip the folders.

## Install Python3
Download the Anaconda Python3 installer from https://www.anaconda.com/download/ the one called Python 3.x 64-bit, unless you know you need something else. Run the installer and... All defaults are okay. For each script to run, be sure to set your working directory to the location of the script.


## Individual Mental Models Data
The file "All_Participants.xlsx" contains all the individual Mental Models data. These are Adjacency matrices of individually created FCMs. Each individual has a unique ID which is linked to the demographic data at "Fishermen Database.xlsx" 

## Model Aggregation
The file "Stripped_Bass_Models.py" contains python code for aggregating the FCMs of individuals. The output would be 4 aggregated modes (Recreational fishers, Commercial fishers, Fisheries managers, and the Diverse Crowd). These aggregated models were used during interviews with experts, where the models structural and dynamic accuracy were evaluated by experts. 

## Dynamic Simulation (FCM Scenario Analysis)
The file "Scenario_Analysis_StripedBass.py" contains python code for replicating the scenario simulations. These outcomes were present in SI Appendix and were used during interviews with experts, where the models structural and dynamic accuracy were evaluated by experts.

## Micro-motif analyes
The file "Stripped_Bass_Motif.py" contains python code for stochastic network analysis that measures the prevalence of complex micro-motifs in a model (i.e., complex micro-structures including bi-directionality, indirect effect, multiple effects, and feedback loop).
