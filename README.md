# Models of Opinion Formation 

## Purpose
This is an ongoing research project in the National Technical University of Athens regarding models of opinion formation. We simulate the behavior of various models such as those proposed by DeGroot, Fredkin-Johnsen, Hegselmann-Krause as well as the K-Nearest Neighbors model. We also study how those models behave when we limit the amount of information that each node can access.

## Dependencies
The following python packages are required to simulate the models:
- numpy
- scipy
- networkx ~1.10 (network creation/display)
- tqdm (progress reporting)
The following packages are needed if you want to use extra features (plots/parallel processing):
- matplotlib
- seaborn (can be ommited if you remove the relevant import statements)
- ipython (for parallel processing)
