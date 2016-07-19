# Opinion Dynamics with Local Interactions ([IJCAI-16](http://ijcai-16.org))
Dimitris Fotakis, Dimitris Palyvos-Giannas and Stratis Skoulakis

This repository contains the code used in the simulations that are described in the paper. The full PDF is available [here](http://www.ijcai.org/Proceedings/16/Papers/047.pdf).

## Abstract
> We study convergence properties of opinion dynamics with local interactions and limited information exchange. We adopt a general model where the agents update their opinions in rounds to a weighted average of the opinions in their neighborhoods. For fixed neighborhoods, we present a simple  randomized protocol that converges in expectation to the stable state of the Friedkin-Johnsen model. For opinion-dependent neighborhoods, we show that the Hegselmann-Krause model converges to a stable state if each agent’s neighborhood is restricted either to a subset of her acquaintances or to a small random subset of agents. Our experimental findings indicate that for a wide range of parameters, the convergence time and the number of opinion clusters of the neighborhood-restricted variants are comparable to those of the standard Hegselmann-Krause model.


## Dependencies
The following python packages are required to simulate the models:
- numpy
- scipy
- networkx *(network creation/display)*
- tqdm *(progress reporting)*
The following packages are needed if you want to use extra features (plots/parallel processing):
- matplotlib
- seaborn *(can be ommited if you remove the relevant import statements)*
- ipyparallel *(for parallel processing)*
