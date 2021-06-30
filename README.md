# SAGA
Surrogate-Assisted Genetic Algorithm for Wrapper Feature Selection
![alt text](https://github.com/Ghaith81/SAGA/blob/master/Flowchart.JPG)

Surrogate-Assisted Genetic Algorithm for Wrapper Feature Selection

Please refer to the Jupyter notebook (Example.ipynb) for an example of using the repository to perform feature selection using synthetic data. In the notebook, feature selection is carried for synthetic data of which the informative features are known. A binary classification dataset of 1000 instances is created of which only the first two features (indexes [0,1]) are informative, and the rest are random noise. Three wrapper feature selection methods are used to identify the infromative features using a Decision Tree classifier:

* SAGA: The hyper-parameter choices of SAGA is based on the paper “Surrogate-Assisted Genetic Algorithm for Wrapper Feature Selection” published in IEEE CEC 2021. Population size is set to 40, 4 surrogate levels, a population reduction rate of 0.5, and a step of 10 generations. CHC is used as the evolutionary algorithm of SAGA with default settings, as follows below.
* CHC: The implementation of a CHC algorithm is according to the paper: “The CHC Adaptive Search Algorithm: How to Have Safe Search When Engaging in Nontraditional Genetic Recombination” The default hyper-parameter choices of d equals 0.25 of the chromosome size and divergence of 0.35, with a population size of 40.
* GA: A classical GA is implemented with 0.9 crossover and 0.1 mutation probabilities. Elitism always migrates the best individual to the next generation population. The population size is also 40 for the GA.


