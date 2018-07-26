# MultiArmedBandit_RL
Implementation of various multi-armed bandits algorithms using Python.

# Algorithms Implemented
The following algorithms are implemented on a 10-arm testbed, as described in [Reinforcement Learning : An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf) by Richard and Sutton.

* Epsilon-Greedy Algorithm
* Softmax Algorithm
* Upper Confidence Bound(UCB1)
* Median Elimination Algorithm(MEA)

# Dependencies

* numpy(used version 1.13.3) 
* matplotlib(used version 1.5.1)

# Description of the Experiment

A 10-arm bandit testbed was generated as described in Section 'The 10-armed Testbed' of the book. The testbed contains 2000 bandit problems with 10 arms each, with the true action values q<sub>∗</sub>(a) for each action/arm in each bandit sampled from a normal distribution N(0,1). When a learning algorithm is applied to any of these arms at time t, action A<sub>t</sub> is selected from each bandit problem and it gets a reward R<sub>t</sub> which
is sampled from N (q<sub>∗</sub>(A<sub>t</sub>),1). The performance of each of the implemented algorithms is measured as ”Average Reward” (or) ”% Optimal Actions picked” at each time step.


# Implementations

The codes for each algorithm and corresponding plots generated can be found in the respective folders.

The 10-arm testbed is implemented separately for each algorithm in the corresponding code file. The epsilon-greedy, softmax and UCB1 algorithms have been implemented for 1000 (time) steps each with varying values of respective parameters. The Median Elimination Algorithm has been implemented for different values of &epsilon; and &delta; (total number of time steps for MEA is determined by the value of these hyperparameters). 
Running it for these 1000(or in the case of MEA, as calculated by values of &epsilon; and &delta;) time steps constitutes 1 run. This is repeated for 2000 independent runs, each time with a different bandit problem. The rewards obtained over all these runs is averaged to get an idea of the algorithm's average behaviour.

Comparison graphs have been plotted and can be found in the UCB and MEA folders.

More descriptions can be found in the respective folders.

# References

* [Reinforcement Learning : An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf) by Richard and Sutton
* [PAC Bounds for Multi-armed Bandit and Markov Decision Processes](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.130.2371&rep=rep1&type=pdf)