# Softmax Algorithm

In this algorithm, the arm to be pulled in a given bandit problem is selected using softmax probabilities. 
At first, all the arms are pulled once to get an initial estimate Q(a). After this the arm pulled (a) is selected as : 

* First, a softmax is computed over the Q-values of all arms(with a temperature hyperparameter).
* An arm is selected for this time step by sampling according to the above calculated softmax probabilities.

# Update for Q(a) : 

N(a)=N(a)+1 (counts number of times this arm has been pulled; initialise to 0)

Q(a)=Q(a)+(reward_obtained-Q(a))/N(a)


# Usage from Linux Terminal

```$ python softmax.py```


# Graphs

Graphs have been generated for the following different values of temperature : 0.01,0.1,1,10.

<img src="https://github.com/SahanaRamnath/MultiArmedBandit_RL/blob/master/Softmax_Method/sfx_opt.png" width=600>

<img src="https://github.com/SahanaRamnath/MultiArmedBandit_RL/blob/master/Softmax_Method/sfx_opt.png" width=600>
