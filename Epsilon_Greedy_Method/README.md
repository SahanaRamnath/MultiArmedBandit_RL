# Epsilon-Greedy Algorithm

In this algorithm, the arm to be pulled in a given bandit problem is selected in an &epsilon;-greedy fashion. 
At first, all the arms are pulled once to get an initial estimate Q(a). After this the arm pulled (a) is selected as : 

* With a probablity of 1-&epsilon; the arm with the highest Q(a) in the current time step is pulled in the next time step.
* With a probablity of &epsilon; , any one of the arms is pulled, with all arms getting an equal probability of &epsilon;/num_arms.

# Update for Q(a) : 

N(a)=N(a)+1 (counts number of times this arm has been pulled; initialise to 0)

Q(a)=Q(a)+(reward_obtained-Q(a))/N(a)

# Usage from Linux Terminal

```$ python eps-greedy.py```

# Graphs

Graphs have been generated for the following different values of &epsilon; : 0(purely greedy),0.01,0.1,0.2 and 1(purely explorative).

<img src="https://github.com/SahanaRamnath/MultiArmedBandit_RL/blob/master/Epsilon_Greedy_Method/eps_reward.png" width=600>

<img src="https://github.com/SahanaRamnath/MultiArmedBandit_RL/blob/master/Epsilon_Greedy_Method/eps_opt.png" width=600>