# UCB1 Algorithm

In this algorithm, the arm to be pulled in a given bandit problem is selected using upper confidence bounds. 
At first, all the arms are pulled once to get an initial estimate Q(a). After this the arm pulled (a) is selected according to the following equation : 

A<sub>t</sub>=argmax<sub>a</sub> [Q<sub>t</sub>(a)+c*sqrt(ln t/N<sub>t</sub>(a))]

The second term in the above expression accounts for the uncertainty is estimating how optimal/maximal the arm is (i.e. the uncertainty in estimating the first term).

# Update for Q(a) : 

N(a)=N(a)+1 (counts number of times this arm has been pulled; initialise to 0)

Q(a)=Q(a)+(reward_obtained-Q(a))/N(a)


# Usage from Linux Terminal

```$ python ucb1.py```


# Graphs

Graphs have been generated for the following different values of c : 0.1,2,5.

<img src="https://github.com/SahanaRamnath/MultiArmedBandit_RL/blob/master/UCB/ucb_reward.png" width=600>

<img src="https://github.com/SahanaRamnath/MultiArmedBandit_RL/blob/master/UCB/ucb_opt.png" width=600>

# Comparison Graph

<img src="https://github.com/SahanaRamnath/MultiArmedBandit_RL/blob/master/UCB/ucb_compare.png" width=600>
