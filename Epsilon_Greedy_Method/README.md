# Epsilon-Greedy Algorithm

In this algorithm, the arm to be pulled in a given bandit problem is selected in an &epsilon;-greedy fashion. 
At first, all the arms are pulled once to get an initial estimate Q(a). After this the arm pulled (a) is selected as : 

* With a probablity of 1-&epsilon; the arm with the highest Q(a) in the current time step is pulled in the next time step.
* With a probablity of &epsilon; , any one of the arms is pulled, with all arms getting an equal probability of &epsilon;/num_arms.

# Update for Q(a) : 

N(a)=N(a)+1 (counts number of times this arm has been pulled; initialise to 0)
Q(a)=Q(a)+(reward_obtained-Q(a))/N(a)

# Graphs

Graphs have been generated for the following different values of &epsilon; : 0(purely greedy),0.01,0.1,0.2 and 1(purely explorative).

![eps_reward](https://user-images.githubusercontent.com/17588365/43283652-187ca81a-9137-11e8-8219-8c3c89b30017.png)

![eps_opt](https://user-images.githubusercontent.com/17588365/43283654-1cef348a-9137-11e8-9a9f-169a30a4ba6d.png)
