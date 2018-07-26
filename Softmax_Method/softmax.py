'''
---------------------------
10 ARM TESTBED WITH SOFTMAX
---------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import random

n_bandit=2000 # number of bandit problems
k=10 # number of arms in each bandit problem
n_pulls=1000 # number of times to pull each arm

q_true=np.random.normal(0,1,(n_bandit,k)) # generating the true means q*(a) for each arm for all bandits
true_opt_arms=np.argmax(q_true,1) # the true optimal arms in each bandit
# each row represents a bandit problem

temperature=[0.01,0.1,1,10]
col=['r','g','k','b']
fig1=plt.figure().add_subplot(111)
fig2=plt.figure().add_subplot(111)

for temp in range(len(temperature)) : 

	Q=np.zeros((n_bandit,k)) # reward estimated
	N=np.ones((n_bandit,k)) # number of times each arm was pulled # each arm is pulled atleast once
	# Pull all arms once
	Qi=np.random.normal(q_true,1) # initial pulling of all arms

	R_temp=[]
	R_temp.append(0)
	R_temp.append(np.mean(Qi))
	R_temp_opt=[]
	
	for pull in range(2,n_pulls+1) : 
		R_pull=[]
		opt_arm_pull=0 # number of pulss of best arm in this time step

		for i in range(n_bandit) : 
			exp_Q=np.exp(Q[i]/temperature[temp])
			sfx_Q=exp_Q/np.sum(exp_Q) # softmax probabilities
			j=np.random.choice(range(k),1,p=sfx_Q) # picks one arm based on softmax probability

			temp_R=np.random.normal(q_true[i][j],1)
			R_pull.append(temp_R)

			if j==true_opt_arms[i] : # To calculate % optimal action
				opt_arm_pull=opt_arm_pull+1

			N[i][j]=N[i][j]+1
			Q[i][j]=Q[i][j]+(temp_R-Q[i][j])/N[i][j]
			'''for var in range(k) : 
				if var!=j : 
					Q[i][var]=Q[i][var]-(temp_R-Q[i][var])*(sfx_Q[var])*0.1'''

		avg_R_pull=np.mean(R_pull)
		R_temp.append(avg_R_pull)
		R_temp_opt.append(float(opt_arm_pull)*100/2000)

	fig1.plot(range(0,n_pulls+1),R_temp,col[temp])
	fig2.plot(range(2,n_pulls+1),R_temp_opt,col[temp])

plt.rc('text',usetex=True)
fig1.title.set_text('Softmax : Average Reward Vs Steps for 10 arms')
fig1.set_ylabel('Average Reward')
fig1.set_xlabel('Steps')
fig1.legend((r"$T=$"+str(temperature[0]),r"$T=$"+str(temperature[1]),r"$T=$"+str(temperature[2]),r"$T=$"+str(temperature[3])))
fig2.title.set_text(r'Softmax : $\%$ Optimal Action Vs Steps for 10 arms')
fig2.set_ylabel(r'$\%$ Optimal Action')
fig2.set_xlabel('Steps')
fig2.set_ylim(0,100)
fig2.legend((r"$T=$"+str(temperature[0]),r"$T=$"+str(temperature[1]),r"$T=$"+str(temperature[2]),r"$T=$"+str(temperature[3])))
plt.show()


