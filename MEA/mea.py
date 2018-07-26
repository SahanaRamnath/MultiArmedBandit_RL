'''
--------------------------------------------------------------------------------------
10 ARM TESTBED WITH MEDIAN ELIMINATION ALGORITHM, AND COMPARISON OF ALL MAB ALGORITHMS
--------------------------------------------------------------------------------------
'''

import numpy as np
import matplotlib.pyplot as plt
import random
import time

start=time.clock()
n_bandit=2000 # number of bandit problems
k=10 # number of arms in each bandit problem

q_true=np.random.normal(0,1,(n_bandit,k)) # generating the true means q*(a) for each arm for all bandits
# each row represents a bandit problem
true_opt_arms=np.argmax(q_true,1) # the true optimal arms in each bandit
true_opt_arm_values=np.max(q_true,1)
true_opt_arm_values=np.reshape(true_opt_arm_values,(n_bandit,1))

eps_delta_pairs=[[1.2,0.8,'g'],[0.6,0.4,'k'],[1.2,0.6,'r']] # along with colours for graphs
fig1=plt.figure().add_subplot(111)
fig2=plt.figure().add_subplot(111)

s1=[]
s2=[]
s3=[]

for eps_delta in eps_delta_pairs : 

	eps_init=eps_delta[0] # epsilon 
	delta_init=eps_delta[1] # delta

	print 'Current eps-delta pair : (',eps_init,delta_init,')'

	# Starting the algo
	l=1
	eps_l=eps_init/4.0
	delta_l=delta_init/2.0
	q_true_l=q_true # q* for the first round
	k_l=k # number of arms in first round
	true_opt_arms_l=true_opt_arms # indicates if optimal arm is present in the lth round

	rewards=[]
	x_graph=0
	opt_arms=[]

	while k_l!=1 : 

		opt_arms_l=0

		#print k_l
		# number of times to sample each arm in the lth round
		sample_l=np.log(3.0/delta_l)*4/(eps_l**2)

		#print int(sample_l),'\n'

		# reward evaluation
		temp_R_l=np.zeros((n_bandit,k_l))
	
		start1=time.clock()
		for cnt in range(int(sample_l)) : 
			temp_R=np.random.normal(q_true_l,1)
			temp_R_l=temp_R_l+temp_R
			rewards.append(np.mean(temp_R))
			x_graph=x_graph+1

		avg_temp_R_l=temp_R_l/int(sample_l)
		s1.append(time.clock()-start1)

		start2=time.clock()
		medians_l=np.median(avg_temp_R_l,1) # finding the medians of each bandit
		s2.append(time.clock()-start2)

		# removal of arms and updates	

		start3=time.clock()
		q_true_l_new=np.zeros((n_bandit,k_l-k_l/2))

		for i in range(n_bandit) : 
			j1=0 # loop variable to update q*
			for j in range(k_l) : 
				if avg_temp_R_l[i][j]>=medians_l[i] : 
					q_true_l_new[i][j1]=q_true_l[i][j]
					j1=j1+1
					if j==true_opt_arms_l[i] : 
						opt_arms_l=opt_arms_l+1
		s3.append(time.clock()-start3)

		opt_arms.append(opt_arms_l*100/float(n_bandit))
		q_true_l=q_true_l_new
		true_opt_arms_l=np.argmax(q_true_l,1)
		#S_l=sorted_arms_numbers_l[:,k_l/2:]
		k_l=k_l-k_l/2 # arms in the (l+1)th round
		eps_l=eps_l*0.75
		delta_l=delta_l*0.5
		l=l+1

	sample_l=np.log(3.0/delta_l)*4/(eps_l**2)
	temp_R_l=np.zeros((n_bandit,k_l))
	for cnt in range(int(sample_l)) : 
		temp_R=np.random.normal(q_true_l,1)
		temp_R_l=temp_R_l+temp_R
		rewards.append(np.mean(temp_R))
		x_graph=x_graph+1
	


	print 'eps=',eps_init,' delta=',delta_init
	print '-----------------------'
	diff=abs(true_opt_arm_values-q_true_l)
	cnt1=n_bandit-np.count_nonzero(diff)
	cnt2=np.count_nonzero(diff<eps_init)-cnt1
	cnt3=np.count_nonzero(diff>eps_init)
	print 'Number of bandit problems where best arm found was the true best arm : ',cnt1,'\n'
	print 'Number of bandit problems where true best arm was eliminated but epsilon condition was met : ',cnt2,'\n'
	print 'Number of bandit problems where true best arm was eliminated and epsilon condition was not met : ',cnt3,'\n\n'

	fig1.plot(range(x_graph),rewards,eps_delta[2])
	fig2.plot(range(1,l),opt_arms,eps_delta[2])

plt.rc('text',usetex=True)
fig1.title.set_text('MEA Average Reward versus Steps')
fig1.set_ylabel('Average Reward')
fig1.set_xlabel('Steps')
fig1.legend((r'$\epsilon$='+str(eps_delta_pairs[0][0])+r', $\delta$='+str(eps_delta_pairs[0][1]),r'$\epsilon$='+str(eps_delta_pairs[1][0])+r', $\delta$='+str(eps_delta_pairs[1][1]),r'$\epsilon$='+str(eps_delta_pairs[2][0])+r', $\delta$='+str(eps_delta_pairs[2][1])),loc='best')
fig2.title.set_text(r'MEA $\%$ Optimal Action Vs Steps')
fig2.set_xlabel('Steps')
fig2.set_ylabel(r'$\%$ Optimal Action')
fig2.set_ylim(0,110)
fig2.legend((r'$\epsilon$='+str(eps_delta_pairs[0][0])+r', $\delta$='+str(eps_delta_pairs[0][1]),r'$\epsilon$='+str(eps_delta_pairs[1][0])+r', $\delta$='+str(eps_delta_pairs[1][1]),r'$\epsilon$='+str(eps_delta_pairs[2][0])+r', $\delta$='+str(eps_delta_pairs[2][1])),loc='best')
plt.show()


print 'Total time taken (over all eps-delta pairs): ',time.clock()-start,'\n\n'

print 'Checking for RDS (averaging over all epsilon-delta pairs)'
print '---------------------------------------------------------'
print 'Average time taken to sample distributions : ',np.mean(s1)
print 'Average time taken to calculate medians for all bandits : ',np.mean(s2)
print 'Average time taken to eliminate half the arms : ',np.mean(s3),'\n'


# Comparison graphs

# MEA
fig3=plt.figure().add_subplot(111)
fig3.plot(range(x_graph),rewards,'r')

# epsilon-greedy, softmax and ucb1
epsilon=0.1 # parameter for epsilon-greedy
temperature=0.1 # parameter for softmax
c=5 # parameter for ucb1
n_pulls=x_graph # number of time steps 

Q_ucb=np.zeros((n_bandit,k)) # reward estimated for ucb1
N_ucb=np.ones((n_bandit,k)) # number of times each arm was pulled for ucb1# each arm is pulled atleast once
	
Q_eps=np.zeros((n_bandit,k)) # reward estimated for epsilon-greedy method
N_eps=np.ones((n_bandit,k)) # number of times each arm was pulled in epsilon-greedy method 

Q_sfx=np.zeros((n_bandit,k)) # reward estimated for softmax method
N_sfx=np.ones((n_bandit,k)) # number of times each arm was pulled in softmax method 

Qi=np.random.normal(q_true,1) # initial pulling of all arms
avg_Qi=np.mean(Qi)

R_ucb=[]
R_ucb.append(0)
R_ucb.append(avg_Qi)

R_eps=[]
R_eps.append(0)
R_eps.append(avg_Qi)

R_sfx=[]
R_sfx.append(0)
R_sfx.append(avg_Qi)

for pull in range(2,n_pulls+1) : 

	print pull

	R_pull_ucb=[]
	R_pull_eps=[]
	R_pull_sfx=[]

	for i in range(n_bandit) : 

		# epsilon-greedy
		if random.random()<epsilon : 
			j_eps=np.random.randint(k)
		else : 
			j_eps=np.argmax(Q_eps[i])

		temp_R_eps=np.random.normal(q_true[i][j_eps],1)
		R_pull_eps.append(temp_R_eps)
		N_eps[i][j_eps]=N_eps[i][j_eps]+1
		Q_eps[i][j_eps]=Q_eps[i][j_eps]+(temp_R_eps-Q_eps[i][j_eps])/N_eps[i][j_eps]

		# softmax
		exp_Q_sfx=np.exp(Q_sfx[i]/temperature)
		sfx_Q_sfx=exp_Q_sfx/np.sum(exp_Q_sfx) # softmax probabilities
		j_sfx=np.random.choice(range(k),1,p=sfx_Q_sfx) # picks one arm based on softmax probability
		temp_R_sfx=np.random.normal(q_true[i][j_sfx],1)
		N_sfx[i][j_sfx]=N_sfx[i][j_sfx]+1
		Q_sfx[i][j_sfx]=Q_sfx[i][j_sfx]+(temp_R_sfx-Q_sfx[i][j_sfx])/N_sfx[i][j_sfx]				
		R_pull_sfx.append(temp_R_sfx)

		# ucb1
		ucb_Q=Q_ucb[i]+np.sqrt(c*np.log(pull)/N_ucb[i])
		j_ucb=np.argmax(ucb_Q)
		temp_R_ucb=np.random.normal(q_true[i][j_ucb],1)
		R_pull_ucb.append(temp_R_ucb)
		N_ucb[i][j_ucb]=N_ucb[i][j_ucb]+1
		Q_ucb[i][j_ucb]=Q_ucb[i][j_ucb]+(temp_R_ucb-Q_ucb[i][j_ucb])/N_ucb[i][j_ucb]

	avg_R_pull_ucb=np.mean(R_pull_ucb)
	R_ucb.append(avg_R_pull_ucb)
	avg_R_pull_eps=np.mean(R_pull_eps)
	R_eps.append(avg_R_pull_eps)
	avg_R_pull_sfx=np.mean(R_pull_sfx)
	R_sfx.append(avg_R_pull_sfx)
	
fig3.plot(range(0,n_pulls+1),R_eps,'g')
fig3.plot(range(0,n_pulls+1),R_ucb,'k')
fig3.plot(range(0,n_pulls+1),R_sfx,'b')

fig3.title.set_text(r'UCB1 versus $\epsilon$-greedy versus Softmax versus MEA')
fig3.set_ylabel('Average Reward')
fig3.set_xlabel('Steps')
fig3.legend((r"MEA $\epsilon$=1.2,$\delta$=0.6",r"$\epsilon$=0.1",r"UCB1 c=5",r"temp=0.1"),loc='best')
plt.show()





