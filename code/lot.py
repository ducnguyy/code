import numpy as np
States=[0,1,2,3,4,5]
num_states=6
Actions=[0,1]#Go left, Stay, Go right
num_actions=2
first=0
second=0
def Reward(first,second):
    if first==second:
        return (first+4)**(-1/2)
    else:
        return (np.abs(second-first))**(1/3)
Expected_Rewards=np.zeros(6) #array 0's 5x1
discount=0.6
num_iterations=50
#Probabilities has the structure: (for each state(for each action(states ended up)))
Probabilities=np.array([
    [
        [1,0,0,0,0,0],
        [1,0,0,0,0,0],
    ],
    [ 
        [1,0,0,0,0,0],
        [0,0.3,0,0.7,0,0]
    ],
    [ 
        [0,1,0,0,0,0],
        [0,0,0.3,0,0.7,0]
    ],
    [
        [0,0,1,0,0,0],
        [0,0,0,0.3,0,0.7]
    ],
    [ 
        [0,0,0,1,0,0],
        [0,0,0,0,1,0]
    ],
    [
        [0,0,0,0,1,0],
        [0,0,0,0,0,1]
    ]
])
#for iter in range(num_iterations):     
#for iter in range(num_test):    
#    for states1 in range(num_states):
#        E=np.zeros(2)
#        for actions in range(num_actions):
#            for states2 in range(num_states):
#                E[actions]+=((Probabilities[states1][actions][states2])*(Reward[states1]+discount*Expected_Rewards[states2])) 
#        Max=np.max(E)
#        Expected_Rewards[states1]=Max
#        Q=
#    print(Expected_Rewards)

 
gamma=0.6
Q = np.zeros((6,2))
for u in range(10):
    for s in range(6):
        # for all states s
        for a in range(2): # for all actions a
            sum_sp=0
            for s_ in range(6): # for all reachable states s'
                sum_sp += (Probabilities[s][a][s_]*(Reward(s,s_) + gamma*max(Q[s_])))
            Q[s][a] = sum_sp
    print(Q)
