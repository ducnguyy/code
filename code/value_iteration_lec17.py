import numpy as np
States=[0,1,2,3,4]
num_states=5
Actions=[0,1,2]#Go left, Stay, Go right
num_actions=3
Reward=np.array([0,0,0,0,1])
Expected_Rewards=np.zeros(5) #array 0's 5x1
discount=0.5
num_iterations=100
num_test=200
#Probabilities has the structure: (for each state(for each action(states ended up))) <=> 5*5*3
Probabilities=np.array([
    [
        [1/2,1/2,0,0,0],
        [1/2,1/2,0,0,0],
        [2/3,1/3,0,0,0]
    ],
    [ 
        [1/3,2/3,0,0,0],
        [1/4,1/2,1/4,0,0],
        [0,2/3,1/3,0,0]
    ],
    [ 
        [0,1/3,2/3,0,0],
        [0,1/4,1/2,1/4,0,],
        [0,0,2/3,1/3,0]
    ],
    [
        [0,0,1/3,2/3,0],
        [0,0,1/4,1/2,1/4],
        [0,0,0,2/3,1/3]
    ],
    [ 
        [0,0,0,1/3,2/3],
        [0,0,0,1/2,1/2],
        [0,0,0,1/2,1/2]
    ]
])
for iter in range(num_iterations):     
#for iter in range(num_test):    
    for states1 in range(num_states):
        E=np.zeros(3)
        for actions in range(num_actions):
            for states2 in range(num_states):
                E[actions]+=((Probabilities[states1][actions][states2])*(Reward[states1]+discount*Expected_Rewards[states2])) 
        Max=np.max(E)
        Expected_Rewards[states1]=Max
    print(Expected_Rewards)