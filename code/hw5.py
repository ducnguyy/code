import numpy as np
States=[0,1,2,3]
num_states=4
Actions=[0,1]#Go up & Down
num_actions=2
Reward=np.array([1,1,1,10])
Expected_Rewards=np.zeros(4) #array 0's 5x1
discount=0.75
num_iterations=100
num_test=200
#Probabilities has the structure: (for each state(for each action(states ended up))) <=> 5*5*3
Probabilities=np.array([
    [
        [0,1,0,0],
        [0,1,0,0],
    ],
    [ 
        [0,0,1,0],
        [1,0,0,0],
    ],
    [ 
        [0,0,0,1],
        [0,1,0,0],
    ],
    [
        [0,0,1,0],
        [0,0,1,0],
    ]
])
for iter in range(num_iterations):     
#for iter in range(num_test):    
    for states1 in range(num_states):
        E=np.zeros(2)
        for actions in range(num_actions):
            for states2 in range(num_states):
                E[actions]+=((Probabilities[states1][actions][states2])*(Reward[states1]+discount*Expected_Rewards[states2])) 
        Max=np.max(E)
        Expected_Rewards[states1]=Max
    print(Expected_Rewards)