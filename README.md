# W251 - Summer 2021 - Homework 11 -- Fun with OpenAI Gym!
### Section 2: Alice Hua

### Questions:

Run with the following command:
```
time docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:rw --privileged -v /data/videos:/tmp/videos hw11
```

1. What parameters did you change, and what values did you use?

- Changed density first layer to 512
- Changed density second layer to 256

Results:
```
465: Episode || Reward:  238.65489618592187 || Average Reward: 197.18086153561296  epsilon:  0.09672876157528969
DQN Training Complete...

real	47m44.369s
user	0m0.564s
sys	0m0.160s

96 : Episode || Reward:  204.20884919637655
97 : Episode || Reward:  237.0537441602595
98 : Episode || Reward:  262.62967074738845
99 : Episode || Reward:  204.30117712024122
Average Reward:  205.1149304522907
Total tests above 200:  61

real	9m11.747s
user	0m0.232s
sys	0m0.104s
``` 

2. Did you try any other changes (like adding layers or changing the epsilon value) that made things better or worse?

- I tried changing the num_epochs but made things a lot worse, negative range.  
The greatest effects seen in the first and second desnity layers.                                                                                                                  

3. Did your changes improve or degrade the model? How close did you get to a test run with 100% of the scores above 200?

- The changes to density layers alone got the test run with 100% of the scores above 200 by episode 90.

4. Based on what you observed, what conclusions can you draw about the different parameters and their values?

- Changes to the layers have the largest effect, the activation function relu is appropriate to use between layers. 

5. What is the purpose of the epsilon value?

- It is a value that causes a new action to be taken every now and then (exploration), but as the model gets better, it decreases the epsilon so there is less randomness (exploitation). 

6. Describe "Q-Learning".

- Q-learning is basic form of reinforcement learning that uses the Q-table as a form of memory for our agent to improve performance.   
- Q-table is of size states x actions where each state-action has a corresponding reward value (Q-values). The higher the values, the higher the rewards.  
- The Q-learning process starts when we initialize the Q-table with 0, the agent chooses the action randomly, the epsilon comes into play to prevent the the agent from taking the same action, missing he optimal strategy (this is our exploration process).
- The three essential components to reinforcement learning that Q-learning comes from are:
	- S = state
	- A = agent
	- R = reward

- Upon this, we also have:
	- epsilon: refer to Q5
	- alpha: step length taken to update estimation of Q(S,A)
	- gamma: discount factor for future rewards

- The equation is therefore:
	- Q_new(S,A) <- (1 - alpha)*Q(S,A) + alpha*(current R + gamma * max(Q(S,A))

### Videos:
Train:
	- https://alicehua-w251-hw3.s3.amazonaws.com/episode0.mp4
	- https://alicehua-w251-hw3.s3.amazonaws.com/episode200.mp4 
	- https://alicehua-w251-hw3.s3.amazonaws.com/episode450.mp4
Test:
	- https://alicehua-w251-hw3.s3.amazonaws.com/testing_run0.mp4
	- https://alicehua-w251-hw3.s3.amazonaws.com/testing_run50.mp4

### References:
- https://github.com/rbraddes/Reinforcement-Learning-Lunar_Lander
- https://www.novatec-gmbh.de/en/blog/introduction-to-q-learning/

