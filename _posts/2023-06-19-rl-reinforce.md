---
layout: post
title:  "RL: Policy gradient"
date:   2023-06-19 10:14:54 +0700
categories: DeepLearning
---

# Introduction

Q-learning is a value-based method in Deep Reinforcement Learning (Deep RL). The aim of Q-learning is to learn the value function and derive a good policy from it. Policy-based methods are another type of methods in which we would like to directly optimize the policy itself. $$ \pi_{\theta}(s) = P{[A \mid s; \theta]} $$. So we parameterize the policy (using a neural network), and this policy can output a probability distribution over actions (a stochastic policy scheme). There are many ways to maximize the local approximation of the objective function such as hill climbing, simulated annealing or evolution strategies. But there is a class of method inside the policy based methods that is called policy gradient method that we optimize using gradient ascent method, in a way that actions leading to higher reward would become more probable, and actions leading to lesser reward would become less probable. Technically, we directly optimize the expected cumulative reward via gradient ascent. The gradient of the the expected reward is calculated with respect to the policy parameters and then the value will be used to update the policy. In value-based method, the policy is implicitly improved by learning the value function. 

# Policy gradient method

Policy gradient method gives several benefits compared to the value-based counterpart. First, it provides ease of integration. Policy gradient methods directly estimate the policy, so we don't need to store additional data such as action values. Second, policy gradient method can output a stochastic policy, unlike a value function. We no longer need to explicitly device exploration/exploitation balance. Since the method outputs a probability distribution over actions, there is a natural degree of exploration. Also, in similar/identical states, we are not stuck at the same action anymore. The agent can prescribe different actions in that same state, giving more freedom and creativity in solving the problem. Third, policy gradient outputs directly a probability distribution over the action space. It can deal with nearly infinite action possibilities, meanwhile methods such as Q-learning assigns a score to each possible action at each time step. Calculating Q-value for all possible actions would become unfeasible. Forth, policy gradient uses a smooth and incremental approach to change policy over time, unlike value-based methods that make drastic changes since they use a max operator (choose what is the best at the moment). Despite those strengths, we need to look into its weakness as well. First, it is easy for a policy gradient method to converge to a local maximum. Second, its step by step nature makes it a slow learner, sometimes less efficient. Third, it also has high variance. 

Specifically, after the agent interacted in one episode, we will consider actions in that episode to be good or bad depending on the status of the game, i.e. whether the agent won that game or not. If the actions were good, then those actions should be sampled more frequently in the future (the action preference for that action increases). The objective function is the expected cumulative reward:

$$ J(\theta) = E {[ R(\tau) ]} $$

$$ R(\tau) = r_{t+1} + \delta r_{t+2} + \delta^2 r_{t+3} + ... $$

The expected return is the weighted average where the weights are the probability for each return.

$$ J(\theta) = \sum_{\tau} P(\tau; \theta) R(\tau) $$

The objective is to find weights $$ \theta $$ that maximize the expected return:

$$ max_{\theta}J(\theta) = E{[R(\tau)]} $$

In each step, we update the weights to get a bit higher:

$$ \theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta) $$

$$ \nabla_{\theta}J(\theta) = E_{\pi_{\theta}} {[\nabla_{\theta} log \pi_{\theta} (a_t \mid s_t) R(\tau)]} $$

# REINFORCE

REINFORCE stands for Reward increment = nonnegative factor times offset reinforcement times characteristic eligibility. This method use result of each episode to update the policy parameter $$ \theta $$. 

REINFORCE is also called the Monte Carlo policy gradient, it is a policy gradient algorithm that estimates the return from an episode and then update the weights $$ \theta $$. First it roll a trajectory out for the current policy. Then it stores the log probability and reward at each step. The discounted cumulative reward is calculated for each step. Then the policy gradient is computed and policy parameter gets to be updated.

$$ \nabla_{\theta} J(\theta) \approx \sum_{t=0} \nabla_{\theta} log \pi_{\theta} (a_t \mid s_t) R(\tau) $$

$$ \theta \leftarrow \theta + \alpha \nabla_{\theta} J(\theta) $$

The reward will decide to increase or decrease the probability of the state action pair and it also depends on the log probability of selecting action $$ a_t $$ in state $$ s_t $$.

# Actor - Critic method

The policy-based methods, such as REINFORCE that we mentioned in the previous section, optimize the policy directly without relying on a value function. It estimates the weights of the policy using gradient ascent but since it is a Monte-Carlo method, we have significant variance in policy gradient estimation. This can result in considerable fluctuation in the policy's evaluation. The policy gradient estimation points out the direction that gives the maximum increase in return. So we can modify our policy weights so that actions yielding good returns are more likely to be chosen. But the high variance leads to slower training time since we need a large number of samples to offset the variance. 

Actor - Critic method is an architecture that blends the advantages of both value-based and policy-based methods. This helps stabilize the training process by mitigating the variance. The actor part influences the agent's action, by guiding with policy-based method (with policy $$ \pi_{\theta} (s) $$). The critic part evaluates the action based on value-based method (using a value function to measure how good the action is in that state: $$ q_w(s,a) $$ ). The critic will also learn to provide better feedback. Together, the actor and critic work to reduce variance, increase stability, and improve the efficiency of the training process in reinforcement learning tasks. 

Specifically, we have the policy function $$ \pi_{\theta}(s) $$ for the actor and the value function $$ q_w(s,a) $$ for the critic. 
- At each time step t, we get the current state $$ s_t $$ from the environment and give it to the actor and critic. The policy function prescribes an action $$ a_t $$ given the state. 

- The critic sees that action, and using $$ s_t $$ and $$ a_t $$, it computes the Q-value of taking that action in that state.

- The action $$ a_t $$ then leads to a new state $$ s_{t+1} $$ and returns the reward $$ r_{t+1} $$

- The actor then uses the Q-value to update its policy parameters. $$ \Delta \theta = \alpha \nabla_{\theta}(log \pi_{\theta}(s,a)) q_w(s,a) $$. In the new state $$ s_{t+1} $$, the actor will gives the next action $$ a_{t+1} $$ based on this new policy.

- The critic now updates the value parameters: $$ \Delta w = \beta (R(s,a) + \delta q_w(s_{t+1}, a_{t+1}) - q_w(s_t, a_t)) \nabla_w q_w(s_t,a_t) $$

An extension of actor - critic method is advantage actor - critic (A2C) in which an advantage function is used to calculate the relative advantage of an action compared to other actions in that same state. Instead of using only the value of the action in that state, the value of the action state pair minus the average value of the state is used: $$ A(s,a) = Q(s,a) - V(s) $$. TD error can be used as an estimator of the advantage function: $$ A(s,a) = r + \delta V(s') - V(s) $$

# Conclusion

Policy Gradient methods mark a significant advance in the field of reinforcement learning, leveraging the strengths of both value-based and policy-based approaches to optimize decision-making processes in complex environments. By directly optimizing the policy using gradient ascent, these methods provide a powerful tool that can handle high-dimensional or continuous action spaces and generate stochastic policies, which is often a limitation in other methodologies.

Furthermore, the advent of Actor-Critic architectures, a hybrid approach of Policy Gradient methods, has addressed one of the primary issues with these techniques, namely the high variance in policy evaluation. By pairing an 'Actor' that determines the agent's behavior with a 'Critic' that assesses the quality of actions taken, Actor-Critic methods have effectively reduced this variance, increasing stability, and enhancing the efficiency of the training process.

However, like all machine learning methods, Policy Gradient methods come with their challenges. They can be computationally expensive, sometimes converge to a local maximum rather than the global optimum, and have been noted to train slower than some alternatives due to the step-by-step progression. Despite these, the overall benefits and successful application of Policy Gradient methods underscore their immense potential in creating intelligent, learning-driven systems.

In conclusion, Policy Gradient methods offer a robust and versatile framework for solving complex reinforcement learning problems. Their inherent capacity to handle a variety of action spaces, their adaptability via Actor-Critic architectures, and their continuous evolution make them an indispensable part of the modern reinforcement learning toolkit. Future research and refinement of these methods will undeniably unlock even more powerful applications, pushing the boundaries of what artificial intelligence can achieve.
