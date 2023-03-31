---
layout: post
title:  "Markov Decision Process (MDP)"
date:   2022-07-13 10:14:54 +0700
categories: MachineLearning
---

# TOC

- [Definition](#define)
- [Example](#ex)
- [Code example](#code)

# Definition <a name="#define"></a>

A Markov decision process is an agent that has access to the following information:

- state space S

- action set A in each state s

- transition probabilities P over the state space at each state (i.e. from one state we know the probability to end up in the next state if we take action a)

- discount factor $$ \gamma $$ to discount future cashflow

- reward function R over the action a and the state s it ends up in

A policy is to map each state to an action. The utility of a policy is the discounted sum of all the rewards on the path of that policy. We discount the value of tomorrow so that money of today worths a bit more than that same amount tomorrow. Here is the discounted sum:

$$ u = r_1 + \gamma r_2 + \gamma^2 r_3 + ... $$

The value of a policy at a state is the expected utility $$ V_{\pi}(s) $$. The Q-value $$ Q_{\pi} (s,a) $$ is the expected utility of taking action a at state s, and then following policy $$ \pi $$. Value at s is either equals 0 (if it is the end), or equals its Q-value otherwise, with Q-value to be the total of probable transitions to all the s' multiplied with its discounted cashflow:

$$ V_{\pi}(s) =
\begin{cases}
    0 & \text{if s ends} \\
    Q_{\pi}(s,\pi(s)) & \text{otherwise} \\
\end{cases}
$$ 

with $$ Q_{\pi}(s,a)=\sum_{s'} P(s,a,s') E{[R(s,a,s') + \gamma V_{\pi} (s')]} $$ (Q-value equals probabilities multiplied by expected value). To evaluate policy, we initialize values at all states to be 0:

$$ V_{\pi}^{(0)} \leftarrow 0 $$ 

Then for each iteration:

$$ V_{\pi}^{(t)}(s) \leftarrow Q^{t-1}(s,\pi(s)) = \sum_{s'} P(s,\pi(s), s') E {[R(s,\pi(s), s') + \gamma V_{\pi}^{(t-1)} (s')]} $$

We iterate until:

$$ max_{s \in S} \mid V_{\pi}^{(t)} (s) - V_{\pi}^{(t-1)}(s) \mid \leq \epsilon $$

The optimal value $$ V_{opt}(s) $$ is the maximum value for each policy. As above,

$$ V_{opt}(s) =
\begin{cases}
    0 & \text{if s ends} \\
    max_{a \in A(s)} Q_{opt}(s,a) & \text{otherwise} \\
\end{cases}
$$ 

with $$ Q_{opt}(s,a)=\sum_{s'} P(s,a,s') E{[R(s,a,s') + \gamma V_{opt} (s')]} $$

Following the similar vein, the optimal policy would be the one that maximize the Q-value with action a:

$$ \pi_{opt}(s) = arg max_{a \in A(s)} Q_{opt}(s,a) $$

Now we iterate for optimal value:

- Initialize $$ V_{opt}^{(0)}(s) \leftarrow 0 $$

- For each state s: $$ V_{opt}^{(t)} \leftarrow  max_{a \in A(s)} Q_{opt}^{(t-1)} (s,a) =  max_{a \in A(s)} \sum_{s'} P(s,a,s') E{[R(s,a,s') + \gamma V_{opt}^{(t-1)} (s')]} $$

# Example <a name="#ex"></a>

We play a game. At each round, you choose to stay or quit. If you quit, you get $$ \$10 $$ and ends the game. If you stay, you get $$ \$4 $$ and $$ \frac{1}{3} $$ probability of ending the game and $$ \frac{2}{3} $$ probability of going to the next round. Let $$ \gamma = 1 $$.

There are two policies: to stay or to quit. The value of policy "quit" is $$ \$10 $$. Let's evaluate the policy of "stay":

$$ V_{\pi} (end) = 0 $$

$$ V_{\pi}(in) = \frac{1}{3} (4 + V_{\pi} (end)) + \frac{2}{3} (4 + V_{\pi}(in)) = 4 + \frac{2}{3} V_{\pi}(in) $$

$$ \Leftrightarrow \frac{1}{3} V_{\pi}(in) = 4 $$

$$ \Leftrightarrow V_{\pi}(in) = 12 > 10 $$

We definitely should stay in the game.

# Code example <a name="code"></a>

At time 0, we set value policy "stay" to be 0. At iteration 1, value (in) = Q-value at 1 = probabilities * expected utility. delta to be the absolute difference between value of previous iteration minus the value of this iteration. If delta is smaller than 0.001, we stop the calculation. As you will see below, the calculation stops at iteration 20, and we have value of policy "stay" to be 11.99 $$ \approx $$ 12


```python
import random
import numpy as np

V = 0
delta = 0
for i in range (100):
    v = V
    V = 1/3 * (4 + 0) + 2/3 * (4 + V)
    delta = np.abs(v-V)
    if delta < 0.001:
        break
    print(i,V)

```

    0 4.0
    1 6.666666666666666
    2 8.444444444444445
    3 9.62962962962963
    4 10.419753086419753
    5 10.946502057613168
    6 11.297668038408778
    7 11.53177869227252
    8 11.687852461515012
    9 11.791901641010009
    10 11.86126776067334
    11 11.907511840448892
    12 11.938341226965928
    13 11.958894151310618
    14 11.972596100873746
    15 11.98173073391583
    16 11.98782048927722
    17 11.991880326184814
    18 11.99458688412321
    19 11.99639125608214
    20 11.997594170721426

