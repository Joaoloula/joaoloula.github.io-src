Title: Reinforcement Learning in the Brain
Date: 2016-04-02 14:54
Category: Neuroscience
Tags: reinforcement learning, model arbitration, episodic memory
Slug: reinforcement-learning
Author: João Loula
publications_src: content/posts/reinforcement-learning/references.bib
Summary: In this post we'll take a look at reinforcement learning, one of the most successful frameworks lately both for enabling AI to perform human-like tasks and for understanding how humans themselves learn these behaviors. 

##Introduction

In this post we'll take a look at reinforcement learning, one of the most successful frameworks lately both for enabling AI to perform human-like tasks and for understanding how humans themselves learn these behaviors. 

The premise is that of an agent in an environment in which it is trying to achieve a certain goal. The agent interacts with the environment in two ways that form a feedback loop:

- It receives as inputs from the environment observations and rewards
- It outputs actions that can in their turn alter the environment

It is the fact that the agent is driven to achieve a certain goal that forces it to extract, from noisy observations and uncertain rewards, a strategy for optimizing their actions. This strategy can be as simple as implementing standard responses to given stimuli and as complicated as building a sophisticated statistical model for the environment. In this post we'll take a look particularly at examples motivated by animal behavior, and discuss what the reinforcement learning framework can offer us in terms of understanding the brain.

##Markov Decision Processes

Suppose we have an agent, say a mouse, in an environment consisting of states it can occupy, possible actions that take it from one state to another, and rewards associated with different states: for example a maze with different kinds of food scattered around it, ranging from delicious to totally unappetizing. It might look a little like this:


<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/reinforcement-learning/maze_mouse.png"/>
</p>

How can we find the path that optimizes the mouse's rewards? Well, we can start from the Bellman equation:

$$Q\left(s_t\right) = \max_{a_t} \{ R(s_t, a_t) + Q \left(s_{t+1}\right)\}$$

What this equation, the principle of dynamic programming, tells us, is that calculating the Q-value of a given node is as easy as starting from the end (where the values are equivalent to the rewards) and working backwards by computing the optimal step at each time point. Following this procedure gives us the optimal path:

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/reinforcement_learning/maze_path_mouse.png"/>
</p>

The real world, however, is a lot messier: for one thing, both state transitions and rewards are usually not deterministic, but rather probabilistic in nature. Things for our mouse might actually look more like this:

<p align="center">
  <img src = "https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/reinforcement-learning/maze_complicated_mouse.png"/>
</p>

The setup is the following: the agent starts in an initial state, and at each time point he can pick one of two actions (take the arrow up or down), which can lead him to different states with some given probability. Each state is also associated with a probabilistic reward (rewards can alternatively be associated not with a state, but rather with a specific state transition) : this kind of system is what's called a Markov Decision Process -- a generalization of Markov Chains allowing for actions (i.e. control of the stochastic system) and rewards.

So, how can an agent go about solving this? Well, we can still take inspiration from Bellman's equation, while keeping running estimates for parameters of interest: given a policy $\pi$ for the agent's decision-making, an immediate reward $r_t$ and transition probabilities for the state space $P\left(s_{t+1}| s_t, a_t\right)$ at time t, we have:

$$ Q_\pi\left(s_t, a_t\right) = r_t + \gamma\sum_{s_{t+1}} P\left(s_{t+1}| s_t, a_t\right) Q_\pi\left(s_{t+1}, \pi \left(s_{t+1}\right)\right) $$


We can immediately see the resemblance to the deterministic case: in fact, the second term in the right-hand side is just an expected value over the different possible transitions, seeing as the problem is now probabilistic in nature. The term $\gamma$ is called a discount factor, and it modulates the importance between immediate and future rewards.

From this equation spring the two most important reinforcement learning algorithm classes for neuroscience.

##Model-free learning

Model-free learning focuses on estimating the left-hand side of the equation: it keeps a table of state-action pair values that is updated through experience, for example by computing a temporal difference:

$$ \delta_t = r_t + \gamma Q (s_{t+1}, a_{t+1}) - Q(s_t, a_t) $$

which can then be used by a [SARSA](https://en.wikipedia.org/wiki/State-Action-Reward-State-Action) algorithm for calculating the new state-action pair values.

Model-free learning pros and cons are:

- <code class="green">Computationally efficient, since decision-making consists of looking up a table.</code>

- <code class="red">Inflexible to changes in the MDP structure (transition probabilities or rewards), since they're not explicited in the model and thus can only be accounted for by relearning state-action values.</code>



<figure>
	<img src="https://raw.githubusercontent.com/Joaoloula/joaoloula.github.io-src/master/content/posts/reinforcement-learning/F3_large.jpg" alt='missing' align='middle' />
	<figcaption> <sup> The most successful prediction of model free RL: the dopamine system. When a stimulus-reward pair is learned, dopaminergic neurons fire at the stimulus onset and not at the reward; when the reward does not succeed the stimulus, we see instead a negative firing rate. This evidence points towards the neural implementation of a TD-like algorithm [@@schulz] </sup>
</figure>

##Model-based learning

Conversely, we can focus on estimating the right side of the equation: this leads to model-based learning. The idea is to keep running estimates of rewards and transition probabilities ($P\left(r_{t}| s_t\right)$, $P\left(s_{t+1}| s_t, a_t\right)$), and thus to have an explicit internal model for the MDP. These estimations can be computed simply by counting immediate events, and can then be strung together at decision time.

Model-based learning pros and cons are:

- <code class="green">Highly flexible: rewards and transition probabilities can be easily re-calculated in case of changes to the environment.</code>

- <code class="red">Computationally expensive, since running estimates of all parameters for an internal model of the MDP must be kept, and used for decision-making computations.</code>

\section*{When to use each learning algorithm class?}

While from the pros and cons listed we could imagine having a grasp on which learning tasks benefit one class of algorithms over the other, a recent study [@@kool] argues that the classically used Daw 2-step task [@@daw] and its variations do not offer a model-free vs. model-based trade-off. It is argued that this is a cause of the following characteristics:

- Second-stage probabilities are not highly distinguishable

- Drift rates for the reward probabilities are too slow

- State transitions are non-deterministic

- Presence of two choices in the second stage

- Observations are binary and thus not highly informative

The article goes on to propose a variant of the Daw task that addresses all these points, thus having a bigger reward range, faster drifts, deterministic transitions, no choices at the second step and continuous rewards. It goes on to show both by simulation and experiments that the trade-off is present in that variant.

##Integrating episodic memory and reinforcement learning

Many problems arise when trying to apply the view of reinforcement learning we presented here to real-world problems solved by the brain : the following issues are of particular concern:
- State spaces are often high-dimesional and continuous, besides being only partially observable
- Observations are sparse
- The Markov property (memorylessness of the stochastic process) is not held: rewards can depend on long strings of state-action pairs.

How then is the brain able to learn whilst generalizing such complex structure from such limited data? A recent paper [@@episodic-learning] argues that episodic memory could help address these concerns.

Episodic memory refers to detailed autobiographical memories, things like memories of your wedding ceremony or of what you had for breakfast this morning. These instances are called *episodes*. 

The idea of the RL model is the following: the value of a state can be approximated by the interpolation of different episodes using a kernel function $K$. For example, supposing all episodes to have a fixed length $t$, if we denote by $ s_{t}^{m}$ the state at time $t$ under the episode $m$, we have:

$$ Q_\pi(s_0, a) = \frac{\sum_{m} R_mK(s_0, s_{t}^{m})}{N}$$

where $R_m$ is the reward for episode $m$, and $N$ is a normalization factor, equal to $\sum_m K(s_0, s_{t}^{m})$. 

The Kernel is at the heart of the model's generalization power: it can be, for example, a gaussian allowing for smoothly combining episodes, or a step function that only averages episodes whose final states are close enough to $s_0$ by some distance metric. This flexibility can capture the structure of different kinds of state spaces.

The temporal dependency problem remains: in order to address it, we must first note that the breaking of the Markov property *inside* an episode poses no problem for the model. We can therefore chunk temporal dependencies inside episodes, using the Markov property only to stitch them together through the Bellman equation.

This might look something like this, by allowing episodes of various lengths $t_m$ and letting the Kernel take those lengths into account: 

$$ Q_\pi(s_0, a) = \frac{1}{N} \sum_{m} K(s_1, s_{t_m}^{m}, t_m) \left[R_m +\gamma^{t_m}\sum_{s}P(s_{t_m+1}^{m}=s| s_{t_m}^{m}, \pi(s_{t_m}^{m}))Q_\pi(s, \pi(s))\right]$$

where $N$ is still a normalization parameter. 

##A link between episodes and MF/MB algorithms?

It is interesting to note the influence of one of the model's parameters, namely the size of the episodes, on the learning algorithm. Take for example the Daw 2-step task, or better yet the variant proposed by [@@kool]. We have two possible starting states, $s_0^A$ and $s_0^B$, each with two possible actions $a_1$ and $a_2$ that lead deterministically to $s_1$ and $s_2$ that will, at a given trial $m$, present rewards $R_1^m$ and $R_2^m$.

If we denote $M$ the set of episodes of length 2, and $M_A$ and $M_B$ the respective subsets of episodes starting from $s_0^A$ and $s_0^B$, the value estimations for the two possible first actions with a constant kernel will look like this:

$$ Q_\pi(s_0^A, a_1) = \frac{1}{|M_A|} \sum_{m \in M_A} R_m^1$$
$$ Q_\pi(s_0^B, a_1) = \frac{1}{|M_B|} \sum_{m \in M_B} R_m^1$$

the formulas for action 2 being analogous. We might, for example, want to add a Kernel term accounting for the recentness of the episode (to track reward drift), but I want to focus on something else for the moment: note that there is no shared term between these formulas, i.e. the value estimation for $s_0^A$ only looks at episodes starting in $A$, and the same for $s_0^B$ In order words, a sudden change in the reward $R_1$, if experienced during a trial starting at $s_0^B$, will not influence the estimation for $Q_\pi(s_0^A, a_1)$ : this kind of insensitivity to reward devaluation is a trademark of model-free learning.

Indeed, these formulas are nothing more than value tables, and are compatible with MF learning if we imagine it being pursued with a simple reward average instead of something like a TD algorithm.

On the other hand, if we set the episode length to one, keeping the same notation, we'll get:

$$ Q_\pi(s_0^A, a_1) = \frac{1}{|M_A|} \sum_{m \in M_A} R_0^A + (P(s_1|s_0^A, a_1)Q_\pi(s_1) + P(s_2|s_0^A, a_1)Q_\pi(s_2))  $$,

but $$R_0^A = 0, Q_\pi(s_i)= \frac{1}{|M|} \sum_{m \in M} R_m^i$$, and thus, since transitions are deterministic:

$$ Q_\pi(s_0^A, a_1) = \frac{1}{|M|} \sum_{m \in M} R_m^1$$
$$ Q_\pi(s_0^B, a_1) = \frac{1}{|M|} \sum_{m \in M} R_m^1$$

it is no surprise that, starting from episodes of length one, we recover the Bellman equation for MDPs, and finally get to an averaging version of MB learning, which presents the same value estimation for action 1, independent of whether we're in $s_0^A$ or $s_0^B$.
