# Threshold-Loss-for-Discrete-Ordered-Labels
This work implements the loss functions that is desinated for discrete ordered labels. Discrete ordered labels are labels that contain ordinal information and the distance between different classes could not be measured numerically. The detail could be found in the paper ["Loss Functions for Preference Levels: Regression with Discrete Ordered Labels"](https://home.ttic.edu/~nati/Publications/RennieSrebroIJCAI05.pdf).

# Learnable Thresholds
Given an N-class classification problem for discrete ordered labels, there would be N-1 thresholds represented in scalars. The goal of the loss is to encourage the model to have $\theta_i < z <\theta_{i}$ given the data is from class $i$, where z is the output of the model for the data. During the training, both thresholds and the model are trained. Through our experiments, we found that two phase training could also further stablize the process. With thresholds being fixed in the first phase, and freeze the model partially and unfreeze the thresholds in the second phase.

There are two methods to update the threshold:

## All-threshold
Update all the thresholds at once given a data from class i
<br />
$loss(z,y) = \sum_{i=0}^{N-2} f(s(i;y)(\theta_i-z))$ <br />
$s(l;y) = 1$ if $l \geq y$, else $s(l;y)=-1$ 

## Immediate threshold
Update neighboring thresholds at once given a data from class i 
<br />
$loss(z,y) = f(z-\theta_{i-1}) + f(\theta_i-z)$

# Citation
RENNIE, Jason DM; SREBRO, Nathan. Loss functions for preference levels: Regression with discrete ordered labels. In: Proceedings of the IJCAI multidisciplinary workshop on advances in preference handling. AAAI Press, Menlo Park, CA, 2005.

I am not the author of this paper.


