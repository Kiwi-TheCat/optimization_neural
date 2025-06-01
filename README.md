# Neural data compression optimization project submission


## List of content
1. Implementation of four gradient descent optimizers where ADAM yielded the best result
- ```main2.m``` (which includes SGD, Momentum, AdaGrad, ADAM)
2. Using the trained auto-encoder weights from ADAM optimization, then add Gaussian noise one layer at a time to investigate which layer is the most vulnerable to weight perturbations (causing the most change in loss)
- ```layer_noise_test_with50samples.m```
3. After adding noise to the weight matrix, continue training, then compare the new converged loss with the initial loss without noise corruption, to find out the resilience of different layers
- ```train_with_noise2.m```
4. Implementation of Swarm Algorithm


## Problem overview