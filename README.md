# Neural data compression optimization project submission


## List of content
1. Implementation of four gradient descent optimizers where ADAM yielded the best result
- ```main2.m``` (which includes SGD, Momentum, AdaGrad, ADAM)
2. Using the trained auto-encoder weights from ADAM optimization, then add Gaussian noise one layer at a time to investigate which layer is the most vulnerable to weight perturbations (causing the most change in loss)
- ```layer_noise_test_with50samples.m```
3. After adding noise to the weight matrix, continue training, then compare the new converged loss with the initial loss without noise corruption, to find out the resilience of different layers
- ```train_with_noise2.m```
4. Implementation of Swarm Algorithm
- ```main_PSO.m```
5. standardised preprocessing -> delta encoding and Z-score normalisation on data_1s
- ```preprocessing_data.m``` (which includes SGD, AdaGrad, ADAM)  
6. Implementation of regularization, fixing the weights during training,  leaky-relu activation or relu, adjustment lr, batch_descend or sample descend (updating after a batch or after a sample)
- ```main_training.m``` (which includes SGD, AdaGrad, ADAM)
7. visualization of weight matrix coefficient changes in a heatmap
- ```plot_history_and_visualization.m``` (which includes SGD, AdaGrad, ADAM)
8. validation stored networks on the 20 test split from 20/80 test/train
- ```plot_validation_mse.m``` 