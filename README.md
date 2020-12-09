# EmotionRecognition

This repository was used to train couple models (ResNet50, CorNET-S) on the AffectNet dataset. 

>AffectNet: A New Database for Facial Expression, Valence, and Arousal Computation in the Wild” IEEE Transactions on Affective Computing,2017

by Ali Mollahosseini, Behzad Hasani, and Mohammad H. Mahoor. The models have been train on a subset of the dataset using only 8 categories. 

You will also find an implementation of the cluster loss from the paper: 

>"Facial Expression Recognition Based on Weighted-Cluster Loss and Deep Transfer Learning Using a Highly Imbalanced Dataset" Sensors, 2020

by Quan T. Ngo and Seokhoon Yoon

## Results

### MNIST with cluster loss
We tried to train the cluster loss on a small CNN to visualize the effect of the cluster loss on the MNIST dataset.

![MNIST_Clusters](/figures/MNIST_cluster_180epochs.png)

Cluster separation at the last layer (2 features)

![MNIST_accurac](/figures/MNIST_accuracy_180epochs.png)

Accuracy plot over epochs

![MNIST_loss](/figures/MNIST_loss_180epochs.png)

Loss plot over epochs