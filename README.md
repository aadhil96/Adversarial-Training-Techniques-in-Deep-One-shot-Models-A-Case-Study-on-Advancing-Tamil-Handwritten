# Adversarial Training Techniques in Deep One-shot Models: A Case Study on Advancing Tamil Handwritten Character Recognition                                                                                                                                                                                     
## ABSTRACT
Deep Convolutional Neural Networks have become the state-of-the-art methods for image classification tasks. However, one of the most significant limitations is they require a lot of labeled data. In many applications, collecting this much data is sometimes not feasible. One-Shot Learning aims to solve this problem, so we want to study how to recognize Tamil letters using one-shot learning with fewer label data. 

One-shot learning is a categorization problem that aims to classify objects given only a limited number of samples, with the ultimate goal of creating a more human-like learning algorithm. Solving the one-shot learning problem by using a special network structure: Siamese Network.

Character recognition using the image processing method is an old trend and this process requires a lot of datasets. The more data we train the more accurate the results will be. most of the time it is hard to get a huge amount of data in a short while.

To overcome this problem, the Oneshot learning method aims to predict the data more accurately than the image processing method, with a few examples. This inspired us to focus on the one-shot learning approach and continue our research.

## METHODOLOGIES
A Siamese network is a class of neural networks that contains one or more identical networks. We feed a pair of inputs to these networks. Each network computes the features of one input. And, then the similarity of features is computed using their difference or the dot product. For the same class input pairs, the target output is 1 and for different classes input pairs, the output is 0. The research is to develop a one-shot learning method for the identification of handwritten characters of Tamil letters.

![metholodology](https://github.com/aadhil96/Adversarial-Training-Techniques-in-Deep-One-shot-Models-A-Case-Study-on-Advancing-Tamil-Handwritten/blob/8f51d9ba0d1a72474b3414e5de0b0fc075b054ff/simense.png)

The model of the Siamese network can be described as CNN architecture with 2 arms, a right arm, and a left arm. The CNN architecture of a single-arm has 9 layers, including Max Pooling and Convolutional layers of different filter sizes, as described in the paper. These 9 layers work as feature selectors for the CNN architecture. Convolutional layers are initialized with weights having 0 mean 0.01 standard deviation, also the bias hyperparameter of these layers is initialized with a mean value of 0.5 and a standard deviation of 0.01.
By doing this, we have converted the classification problem to a similarity problem. We are training the network to minimize the distance between samples of the same class and increase the inter-class distance. There are multiple kinds of similarity functions through which the Siamese network can be trained like Contrastive loss.

### Contrastive loss
In Contrastive loss, pairs of images are taken. For same class pairs, distance is less between them. For different pairs, the distance is more. Although binary cross-entropy seems like a perfect loss function for our problem, contrastive loss does a better job differentiating between image pairs. 

L = Y * D^2 + (1-Y) * max(margin — D, 0)^2

D is the distance between image features. ‘margin’ is a parameter that helps us push different classes apart.

For validation, the Siemens Network model generates a similarity score between 0 and 1. But just looking at the score it's difficult to ascertain whether the model is really able to recognize similar characters and distinguish dissimilar ones. A nice way to judge the model is N-way one-shot learning. 
For each N-way testing, 50 trials were performed and the average accuracy was computed over these 50 trials.

