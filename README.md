#Digit Recognizer

This Digit Recognizer package provides an example of the use of supervised (machine) learning methods to classify a large dataset. In the present case, the dataset consists in a large set of images representing handwritten digits (0 to 9). The goal is to be able to predict to which class (i.e. which digit) a given instance (image) belongs to. 

This dataset has been studied extensively (see [Yann LeCun's webpage][MNIST]); current best efforts with state of the art classification methods achieve remarkable success rates. The aim of this digit recognizer is not to improve on those results but to provide a testbed for understanding the output of a classifier and making the most out of it. Here, decision trees in the form of random forests will be used since they can readily handle categorical data in a multiclass classifcation problem. 

The whole process is divided in two steps:

1. A feature extraction algorithm processes all the images and computes a number of observables in order to reduce the complexity of the problem.
2. Using those observables, the random forest classifer is trained and then used to classify the images according to the digits they represent. 

The functionality of the random forest classifier is enhanced by providing _scores_ (or probability estimates) of each digit for every image. Those scores are then calibrated to the true classifier response. Instead of being able to only predict the most likely digit an image represents (this image represents a 3), the calibration procedure provides us with the probabilities for an image to represent a given digit (10% chance it's a 0, 1% chance it's a 1, 5% chance it's a 2, 65% chance it's a 3, etc. ).
















[MNIST]: http://yann.lecun.com/exdb/mnist/
