# MScdissertation-code
Contains the code for the dissertation "A Study on Human Pose Classification Using Convolutional Neural Networks and Tensor Regression"

## CNN Application (in code/CNNapplication/)
Hyperparameter optimization for Trial A was done using Hyperparameter_Opt_TrialA.py, and for Trial B was done using Hyperparameter_Opt_TrialB.py. From there, we use ThreeDancersTrainingTest.py for Trial A and TrainingOnTwoDancers_TestOnOne.py for Trial B to run the optimal sets of hyperparameters.

## TR Application (in code/TRapplication/)
Hyperparameter optimization for Trials A & B was done using Chapter_4_application_withvalidationset.py , where the main runs would be done using Chapter_4_application.py or Chapter_4_application_withclassweights.py depending on if I am adding class weights or not. The other .py files are custom-made that need to be imported for the TR application to do multi-processing with concurrent.futures package.

## Notes for general use
At times, need to decomment/change some text depending on what you are doing in the code. For example, if testing for Trial B, I would decomment the part where I am loading the data for two dancers as training and one dancer as test and comment out the three dancers as training/test part. Same for the CNN application, depending on whether I am resizing the images, changing the optimization algorithm or CNN architecture.
