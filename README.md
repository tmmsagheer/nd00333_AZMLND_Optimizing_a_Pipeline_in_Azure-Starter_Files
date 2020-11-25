# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**"This dataset is the UCI Bank Marketing dataset and the goal is to predict if the client will subscribe to a term deposit with the bank."**

**The best performing model was Voting Ensemble found using AutoML. It had an accuracy of 91.5% whereas the logistic regression model tuned with Hyperdrive had best accuracy of 91.18% "**

## Scikit-learn Pipeline
**1. Clean Data**
**2. Split the cleansed data into test and train datasets**
**3. Run a logistic regression model with the regularization strength and maximum iterations supplied by Hyperdrive**
**4. Calculate the accuracy of a model and log it**
**5. Save the model data to output folder**

**What are the benefits of the parameter sampler you chose?**
**Chosen Sampler:** Random Parameter Sampler which does a random sweep of the given arguments.
The Arguments:
 -- a uniform ditribution of values between 0 and +1 for regularization strength
 -- number of maximum iterations between a range of 1 and 100
**What are the benefits of the early stopping policy you chose?**
**Chosen Policy:** Bandit Policy
Metrics:
 -- 50 Evaluation Intervals
 -- Slack Factor of 10
Due to the metric values, the bandit policy checks if the accuracy of the current model after 50 intervals is within the 10% of all other models, else it terminates the particular model run

## AutoML
**The AutoML model was Voting Ensemble that predicts based on the weighted average of predicted class probabilities**

## Pipeline comparison
**The logistic regression classifies based on threshold and Voting Ensemble works based on average of probabilities.
The difference in Accuracy is not much but the effort and time taken were relatively less using AutoML.
Also the AutoML ran more than 40 different experiments**

## Future work
**AutoML could be set up with a different metric than accuracy to identify the best performing model**
