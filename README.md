# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
**This dataset is the UCI Bank Marketing dataset provided here ().
The goal is to predict if the client will subscribe to a term deposit with the bank (labelled in column 'y').**

**Some details on the dataset:**

-- The dataset is from a Portuguese bank

-- The incidence of customer term deposit subscription is marked as 'yes or no' in column 'y.'

-- There are a total of 45211 rows and 21 columns

-- There are 0 missing rows but nine duplicate columns

-- The duplicates are because, the same customers might be called more than once to confirm their term deposit subscription

-- There are 10 Numerical, 10 Categorical Columns and one binary column

-- Employment Variation Rate (emp.var.rate), Previous Marketing Campain Outcome (poutcome) and Euro Inter-Bank Offered Rate (euribor3m) have good correlation to the target variable 'y'

**Some details on the ML ALgorithm:**

-- We use Logistic Regression to classify the customer outcome

-- Logistic Regression is used to model the probability of a particular class or event based on the various features available in Data

-- In here we use, most of the features as is (age, marital, default, housing, loan, month, day_of_week, duration, campaign, pdays, previous, poutcome, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed)

-- These features are one-hot encoded (job_admin., job_blue-collar, job_entrepreneur, job_housemaid, job_management, job_retired, job_self-employed, job_services, job_student, job_technician, job_unemployed, job_unknown, contact_cellular, contact_telephone, education_basic.4y, education_basic.6y, education_basic.9y, education_high.school, education_illiterate, education_professional.course, education_university.degree, education_unknown)

-- The logistic regression algorithm tries to calculate the probability of the final class based on the weights associated with each feature

-- If the final probability is <=0.5 the customer is considered to not opt for a term deposit, else the customer is said to opt for a term deposit

**Some details on the hyperparameters:**

-- The hyperparameters of focus were *The Inverse of Regularization Strength* and *The number of maximum iterations*

***The Inverse of Regularization Strength (C)***: We use the default setting of L2 regularization also known as ridge regularization from Scikit-Learn module. L2 regularization makes the weights of less important features to be small values, and the higher the value of regularization strength C, the smaller the feature weights will be.

***The number of maximum iterations (max_iter)***:  The number of Iterations taken for the default liblinear Solver to converge. Liblinear uses a one-vs-the-rest type of classification.

**We have used Hyperdrive from Azure Machine Learning to tune the Hyperparameters. Hyperdrive lets us programmatically define: a search space, early termination policy, run experiments with different settings, etc.**

**Here we have set the following parameters for Hyperdrive:**

**Search Space**: For C we set a uniform distribution between 0 and 1.0. For max_iter we use values between 1 and 100

**Chosen Sampler:** Random Parameter Sampler which does a random sweep of the given arguments. It does not search through the entire space like Grid Sampler or does not care about the previous outcome like the Bayesian Sampler. Hence it is lightweight and easy to execute with minimal resources. Some people use Random Sampler to arrive at the best region in the search space initially and then use Grid or Bayesian sampling in the narrowed search space. Also, Random Search allows early termination while Bayesian Sampler does not. The choice of this policy is significant for us, as we strive to minimize the use of cloud resources.

The Arguments:

-- a uniform distribution of values between 0 and +1 for regularization strength

-- the number of maximum iterations between a range of 1 and 100
 
**Chosen Policy:** Bandit Policy. is our choice. Bandit policy terminates a run based on the outcome of the slack factor calculation on the primary metric over the evaluation runs specified. Median stopping policy calculates the running averages and truncation policy cancels the percentage of lowest-performing runs. The slack factor is more comfortable to explain how a particular run was evaluated compared to others. 

Metrics:

 -- 50 Evaluation Intervals
 
 -- Slack Factor of 10
 
Due to the metric values, the bandit policy checks if the Accuracy of the current model after 50 intervals is within the 10% of all other models, else it terminates the particular model run


## AutoML
**The AutoML model was Voting Ensemble that predicts based on the weighted average of predicted class probabilities**

The AutoML feature checked for the following:

-- Class Imbalance: The positive customer class was heavily imbalanced (~10% of the training data)

-- Missing Values: None

-- High Cardinality: No

The Voting Ensemble used ***MaxAbs Scaler (Scaling the features) with XGBoost (Extreme Gradient Boosting a modified Stochastic Gradient Boosting model for classification)***
**The Feature Importance**: Globally ***Duration*** and ***Employed Net Rate*** features were found to be influential. 

 
**The best performing model was Voting Ensemble found using AutoML. It had an accuracy of 91.5% whereas the logistic regression model tuned with Hyperdrive had the best Accuracy of 91.18% "**

## Scikit-learn Pipeline
**1. Clean Data**

**2. Split the cleansed data into test and train datasets**

**3. Run a logistic regression model with the regularization strength and maximum iterations supplied by Hyperdrive**

**4. Calculate the Accuracy of a model and log it**

**5. Save the model data to output folder**


## Pipeline comparison

**The logistic regression classifies based on threshold and Voting Ensemble works based on average of probabilities.**

**Jargons Explained**

***Threshold***: The value with which the class probability output from Logistic Regression Estimator is compared. If the probability <= Threshold, the entry is classified as Negative and if the probability > Threshold, the entry is classified as Positive

***Average of Probabilities***:
Multiple models are given the same input, and their output probability estimates are stored. Finally, all the probabilities are averaged to predict the class outcome of the entry.

***Difference***:
The difference in Accuracy is not much, but the effort and time taken were relatively less using AutoML. AutoMl also provides additional insights by automatically bundling Explanations of Feature Importance and also logs other useful metrics apart from the metric chosen by the user. This is helpful when we want to compare models.**

## Future work

**AutoML could be set up with a different metric (AUC, ROC, Precision & Recall, etc.) than Accuracy to identify the best performing model. Relying on Accuracy alone might lead us to choose an underperforming model.**
