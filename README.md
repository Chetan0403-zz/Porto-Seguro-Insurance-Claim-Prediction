# Porto-Seguro-Insurance-Claim-Prediction
Porto Seguro, One of Brazil's largest auto and homeowner insurance companies, has an ongoing Kaggle competition challenging to build a model that predicts the probability that a driver will initiate an auto insurance claim in the next year. The R scripts included in this repository earn top 10% spot with very minimal feature engineering and building basic ensembles. These scripts can be run on a regular CPU and does not need any GPU support.

Data for this competition can be found at the following URL:
https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/data

Download the train.7z and test.7z files to the working directory. Shapes of train and test data are (595212,59) and (892816,58) respectively. The metric of evaluation is Normalized Gini. More details about this metric can be found here -
https://www.kaggle.com/batzner/gini-coefficient-an-intuitive-explanation

The R script 'PS Data Prep v1.R' focuses on building useful features from raw data that can potentially go into making accurate predictions. Specifically, the following are attempted - 
1. Re-scaling continuous variables from categorical variables. For example, scaling each value a numerical feature of driver by the average value of that feature belonging to the same category of drivers. This helps rank / position drivers within certain categories
2. Building clusters of similar sets of features together.
3. Predicting posterior probabilility for highly cardinal features. This link gives more information about this:
   https://dl.acm.org/citation.cfm?id=507538&dl=&preflayout=tabs
4. Converting categorical features into one-hot encoding
5. Attempting to address class imbalance through SMOTE

The options to create lots of new features are limited since the sponsors have not provided (for confidentiality reasons) any information about the features. Thus, one does not have true knowledge what the raw features mean or represent.

The R script 'PS training + ensemble v1.R' creates two training models a. XGBoost Model with hyper-parameters already tuned using Bayesian optimization. Basic script for Bayesian Optimization is included (hyper-opt.R) b. A LightGBM model with minimal hyper parameter tuning. These two models can further be stacked with another model (more like XGBoost) or the average of their probabilities taken for final submission.
