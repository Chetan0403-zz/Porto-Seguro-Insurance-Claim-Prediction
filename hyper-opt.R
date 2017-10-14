# XGBoost
library(rBayesianOptimization)

x_train <- train[,-2] 
y_train <- as.integer(factor(train[,2]))-1
dtrain <- xgb.DMatrix(data = data.matrix(subset(data_xgb, !is.na(target))[,-2]), 
                      label = data.matrix(subset(data_xgb, !is.na(target))[,2]))

cv_folds <- KFold(y_train, nfolds = 5,stratified = TRUE, seed = 0)

xgb_cv_bayes <- function(max_depth, min_child_weight) {
  cv <- xgb.cv(params = list(eta = 0.1,
                             max_depth = max_depth,
                             min_child_weight = min_child_weight,
                             subsample = 0.8, 
                             colsample_bytree = 0.8,
                             gamma = 0
                             #lambda = 1, 
                             #alpha = 0,
  ),
  objective = 'binary:logistic',
  feval = xgb_normalizedgini,
  data = dtrain,
  nround = 1500,
  folds = cv_folds, 
  prediction = TRUE, 
  showsd = TRUE,
  #num_class = 2,
  early_stopping_rounds = 50, 
  print_every_n = 25,
  seed = 50,
  maximize = TRUE)
  
  list(Score = cv$evaluation_log[,max(test_NormalizedGini_mean)],
       Pred = cv$pred)
}

OPT_Res <- BayesianOptimization(xgb_cv_bayes,
                                bounds = list(max_depth = c(5L, 9L),
                                              min_child_weight = c(5L,9L)),
                                init_grid_dt = NULL, init_points = 10, n_iter = 50,
                                acq = "ucb", kappa = 1.0, eps = 0.0,
                                verbose = TRUE)