setwd ("C:/Users/Chetan Bhat/Dropbox/2. Data Science/Data Sciences/Kaggle/3. [10-10-2017] Porto Seguro Safe Driver Prediction")

library(dplyr)
library(xgboost)
library(lightgbm)
library(caret)

## Fixed list of indices for stacking procedure. This should stay fixed across all models
k=5
id <- createFolds(subset(data_xgb, !is.na(target))[,2], k=k, list=FALSE, returnTrain=TRUE)
#id <- sample(1:k,nrow(subset(data_xgb, !is.na(target))),replace=TRUE)
list <- 1:k
##

## Set model parameters
xgb_param <- list(booster="gbtree",
              objective="binary:logistic",
              eta = 0.02,
              gamma = 0,
              max_depth = 5,
              min_child_weight = 7,
              subsample = 1.0,
              colsample_bytree = 0.3681)

detach("package:lightgbm", unload = TRUE) # Lightgbm still loaded causes xgb training issue. 

#################################
## XGBoost Meta features creation
#################################
data_xgb_meta <- data.frame()

for(i in 1:k)
{
  train <- subset(subset(data_xgb, !is.na(target)), id %in% list[-i])
  test <- subset(subset(data_xgb, !is.na(target)), id %in% c(i))
  
  xgb <- xgboost(data = data.matrix(train[-2])
                       ,label = data.matrix(train[,2])
                       ,params = xgb_param
                       ,feval = xgb_normalizedgini
                       ,nrounds = 773
                       ,print_every_n = 50
                       ,maximize = TRUE)
  
  pred <- predict(xgb_train, data.matrix(test[,-2]))
  pred <- cbind(test[,c(1:2)], data.frame(pred))
  data_xgb_meta <- rbind(data_xgb_meta, pred)
}

rm("train", "test", "pred", "xgb")
gc()

colnames(data_xgb_meta) <- c("driver_id", "actual", "xgb_pred")
gini <- NormalizedGini(data_xgb_meta$actual, data_xgb_meta$xgb_pred)
print(paste0("xgb gini is ", round(gini,4)))

# Create test xgb meta
xgb <- xgboost(data = data.matrix(subset(data_xgb, !is.na(target))[-2])
               ,label = data.matrix(subset(data_xgb, !is.na(target))[,2])
               ,params = xgb_param
               ,feval = xgb_normalizedgini
               ,nrounds = 150
               ,print_every_n = 50
               ,maximize = TRUE)

test_xgb_meta <- predict(xgb, data.matrix(subset(data_xgb, is.na(target))[-2]))
test_xgb_meta <- cbind(subset(data_xgb, is.na(target))[,1], data.frame(test_xgb_meta))
colnames(test_xgb_meta) <- c("driver_id", "xgb_pred")

rm("xgb")

## XGBoost cross-validation
detach("package:lightgbm", unload = TRUE) 
train <- subset(data_xgb, !is.na(data_xgb$target))
test <- subset(data_xgb, is.na(data_xgb$target))
dtrain_xgb <- xgb.DMatrix(data = data.matrix(train[,-2]), label = data.matrix(train[,2]))
cvFolds <- createFolds(train$target, k=5, list=TRUE, returnTrain=FALSE)
rm("train", "test")
gc()

param <- list(booster="gbtree",
              objective="binary:logistic",
              eta = 0.02,
              gamma = 0,
              max_depth = 5,
              min_child_weight = 7,
              subsample = 1.0,
              colsample_bytree = 0.3681)

xgb_cv <- xgb.cv(data = dtrain_xgb,
                 params = param,
                 nrounds = 5000,
                 feval = xgb_normalizedgini,
                 maximize = TRUE,
                 prediction = TRUE,
                 folds = cvFolds,
                 print_every_n = 50,
                 early_stopping_round = 50)

###################################
## LightGBM meta features creation
###################################
require(lightgbm)

train_lgb_meta <- data.frame()
lgb.unloader(wipe = TRUE)

for(i in 1:k)
{
  train <- subset(subset(data_lgb, !is.na(target)), id %in% list[-i])
  train[is.na(train)] <- -1
  test <- subset(subset(data_lgb, !is.na(target)), id %in% c(i))
  
  lgb <- lgb.train(list()
                   ,120
                   ,data = lgb.Dataset(data = as.matrix(train[-2]), label = as.matrix(train[,2]))
                   #,label = as.matrix(train[,2])
                   ,learning_rate = 0.1
                   ,boosting = "gbdt"
                   ,objective = "binary"
                   ,eval = lgb_normalizedgini
                   ,eval_freq = 20
                   #,num_leaves = 12
                   ,max_depth = 6
                   ,max_bin = 6
                   ,bagging_fraction = 1
                   ,num_threads = 2
                   ,min_data = 20)
  
  pred <- predict(lgb, as.matrix(test[,-2]))
  pred <- cbind(test[,c(1:2)], data.frame(pred))
  train_lgb_meta <- rbind(train_lgb_meta, pred)
}

rm("train", "test", "pred", "lgb")
gc()

colnames(train_lgb_meta) <- c("driver_id", "actual", "lgb_pred")
gini <- NormalizedGini(train_lgb_meta$actual, train_lgb_meta$lgb_pred)
print(paste0("lgb gini is ", round(gini,4)))

# Create test xgb meta
lgb <- lgb.train(list()
                 ,120
                 ,data = lgb.Dataset(data = subset(data_xgb, !is.na(target))[-2], 
                                     label = as.matrix(subset(data_xgb, !is.na(target))[,2]))
                 ,learning_rate = 0.1
                 ,boosting = "gbdt"
                 ,objective = "binary"
                 ,eval = lgb_normalizedgini
                 ,eval_freq = 20
                 #,num_leaves = 12
                 ,max_depth = 6
                 ,max_bin = 6
                 ,bagging_fraction = 1
                 ,num_threads = 2
                 ,min_data = 20)

test_lgb_meta <- predict(lgb, as.matrix(subset(data_xgb, is.na(target))[-2]))
test_lgb_meta <- cbind(subset(data_lgb, is.na(target))[,1], data.frame(test_lgb_meta))
colnames(test_lgb_meta) <- c("driver_id", "lgb_pred")

rm("lgb")

## LightGBM CV
# train <- subset(data_lgb, !is.na(data_lgb$target))
# x_train <- as.matrix(train[,-2])
# x_train[is.na(x_train)] <- -1
# y_train <- as.matrix(train[,2])
# 
# lgb.unloader(wipe = TRUE)
# lgb_cv <- lgb.cv(list()
#                  ,5000
#                  ,data = x_train
#                  ,label = y_train
#                  ,learning_rate = 0.1
#                  ,boosting = "gbdt"
#                  ,objective = "binary"
#                  ,nfold = 5
#                  ,early_stopping_rounds = 20
#                  ,eval = lgb_normalizedgini
#                  ,eval_freq = 20
#                  #,num_leaves = 12
#                  ,max_depth = 7
#                  ,max_bin = 7
#                  ,bagging_fraction = 1
#                  ,num_threads = 2
#                  ,min_data = 20)

