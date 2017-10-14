setwd ("C:/Users/Chetan Bhat/Dropbox/2. Data Science/Data Sciences/Kaggle/3. [10-10-2017] Porto Seguro Safe Driver Prediction")

# Data wrangling
library(lightgbm)
library(dplyr)
library(data.table)
library(tibble)
library(lazyeval)
library(caret)
library(dummies)
library(ROSE) # For Addressing class imbalance
library(DMwR) # For ADdressing class imbalance

## Define eval metric: Normalized gini
NormalizedGini <- function(aa, pp) {
  Gini <- function(a, p) {
    if (length(a) !=  length(p)) stop("Actual and Predicted need to be equal lengths!")
    temp.df <- data.frame(actual = a, pred = p, range=c(1:length(a)))
    temp.df <- temp.df[order(-temp.df$pred, temp.df$range),]
    t <- sum(as.numeric(cumsum(temp.df$actual)))/sum(temp.df$actual)
    t = t - (length(a) + 1) / 2
    return(t/length(a))
  }
  Gini(aa,pp) / Gini(aa,aa)
}

xgb_normalizedgini <- function(preds, dtrain){
  actual <- getinfo(dtrain, "label")
  score <- NormalizedGini(actual, preds)
  return(list(metric = "NormalizedGini", value = score))
}

lgb_normalizedgini <- function(preds, dtrain){
  actual <- getinfo(dtrain, "label")
  score <- NormalizedGini(actual, preds)
  return(list(name = "NormalizedGini", value = score, higher_better = TRUE))
}

## Load data and Perform basic hygiene
raw_train <- as.tibble(fread('./1. Input/train.csv', na.strings=c("-1","-1.0")))
raw_test <- as.tibble(fread('./1. Input/test.csv', na.strings=c("-1","-1.0")))

data <- bind_rows(raw_train,raw_test)
colnames(data)[colnames(data) == "id"] <- "driver_id"
colnames(data) <- gsub("ps_","",names(data)) #removing 'ps_' from column names for brevity

# Dropping calc features from the dataset (Initial run of XGBoost showed no importance)
data <- data[, !(names(data) %in% names(data)[grepl("calc", names(data))])]

# Handling data-types adequately
# Convert non-categorical features into numeric
colnames_num <- names(data)[!(names(data) %in% names(data)[grepl("_cat",names(data))])]
colnames_num <- colnames_num[!(colnames_num %in% c("driver_id", "target"))]
data[,colnames_num] <- apply(data[,colnames_num], 2, function(x) as.numeric(x)) 

## Creating features already tested at Kaggle
#1. Kernel - XGBoost starter [0.280]
data$amount_nas <- rowSums(data == -1, na.rm = T)
data$high_nas <- ifelse(data$amount_nas>4,1,0)
data$car_13_reg_03 <- data$car_13*data$reg_03
data$reg_mult <- data$reg_01*data$reg_02*data$reg_03
data$ind_bin_sum <- data$ind_06_bin+data$ind_07_bin+data$ind_08_bin+data$ind_09_bin+
  data$ind_10_bin+data$ind_11_bin+data$ind_12_bin+data$ind_13_bin+data$ind_16_bin+data$ind_17_bin+
  data$ind_18_bin
data$car_mult <- data$car_01_cat*data$car_02_cat*data$car_03_cat*data$car_04_cat*data$car_05_cat*data$car_06_cat*
  data$car_07_cat*data$car_08_cat*data$car_09_cat*data$car_10_cat*data$car_11_cat # not in kernel
data$car_mult <- as.numeric(data$car_mult)
data$reg_add <- data$reg_01 + data$reg_02 # not in kernel

#2. Kernel - Kinetic energy + Forza baseline
kinetic <- function(row)
{
  row <- unlist(row, use.names = FALSE)
  len <- length(row)
  row <- as.vector(table(row))
  return(sum((row/len)^2))
}
data$ind_kinetic <- apply(data[,names(data)[grepl("ind", names(data))]], 1, function(x) kinetic(x))
data$car_kinetic <- apply(data[,names(data)[grepl("car", names(data))]], 1, function(x) kinetic(x))

## Category level rescaling of continuous features
category_cols <- c('car_01_cat','car_06_cat','car_11_cat', 'reg_01', 'reg_02', 'reg_add', 'ind_bin_sum')
scale_cols <- c('ind_01', 'ind_03', 'ind_15', 'car_11', 'car_12', 'car_13', 'car_14', 'car_15')

# Rescale function
rescale <- function(df, cat_col, scale_col){
  col <- df %>%
    group_by_(cat_col) %>%
    mutate_(y = interp(~mean(x), x = as.name(scale_col))) %>%
    ungroup 
  
  col <- col[,c(scale_col,'y')]
  col[,paste0(scale_col,"_",cat_col,"_scale")] <- col[,1]/col$y
  col <- col[,3]
  return(col)
}

# Create rescaled data
rescaled <- data$driver_id
for (i in 1:length(category_cols))
{
  for (j in 1:length(scale_cols))
  {
    rescaled <- cbind(rescaled,rescale(data, category_cols[i], scale_cols[j]))
  }
}
rescaled <- rescaled[,-1]

## Clustering similar sets of features together
# ind continuous
ind <- data[, names(data)[grepl("ind", names(data))]]
ind <- ind[, !(names(ind) %in% names(ind)[grepl("_cat|bin", names(ind))])] # Dropping cat columns
ind[is.na(ind)] <- -1 # k-means clustering doesn't take NA/NaN rows

# car continuous
car <- data[, names(data)[grepl("car", names(data))]]
car <- car[, !(names(car) %in% names(car)[grepl("_cat", names(car))])] # Dropping cat columns
car$car_13_reg_03 <- NULL
car[is.na(car)] <- -1

# reg continuous
reg <- data[, names(data)[grepl("reg", names(data))]]
reg <- reg[, !(names(reg) %in% names(reg)[grepl("_cat", names(reg))])] # Dropping cat columns
reg$car_13_reg_03 <- NULL
reg$reg_mult <- NULL
reg$reg_add <- NULL
reg[is.na(reg)] <- -1

# Output cluster indexes based on number of clusters provided
createclus <- function(df, nc){
  
  k.means.fit <- kmeans(df, nc, iter.max = 20)
  df.name <- deparse(substitute(df))
  t <- data.frame(k.means.fit$cluster)
  colnames(t)[colnames(t) == "k.means.fit.cluster"] <- paste0(df.name, "_clus_num")
  return(t)
  
}

ind <- createclus(ind, 10)
car <- createclus(car, 10)
reg <- createclus(reg, 5)

# Plot of within group sum of squares with no. of clusters: Used to assess optimal number of clusters
# wssplot <- function(df, nc=15, seed=1234){
#   wss <- (nrow(df)-1)*sum(apply(df,2,var))
#   for (i in 2:nc){
#     set.seed(seed)
#     wss[i] <- sum(kmeans(df, centers=i)$withinss)}
#   plot(1:nc, wss, type="b", xlab="Number of Clusters",
#        ylab="Within groups sum of squares")}
# 
# wssplot(ind_cat_bin, nc=15)

## Convert categorical to one hot
one_hot <- data[, names(data)[grepl("cat", names(data))]]
one_hot$car_11_cat <- NULL # has 104 level. Dataset becomes too large if one-hot encoded
one_hot <- dummy.data.frame(one_hot, sep="_")
one_hot <- dummy.data.frame(one_hot, names = names(one_hot), sep="_")
one_hot <- one_hot[, !(names(one_hot) %in% names(one_hot)[grepl("_NA", names(one_hot))])]

## Putting all datasets together
data_xgb <- cbind(data 
                  ,ind, reg, car 
                  ,rescaled) #xgb unable to handle 100+ features

data_lgb <- cbind(data[, !(names(data) %in% names(data)[grepl("_cat", names(data))])] #original without categoricals
                    ,ind, reg, car 
                    ,rescaled
                    ,one_hot)

## Deleting other tables to free up memory
rm("one_hot", "data", "car", "ind", "reg", "temp", "rescaled", "raw_train", "raw_test")
gc()
