
getwd()

library(caret)
library(mlbench)
library(AppliedPredictiveModeling)
#Load required Libraries
library(tidyverse)
library(ggplot2)
library(lattice)
library(caret)
library(dplyr)
library(tidyr)
library(ggplot2)
library(corrplot)
library(mltools)
library(smotefamily)
library(VIM)
library(moments)
library(naniar)
library(MASS)
library(mda)
library(pROC)
library(kernlab)


mushroom <- read.csv("secondary_data.csv", sep = ";" )
summary(mushroom)
str(mushroom)
dim(mushroom)
table(mushroom$class)

#Find for missing Values ##########################################################
sum(mushroom == "")
mushroom[mushroom == ""] <- NA
# Recheck for missing values after replacing empty strings
sapply(mushroom[, sapply(mushroom, is.character)], function(x) sum(is.na(x)))


# Remove these columns from the dataset which has more than 60% of missing data#####
missing_proportions <- colSums(is.na(mushroom)) / nrow(mushroom)
# Identify columns with more than 60% missing values
columns_to_remove <- names(missing_proportions[missing_proportions > 0.6])
mushroom <- dplyr::select(mushroom, -all_of(columns_to_remove))

dim(mushroom)
names(mushroom)

# # Perform KNN imputation #######################################################
# # Apply KNN imputation on the dataset
# # You can specify which columns to include, or use the whole dataset
# mushroom_knn_imputed <- kNN(mushroom,
#                             variable = c("cap.surface", "gill.attachment", "gill.spacing", "ring.type"),
#                             k = 5)
# # Save the KNN-imputed dataframe to a CSV file
# write.csv(mushroom_knn_imputed, "mushroom_knn_imputed.csv", row.names = FALSE)
# dim(mushroom_knn_imputed)

mushroom_knn_imputed <- read.csv("mushroom_knn_imputed.csv")
# Extract only the imputed columns (without the ".imp" suffix)
mushroom$cap.surface <- mushroom_knn_imputed$cap.surface
mushroom$gill.attachment <- mushroom_knn_imputed$gill.attachment
mushroom$gill.spacing <- mushroom_knn_imputed$gill.spacing
mushroom$stem.surface <- mushroom_knn_imputed$stem.surface
mushroom$ring.type <- mushroom_knn_imputed$ring.type
# Now, the mushroom_knn_imputed dataset has imputed missing values using KNN
str(mushroom)
sum(mushroom == "")
sapply(mushroom[, sapply(mushroom, is.character)], function(x) sum(is.na(x)))

# Create a subset of the predictor variables (exclude 'Class')
mushroom_predictors <- mushroom[, -which(names(mushroom) == "class")]
dim(mushroom_predictors)
dim(mushroom_predictors[, sapply(mushroom_predictors, is.numeric)])

numeric_vars <- mushroom_predictors[, sapply(mushroom_predictors, is.numeric)]

# Calculate the skewness of continuous numerical predictors
skewness_values <- apply(numeric_vars, 2, skewness)
skewness_table <- data.frame(
  Predictor = colnames(numeric_vars),
  Skewness = skewness_values
)
skewness_table

##categorical_vars##############################################################
categorical_vars <- mushroom_predictors[, sapply(mushroom_predictors, is.character)]
dim(categorical_vars)
# Convert Binary Variables using Label Encoding (0/1)
categorical_vars$does.bruise.or.bleed <- ifelse(categorical_vars$does.bruise.or.bleed == "t", 1, 0)
categorical_vars$has.ring <- ifelse(categorical_vars$has.ring == "t", 1, 0)
str(categorical_vars)
mushroom_predictors[, names(categorical_vars)] <- categorical_vars
str(mushroom_predictors)
length(mushroom_predictors[, sapply(mushroom_predictors, is.character)])

# Ensure that the categorical variables are factors
mushroom_predictors <- mushroom_predictors %>%
  mutate_if(is.character, as.factor)
# Use model.matrix for one-hot encoding
mushroom_predictors <- as.data.frame(model.matrix(~ . - 1, data = mushroom_predictors))
# Convert all factor variables to numeric
mushroom_predictors <- mushroom_predictors %>%
  mutate_if(is.factor, ~ as.numeric(as.factor(.)))
str(mushroom_predictors)

#remove highly correlated and nearzero variance predictors
nzv <- nearZeroVar(mushroom_predictors, saveMetrics = TRUE)
str(nzv)
rownames(nzv[nzv$nzv == TRUE, ])
mushroom_predictors <- mushroom_predictors[, !nzv$nzv]
str(mushroom_predictors)

corr_matrix <- cor(mushroom_predictors)
corrplot(corr_matrix, order = "hclust")
high_corr <- findCorrelation(corr_matrix, cutoff = 0.9) # Set your cutoff (e.g., 0.9)
length(high_corr)
high_corr
colnames(mushroom_predictors)[high_corr]
str(high_corr)
mushroom_predictors <- mushroom_predictors[, -high_corr]
str(mushroom_predictors)

# write.csv(mushroom_predictors, "mushroom_predictors_final.csv", row.names = FALSE)

###Transformations on continuous numerical variable######################################
trans_df <- preProcess(mushroom_predictors[, c("cap.diameter", "stem.width", "stem.height")], method = c("BoxCox", "center", "scale"))  ## need {caret} package
# apply the transformation to your dataset, you’ll need to use the predict() function,
mushroom_transformed <- predict(trans_df, mushroom_predictors)
str(mushroom_transformed)

# Calculate skewness values for the selected predictors
skewness_values <- apply(mushroom_transformed[, c("cap.diameter", "stem.width", "stem.height")], 2, skewness)
# Create a table with predictor names and their corresponding skewness values
skewness_table <- data.frame(
  Predictor = colnames(mushroom_transformed[, c("cap.diameter", "stem.width", "stem.height")]),
  Skewness = skewness_values
)
skewness_table

###Reduce the outliers using spatial sign 
# Apply Spatial Sign transformation
spatial_sign_transformed <- spatialSign(mushroom_transformed[, c("cap.diameter", "stem.width", "stem.height")])
mushroom_transformed[, c("cap.diameter", "stem.width", "stem.height")] <- spatial_sign_transformed
str(mushroom_transformed)

# Stratified random split (75% training, 25% testing)
# Combine predictors and target variable into a single dataset
mushroom_data <- data.frame(mushroom_transformed, class = as.factor(mushroom$class))
dim(mushroom_data)
# Set the seed for reproducibility
set.seed(123)

# Create stratified random split (75% training, 25% testing) based on the 'injury' class
train_index <- createDataPartition(mushroom_data$class, p = 0.75, list = FALSE, times = 1)
training_set <- mushroom_data[train_index, ]
testing_set <- mushroom_data[-train_index, ]
training_labels <- training_set$class
testing_labels <- testing_set$class

# Print dimensions of training and testing sets
dim(training_set)
dim(testing_set)

# Check distribution of classes in training and testing sets
table(training_set$class)
table(testing_set$class)

#####Resampling: 10-fold Cross Validation###
train_control <- trainControl(method = "cv",
                              number = 10,
                              summaryFunction = twoClassSummary,
                              classProbs = TRUE)


############ Logistic Regression ###############
lrFull <- train(x = training_set[, -ncol(training_set)],
                y = training_set$class,
                method = "glm",
                metric = "ROC",
                trControl = train_control)

lrFull
summary(lrFull)
# plot(lrFull)
# Predict on testing set
lr_predictions <- predict(lrFull, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(lr_predictions, testing_set$class)

# ROC Curve plot
lr_predicted_prob <- predict(lrFull, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, lr_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - Logistic Regression", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc


############ Linear Discriminant Analysis############
lda_model <- train( x = training_set[, -ncol(training_set)],
                    y = training_set$class,
                    method = "lda",
                    metric = "ROC",
                    trControl = train_control
)
print(lda_model)
# plot(lda_model)
# Predict on testing set
lda_predictions <- predict(lda_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(lda_predictions, testing_set$class)

# ROC Curve plot
lda_predicted_prob <- predict(lda_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, lda_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - LInear Discriminant Analysis", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc

############ Quadratic Discriminant Analysis ###########
qda_model <- train( x = training_set[, -ncol(training_set)],
                    y = training_set$class,
                    method = "qda",
                    metric = "ROC",
                    trControl = train_control
)
print(qda_model)
# plot(qda_model)
# Predict on testing set
qda_predictions <- predict(qda_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(qda_predictions, testing_set$class)

# ROC Curve plot
qda_predicted_prob <- predict(qda_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, qda_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - Quadratic Discriminant Analysis", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc



########Neural Network########
nnet_model <- train( x = training_set[, -ncol(training_set)],
                     y = training_set$class,
                     method = "nnet",
                     tuneGrid = expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2)),
                     metric = "ROC",
                     preProc = c("center", "scale"),
                     trControl = train_control,
                     trace = FALSE)


print(nnet_model)
plot(nnet_model)
# Predict on testing set
nnet_predictions <- predict(nnet_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(nnet_predictions, testing_set$class)

# ROC Curve plot
nnet_predicted_prob <- predict(nnet_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, nnet_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - Quadratic Discriminant Analysis", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc


##### Mixture Discriminant Analysis ##########
mdaGrid <- expand.grid(
  .subclasses = seq(1, 20, by = 1)  # Test 1 to 10 subclasses per class
)

mda_model <- train( x = training_set[, -ncol(training_set)],
                    y = training_set$class,
                    method = "mda",
                    tuneGrid = mdaGrid,
                    metric = "ROC",
                    trControl = train_control,
                    trace = FALSE)


print(mda_model)
plot(mda_model)
# Predict on testing set
mda_predictions <- predict(mda_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(mda_predictions, testing_set$class)

# ROC Curve plot
mda_predicted_prob <- predict(mda_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, mda_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - Mixture Discriminant Analysis", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc



# Mixture Discriminant Analysis (MDA)
mda_model <- train(
  x = filtered_train[, -ncol(filtered_train)], 
  y = filtered_train$Diagnosis,
  method = "mda",
  tuneGrid = expand.grid(.subclasses = seq(1, 10, by = 1)), 
  metric = "ROC",
  trControl = trainControl(method = "repeatedcv", 
                           number = 10, 
                           repeats = 5, 
                           classProbs = TRUE, 
                           summaryFunction = twoClassSummary)
)
print(mda_model)
plot(mda_model)
summary(mda_model)
filtered_test$Diagnosis <- factor(filtered_test$Diagnosis, levels = c("B", "M"))
predicted_labels <- predict(mda_model, newdata = filtered_test[, -ncol(filtered_test)])
predicted_probs <- predict(mda_model, newdata = filtered_test[, -ncol(filtered_test)], type = "prob")
conf_matrix <- confusionMatrix(factor(predicted_labels), factor(filtered_test$Diagnosis), positive = "M")
print(conf_matrix)
roc_auc <- roc(filtered_test$Diagnosis, predicted_probs[, "M"], levels = c("B", "M"))
plot(roc_auc, main = "ROC Curve - MDA", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)

######### Regularized Discriminant ANalysis ##############
rdaGrid <- expand.grid(
  .gamma = seq(0, 1, length = 10),  # Mix of LDA and QDA
  .lambda = seq(0, 1, length = 15) # Regularization strength
)
rda_model <- train( x = training_set[, -ncol(training_set)],
                    y = training_set$class,
                    method = "rda",
                    tuneGrid = rdaGrid,
                    metric = "ROC",
                    trControl = train_control,
                    trace = FALSE)

print(rda_model)
plot(rda_model)
# Predict on testing set
rda_predictions <- predict(rda_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(rda_predictions, testing_set$class)

# ROC Curve plot
rda_predicted_prob <- predict(rda_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, rda_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - Regularized Discriminant Analysis", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc


############ Partial Least Square Discriminant Analysis ###############
plsda_model <- train(x = training_set[, -ncol(training_set)],
                     y = training_set$class,
                     method = "pls",
                     tuneGrid = expand.grid(.ncomp = 1:43),
                     metric = "ROC",
                     preProc = c("center", "scale"),
                     trControl = train_control)

plsda_model
summary(plsda_model)
plot(plsda_model)
# Predict on testing set
pls_predictions <- predict(plsda_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(pls_predictions, testing_set$class)

# ROC Curve plot
pls_predicted_prob <- predict(plsda_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, pls_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - PLS Discriminant Analysis", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc

############ Penalized Models #############
penalizedGrid <- expand.grid(
  .alpha = c(0, 0.1, 0.2, 0.4, 0.6, 0.8, 1),  # Test Ridge, Lasso, and Elastic Net
  .lambda = seq(0.01, 0.2, length = 10)       # Regularization strength
)

glmnet_model <- train(x = training_set[, -ncol(training_set)],
                      y = training_set$class,
                      method = "glmnet",
                      tuneGrid = penalizedGrid,
                      metric = "ROC",
                      preProc = c("center", "scale"),
                      trControl = train_control)

glmnet_model
summary(glmnet_model)
plot(glmnet_model)
# Predict on testing set
glmnet_predictions <- predict(glmnet_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(pls_predictions, testing_set$class)

# ROC Curve plot
glmnet_predicted_prob <- predict(glmnet_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, glmnet_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - Penalized Model", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc


############## K-Nearest Neighbours ############
knn_model <- train(x = training_set[, -ncol(training_set)],
                   y = training_set$class,
                   method = "knn",
                   tuneGrid = data.frame(.k = 1:20),
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   trControl = train_control)

knn_model
summary(knn_model)
plot(knn_model)
# Predict on testing set
knn_predictions <- predict(knn_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(knn_predictions, testing_set$class)

# ROC Curve plot
knn_predicted_prob <- predict(knn_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, knn_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - K-Nearest Neighbors", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc
varImp(knn_model)
plot(varImp(knn_model), main = "Top 5 VarImp Predictors", top = 5, scales = list(y = list(cex = .95)))

######### Support Vector Machine ############
svmGrid <- expand.grid(
  .sigma = sigest(as.matrix(training_set[, -ncol(training_set)]))[1],  # Use `sigest` to estimate sigma
  .C = 2^(seq(-4, 4))  # Test C values from 2^-4 to 2^4
)

svm_model <- train(x = training_set[, -ncol(training_set)],
                   y = training_set$class,
                   method = "svmRadial",
                   tuneGrid = svmGrid,
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   trControl = train_control)

svm_model
summary(svm_model)
plot(svm_model)
# Predict on testing set
svm_predictions <- predict(svm_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(svm_predictions, testing_set$class)

# ROC Curve plot
svm_predicted_prob <- predict(svm_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, svm_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - Penalized Model", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc

#VaryImp Predictors
varImp(svm_model)
plot(varImp(svm_model), main = "Top 5 VarImp Predictors", top = 5, scales = list(y = list(cex = .95)))

############ Naive Bayes  ###################
nbGrid <- expand.grid(
  .fL = c(0, 1, 2),           # Test different levels of Laplace smoothing
  .usekernel = c(TRUE, FALSE),  # Test with and without kernel density estimation
  .adjust = c(1, 1.5, 2)       # Test different smoothing levels for kernel density
)


nb_model <- train(x = training_set[, -ncol(training_set)],
                  y = training_set$class,
                  method = "nb",
                  tuneGrid = nbGrid,
                  metric = "ROC",
                  preProc = c("center", "scale"),
                  trControl = train_control)

nb_model
summary(nb_model)
plot(nb_model)
# Predict on testing set
nb_predictions <- predict(nb_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(nb_predictions, testing_set$class)

# ROC Curve plot
nb_predicted_prob <- predict(nb_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, nb_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - Penalized Model", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc


############# FDA ########
fda_model <- train( x = training_set[, -ncol(training_set)],
                    y = training_set$class,
                    method = "fda",
                    tuneGrid = expand.grid(.degree = 1:2, .nprune = 2:43),
                    metric = "ROC",
                    preProc = c("center", "scale"),
                    trControl = train_control
)
print(fda_model)
summary(fda_model)
plot(fda_model)
# Predict on testing set
fda_predictions <- predict(fda_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(fda_predictions, testing_set$class)

# ROC Curve plot
fda_predicted_prob <- predict(fda_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, fda_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - Flexible Discriminant Analysis", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc



###### SVM 2#####

svmGrid <- expand.grid(
  .sigma = sigest(as.matrix(training_set[, -ncol(training_set)]))[1],  # Use `sigest` to estimate sigma
  .C = 2^(seq(-4, 4))  # Test C values from 2^-4 to 2^6
)

svm_model <- train(x = training_set[, -ncol(training_set)],
                   y = training_set$class,
                   method = "svmRadial",
                   tuneGrid = svmGrid,
                   metric = "ROC",
                   preProc = c("center", "scale"),
                   trControl = train_control)

svm_model
summary(svm_model)
plot(svm_model)
# Predict on testing set
svm_predictions <- predict(svm_model, newdata = testing_set[, -ncol(testing_set)])
confusionMatrix(svm_predictions, testing_set$class)

# ROC Curve plot
svm_predicted_prob <- predict(svm_model, newdata = testing_set[, -ncol(testing_set)], type = "prob")
roc_auc <- roc(testing_set$class, svm_predicted_prob[, "e"], levels = c("p", "e"))
plot(roc_auc, main = "ROC Curve - Support Vector Machine", col = "blue")
legend("bottomright", legend = c(paste("AUC:", round(auc(roc_auc), 3))), col = "blue", lty = 1)
roc_auc

#VaryImp Predictors
varImp(svm_model)
plot(varImp(svm_model), main = "Top 5 Variable Important Predictors", top = 5, scales = list(y = list(cex = .95)))
