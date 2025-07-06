# ------------------------------------------
# Load libraries
# ------------------------------------------
library(nnet)          # multinom
library(rpart)         # decision tree
library(rpart.plot)    # tree plot
library(randomForest)  # random forest
library(xgboost)       # xgboost
library(caret)         # confusion matrix
library(ggplot2)       # plot
library(dplyr)         # tidy manipulation
library(reshape2)     # reshape data for plotting
library(forcats)        # factor manipulation
# ------------------------------------------
# Load cleaned dataset
# ------------------------------------------
df <- read.csv("cleaned_data.csv")

str(df)
# ------------------------------------------
# Change Respective Variables to Factor
# ------------------------------------------
df$Marital.status <- as.factor(df$Marital.status)
df$Nacionality <- as.factor(df$Nacionality)
df$Displaced <- as.factor(df$Displaced)
df$Gender <- as.factor(df$Gender)
df$International <- as.factor(df$International)
df$Application.mode <- as.factor(df$Application.mode)
df$Application.order <- as.factor(df$Application.order)
df$Course <- as.factor(df$Course)
df$Daytime.evening.attendance <- as.factor(df$Daytime.evening.attendance)
df$Previous.qualification <- as.factor(df$Previous.qualification)
df$Mother.s.qualification <- as.factor(df$Mother.s.qualification)
df$Father.s.qualification <- as.factor(df$Father.s.qualification)
df$Mother.s.occupation <- as.factor(df$Mother.s.occupation)
df$Father.s.occupation <- as.factor(df$Father.s.occupation)
df$Educational.special.needs <- as.factor(df$Educational.special.needs)
df$Debtor <- as.factor(df$Debtor)
df$Tuition.fees.up.to.date <- as.factor(df$Tuition.fees.up.to.date)
df$Scholarship.holder <- as.factor(df$Scholarship.holder)
df$Target <- as.factor(df$Target)

str(df)

# ------------------------------------------
# Data Manipulation (To avoid High-cardinality Factors)
# ------------------------------------------
df$Application.mode <- fct_lump(df$Application.mode, n = 10)
df$Application.order <- fct_lump(df$Application.order, n = 5) # Optional: could stay numeric if ordinal
df$Course <- fct_lump(df$Course, n = 10)
df$Previous.qualification <- fct_lump(df$Previous.qualification, n = 8)
df$Nacionality <- fct_lump(df$Nacionality, n = 10)
df$Mother.s.qualification <- fct_lump(df$Mother.s.qualification, n = 8)
df$Father.s.qualification <- fct_lump(df$Father.s.qualification, n = 8)
df$Mother.s.occupation <- fct_lump(df$Mother.s.occupation, n = 8)
df$Father.s.occupation <- fct_lump(df$Father.s.occupation, n = 8)

#check levels
sapply(df[, sapply(df, is.factor)], nlevels)
# ------------------------------------------
# Splitting the Data (80-20 split)
# ------------------------------------------
set.seed(123)  # For reproducibility
split <- createDataPartition(df$Target, p = 0.8, list = FALSE)
train_data <- df[split, ]
test_data <- df[-split, ]

# ------------------------------------------
# !!! Model Training and Evaluation !!!
# ------------------------------------------
# ------------------------------------------
# 1. Multinomial Logistic Regression
# ------------------------------------------
multi_model <- multinom(Target ~ ., data = train_data)
pred_multi <- predict(multi_model, newdata = test_data)
acc_multi <- mean(pred_multi == test_data$Target)
cat("Multinomial Logistic Regression Accuracy:", acc_multi, "\n")
# ------------------------------------------
# 2. Decision Tree
# ------------------------------------------
tree_model <- rpart(Target ~ ., data = train_data, method = "class",
                    control = rpart.control(maxdepth = 5, cp = 0.01))
pred_tree <- predict(tree_model, newdata = test_data, type = "class")
acc_tree <- mean(pred_tree == test_data$Target)
cat("Decision Tree Accuracy:", acc_tree, "\n")

# ------------------------------------------
# 3. Random Forest
# ------------------------------------------
rf_model <- randomForest(Target ~ ., data = train_data, ntree = 100)
pred_rf <- predict(rf_model, newdata = test_data)
acc_rf <- mean(pred_rf == test_data$Target)
cat("Random Forest Accuracy:", acc_rf, "\n")

# ------------------------------------------
# 4. XGBoost
# ------------------------------------------
train_matrix <- model.matrix(Target ~ . - 1, data = train_data)
test_matrix <- model.matrix(Target ~ . - 1, data = test_data)
dtrain <- xgb.DMatrix(data = train_matrix, label = as.numeric(train_data$Target) - 1)
dtest <- xgb.DMatrix(data = test_matrix, label = as.numeric(test_data$Target) - 1)
params <- list(objective = "multi:softmax", num_class = length(unique(train_data$Target)), eval_metric = "mlogloss")
xgb_model <- xgboost(data = dtrain, params = params, nrounds = 100, verbose = 0)
pred_xgb <- predict(xgb_model, newdata = dtest)
acc_xgb <- mean(pred_xgb == (as.numeric(test_data$Target) - 1))
cat("XGBoost Accuracy:", acc_xgb, "\n")


# ------------------------------------------
# Confusion Matrix for each model
# ------------------------------------------
confusion_multi <- confusionMatrix(factor(pred_multi), test_data$Target)
confusion_tree <- confusionMatrix(factor(pred_tree), test_data$Target)
confusion_rf <- confusionMatrix(factor(pred_rf), test_data$Target)
#XGB
target_levels <- levels(train_data$Target)
pred_xgb_factor <- factor(target_levels[pred_xgb + 1], levels = target_levels)
confusion_xgb <- confusionMatrix(pred_xgb_factor, test_data$Target)

# ------------------------------------------
# Print Confusion Matrices
# ------------------------------------------
print("Confusion Matrix for Multinomial Logistic Regression:")
print(confusion_multi)
print("Confusion Matrix for Decision Tree:")
print(confusion_tree)
print("Confusion Matrix for Random Forest:")
print(confusion_rf)
print("Confusion Matrix for XGBoost:")
print(confusion_xgb)

# ------------------------------------------
# Plot Confusion Matrices
# ------------------------------------------
plot_confusion_matrix <- function(confusion_matrix, title) {
  cm_data <- as.data.frame(confusion_matrix$table)
  ggplot(cm_data, aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = Freq), color = "white") +
    scale_fill_gradient(low = "blue", high = "red") +
    geom_text(aes(label = Freq), color = "black") +
    labs(title = title, x = "Actual", y = "Predicted") +
    theme_minimal()
}
plot_multi <- plot_confusion_matrix(confusion_multi, "Multinomial Logistic Regression")
plot_tree <- plot_confusion_matrix(confusion_tree, "Decision Tree")
plot_rf <- plot_confusion_matrix(confusion_rf, "Random Forest")
plot_xgb <- plot_confusion_matrix(confusion_xgb, "XGBoost")

# Save plots
ggsave("confusion_matrix_multi.png", plot = plot_multi, width = 8, height = 6)
ggsave("confusion_matrix_tree.png", plot = plot_tree, width = 8, height = 6)
ggsave("confusion_matrix_rf.png", plot = plot_rf, width = 8, height = 6)
ggsave("confusion_matrix_xgb.png", plot = plot_xgb, width = 8, height = 6)

# ------------------------------------------
# Plot Accuracy Scores of Models with Horizontal Bar Graph
# ------------------------------------------
accuracy_scores <- data.frame(
  Model = c("Multinomial Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"),
  Accuracy = c(acc_multi, acc_tree, acc_rf, acc_xgb)
)
ggplot(accuracy_scores, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Model Accuracy Comparison", x = "Model", y = "Accuracy") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")
# Save accuracy plot
ggsave("model_accuracy_comparison.png", width = 8, height = 6)

# ------------------------------------------
# Plot Precision scores of Models with Horizontal Bar Graph
# ------------------------------------------
precision_scores <- data.frame(
  Model = c("Multinomial Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"),
  Precision = c(
    mean(confusion_multi$byClass[, "Precision"]),
    mean(confusion_tree$byClass[, "Precision"]),
    mean(confusion_rf$byClass[, "Precision"]),
    mean(confusion_xgb$byClass[, "Precision"])
  )
)
ggplot(precision_scores, aes(x = Model, y = Precision, fill = Model)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Model Precision Comparison", x = "Model", y = "Precision") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")
# Save precision plot
ggsave("model_precision_comparison.png", width = 8, height = 6)

# ------------------------------------------
# Plot Recall scores of Models with Horizontal Bar Graph
# ------------------------------------------
recall_scores <- data.frame(
  Model = c("Multinomial Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"),
  Recall = c(
    mean(confusion_multi$byClass[, "Recall"]),
    mean(confusion_tree$byClass[, "Recall"]),
    mean(confusion_rf$byClass[, "Recall"]),
    mean(confusion_xgb$byClass[, "Recall"])
  )
)
ggplot(recall_scores, aes(x = Model, y = Recall, fill = Model)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Model Recall Comparison", x = "Model", y = "Recall") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")
# Save recall plot
ggsave("model_recall_comparison.png", width = 8, height = 6)

# ------------------------------------------
# Plot F1 scores of Models with Horizontal Bar Graph
# ------------------------------------------
f1_scores <- data.frame(
  Model = c("Multinomial Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"),
  F1 = c(
    mean(confusion_multi$byClass[, "F1"]),
    mean(confusion_tree$byClass[, "F1"]),
    mean(confusion_rf$byClass[, "F1"]),
    mean(confusion_xgb$byClass[, "F1"])
  )
)
ggplot(f1_scores, aes(x = Model, y = F1, fill = Model)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Model F1 Score Comparison", x = "Model", y = "F1 Score") +
  theme_minimal() +
  scale_fill_brewer(palette = "Set3")
# Save F1 score plot
ggsave("model_f1_comparison.png", width = 8, height = 6)

# ------------------------------------------
# Print table of Accuracy, Precision, Recall, and F1 scores of models
# ------------------------------------------
results_table <- data.frame(
  Model = c("Multinomial Logistic Regression", "Decision Tree", "Random Forest", "XGBoost"),
  Accuracy = c(acc_multi, acc_tree, acc_rf, acc_xgb),
  Precision = c(
    mean(confusion_multi$byClass[, "Precision"]),
    mean(confusion_tree$byClass[, "Precision"]),
    mean(confusion_rf$byClass[, "Precision"]),
    mean(confusion_xgb$byClass[, "Precision"])
  ),
  Recall = c(
    mean(confusion_multi$byClass[, "Recall"]),
    mean(confusion_tree$byClass[, "Recall"]),
    mean(confusion_rf$byClass[, "Recall"]),
    mean(confusion_xgb$byClass[, "Recall"])
  ),
  F1 = c(
    mean(confusion_multi$byClass[, "F1"]),
    mean(confusion_tree$byClass[, "F1"]),
    mean(confusion_rf$byClass[, "F1"]),
    mean(confusion_xgb$byClass[, "F1"])
  )
)
print("Model Performance Summary:")
print(results_table)