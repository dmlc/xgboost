# If you need the development version of caret, uncomment and run:
# if (requireNamespace("devtools", quietly = TRUE)) {
#   devtools::install_github("topepo/caret/pkg/caret")
# }

if (!requireNamespace("caret", quietly = TRUE)) {
  message("Package 'caret' is required for this demo. Please install it.")
  stop()
}
if (!requireNamespace("e1071", quietly = TRUE)) {
  message("Package 'e1071' is required for this demo. Please install it.")
  stop()
}
if (!requireNamespace("vcd", quietly = TRUE)) {
  message("Package 'vcd' is required for this demo. Please install it.")
  stop()
}
library(xgboost)
library(data.table)
library(caret)

# Load Arthritis dataset in memory.
data(Arthritis, package = "vcd")
# Create a copy of the dataset with data.table package (data.table is 100% compliant with R dataframe but its syntax is a lot more consistent and its performance are really good).
df <- data.table(Arthritis, keep.rownames = FALSE)

# Let's add some new categorical features to see if it helps. Of course these feature are highly correlated to the Age feature. Usually it's not a good thing in ML, but Tree algorithms (including boosted trees) are able to select the best features, even in case of highly correlated features.
# For the first feature we create groups of age by rounding the real age. Note that we transform it to factor (categorical data) so the algorithm treat them as independant values.
df[, AgeDiscret := as.factor(round(Age / 10, 0))]

# Here is an even stronger simplification of the real age with an arbitrary split at 30 years old. I choose this value based on nothing. We will see later if simplifying the information based on arbitrary values is a good strategy (I am sure you already have an idea of how well it will work!).
df[, AgeCat := as.factor(ifelse(Age > 30, "Old", "Young"))]

# We remove ID as there is nothing to learn from this feature (it will just add some noise as the dataset is small).
df[, ID := NULL]

#-------------Basic Training using XGBoost in caret Library-----------------
# Set up control parameters for caret::train
# Here we use 10-fold cross-validation, repeating twice, and using random search for tuning hyper-parameters.
fitControl <- caret::trainControl(method = "repeatedcv", number = 10, repeats = 2, search = "random")
# train a xgbTree model using caret::train
model <- caret::train(factor(Improved)~., data = df, method = "xgbTree", trControl = fitControl)

# Instead of tree for our boosters, you can also fit a linear regression or logistic regression model using xgbLinear
# model <- caret::train(factor(Improved)~., data = df, method = "xgbLinear", trControl = fitControl)

# See model results
print(model)
