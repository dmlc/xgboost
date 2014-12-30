require(xgboost)
require(Matrix)
require(data.table)
require(vcd) #Available in Cran. Used for its dataset with categorical values.

# According to its documentation, Xgboost works only on numbers.
# Sometimes the dataset we have to work on have categorical data. 
# A categorical variable is one which have a fixed number of values. By exemple, if for each observation a variable called "Colour" can have only "red", "blue" or "green" as value, it is a categorical variable.
#
# In R, categorical variable is called Factor. 
# Type ?factor in console for more information.
#
# In this demo we will see how to transform a dense dataframe with categorical variables to a sparse matrix before analyzing it in Xgboost.
# The method we are going to see is usually called "one hot encoding".

#load Arthritis dataset in memory.
data(Arthritis)

# create a copy of the dataset with data.table package (data.table is 100% compliant with R dataframe but its syntax is a lot more consistent and its performance are really good).
df <- data.table(Arthritis, keep.rownames = F)

# Let's have a look to the data.table
cat("Print the dataset\n")
print(df)

# 2 columns have factor type, one has ordinal type (ordinal variable is a categorical variable with values wich can be ordered, here: None > Some > Marked).
cat("Structure of the dataset\n")
str(df)

# We remove the Age column which has no interest for the purpose of this demo.
df[,Age:= NULL]

# List the different values for the column Treatment: Placebo, Treated.
cat("Values of the categorical feature Treatment\n")
print(levels(df[,Treatment]))

# Next step, we will transform the categorical data to dummy variables.
# This method is also called dummy encoding.
# The purpose is to transform each value of each categorical feature in one binary feature.
#
# For example, the column Treatment will be replaced by two columns, Placebo, and Treated. Each of them will be binary, meaning that it will contain the value 1 in the new column Placebo and 0 in the new column  Treated, for observations which had the value Placebo in column Treatment before the transformation.
#
# Formulae Improved~.-1 means transform all categorical features but column Improved to binary values.
# Column Improved is excluded because it will be our output column, the one we want to predict.
sparse_matrix = sparse.model.matrix(Improved~.-1, data = df)

cat("Encoding of the sparse Matrix\n")
print(sparse_matrix)

# Create the output vector (not sparse)
# 1. Set, for all rows, field in Y column to 0; 
# 2. set Y to 1 when Improved == Marked; 
# 3. Return Y column
output_vector = df[,Y:=0][Improved == "Marked",Y:=1][,Y]

# Following is the same process as other demo
cat("Learning...\n")
bst <- xgboost(data = sparse_matrix, label = output_vector, max.depth = 3,
               eta = 1, nround = 2,objective = "binary:logistic")
xgb.dump(bst, 'xgb.model.dump', with.stats = T)

# sparse_matrix@Dimnames[[2]] represents the column names of the sparse matrix.
importance = xgb.importance(sparse_matrix@Dimnames[[2]], 'xgb.model.dump')
print(importance)
# According to the matrix below, the most important feature in this dataset to predict if the treatment will work is having received a Placebo or not.
