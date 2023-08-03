
Understand your dataset with XGBoost
====================================

Introduction
------------

The purpose of this Vignette is to show you how to use **XGBoost** to discover and understand your own dataset better.

This Vignette is not about predicting anything (see [XGBoost presentation](https://github.com/dmlc/xgboost/blob/master/R-package/vignettes/xgboostPresentation.Rmd)). We will explain how to use **XGBoost** to highlight the *link* between the *features* of your data and the *outcome*.

Package loading:


```r
require(xgboost)
require(Matrix)
require(data.table)
if (!require('vcd')) install.packages('vcd')
```

> **VCD** package is used for one of its embedded dataset only.

Preparation of the dataset
--------------------------

### Numeric VS categorical variables


**XGBoost** manages only `numeric` vectors.

What to do when you have *categorical* data?

A *categorical* variable has a fixed number of different values. For instance, if a variable called *Colour* can have only one of these three values, *red*, *blue* or *green*, then *Colour* is a *categorical* variable.

> In **R**, a *categorical* variable is called `factor`.
>
> Type `?factor` in the console for more information.

To answer the question above we will convert *categorical* variables to `numeric` ones.

### Conversion from categorical to numeric variables

#### Looking at the raw data

In this Vignette we will see how to transform a *dense* `data.frame` (*dense* = the majority of the matrix is non-zero) with *categorical* variables to a very *sparse* matrix (*sparse* = lots of zero entries in the matrix) of `numeric` features.

The method we are going to see is usually called [one-hot encoding](http://en.wikipedia.org/wiki/One-hot).

The first step is to load the `Arthritis` dataset in memory and wrap it with the `data.table` package.


```r
data(Arthritis)
df <- data.table(Arthritis, keep.rownames = FALSE)
```

> `data.table` is 100% compliant with **R** `data.frame` but its syntax is more consistent and its performance for large dataset is [best in class](http://stackoverflow.com/questions/21435339/data-table-vs-dplyr-can-one-do-something-well-the-other-cant-or-does-poorly) (`dplyr` from **R** and `Pandas` from **Python** [included](https://github.com/Rdatatable/data.table/wiki/Benchmarks-%3A-Grouping)). Some parts of **XGBoost's** **R** package use `data.table`.

The first thing we want to do is to have a look to the first lines of the `data.table`:


```r
head(df)
```

```
##    ID Treatment  Sex Age Improved
## 1: 57   Treated Male  27     Some
## 2: 46   Treated Male  29     None
## 3: 77   Treated Male  30     None
## 4: 17   Treated Male  32   Marked
## 5: 36   Treated Male  46   Marked
## 6: 23   Treated Male  58   Marked
```

Now we will check the format of each column.


```r
str(df)
```

```
## Classes 'data.table' and 'data.frame':	84 obs. of  5 variables:
##  $ ID       : int  57 46 77 17 36 23 75 39 33 55 ...
##  $ Treatment: Factor w/ 2 levels "Placebo","Treated": 2 2 2 2 2 2 2 2 2 2 ...
##  $ Sex      : Factor w/ 2 levels "Female","Male": 2 2 2 2 2 2 2 2 2 2 ...
##  $ Age      : int  27 29 30 32 46 58 59 59 63 63 ...
##  $ Improved : Ord.factor w/ 3 levels "None"<"Some"<..: 2 1 1 3 3 3 1 3 1 1 ...
##  - attr(*, ".internal.selfref")=<externalptr>
```

2 columns have `factor` type, one has `ordinal` type.

> `ordinal` variable :
>
> * can take a limited number of values (like `factor`) ;
> * these values are ordered (unlike `factor`). Here these ordered values are: `Marked > Some > None`

#### Creation of new features based on old ones

We will add some new *categorical* features to see if it helps.

##### Grouping per 10 years

For the first features we create groups of age by rounding the real age.

Note that we transform it to `factor` so the algorithm treats these age groups as independent values.

Therefore, 20 is not closer to 30 than 60. In other words, the distance between ages is lost in this transformation.


```r
head(df[,AgeDiscret := as.factor(round(Age/10,0))])
```

```
##    ID Treatment  Sex Age Improved AgeDiscret
## 1: 57   Treated Male  27     Some          3
## 2: 46   Treated Male  29     None          3
## 3: 77   Treated Male  30     None          3
## 4: 17   Treated Male  32   Marked          3
## 5: 36   Treated Male  46   Marked          5
## 6: 23   Treated Male  58   Marked          6
```

##### Randomly split into two groups

The following is an even stronger simplification of the real age with an arbitrary split at 30 years old. I choose this value **based on nothing**. We will see later if simplifying the information based on arbitrary values is a good strategy (you may already have an idea of how well it will work...).


```r
head(df[,AgeCat:= as.factor(ifelse(Age > 30, "Old", "Young"))])
```

```
##    ID Treatment  Sex Age Improved AgeDiscret AgeCat
## 1: 57   Treated Male  27     Some          3  Young
## 2: 46   Treated Male  29     None          3  Young
## 3: 77   Treated Male  30     None          3  Young
## 4: 17   Treated Male  32   Marked          3    Old
## 5: 36   Treated Male  46   Marked          5    Old
## 6: 23   Treated Male  58   Marked          6    Old
```

##### Risks in adding correlated features

These new features are highly correlated to the `Age` feature because they are simple transformations of this feature.

For many machine learning algorithms, using correlated features is not a good idea. It may sometimes make prediction less accurate, and most of the time make interpretation of the model almost impossible. GLM, for instance, assumes that the features are uncorrelated.

Fortunately, decision tree algorithms (including boosted trees) are very robust to these features. Therefore we don't have to do anything to manage this situation.

##### Cleaning data

We remove ID as there is nothing to learn from this feature (it would just add some noise).


```r
df[,ID:=NULL]
```

We will list the different values for the column `Treatment`:


```r
levels(df[,Treatment])
```

```
## [1] "Placebo" "Treated"
```


#### One-hot encoding

Next step, we will transform the categorical data to dummy variables.
This is the [one-hot encoding](http://en.wikipedia.org/wiki/One-hot) step.

The purpose is to transform each value of each *categorical* feature into a *binary* feature `{0, 1}`.

For example, the column `Treatment` will be replaced by two columns, `Placebo`, and `Treated`. Each of them will be *binary*. Therefore, an observation which has the value `Placebo` in column `Treatment` before the transformation will have the value `1` in the new column `Placebo` and the value `0` in the new column `Treated` after the transformation. The column `Treatment` will disappear during the one-hot encoding.

Column `Improved` is excluded because it will be our `label` column, the one we want to predict.


```r
sparse_matrix <- sparse.model.matrix(Improved~.-1, data = df)
head(sparse_matrix)
```

```
## 6 x 10 sparse Matrix of class "dgCMatrix"
##                       
## 1 . 1 1 27 1 . . . . 1
## 2 . 1 1 29 1 . . . . 1
## 3 . 1 1 30 1 . . . . 1
## 4 . 1 1 32 1 . . . . .
## 5 . 1 1 46 . . 1 . . .
## 6 . 1 1 58 . . . 1 . .
```

> Formulae `Improved~.-1` used above means transform all *categorical* features but column `Improved` to binary values. The `-1` is here to remove the first column which is full of `1` (this column is generated by the conversion). For more information, you can type `?sparse.model.matrix` in the console.

Create the output `numeric` vector (not as a sparse `Matrix`):


```r
output_vector <- df[,Improved] == "Marked"
```

1. set `Y` vector to `0`;
2. set `Y` to `1` for rows where `Improved == Marked` is `TRUE` ;
3. return `Y` vector.

Build the model
---------------

The code below is very usual. For more information, you can look at the documentation of `xgboost` function (or at the vignette [XGBoost presentation](https://github.com/dmlc/xgboost/blob/master/R-package/vignettes/xgboostPresentation.Rmd)).


```r
bst <- xgboost(data = sparse_matrix, label = output_vector, max.depth = 4,
               eta = 1, nthread = 2, nrounds = 10,objective = "binary:logistic")
```

```
## [1]	train-logloss:0.485466
## [2]	train-logloss:0.438534
## [3]	train-logloss:0.412250
## [4]	train-logloss:0.395828
## [5]	train-logloss:0.384264
## [6]	train-logloss:0.374028
## [7]	train-logloss:0.365005
## [8]	train-logloss:0.351233
## [9]	train-logloss:0.341678
## [10]	train-logloss:0.334465
```

You can see some `train-logloss: 0.XXXXX` lines followed by a number. It decreases. Each line shows how well the model explains your data. Lower is better.

A model which fits too well may [overfit](http://en.wikipedia.org/wiki/Overfitting) (meaning it copy/paste too much the past, and won't be that good to predict the future).

Feature importance
------------------

## Measure feature importance


### Build the feature importance data.table

In the code below, `sparse_matrix@Dimnames[[2]]` represents the column names of the sparse matrix. These names are the original values of the features (remember, each binary column == one value of one *categorical* feature).


```r
importance <- xgb.importance(feature_names = sparse_matrix@Dimnames[[2]], model = bst)
head(importance)
```

```
##             Feature        Gain      Cover  Frequency
## 1:              Age 0.735014935 0.58954398 0.64150943
## 2: TreatmentPlacebo 0.203494186 0.28314940 0.18867925
## 3:      AgeDiscret6 0.045206923 0.04651722 0.05660377
## 4:          SexMale 0.011222184 0.05464769 0.07547170
## 5:      AgeDiscret5 0.003193737 0.01044677 0.01886792
## 6:      AgeDiscret4 0.001868035 0.01569495 0.01886792
```

> The column `Gain` provides the information we are looking for.
>
> As you can see, features are classified by `Gain`.

`Gain` is the improvement in accuracy brought by a feature to the branches it is on. The idea is that before adding a new split on a feature X to the branch there were some wrongly classified elements; after adding the split on this feature, there are two new branches, and each of these branches is more accurate (one branch saying if your observation is on this branch then it should be classified as `1`, and the other branch saying the exact opposite).

`Cover` measures the relative quantity of observations concerned by a feature.

`Frequency` is a simpler way to measure the `Gain`. It just counts the number of times a feature is used in all generated trees. You should not use it (unless you know why you want to use it).

### Plotting the feature importance


All these things are nice, but it would be even better to plot the results.


```r
xgb.plot.importance(importance_matrix = importanceRaw)
```

Running this line of code, you should get a bar chart showing the importance of the 6 features (containing the same data as the output we saw earlier, but displaying it visually for easier consumption).  Note that `xgb.ggplot.importance` is also available for all the ggplot2 fans!

According to the plot above, the most important features in this dataset to predict if the treatment will work are :

* An individual's age;
* having received a placebo or not;
* then we see our generated features (AgeDiscret). We can see that their contribution is very low.
* Gender is fourth with a very low importance

### Do these results make sense?


Let's check some **Chi2** between each of these features and the label.

Higher **Chi2** means better correlation.


```r
c2 <- chisq.test(df$Age, output_vector)
print(c2)
```

```
## 
## 	Pearson's Chi-squared test
## 
## data:  df$Age and output_vector
## X-squared = 35.475, df = 35, p-value = 0.4458
```

The Pearson correlation between Age and illness disappearing is **35.48**.


```r
c2 <- chisq.test(df$AgeDiscret, output_vector)
print(c2)
```

```
## 
## 	Pearson's Chi-squared test
## 
## data:  df$AgeDiscret and output_vector
## X-squared = 8.2554, df = 5, p-value = 0.1427
```

Our first simplification of Age gives a Pearson correlation is **8.26**.


```r
c2 <- chisq.test(df$AgeCat, output_vector)
print(c2)
```

```
## 
## 	Pearson's Chi-squared test with Yates' continuity correction
## 
## data:  df$AgeCat and output_vector
## X-squared = 2.3571, df = 1, p-value = 0.1247
```

The perfectly random split I did between young and old at 30 years old has a low correlation of **2.36**. This suggests that, for the particular illness we are studying, the age at which you are vulnerable to this disease is likely very different from 30.

Conclusion
----------

As you can see, in general *destroying information by simplifying it won't improve your model*. **Chi2** just demonstrates that.

But in more complex cases, creating a new feature from an existing one may help the algorithm and improve the model.

The case studied here is not complex enough to show that. Check [Kaggle website](http://www.kaggle.com/) for some challenging datasets.

Moreover, you can see that even if we have added some new features which are not very useful/highly correlated with other features, the boosting tree algorithm was still able to choose the best one (which in this case is the Age).

Linear models may not perform as well.

Special Note: What about Random Forests™?
-----------------------------------------

As you may know, the [Random Forests](http://en.wikipedia.org/wiki/Random_forest) algorithm is cousin with boosting and both are part of the [ensemble learning](http://en.wikipedia.org/wiki/Ensemble_learning) family.

Both train several decision trees for one dataset. The *main* difference is that in Random Forests, trees are independent and in boosting, the `N+1`-st tree focuses its learning on the loss (<=> what has not been well modeled by the tree `N`).

This difference can have an impact on a corner case in feature importance analysis: the *correlated features*.

Imagine two features perfectly correlated, feature `A` and feature `B`. For one specific tree, if the algorithm needs one of them, it will choose randomly (true in both boosting and Random Forests).

However, in Random Forests this random choice will be done for each tree, because each tree is independent from the others. Therefore, approximatel (and depending on your parameters) 50% of the trees will choose feature `A` and the other 50% will choose feature `B`. So the *importance* of the information contained in `A` and `B` (which is the same, because they are perfectly correlated) is diluted in `A` and `B`. So you won't easily know this information is important to predict what you want to predict! It is even worse when you have 10 correlated features...

In boosting, when a specific link between feature and outcome have been learned by the algorithm, it will try to not refocus on it (in theory it is what happens, reality is not always that simple). Therefore, all the importance will be on feature `A` or on feature `B` (but not both). You will know that one feature has an important role in the link between the observations and the label. It is still up to you to search for the correlated features to the one detected as important if you need to know all of them.

If you want to try Random Forests algorithm, you can tweak XGBoost parameters!

**Warning**: this is still an experimental parameter.

For instance, to compute a model with 1000 trees, with a 0.5 factor on sampling rows and columns:


```r
data(agaricus.train, package='xgboost')
data(agaricus.test, package='xgboost')
train <- agaricus.train
test <- agaricus.test

#Random Forest - 1000 trees
bst <- xgboost(data = train$data, label = train$label, max.depth = 4, num_parallel_tree = 1000, subsample = 0.5, colsample_bytree =0.5, nrounds = 1, objective = "binary:logistic")
```

```
## [0]	train-logloss:0.455762
```

```r
#Boosting - 3 rounds
bst <- xgboost(data = train$data, label = train$label, max.depth = 4, nrounds = 3, objective = "binary:logistic")
```

```
## [1]	train-logloss:0.444882 
## [2]	train-logloss:0.302428 
## [3]	train-logloss:0.212847 
```

> Note that the parameter `round` is set to `1`.

> [**Random Forests**](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_papers.htm) is a trademark of Leo Breiman and Adele Cutler and is licensed exclusively to Salford Systems for the commercial release of the software.
