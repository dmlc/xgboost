# Understand your dataset with XGBoost

## Introduction

The purpose of this vignette is to show you how to use **XGBoost** to
discover and understand your own dataset better.

This vignette is not about predicting anything (see [XGBoost
presentation](https://github.com/dmlc/xgboost/blob/master/R-package/vignettes/xgboostPresentation.Rmd)).
We will explain how to use **XGBoost** to highlight the *link* between
the *features* of your data and the *outcome*.

Package loading:

    require(xgboost)
    require(Matrix)
    require(data.table)
    if (!require('vcd')) {
      install.packages('vcd')
    }

> **VCD** package is used for one of its embedded dataset only.

## Preparation of the dataset

### Numeric v.s. categorical variables

**XGBoost** manages only `numeric` vectors.

What to do when you have *categorical* data?

A *categorical* variable has a fixed number of different values. For
instance, if a variable called *Colour* can have only one of these three
values, *red*, *blue* or *green*, then *Colour* is a *categorical*
variable.

> In **R**, a *categorical* variable is called `factor`.
>
> Type `?factor` in the console for more information.

To answer the question above we will convert *categorical* variables to
`numeric` ones.

### Conversion from categorical to numeric variables

#### Looking at the raw data

+In this Vignette we will see how to transform a *dense* `data.frame`
(*dense* = the majority of the matrix is non-zero) with *categorical*
variables to a very *sparse* matrix (*sparse* = lots of zero entries in
the matrix) of `numeric` features.

The method we are going to see is usually called [one-hot
encoding](https://en.wikipedia.org/wiki/One-hot).

The first step is to load the `Arthritis` dataset in memory and wrap it
with the `data.table` package.

    data(Arthritis)
    df <- data.table(Arthritis, keep.rownames = FALSE)

> `data.table` is 100% compliant with **R** `data.frame` but its syntax
> is more consistent and its performance for large dataset is [best in
> class](https://stackoverflow.com/questions/21435339/data-table-vs-dplyr-can-one-do-something-well-the-other-cant-or-does-poorly)
> (`dplyr` from **R** and `Pandas` from **Python**
> [included](https://github.com/Rdatatable/data.table/wiki/Benchmarks-%3A-Grouping)).
> Some parts of **XGBoost’s** **R** package use `data.table`.

The first thing we want to do is to have a look to the first few lines
of the `data.table`:

    head(df)

    ##    ID Treatment  Sex Age Improved
    ## 1: 57   Treated Male  27     Some
    ## 2: 46   Treated Male  29     None
    ## 3: 77   Treated Male  30     None
    ## 4: 17   Treated Male  32   Marked
    ## 5: 36   Treated Male  46   Marked
    ## 6: 23   Treated Male  58   Marked

Now we will check the format of each column.

    str(df)

    ## Classes 'data.table' and 'data.frame':   84 obs. of  5 variables:
    ##  $ ID       : int  57 46 77 17 36 23 75 39 33 55 ...
    ##  $ Treatment: Factor w/ 2 levels "Placebo","Treated": 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ Sex      : Factor w/ 2 levels "Female","Male": 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ Age      : int  27 29 30 32 46 58 59 59 63 63 ...
    ##  $ Improved : Ord.factor w/ 3 levels "None"<"Some"<..: 2 1 1 3 3 3 1 3 1 1 ...
    ##  - attr(*, ".internal.selfref")=<externalptr>

2 columns have `factor` type, one has `ordinal` type.

> `ordinal` variable :
>
> -   can take a limited number of values (like `factor`) ;
> -   these values are ordered (unlike `factor`). Here these ordered
>     values are: `Marked > Some > None`

#### Creation of new features based on old ones

We will add some new *categorical* features to see if it helps.

##### Grouping per 10 years

For the first features we create groups of age by rounding the real age.

Note that we transform it to `factor` so the algorithm treats these age
groups as independent values.

Therefore, 20 is not closer to 30 than 60. In other words, the distance
between ages is lost in this transformation.

    head(df[, AgeDiscret := as.factor(round(Age / 10, 0))])

    ##    ID Treatment  Sex Age Improved AgeDiscret
    ## 1: 57   Treated Male  27     Some          3
    ## 2: 46   Treated Male  29     None          3
    ## 3: 77   Treated Male  30     None          3
    ## 4: 17   Treated Male  32   Marked          3
    ## 5: 36   Treated Male  46   Marked          5
    ## 6: 23   Treated Male  58   Marked          6

##### Randomly split into two groups

The following is an even stronger simplification of the real age with an
arbitrary split at 30 years old. I choose this value **based on
nothing**. We will see later if simplifying the information based on
arbitrary values is a good strategy (you may already have an idea of how
well it will work…).

    head(df[, AgeCat := as.factor(ifelse(Age > 30, "Old", "Young"))])

    ##    ID Treatment  Sex Age Improved AgeDiscret AgeCat
    ## 1: 57   Treated Male  27     Some          3  Young
    ## 2: 46   Treated Male  29     None          3  Young
    ## 3: 77   Treated Male  30     None          3  Young
    ## 4: 17   Treated Male  32   Marked          3    Old
    ## 5: 36   Treated Male  46   Marked          5    Old
    ## 6: 23   Treated Male  58   Marked          6    Old

##### Risks in adding correlated features

These new features are highly correlated to the `Age` feature because
they are simple transformations of this feature.

For many machine learning algorithms, using correlated features is not a
good idea. It may sometimes make prediction less accurate, and most of
the time make interpretation of the model almost impossible. GLM, for
instance, assumes that the features are uncorrelated.

Fortunately, decision tree algorithms (including boosted trees) are very
robust to these features. Therefore we don’t have to do anything to
manage this situation.

##### Cleaning data

We remove ID as there is nothing to learn from this feature (it would
just add some noise).

    df[, ID := NULL]

We will list the different values for the column `Treatment`:

    levels(df[, Treatment])

    ## [1] "Placebo" "Treated"

#### Encoding categorical features

Next step, we will transform the categorical data to dummy variables.
Several encoding methods exist, e.g., [one-hot
encoding](https://en.wikipedia.org/wiki/One-hot) is a common approach.
We will use the [dummy contrast
coding](https://stats.oarc.ucla.edu/r/library/r-library-contrast-coding-systems-for-categorical-variables/)
which is popular because it produces “full rank” encoding (also see
[this blog post by Max
Kuhn](http://appliedpredictivemodeling.com/blog/2013/10/23/the-basics-of-encoding-categorical-data-for-predictive-models)).

The purpose is to transform each value of each *categorical* feature
into a *binary* feature `{0, 1}`.

For example, the column `Treatment` will be replaced by two columns,
`TreatmentPlacebo`, and `TreatmentTreated`. Each of them will be
*binary*. Therefore, an observation which has the value `Placebo` in
column `Treatment` before the transformation will have the value `1` in
the new column `TreatmentPlacebo` and the value `0` in the new column
`TreatmentTreated` after the transformation. The column
`TreatmentPlacebo` will disappear during the contrast encoding, as it
would be absorbed into a common constant intercept column.

Column `Improved` is excluded because it will be our `label` column, the
one we want to predict.

    sparse_matrix <- sparse.model.matrix(Improved ~ ., data = df)[, -1]
    head(sparse_matrix)

    ## 6 x 9 sparse Matrix of class "dgCMatrix"
    ##   TreatmentTreated SexMale Age AgeDiscret3 AgeDiscret4 AgeDiscret5 AgeDiscret6
    ## 1                1       1  27           1           .           .           .
    ## 2                1       1  29           1           .           .           .
    ## 3                1       1  30           1           .           .           .
    ## 4                1       1  32           1           .           .           .
    ## 5                1       1  46           .           .           1           .
    ## 6                1       1  58           .           .           .           1
    ##   AgeDiscret7 AgeCatYoung
    ## 1           .           1
    ## 2           .           1
    ## 3           .           1
    ## 4           .           .
    ## 5           .           .
    ## 6           .           .

> Formula `Improved ~ .` used above means transform all *categorical*
> features but column `Improved` to binary values. The `-1` column
> selection removes the intercept column which is full of `1` (this
> column is generated by the conversion). For more information, you can
> type `?sparse.model.matrix` in the console.

Create the output `numeric` vector (not as a sparse `Matrix`):

    output_vector <- df[, Improved] == "Marked"

1.  set `Y` vector to `0`;
2.  set `Y` to `1` for rows where `Improved == Marked` is `TRUE` ;
3.  return `Y` vector.

## Build the model

The code below is very usual. For more information, you can look at the
documentation of `xgboost` function (or at the vignette [XGBoost
presentation](https://github.com/dmlc/xgboost/blob/master/R-package/vignettes/xgboostPresentation.Rmd)).

    bst <- xgboost(data = sparse_matrix, label = output_vector, max_depth = 4,
                   eta = 1, nthread = 2, nrounds = 10, objective = "binary:logistic")

    ## [1]  train-logloss:0.485466 
    ## [2]  train-logloss:0.438534 
    ## [3]  train-logloss:0.412250 
    ## [4]  train-logloss:0.395828 
    ## [5]  train-logloss:0.384264 
    ## [6]  train-logloss:0.374028 
    ## [7]  train-logloss:0.365005 
    ## [8]  train-logloss:0.351233 
    ## [9]  train-logloss:0.341678 
    ## [10] train-logloss:0.334465

You can see some `train-logloss: 0.XXXXX` lines followed by a number. It
decreases. Each line shows how well the model explains the data. Lower
is better.

A small value for training error may be a symptom of
[overfitting](https://en.wikipedia.org/wiki/Overfitting), meaning the
model will not accurately predict unseen values.

## Feature importance

## Measure feature importance

### Build the feature importance data.table

Remember, each binary column corresponds to a single value of one of
*categorical* features.

    importance <- xgb.importance(feature_names = colnames(sparse_matrix), model = bst)
    head(importance)

    ##             Feature        Gain      Cover  Frequency
    ## 1:              Age 0.622031769 0.67251696 0.67241379
    ## 2: TreatmentTreated 0.285750540 0.11916651 0.10344828
    ## 3:          SexMale 0.048744022 0.04522028 0.08620690
    ## 4:      AgeDiscret6 0.016604639 0.04784639 0.05172414
    ## 5:      AgeDiscret3 0.016373781 0.08028951 0.05172414
    ## 6:      AgeDiscret4 0.009270557 0.02858801 0.01724138

> The column `Gain` provides the information we are looking for.
>
> As you can see, features are classified by `Gain`.

`Gain` is the improvement in accuracy brought by a feature to the
branches it is on. The idea is that before adding a new split on a
feature X to the branch there were some wrongly classified elements;
after adding the split on this feature, there are two new branches, and
each of these branches is more accurate (one branch saying if your
observation is on this branch then it should be classified as `1`, and
the other branch saying the exact opposite).

`Cover` is related to the second order derivative (or Hessian) of the
loss function with respect to a particular variable; thus, a large value
indicates a variable has a large potential impact on the loss function
and so is important.

`Frequency` is a simpler way to measure the `Gain`. It just counts the
number of times a feature is used in all generated trees. You should not
use it (unless you know why you want to use it).

### Plotting the feature importance

All these things are nice, but it would be even better to plot the
results.

    xgb.plot.importance(importance_matrix = importance)

<img src="discoverYourData_files/figure-markdown_strict/unnamed-chunk-12-1.png" style="display: block; margin: auto;" />

Running this line of code, you should get a bar chart showing the
importance of the 6 features (containing the same data as the output we
saw earlier, but displaying it visually for easier consumption). Note
that `xgb.ggplot.importance` is also available for all the ggplot2 fans!

> Depending of the dataset and the learning parameters you may have more
> than two clusters. Default value is to limit them to `10`, but you can
> increase this limit. Look at the function documentation for more
> information.

According to the plot above, the most important features in this dataset
to predict if the treatment will work are :

-   An individual’s age;
-   Having received a placebo or not;
-   Gender;
-   Our generated feature AgeDiscret. We can see that its contribution
    is very low.

### Do these results make sense?

Let’s check some **Chi2** between each of these features and the label.

Higher **Chi2** means better correlation.

    c2 <- chisq.test(df$Age, output_vector)
    print(c2)

    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  df$Age and output_vector
    ## X-squared = 35.475, df = 35, p-value = 0.4458

The Pearson correlation between Age and illness disappearing is
**35.47**.

    c2 <- chisq.test(df$AgeDiscret, output_vector)
    print(c2)

    ## 
    ##  Pearson's Chi-squared test
    ## 
    ## data:  df$AgeDiscret and output_vector
    ## X-squared = 8.2554, df = 5, p-value = 0.1427

Our first simplification of Age gives a Pearson correlation of **8.26**.

    c2 <- chisq.test(df$AgeCat, output_vector)
    print(c2)

    ## 
    ##  Pearson's Chi-squared test with Yates' continuity correction
    ## 
    ## data:  df$AgeCat and output_vector
    ## X-squared = 2.3571, df = 1, p-value = 0.1247

The perfectly random split we did between young and old at 30 years old
has a low correlation of **2.36**. This suggests that, for the
particular illness we are studying, the age at which someone is
vulnerable to this disease is likely very different from 30.

Moral of the story: don’t let your *gut* lower the quality of your
model.

In *data science*, there is the word *science* :-)

## Conclusion

As you can see, in general *destroying information by simplifying it
won’t improve your model*. **Chi2** just demonstrates that.

But in more complex cases, creating a new feature from an existing one
may help the algorithm and improve the model.

+The case studied here is not complex enough to show that. Check [Kaggle
website](https://www.kaggle.com/) for some challenging datasets.

Moreover, you can see that even if we have added some new features which
are not very useful/highly correlated with other features, the boosting
tree algorithm was still able to choose the best one (which in this case
is the Age).

Linear models may not perform as well.

## Special Note: What about Random Forests™?

As you may know, the [Random
Forests](https://en.wikipedia.org/wiki/Random_forest) algorithm is
cousin with boosting and both are part of the [ensemble
learning](https://en.wikipedia.org/wiki/Ensemble_learning) family.

Both train several decision trees for one dataset. The *main* difference
is that in Random Forests, trees are independent and in boosting, the
`N+1`-st tree focuses its learning on the loss (&lt;=&gt; what has not
been well modeled by the tree `N`).

This difference can have an impact on a edge case in feature importance
analysis: *correlated features*.

Imagine two features perfectly correlated, feature `A` and feature `B`.
For one specific tree, if the algorithm needs one of them, it will
choose randomly (true in both boosting and Random Forests).

However, in Random Forests this random choice will be done for each
tree, because each tree is independent from the others. Therefore,
approximately (and depending on your parameters) 50% of the trees will
choose feature `A` and the other 50% will choose feature `B`. So the
*importance* of the information contained in `A` and `B` (which is the
same, because they are perfectly correlated) is diluted in `A` and `B`.
So you won’t easily know this information is important to predict what
you want to predict! It is even worse when you have 10 correlated
features…

In boosting, when a specific link between feature and outcome have been
learned by the algorithm, it will try to not refocus on it (in theory it
is what happens, reality is not always that simple). Therefore, all the
importance will be on feature `A` or on feature `B` (but not both). You
will know that one feature has an important role in the link between the
observations and the label. It is still up to you to search for the
correlated features to the one detected as important if you need to know
all of them.

If you want to try Random Forests algorithm, you can tweak XGBoost
parameters!

For instance, to compute a model with 1000 trees, with a 0.5 factor on
sampling rows and columns:

    data(agaricus.train, package = 'xgboost')
    data(agaricus.test, package = 'xgboost')
    train <- agaricus.train
    test <- agaricus.test

    #Random Forest - 1000 trees
    bst <- xgboost(
        data = train$data
        , label = train$label
        , max_depth = 4
        , num_parallel_tree = 1000
        , subsample = 0.5
        , colsample_bytree = 0.5
        , nrounds = 1
        , objective = "binary:logistic"
    )

    ## [1]  train-logloss:0.456201

    #Boosting - 3 rounds
    bst <- xgboost(
        data = train$data
        , label = train$label
        , max_depth = 4
        , nrounds = 3
        , objective = "binary:logistic"
    )

    ## [1]  train-logloss:0.444882 
    ## [2]  train-logloss:0.302428 
    ## [3]  train-logloss:0.212847

> Note that the parameter `round` is set to `1`.

> [**Random
> Forests**](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_papers.htm)
> is a trademark of Leo Breiman and Adele Cutler and is licensed
> exclusively to Salford Systems for the commercial release of the
> software.
