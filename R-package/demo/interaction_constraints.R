library(xgboost)
library(data.table)

set.seed(1024)

# Function to obtain a list of interactions fitted in trees
treeInteractions <- function(in.tree, in.depth){
  temp.tree <- copy(in.tree)
  if (in.depth < 2) return(list())
  if (nrow(in.tree) == 1) return(list())

  # Attach parent nodes
  for (i in 2:in.depth){
    if (i == 2) temp.tree[, ID_merge:=ID] else temp.tree[, ID_merge:=get(paste0('parent_',i-2))]
    temp.parents_left <- temp.tree[!is.na(Split), list(i.id=ID, i.feature=Feature, ID_merge=Yes)]
    temp.parents_right <- temp.tree[!is.na(Split), list(i.id=ID, i.feature=Feature, ID_merge=No)]

    setorderv(temp.tree, 'ID_merge')
    setorderv(temp.parents_left, 'ID_merge')
    setorderv(temp.parents_right, 'ID_merge')

    temp.tree <- merge(temp.tree, temp.parents_left, by='ID_merge', all.x=T)
    temp.tree[!is.na(i.id), c(paste0('parent_',i-1), paste0('parent_feat_',i-1)):=list(i.id,i.feature)]
    temp.tree[, c('i.id','i.feature'):=NULL]

    temp.tree <- merge(temp.tree, temp.parents_right, by='ID_merge', all.x=T)
    temp.tree[!is.na(i.id), c(paste0('parent_',i-1), paste0('parent_feat_',i-1)):=list(i.id,i.feature)]
    temp.tree[, c('i.id','i.feature'):=NULL]
  }

  # Extract nodes with interactions
  temp <- temp.tree[!is.na(Split) & !is.na(parent_1)]
  temp <- temp[, c('Feature',paste0('parent_feat_',1:(in.depth-1))), with=F]
  temp <- split(temp, 1:nrow(temp))
  temp.int <- lapply(temp, as.character)

  # Remove NAs (no parent interaction)
  temp.int <- lapply(temp.int, function(x) x[!is.na(x)])

  # Remove non-interactions (same variable)
  temp.int <- lapply(temp.int, unique)
  temp <- sapply(temp.int, length)
  temp.int <- temp.int[temp > 1]
  temp.int <- unique(lapply(temp.int, sort))

  return(temp.int)
}

# Generate sample data
x <- list()
for (i in 1:10){
  x[[i]] = i*rnorm(1000, 10)
}
x <- as.data.table(x)

y = -1*x[, rowSums(.SD)] + x[['V1']]*x[['V2']] + x[['V3']]*x[['V4']]*x[['V5']] + rnorm(1000, 0.001) + 3*sin(x[['V7']])

train = as.matrix(x)

# Fit model with interaction constraints
bst = xgboost(data = train, label = y, max_depth = 4,
              eta = 0.1, nthread = 2, nrounds = 1000,
              int_constraints_list = list(c('V1','V2'),c('V3','V4','V5')),
              split_evaluator = 'interaction')

temp <- xgb.model.dt.tree(colnames(train), bst)
temp.int <- treeInteractions(temp, 4)  # limited interactions

# Fit model without interaction constraints
bst2 = xgboost(data = train, label = y, max_depth = 4,
               eta = 0.1, nthread = 2, nrounds = 1000)

temp <- xgb.model.dt.tree(colnames(train), bst2)
temp.int2 <- treeInteractions(temp, 4)  # much more interactions
