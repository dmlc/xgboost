library(xgboost)
library(data.table)

set.seed(1024)

# Function to obtain a list of interactions fitted in trees, requires input of maximum depth
treeInteractions <- function(input_tree, input_max_depth){
  trees <- copy(input_tree)  # copy tree input to prevent overwriting
  if (input_max_depth < 2) return(list())  # no interactions if max depth < 2
  if (nrow(input_tree) == 1) return(list())

  # Attach parent nodes
  for (i in 2:input_max_depth){
    if (i == 2) trees[, ID_merge:=ID] else trees[, ID_merge:=get(paste0('parent_',i-2))]
    parents_left <- trees[!is.na(Split), list(i.id=ID, i.feature=Feature, ID_merge=Yes)]
    parents_right <- trees[!is.na(Split), list(i.id=ID, i.feature=Feature, ID_merge=No)]

    setorderv(trees, 'ID_merge')
    setorderv(parents_left, 'ID_merge')
    setorderv(parents_right, 'ID_merge')

    trees <- merge(trees, parents_left, by='ID_merge', all.x=T)
    trees[!is.na(i.id), c(paste0('parent_', i-1), paste0('parent_feat_', i-1)):=list(i.id, i.feature)]
    trees[, c('i.id','i.feature'):=NULL]

    trees <- merge(trees, parents_right, by='ID_merge', all.x=T)
    trees[!is.na(i.id), c(paste0('parent_', i-1), paste0('parent_feat_', i-1)):=list(i.id, i.feature)]
    trees[, c('i.id','i.feature'):=NULL]
  }

  # Extract nodes with interactions
  interaction_trees <- trees[!is.na(Split) & !is.na(parent_1), 
                             c('Feature',paste0('parent_feat_',1:(input_max_depth-1))), with=F]
  interaction_trees_split <- split(interaction_trees, 1:nrow(interaction_trees))
  interaction_list <- lapply(interaction_trees_split, as.character)

  # Remove NAs (no parent interaction)
  interaction_list <- lapply(interaction_list, function(x) x[!is.na(x)])

  # Remove non-interactions (same variable)
  interaction_list <- lapply(interaction_list, unique)  # remove same variables
  interaction_length <- sapply(interaction_list, length)
  interaction_list <- interaction_list[interaction_length > 1]
  interaction_list <- unique(lapply(interaction_list, sort))
  return(interaction_list)
}

# Generate sample data
x <- list()
for (i in 1:10){
  x[[i]] = i*rnorm(1000, 10)
}
x <- as.data.table(x)

y = -1*x[, rowSums(.SD)] + x[['V1']]*x[['V2']] + x[['V3']]*x[['V4']]*x[['V5']] + rnorm(1000, 0.001) + 3*sin(x[['V7']])

train = as.matrix(x)

# Interaction constraint list (column names form)
interaction_list <- list(c('V1','V2'),c('V3','V4','V5'))

# Convert interaction constraint list into feature index form
cols2ids <- function(object, col_names) {
  LUT <- seq_along(col_names) - 1
  names(LUT) <- col_names
  rapply(object, function(x) LUT[x], classes="character", how="replace")
}
interaction_list_fid = cols2ids(interaction_list, colnames(train))

# Fit model with interaction constraints
bst = xgboost(data = train, label = y, max_depth = 4,
              eta = 0.1, nthread = 2, nrounds = 1000,
              interaction_constraints = interaction_list_fid)

bst_tree <- xgb.model.dt.tree(colnames(train), bst)
bst_interactions <- treeInteractions(bst_tree, 4)  # interactions constrained to combinations of V1*V2 and V3*V4*V5

# Fit model without interaction constraints
bst2 = xgboost(data = train, label = y, max_depth = 4,
               eta = 0.1, nthread = 2, nrounds = 1000)

bst2_tree <- xgb.model.dt.tree(colnames(train), bst2)
bst2_interactions <- treeInteractions(bst2_tree, 4)  # much more interactions

# Fit model with both interaction and monotonicity constraints
bst3 = xgboost(data = train, label = y, max_depth = 4,
               eta = 0.1, nthread = 2, nrounds = 1000,
               interaction_constraints = interaction_list_fid,
               monotone_constraints = c(-1,0,0,0,0,0,0,0,0,0))

bst3_tree <- xgb.model.dt.tree(colnames(train), bst3)
bst3_interactions <- treeInteractions(bst3_tree, 4)  # interactions still constrained to combinations of V1*V2 and V3*V4*V5

# Show monotonic constraints still apply by checking scores after incrementing V1
x1 <- sort(unique(x[['V1']]))
for (i in 1:length(x1)){
  testdata <- copy(x[, -c('V1')])
  testdata[['V1']] <- x1[i]
  testdata <- testdata[, paste0('V',1:10), with=F]
  pred <- predict(bst3, as.matrix(testdata))
  
  # Should not print out anything due to monotonic constraints
  if (i > 1) if (any(pred > prev_pred)) print(i)
  prev_pred <- pred 
}
