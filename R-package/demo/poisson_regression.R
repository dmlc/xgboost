data(mtcars)
head(mtcars)
bst = xgboost(data=as.matrix(mtcars[,-11]),label=mtcars[,11],
              objective='count:poisson',nrounds=5)
pred = predict(bst,as.matrix(mtcars[,-11]))
sqrt(mean((pred-mtcars[,11])^2))

