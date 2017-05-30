site <- 'http://cran.r-project.org'
if (!require('dummies'))
    install.packages('dummies', repos=site)
if (!require('insuranceData'))
    install.packages('insuranceData', repos=site)

library(dummies)
library(insuranceData)

data(AutoClaims)
data = AutoClaims

data$STATE = as.factor(data$STATE)
data$CLASS = as.factor(data$CLASS)
data$GENDER = as.factor(data$GENDER)

data.dummy <- dummy.data.frame(data, dummy.class='factor', omit.constants=T);
write.table(data.dummy, 'autoclaims.csv', sep=',', row.names=F, col.names=F, quote=F)
