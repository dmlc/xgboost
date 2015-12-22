## Understand How XGBoost found Each Other##

We know that XGBoost relies on rabit library for communication. In this essay, we analyze how XGBoost utilizes rabit to discover the other node. 

XGBoost does not include distributed computing logic in its core library. Instead, it utilizes rabit to submit XGBoost as the distributed job and through rabit node, XGBoost can do allreduce, etc.


