# Test java package of xgboost
set -e
cd java
./create_wrap.sh
cd xgboost4j
mvn clean install -DskipTests=true
mvn test
