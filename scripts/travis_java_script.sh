# Test java package of xgboost
cd java
./create_wrap.sh
cd xgboost4j
mvn clean install -DskipTests=true
mvn test
