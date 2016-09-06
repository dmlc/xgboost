mkdir bin

for /f %%i in ('%HADOOP_HOME%\bin\hadoop classpath') do set CPATH=%%i
%JAVA_HOME%/bin/javac -cp %CPATH% -d bin src/main/java/org/apache/hadoop/yarn/dmlc/*.java
%JAVA_HOME%/bin/jar cf dmlc-yarn.jar -C bin .
