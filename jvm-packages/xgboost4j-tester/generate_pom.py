import sys

pom_template = """
<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>ml.dmlc</groupId>
  <artifactId>xgboost4j-tester_{scala_binary_version}</artifactId>
  <version>1.0-SNAPSHOT</version>

  <name>xgboost4j-tester</name>

  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.source>{maven_compiler_source}</maven.compiler.source>
    <maven.compiler.target>{maven_compiler_target}</maven.compiler.target>
    <junit.version>4.13.2</junit.version>
    <spark.version>{spark_version}</spark.version>
    <scala.version>{scala_version}</scala.version>
    <scalatest.version>3.2.15</scalatest.version>
    <scala.binary.version>{scala_binary_version}</scala.binary.version>
    <kryo.version>5.5.0</kryo.version>
  </properties>

  <dependencies>
   <dependency>
      <groupId>com.esotericsoftware</groupId>
      <artifactId>kryo</artifactId>
      <version>${{kryo.version}}</version>
    </dependency>
    <dependency>
      <groupId>org.scala-lang</groupId>
      <artifactId>scala-compiler</artifactId>
      <version>${{scala.version}}</version>
    </dependency>
    <dependency>
      <groupId>org.scala-lang</groupId>
      <artifactId>scala-reflect</artifactId>
      <version>${{scala.version}}</version>
    </dependency>
    <dependency>
      <groupId>org.scala-lang</groupId>
      <artifactId>scala-library</artifactId>
      <version>${{scala.version}}</version>
    </dependency>
    <dependency>
      <groupId>commons-logging</groupId>
      <artifactId>commons-logging</artifactId>
      <version>1.2</version>
    </dependency>
    <dependency>
      <groupId>org.scalatest</groupId>
      <artifactId>scalatest_${{scala.binary.version}}</artifactId>
      <version>${{scalatest.version}}</version>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-core_${{scala.binary.version}}</artifactId>
      <version>${{spark.version}}</version>
      <scope>provided</scope>
    </dependency>
    <dependency>
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-sql_${{scala.binary.version}}</artifactId>
      <version>${{spark.version}}</version>
      <scope>provided</scope>
    </dependency>
    <dependency>
      <groupId>org.apache.spark</groupId>
      <artifactId>spark-mllib_${{scala.binary.version}}</artifactId>
      <version>${{spark.version}}</version>
      <scope>provided</scope>
    </dependency>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>${{junit.version}}</version>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>ml.dmlc</groupId>
      <artifactId>xgboost4j_${{scala.binary.version}}</artifactId>
      <version>{xgboost4j_version}</version>
    </dependency>
    <dependency>
      <groupId>ml.dmlc</groupId>
      <artifactId>xgboost4j_${{scala.binary.version}}</artifactId>
      <version>{xgboost4j_version}</version>
      <classifier>tests</classifier>
      <type>test-jar</type>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>ml.dmlc</groupId>
      <artifactId>xgboost4j-spark_${{scala.binary.version}}</artifactId>
      <version>{xgboost4j_version}</version>
    </dependency>
    <dependency>
      <groupId>ml.dmlc</groupId>
      <artifactId>xgboost4j-example_${{scala.binary.version}}</artifactId>
      <version>{xgboost4j_version}</version>
    </dependency>
  </dependencies>

  <build>
    <plugins>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-assembly-plugin</artifactId>
        <configuration>
          <descriptorRefs>
            <descriptorRef>jar-with-dependencies</descriptorRef>
          </descriptorRefs>
          <archive>
            <manifest>
              <mainClass>ml.dmlc.xgboost4j.tester.App</mainClass>
            </manifest>
          </archive>
        </configuration>
        <executions>
          <execution>
            <phase>package</phase>
            <goals>
              <goal>single</goal>
            </goals>
          </execution>
        </executions>
      </plugin>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <configuration>
          <dependenciesToScan>
            <dependency>ml.dmlc:xgboost4j_${{scala.binary.version}}</dependency>
          </dependenciesToScan>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
"""

if __name__ == '__main__':
  if len(sys.argv) != 7:
    print('Usage: {} [xgboost4j version] [maven compiler source level] [maven compiler target level] [spark version] [scala version] [scala binary version]'.format(sys.argv[0]))
    sys.exit(1)
  with open('pom.xml', 'w') as f:
    print(pom_template.format(xgboost4j_version=sys.argv[1],
                              maven_compiler_source=sys.argv[2],
                              maven_compiler_target=sys.argv[3],
                              spark_version=sys.argv[4],
                              scala_version=sys.argv[5],
                              scala_binary_version=sys.argv[6]), file=f)
