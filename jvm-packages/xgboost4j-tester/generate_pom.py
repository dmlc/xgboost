import sys

pom_template = """
<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>ml.dmlc</groupId>
  <artifactId>xgboost4j-tester_2.12</artifactId>
  <version>1.0-SNAPSHOT</version>

  <name>xgboost4j-tester_2.12</name>

  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.source>{maven_compiler_source}</maven.compiler.source>
    <maven.compiler.target>{maven_compiler_target}</maven.compiler.target>
    <spark.version>{spark_version}</spark.version>
    <scala.version>{scala_version}</scala.version>
    <scala.binary.version>{scala_binary_version}</scala.binary.version>
  </properties>

  <dependencies>
    <dependency>
      <groupId>com.esotericsoftware.kryo</groupId>
      <artifactId>kryo</artifactId>
      <version>2.21</version>
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
      <groupId>com.typesafe.akka</groupId>
      <artifactId>akka-actor_${{scala.binary.version}}</artifactId>
      <version>2.5.23</version>
      <scope>compile</scope>
    </dependency>
    <dependency>
      <groupId>com.typesafe.akka</groupId>
      <artifactId>akka-testkit_${{scala.binary.version}}</artifactId>
      <version>2.5.23</version>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>org.scalatest</groupId>
      <artifactId>scalatest_${{scala.binary.version}}</artifactId>
      <version>3.0.8</version>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>org.scalactic</groupId>
      <artifactId>scalactic_${{scala.binary.version}}</artifactId>
      <version>3.0.8</version>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>org.apache.commons</groupId>
      <artifactId>commons-lang3</artifactId>
      <version>3.4</version>
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
      <version>4.11</version>
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
      <!-- clean lifecycle, see https://maven.apache.org/ref/current/maven-core/lifecycles.html#clean_Lifecycle -->
      <plugin>
        <artifactId>maven-clean-plugin</artifactId>
        <version>3.1.0</version>
      </plugin>
      <!-- default lifecycle, jar packaging: see https://maven.apache.org/ref/current/maven-core/default-bindings.html#Plugin_bindings_for_jar_packaging -->
      <plugin>
        <artifactId>maven-resources-plugin</artifactId>
        <version>3.0.2</version>
      </plugin>
      <plugin>
        <artifactId>maven-compiler-plugin</artifactId>
        <version>3.8.0</version>
      </plugin>
      <plugin>
        <artifactId>maven-jar-plugin</artifactId>
        <version>3.0.2</version>
      </plugin>
      <plugin>
        <artifactId>maven-install-plugin</artifactId>
        <version>2.5.2</version>
      </plugin>
      <plugin>
        <artifactId>maven-deploy-plugin</artifactId>
        <version>2.8.2</version>
      </plugin>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-assembly-plugin</artifactId>
        <version>2.4</version>
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
      <!-- site lifecycle, see https://maven.apache.org/ref/current/maven-core/lifecycles.html#site_Lifecycle -->
      <plugin>
        <artifactId>maven-site-plugin</artifactId>
        <version>3.7.1</version>
      </plugin>
      <plugin>
        <artifactId>maven-project-info-reports-plugin</artifactId>
        <version>3.0.0</version>
      </plugin>
      <plugin>
        <groupId>org.apache.maven.plugins</groupId>
        <artifactId>maven-surefire-plugin</artifactId>
        <version>2.22.1</version>
        <configuration>
          <dependenciesToScan>
            <dependency>ml.dmlc:xgboost4j_2.12</dependency>
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
