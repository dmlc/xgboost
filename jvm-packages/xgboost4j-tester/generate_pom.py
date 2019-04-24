import sys

pom_template = """
<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>ml.dmlc</groupId>
  <artifactId>xgboost4j-tester</artifactId>
  <version>1.0-SNAPSHOT</version>

  <name>xgboost4j-tester</name>

  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.source>{maven_compiler_source}</maven.compiler.source>
    <maven.compiler.target>{maven_compiler_target}</maven.compiler.target>
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
        <version>2.3.11</version>
        <scope>compile</scope>
    </dependency>
    <dependency>
        <groupId>com.typesafe.akka</groupId>
        <artifactId>akka-testkit_${{scala.binary.version}}</artifactId>
        <version>2.3.11</version>
        <scope>test</scope>
    </dependency>
    <dependency>
        <groupId>org.scalatest</groupId>
        <artifactId>scalatest_${{scala.binary.version}}</artifactId>
        <version>3.0.0</version>
        <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.11</version>
      <scope>test</scope>
    </dependency>
    <dependency>
      <groupId>ml.dmlc</groupId>
      <artifactId>xgboost4j</artifactId>
      <version>{xgboost4j_version}</version>
    </dependency>
    <dependency>
      <groupId>ml.dmlc</groupId>
      <artifactId>xgboost4j</artifactId>
      <version>{xgboost4j_version}</version>
      <classifier>tests</classifier>
      <type>test-jar</type>
      <scope>test</scope>
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
            <dependency>ml.dmlc:xgboost4j</dependency>
          </dependenciesToScan>
        </configuration>
      </plugin>
    </plugins>
  </build>
</project>
"""

if __name__ == '__main__':
  if len(sys.argv) != 6:
    print('Usage: {} [xgboost4j version] [maven compiler source level] [maven compiler target level] [scala version] [scala binary version]'.format(sys.argv[0]))
    sys.exit(1)
  with open('pom.xml', 'w') as f:
    print(pom_template.format(xgboost4j_version=sys.argv[1],
                              maven_compiler_source=sys.argv[2],
                              maven_compiler_target=sys.argv[3],
                              scala_version=sys.argv[4],
                              scala_binary_version=sys.argv[5]), file=f)
