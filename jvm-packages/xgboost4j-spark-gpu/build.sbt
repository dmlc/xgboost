version := "1.4.1-spark3.2"

scalaVersion := "2.12.15"

resolvers += "Apache Spark RC Repository" at "https://repository.apache.org/content/repositories/orgapachespark-1392"

organization := "ml.dmlc"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.2.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.2.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.2.0" % "provided",
  "ml.dmlc" %% "xgboost4j-gpu" % "1.4.1",
)

Compile / scalaSource := baseDirectory.value / "src" / "main" / "scala"
