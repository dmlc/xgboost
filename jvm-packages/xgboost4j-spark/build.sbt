version := "1.4.1-spark3.2"

scalaVersion := "2.12.15"

organization := "ml.dmlc"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "3.2.0" % "provided",
  "org.apache.spark" %% "spark-sql" % "3.2.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "3.2.0" % "provided",
  "ml.dmlc" %% "xgboost4j" % "1.4.1",
)

Compile / scalaSource := baseDirectory.value / "src" / "main" / "scala"

publishMavenStyle := true
