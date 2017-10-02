import sbt._
import Keys._

object Dependencies {
  val commonDependencies: Seq[ModuleID] = Seq(
    "commons-logging" % "commons-logging" % "1.2",
    "org.scalatest" %% "scalatest" % "3.0.0" % "test"
  )

  val akkaVersion = "2.3.11"

  val coreDependencies: Seq[ModuleID] = commonDependencies ++ Seq(
    "com.esotericsoftware.kryo" % "kryo" % "2.21",
    "com.typesafe.akka" %% "akka-actor" % akkaVersion,
    "com.typesafe.akka" %% "akka-testkit" % akkaVersion % "test",
    "junit" % "junit" % "4.11" % "test"
  )

  val sparkVersion = "2.1.0"

  val sparkDependencies: Seq[ModuleID] = commonDependencies ++ Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion,
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion,
    "org.apache.spark" %% "spark-streaming" % sparkVersion)

  val flinkVersion = "1.3.1"

  val flinkDependencies: Seq[ModuleID] = commonDependencies ++ Seq(
    "org.apache.flink" %% "flink-scala" % flinkVersion,
    "org.apache.flink" %% "flink-clients" % flinkVersion,
    "org.apache.flink" %% "flink-ml" % flinkVersion
  )

  val exampleDependencies: Seq[ModuleID] =
    coreDependencies ++ sparkDependencies ++ flinkDependencies
}