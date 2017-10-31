import sbt._

object Dependencies {
  val commonDependencies: Seq[ModuleID] = Seq(
    "commons-logging" % "commons-logging" % "1.2",
    "org.scalatest" %% "scalatest" % "3.0.0" % "test"
  )

  val flakkaVersion = "2.3-custom"

  val coreDependencies: Seq[ModuleID] = commonDependencies ++ Seq(
    // TODO: remove Kryo dependency.
    "com.esotericsoftware" % "kryo-shaded" % "3.0.3",
    "com.data-artisans" %% "flakka-actor" % flakkaVersion,
    "com.data-artisans" %% "flakka-testkit" % flakkaVersion % "test",
    "junit" % "junit" % "4.11" % "test"
  )

  val sparkVersion = "2.1.0"

  val sparkDependencies: Seq[ModuleID] = commonDependencies ++ Seq(
    ("org.apache.spark" %% "spark-core" % sparkVersion).
        exclude("commons-beanutils", "commons-beanutils-core").
        exclude("com.esotericsoftware.minlog", "minlog").
        exclude("commons-collections", "commons-collections").
        exclude("commons-logging", "commons-logging").
        exclude("org.glassfish.hk2.external", "*").
        exclude("org.apache.hadoop", "hadoop-yarn-api").
        exclude("org.slf4j", "jcl-over-slf4j"),
    "org.apache.spark" %% "spark-sql" % sparkVersion,
    "org.apache.spark" %% "spark-mllib" % sparkVersion)

  val flinkVersion = "1.3.2"

  val flinkDependencies: Seq[ModuleID] = commonDependencies ++ Seq(
    "org.apache.flink" %% "flink-scala" % flinkVersion,
    "org.apache.flink" %% "flink-ml" % flinkVersion
  ).map(_.
      exclude("com.esotericsoftware.kryo", "kryo").
      exclude("net.java.dev.jets3t", "jets3t"))

  val exampleDependencies: Seq[ModuleID] =
    coreDependencies ++ sparkDependencies ++ flinkDependencies
}
