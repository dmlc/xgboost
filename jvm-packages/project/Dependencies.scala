/*
 Copyright (c) 2014 by Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 */

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
    ("org.apache.spark" %% "spark-core" % sparkVersion % "provided")
        .exclude("commons-beanutils", "commons-beanutils-core")
        .exclude("com.esotericsoftware.minlog", "minlog")
        .exclude("commons-collections", "commons-collections")
        .exclude("commons-logging", "commons-logging")
        .exclude("org.glassfish.hk2.external", "*")
        .exclude("org.apache.hadoop", "hadoop-yarn-api")
        .exclude("org.slf4j", "jcl-over-slf4j"),
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided"
  )

  val flinkVersion = "1.3.2"

  val flinkDependencies: Seq[ModuleID] = commonDependencies ++ Seq(
    "org.apache.flink" %% "flink-scala" % flinkVersion % "provided",
    "org.apache.flink" %% "flink-ml" % flinkVersion % "provided"
  ).map(
    _.exclude("com.esotericsoftware.kryo", "kryo")
        .exclude("net.java.dev.jets3t", "jets3t")
  )

  val exampleDependencies: Seq[ModuleID] =
    coreDependencies ++ sparkDependencies ++ flinkDependencies
}
