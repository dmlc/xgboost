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
    // TODO: Remove this dependency. Only Spark part
    // should depend on Kryo
    "com.esotericsoftware" % "kryo-shaded" % "4.0.2",
    "commons-logging" % "commons-logging" % "1.2",
    "org.scalatest" %% "scalatest" % "3.0.8" % Test,
    "org.scalactic" %% "scalactic" % "3.0.8" % Test
  )

  val akkaVersion = "2.5.23"

  val coreDependencies: Seq[ModuleID] = commonDependencies ++ Seq(
    "com.typesafe.akka" %% "akka-actor" % akkaVersion,
    "com.typesafe.akka" %% "akka-testkit" % akkaVersion % Test,

    "junit" % "junit" % "4.11" % Test
  )

  val sparkVersion = "2.4.3"

  val sparkDependencies: Seq[ModuleID] = commonDependencies ++ Seq(
    "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
    "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided"
  )

  val flinkVersion = "1.7.2"

  val flinkDependencies: Seq[ModuleID] = commonDependencies ++ Seq(
    "org.apache.commons" % "commons-lang3" % "3.4",
    "org.apache.hadoop" % "hadoop-common" % "2.7.3",
    "org.apache.flink" %% "flink-scala" % flinkVersion,
    "org.apache.flink" %% "flink-clients" % flinkVersion,
    "org.apache.flink" %% "flink-ml" % flinkVersion
  )

  val exampleDependencies: Seq[ModuleID] = commonDependencies ++ Seq(
    "org.apache.commons" % "commons-lang3" % "3.4"
  )
 }