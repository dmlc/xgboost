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

val compileJni = taskKey[Unit]("Compiles XGBoost JNI bindings")

lazy val core = (project in file("xgboost4j"))
    .settings(name := "xgboost4j-core")
    .settings(Common.settings: _*)
    .settings(
      libraryDependencies ++= Dependencies.coreDependencies,
      compileJni := {
        import sys.process._
        val rc = Seq("python", "create_jni.py").!
        if (rc != 0) {
          sys.error(s"Failed to compile JNI bindings (exit code $rc)!")
        }
      },
      compile in Compile := (compile in Compile).dependsOn(compileJni).value
    )

lazy val spark = (project in file("xgboost4j-spark"))
    .dependsOn(core % "test->test;compile->compile")
    .settings(Common.settings: _*)
    .settings(name := "xgboost4j-spark")
    .settings(Common.settings: _*)
    .settings(
      libraryDependencies ++= Dependencies.sparkDependencies,

      // Not supported by [[PerTest]] trait.
      parallelExecution in Test := false
    )

lazy val flink = (project in file("xgboost4j-flink"))
    .dependsOn(core)
    .settings(name := "xgboost4j-flink")
    .settings(Common.settings: _*)
    .settings(libraryDependencies ++= Dependencies.flinkDependencies)

lazy val examples = (project in file("xgboost4j-examples"))
    .disablePlugins(sbtassembly.AssemblyPlugin)
    .dependsOn(core, spark, flink)
    .settings(name := "xgboost4j-examples")
    .settings(Common.settings: _*)
    .settings(libraryDependencies ++= Dependencies.exampleDependencies)

lazy val root = (project in file("."))
    .disablePlugins(sbtassembly.AssemblyPlugin)
    .aggregate(core, spark, flink, examples)
    .settings(name := "xgboost4j")
    .settings(Common.settings: _*)
