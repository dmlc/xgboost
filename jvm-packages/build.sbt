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

ThisBuild / organization := "ml.dmlc"

val compileJni = taskKey[Unit]("Compiles XGBoost JNI bindings")

lazy val xgboost4j = (project in file("xgboost4j"))
  .settings(name := "xgboost4j")
  .settings(Common.settings: _*)
  .settings(
    checkstyleConfigLocation := CheckstyleConfigLocation.File("checkstyle.xml"),
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


lazy val xgboost4jSpark = (project in file("xgboost4j-spark"))
  .dependsOn(xgboost4j % "test->test; compile->compile")
  .settings(name := "xgboost4j-spark")
  .settings(Common.settings: _*)
  .settings(
    checkstyleConfigLocation := CheckstyleConfigLocation.File("checkstyle.xml"),
    libraryDependencies ++= Dependencies.sparkDependencies,

    parallelExecution in Test := false
  )

lazy val xgboost4jFlink = (project in file("xgboost4j-flink"))
  .dependsOn(xgboost4j)
  .settings(name := "xgboost4j-flink")
  .settings(Common.settings)
  .settings(
    checkstyleConfigLocation := CheckstyleConfigLocation.File("checkstyle.xml"),
    libraryDependencies ++= Dependencies.flinkDependencies
  )

lazy val xgboost4jExamples = (project in file("xgboost4j-examples"))
  .disablePlugins(sbtassembly.AssemblyPlugin)
  .dependsOn(xgboost4j, xgboost4jSpark, xgboost4jFlink)
  .settings(name := "xgboost4j-examples")
  .settings(Common.settings: _*)
  .settings(
    checkstyleConfigLocation := CheckstyleConfigLocation.File("checkstyle.xml"),
    libraryDependencies ++= Dependencies.exampleDependencies
  )

lazy val root = (project in file("."))
  .aggregate(xgboost4j, xgboost4jSpark, xgboost4jFlink, xgboost4jExamples)
  .disablePlugins(sbtassembly.AssemblyPlugin)
  .settings(name := "xgboost4j")
  .settings(Common.settings: _*)
  .settings(
    crossScalaVersions := Nil,
    checkstyleConfigLocation := CheckstyleConfigLocation.File("checkstyle.xml")
  )
