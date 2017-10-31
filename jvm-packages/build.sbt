val compileJni = taskKey[Unit]("Compiles XGBoost JNI bindings")

lazy val core = (project in file("xgboost4j")).
    settings(Common.settings: _*).
    settings(
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

lazy val spark = (project in file("xgboost4j-spark")).
    dependsOn(core % "test->test;compile->compile").
    settings(Common.settings: _*).
    settings(
      libraryDependencies ++= Dependencies.sparkDependencies,

      // Not supported by [[PerTest]] trait.
      parallelExecution in Test := false,

      assemblyMergeStrategy in assembly := (_ => MergeStrategy.first)
    )

lazy val flink = (project in file("xgboost4j-flink")).
    dependsOn(core).
    settings(Common.settings: _*).
    settings(libraryDependencies ++= Dependencies.flinkDependencies).

lazy val example = (project in file("xgboost4j-examples")).
    disablePlugins(sbtassembly.AssemblyPlugin).
    dependsOn(core, spark, flink).
    settings(Common.settings: _*).
    settings(libraryDependencies ++= Dependencies.exampleDependencies)

lazy val root = (project in file(".")).
    disablePlugins(sbtassembly.AssemblyPlugin).
    aggregate(core, spark, flink, example).
    settings(name := "xgboost").
    settings(Common.settings: _*)
