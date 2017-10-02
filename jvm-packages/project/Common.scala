import sbt._
import Keys._

object Common {
  val appVersion = "0.7"

  val settings: Seq[Def.Setting[_]] = Seq(
    version := appVersion,
    scalaVersion := "2.11.11",
    javacOptions ++= Seq("-source", "1.7", "-target", "1.7"),
    scalacOptions ++= Seq("-deprecation", "-unchecked"),
    resolvers += Opts.resolver.mavenLocalFile,
    resolvers ++= Seq(DefaultMavenRepository,
      Resolver.defaultLocal,
      Resolver.mavenLocal,
      Resolver.jcenterRepo)
  )
}