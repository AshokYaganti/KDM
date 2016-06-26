name := "Spark-W2Vec"

version := "1.0"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "2.0.0-preview",
  "org.apache.spark" % "spark-mllib_2.11" % "2.0.0-preview",
  "org.apache.spark" % "spark-sql_2.11" % "2.0.0-preview",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0",
  "edu.stanford.nlp" % "stanford-corenlp" % "3.6.0" classifier "models",
  "edu.stanford.nlp" % "stanford-parser" % "3.6.0",
  "com.google.protobuf" % "protobuf-java" % "2.6.1"

)
