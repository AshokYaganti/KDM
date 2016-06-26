
import org.apache.spark.ml.feature.{StopWordsRemover, HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession


/**
  * Created by Mayanka on 17-Jun-16.
  */
object SparkNLPMain {

  def main(args: Array[String]) {
    val spark = SparkSession
      .builder
      .appName("TfIdfExample")
      .master("local[*]")
      .getOrCreate()


    // $example on$
    val sentenceData = spark.createDataFrame(Seq(
      (0, "The most important actions that one can perform on a website also tend to be the ones "),
      (0, "Video-sharing sites need to be able to associate unique upvotes with users. Using CSRF"),
      (1, "As an example, my banking website, example.com, does not protect itself against CSRF.")
    )).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)

    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filteredWords")
    val processedWordData= remover.transform(wordsData)

    val hashingTF = new HashingTF()
      .setInputCol("filteredWords").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(processedWordData)
    // alternatively, CountVectorizer can also be used to get term frequency vectors

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("filteredWords","features", "label").take(3).foreach(println)


    spark.stop()

  }

}
