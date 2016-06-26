package mlpipeline

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.feature.{NGram, StopWordsRemover, Tokenizer, Word2Vec}
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by Mayanka on 21-Jun-16.
  */
object SparkW2VML {
  def main(args: Array[String]) {

    // Configuration
    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

    val sc = new SparkContext(sparkConf)

    val spark = SparkSession
      .builder
      .appName("SparkW2VML")
      .master("local[*]")
      .getOrCreate()


    // Turn off Info Logger for Console
    Logger.getLogger("org").setLevel(Level.OFF);
    Logger.getLogger("akka").setLevel(Level.OFF);

    // Read the file into RDD[String]
    val input = sc.textFile("data/sample").map(line => {
      //Getting Lemmatized Form of the word using CoreNLP
      val lemma = CoreNLP.returnLemma(line)
      (0, lemma)
    })


    //Creating DataFrame from RDD

    val sentenceData = spark.createDataFrame(input).toDF("labels", "sentence")

    //Tokenizer
    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)

    //Stop Word Remover
    val remover = new StopWordsRemover()
      .setInputCol("words")
      .setOutputCol("filteredWords")
    val processedWordData = remover.transform(wordsData)

    //NGram
    val ngram = new NGram().setInputCol("filteredWords").setOutputCol("ngrams")
    val ngramDataFrame = ngram.transform(processedWordData)
    ngramDataFrame.take(3).foreach(println)
    println(ngramDataFrame.printSchema())

    //TFIDF TopWords
    val topWords = TFIDF.getTopTFIDFWords(sc, ngramDataFrame.select("filteredWords").rdd)
    println("TOP WORDS: \n\n"+ topWords.mkString("\n"))


    //Word2Vec Model Generation

    val word2Vec = new Word2Vec()
      .setInputCol("filteredWords")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2Vec.fit(ngramDataFrame)

    //Finding synonyms for TOP TFIDF Words using Word2Vec Model
    topWords.foreach(f => {
      println(f._1+"  : ")
      val result = model.findSynonyms(f._1, 3)
      result.take(3).foreach(println)
      println
    })

    spark.stop()
  }

}
