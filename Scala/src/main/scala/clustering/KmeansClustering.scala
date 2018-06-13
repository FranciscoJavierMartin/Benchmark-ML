package clustering

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession
import utils.Cronometer

object KmeansClustering {
  def main(args: Array[String]): Unit = {

    /*
      args(0) -> appName
      args(1) -> master
      args(2) -> dataset
      args(3) -> output file
     */

    if(args.length!=4){
      throw new IllegalArgumentException("spark-submit --class <mainClass>"+
        " --master <master> target/scala-2.11/scalasparkml_2.11-0.1.jar"+
        " <appName> <master> <dataset> <outputFile>")
    }

    val spark = SparkSession
      .builder()
      .master(args(1))
      .appName(args(0))
      .getOrCreate()

    val cronometer=new Cronometer()

    // Load the data stored in LIBSVM format as a DataFrame.
    val data = spark.read.format("libsvm").load(args(2))

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3),seed = 1234L)

    // Trains a k-means model.
    val kmeans = new KMeans().setK(2).setSeed(1L)
    val model = kmeans.fit(trainingData)

    // Make predictions
    val predictions = model.transform(testData)

    cronometer.appendTime(args(3),args)
    cronometer.printTime()

    spark.stop()
  }
}
