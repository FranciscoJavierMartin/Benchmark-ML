package clustering

import org.apache.spark.ml.classification.{LinearSVC, MultilayerPerceptronClassifier}
import org.apache.spark.sql.SparkSession
import utils.Cronometer

object LinearSupportVectorMachine {

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

    val lsvc = new LinearSVC()
      .setMaxIter(10)
      .setRegParam(0.1)

    // Fit the model
    val lsvcModel = lsvc.fit(data)

    cronometer.appendTime(args(3),args)
    cronometer.printTime()

    spark.stop()
  }

}
