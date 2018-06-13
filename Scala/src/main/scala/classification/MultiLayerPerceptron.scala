package clustering

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{MultilayerPerceptronClassifier, RandomForestClassifier}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession
import utils.Cronometer

object MultiLayerPerceptron {
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

    // Split the data into train and test
    val splits = data.randomSplit(Array(0.7, 0.3), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](4, 5, 4, 3)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    // train the model
    val model = trainer.fit(train)

    // compute accuracy on the test set
    val result = model.transform(test)

    cronometer.appendTime(args(3),args)
    cronometer.printTime()

    spark.stop()
  }
}
