package clustering

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.sql.SparkSession
import utils.Cronometer

object RandomForest {
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

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(data)
    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(4)
      .fit(data)

    // Split the data into training and test sets (30% held out for testing).
    val  Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3),seed = 1234L)

    // Train a RandomForest model.
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("indexedFeatures")
      .setNumTrees(10)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and forest in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer, featureIndexer, rf, labelConverter))

    // Train model. This also runs the indexers.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    cronometer.appendTime(args(3),args)
    cronometer.printTime()

    spark.stop()
  }
}
