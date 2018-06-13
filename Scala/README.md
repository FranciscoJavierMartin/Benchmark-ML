# Scala algorithms to compare performance of some machine learning algorithms include on Apache Spark

## Previous considerations
This code was developed using Scala 2.11.12 and Apache Spark 2.3.0 and was tested on Linux system.

## Usage
1. ```sbt clean compile package```
2. 
~~~
spark-submit --class <mainClass> --master <master> target/scala-2.11/scalasparkml_2.11-0.1.jar <appName> <master> <dataset> <outputFile>
~~~

The mainClass to use are:
- classification.DecissionTree
- classification.GradientBoostTree
- classification.LinearSupportVectorMachine
- classification.MultiLayerPerceptron
- classification.NaiveBayesClassifier
- classification.RandomForest
- clustering.BisectingKMeansClustering
- clustering.GaussianMixtureModel
- clustering.KmeansClustering
- clustering.LatentDirichletAllocation