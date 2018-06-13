package clustering;

import org.apache.spark.ml.clustering.GaussianMixture;
import org.apache.spark.ml.clustering.GaussianMixtureModel;
import org.apache.spark.ml.clustering.LDA;
import org.apache.spark.ml.clustering.LDAModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

import static java.nio.charset.StandardCharsets.UTF_8;

public class LatentDirichletAllocation {

    public static void main(String[] args) {
        /*
            args[0] -> Appname
            args[1] -> master
            args[2] -> dataset
            args[3] -> outputFile
         */

        if(args.length!=4){
            throw new IllegalArgumentException("Spark-submit --class <mainClass> --master <master> "+
                    "target/SparkMLMaven-1.0-SNAPSHOT.jar <appName> <master> <dataset> <outputFile>");
        }

        SparkSession spark = SparkSession
                .builder()
                .appName(args[0])
                .getOrCreate();

        /*
            If master cause problem to assign nodes,
            please use the following code replacing the former.
         */
        /*SparkSession spark = SparkSession
                .builder()
                .master(args[1])
                .appName(args[0])
                .getOrCreate();*/

        long start_time = System.currentTimeMillis();

        // Load the data stored in LIBSVM format as a DataFrame.
        Dataset<Row> data = spark
                .read()
                .format("libsvm")
                .load(args[2]);

        // Split the data into training and test sets (30% held out for testing)
        Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3},1234L);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Trains a LDA model.
        LDA lda = new LDA().setK(10).setMaxIter(10);
        LDAModel model = lda.fit(trainingData);

        // Make predictions
        Dataset<Row> predictions = model.transform(testData);


        String textToSave =
                args[0] + '\n'
                        + args[1] + '\n'
                        + (System.currentTimeMillis() - start_time) + " milliseconds\n";
        try {
            Files.write(Paths.get(args[3]), (textToSave).getBytes(UTF_8), StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            System.out.println(e);
        }

        spark.stop();
    }
}
