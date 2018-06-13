package classification;

import org.apache.spark.ml.classification.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

import static java.nio.charset.StandardCharsets.UTF_8;

public class NaivesBayesClassifier {

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

        // Split the data into train and test
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3}, 1234L);
        Dataset<Row> train = splits[0];
        Dataset<Row> test = splits[1];

        // create the trainer and set its parameters
        NaiveBayes nb = new NaiveBayes();

        // train the model
        NaiveBayesModel model = nb.fit(train);

        // Select example rows to display.
        Dataset<Row> predictions = model.transform(test);


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
