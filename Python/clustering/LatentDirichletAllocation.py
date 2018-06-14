from pyspark.ml.clustering import LDA
from pyspark.sql import SparkSession
import time
import sys

def appendTime(args,start_time):
    """Append the training time in a file
    
    Arguments:
        args {[String]} -- [main program's args]
        start_time {time} -- [time when the train started]
    """
    total_time=time.time()-start_time
    myfile= open(args[4], "a")
    myfile.write(f"Python AppName: {args[1]} Master: {args[2]} Time: {total_time} seconds\n")
    myfile.write(f"Training time: {total_time//3600} hours, {(total_time//60)%60} minutes, {(total_time)%60} seconds\n\n")
    myfile.close()

def main(args):
    spark=SparkSession\
            .builder\
            .master(args[2])\
            .appName(args[1])\
            .getOrCreate()
    
    start_computing_time = time.time()

    # Load the data stored in LIBSVM format as a DataFrame.
    data = spark.read.format("libsvm").load(args[3])

    (trainingData, testData) = data.randomSplit([0.7, 0.3],seed=1234)

    # Trains a LDA model.
    lda = LDA(k=10, maxIter=10)
    model = lda.fit(trainingData)
    
    # Make predictions
    predictions = model.transform(testData)

    appendTime(sys.argv,start_computing_time)

    spark.stop()

if __name__=='__main__':

    if(len(sys.argv) != 5):
        print("Usage: spark-submit <script-name> <appName> <master> <dataset> <output-file>", file=sys.stderr)
        exit(-1)

    main(sys.argv)