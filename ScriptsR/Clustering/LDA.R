# Load training data
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))

args = commandArgs(trailingOnly=TRUE)

if(length(args)!=4){
  stop("Faltan argumentos: spark-submit LDA.R appName master dataset outputFile")
}

sparkR.session(master = args[2],appName = args[1])

start_time <- proc.time()

training <- read.df(args[3], source = "libsvm")

# Fit a latent dirichlet allocation model with spark.lda
model <- spark.lda(training, k = 10, maxIter = 10)

# Model summary
#summary(model)

fila=c(args[1],args[2],(proc.time()-start_time)[3],"seconds","")

write(x = fila,sep=" ",file=args[4],append=TRUE)

sparkR.session.stop()

# Prediction
#predictions <- predict(model, test)
#head(predictions)
