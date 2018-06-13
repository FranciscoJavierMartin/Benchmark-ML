library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))

args = commandArgs(trailingOnly=TRUE)

if(length(args)!=4){
  stop("Faltan argumentos: spark-submit GaussianMixtureModel.R appName master dataset outputFile")
}

sparkR.session(master = args[2],appName = args[1])

start_time <- proc.time()

df <- read.df(args[3], source = "libsvm")

df_list <- randomSplit(df, c(7, 3), 1234)

training <- df_list[[1]]
test <- df_list[[2]]

# Fit a gaussian mixture clustering model with spark.gaussianMixture
model <- spark.gaussianMixture(training, ~ features, k = 2)

# Model summary
#summary(model)
predictions <- predict(model, test)

fila=c(args[1],args[2],(proc.time()-start_time)[3],"seconds","")

write(x = fila,sep=" ",file=args[4],append=TRUE)

sparkR.session.stop()

# Prediction
#predictions <- predict(model, test)
#head(predictions)
