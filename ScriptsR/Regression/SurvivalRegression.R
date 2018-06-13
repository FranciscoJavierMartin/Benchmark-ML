# Use the ovarian dataset available in R survival package
library(survival)

# Fit an accelerated failure time (AFT) survival regression model with spark.survreg
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))

sparkR.session(master = "local[*]",appName = "args[1]")

training <- read.df("../../data/classification.1000.4.txt", source = "libsvm")

aftModel <- spark.survreg(training, Surv(futime, fustat) ~ features)

# Model summary
summary(aftModel)

# Prediction
#aftPredictions <- predict(aftModel, aftTestDF)
#head(aftPredictions)