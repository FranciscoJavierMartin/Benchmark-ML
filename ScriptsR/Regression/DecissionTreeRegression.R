# Load training data
library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))

sparkR.session(master = "local[*]",appName = "args[1]")

training <- read.df("../../data/classification.1000.4.txt", source = "libsvm")

# Fit a DecisionTree regression model with spark.decisionTree
model <- spark.decisionTree(training, label ~ features, "regression")

# Model summary
summary(model)

# Prediction
#predictions <- predict(model, test)
#head(predictions)