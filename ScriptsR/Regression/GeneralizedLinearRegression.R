library(SparkR, lib.loc = c(file.path(Sys.getenv("SPARK_HOME"), "R", "lib")))

sparkR.session(master = "local[*]",appName = "args[1]")

training <- read.df("../../data/classification.1000.4.txt", source = "libsvm")

gaussianGLM <- spark.glm(training, label ~ features, family = "gaussian")

# Model summary
summary(gaussianGLM)

# Prediction
#gaussianPredictions <- predict(gaussianGLM, gaussianTestDF)
#head(gaussianPredictions)
#
## Fit a generalized linear model with glm (R-compliant)
#gaussianGLM2 <- glm(label ~ features, gaussianDF, family = "gaussian")
#summary(gaussianGLM2)
#
## Fit a generalized linear model of family "binomial" with spark.glm
#training2 <- read.df("data/mllib/sample_multiclass_classification_data.txt", source = "libsvm")
#training2 <- transform(training2, label = cast(training2$label > 1, "integer"))
#df_list2 <- randomSplit(training2, c(7, 3), 2)
#binomialDF <- df_list2[[1]]
#binomialTestDF <- df_list2[[2]]
#binomialGLM <- spark.glm(binomialDF, label ~ features, family = "binomial")
#
## Model summary
#summary(binomialGLM)
#
## Prediction
#binomialPredictions <- predict(binomialGLM, binomialTestDF)
#head(binomialPredictions)
#
## Fit a generalized linear model of family "tweedie" with spark.glm
#training3 <- read.df("data/mllib/sample_multiclass_classification_data.txt", source = "libsvm")
#tweedieDF <- transform(training3, label = training3$label * exp(randn(10)))
#tweedieGLM <- spark.glm(tweedieDF, label ~ features, family = "tweedie",
#                        var.power = 1.2, link.power = 0)
#
## Model summary
#summary(tweedieGLM)