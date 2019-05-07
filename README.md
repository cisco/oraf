# ORaF (Optimized Random Forest on Apache Spark)

ORaF is a library which aims to improve the performance of distributed random forest training on large datasets in Spark MLlib. ORaF is a fork of the random forest algorithm in Mllib and as such has an almost identical interface.

To optimize the training process, we introduce a local training phase with improved task scheduling. We complete the tree induction of sufficiently small nodes in-memory on a single executor. Additionally, we group these nodes into larger and more balanced local training tasks using bin packing and effectively schedule the processing of these tasks into batches by computing their expected duration. Our algorithm speeds up the training process significantly (**more than 100x**), enables the training of deeper decision trees and mitigates runtime memory issues which allows reliable horizontal scaling (we are able to train a model on a billion rows).

## Benchmark

A thorough explanation of the used methods and detailed experiments can be found in the authors' thesis: [Distributed Algorithms for Decision Forest Training in the Network Traffic Classification Task](https://dspace.cvut.cz/bitstream/handle/10467/76092/F3-BP-2018-Starosta-Radek-thesis.pdf). 

The following plot compares the performance of MLlib and ORaF on two datasets (10M rows and 30M rows) originating from network telemetry. The dimension of feature vectors was 357, and the rows were labeled into 153 strongly imbalanced classes. The experiment ran on an AWS EMR cluster of 11 r4.2xlarge instances, and the experiment setup along with all hyperparameters is described in much greater detail in the thesis on pages 33-35.

![MLlib vs ORaF benchmark showing 100-fold performance increase](img/mllib_oraf1.png?raw=true "MLlib vs ORaF benchmark showing 100-fold performance increase")

ORaF is 40x faster than MLLib on the 10M dataset and more than 100x faster on the 30M dataset.

## Installation

Use `mvn package` to build the project to jar file in Maven. You can also download a prebuilt jar file in the releases tab.

We plan to add ORaF to https://spark-packages.org/ soon.

Currently, ORaF depends and was tested on Apache Spark 2.4.0. We will try to update the dependency regularly to more recent Spark versions. If you would like to try ORaF on a version of Spark that we do not officially support yet, feel free to try it. In our experience, the jar file usually works even on slightly different minor or patch versions of Spark.

## Example

The interface is almost identical to the original RandomForestClassifier / RandomForestRegressor classes (see [RandomForestClassifier](https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forest-classifier)). It includes all of the fundamental methods for training, saving/loading models and inference, but we don't support computing classification probabilities and feature importance (see [Removal of ImpurityStats](#removal-of-impuritystats-from-final-models)).

    import org.apache.spark.ml.classification.OptimizedRandomForestClassifier
    
    val orf = new OptimizedRandomForestClassifier()
        .setImpurity("entropy")
        .setMaxDepth(30)
        .setNumTrees(5)
        .setMaxMemoryMultiplier(2.0)

    // trainingData is a Dataset containing "label" (Double) and "features" (ml.Vector) columns
    val model = orf.fit(trainingData)

    // testData is a Dataset with a "features" (ml.Vector) column, predictions are filled into a new "prediction" column 
    val dataWithPredictions = model.transform(testData)

Note that ORaF is implemented in the org.apache.spark package structure because it has various package private dependencies to Spark MLlib. There are no naming conflicts with Spark. If there is demand, we will figure out how to address this better in future major versions. 

## Old MLlib interface example

The training interface is again identical to the MLlib RandomForest class (see [MLlib ensembles](https://spark.apache.org/docs/latest/mllib-ensembles.html)). This interface returns the same models as the new ml interface (OptimizedRandomForestClassificationModel / OptimizedRandomForestRegressionModel), as the old model is unable to store trees deeper than 30 levels because of node indexing.

    import org.apache.spark.mllib.tree.configuration.OptimizedForestStrategy
    import org.apache.spark.mllib.tree.OptimizedRandomForest
    
    val strategy = new OptimizedForestStrategy(algo = Classification,
                                               impurity = Entropy,
                                               maxDepth = 30,
                                               numClasses = 3,
                                               numTrees = 5
                                               maxMemoryMultiplier = 2.0)

    // trainingData is an RDD of LabeledPoints
    val (model, statistics) = OptimizedRandomForest.trainClassifier(
          input = trainingData,
          featureSubsetStrategy = "sqrt",
          strategy = strategy,
          numTrees = 5)

    // testData is an RDD of mllib.Vectors
    val dataWithPredictions = testData.map { point =>
        (point, model.predict(point.features))
    }

## Additional parameters

These parameters can be set in the OptimizedForestStrategy object (RDD MLlib interface), or in the OptimizedRandomForestClassifier / Regressor class (DataFrame ml interface).

- maxMemoryMultiplier (Double)
    - This parameter affects the threshold deciding whether a task is small enough to be trained locally. It is used to multiply the estimate of the tasks memory consumption (the larger the value, the smaller the task has to be for it to be selected for local training). The default value is 4.0, which is very conservative. Increasing this parameter can also help to balance the tasks if your dataset isn't very large and the training doesn't utilize the cluster fully.
- timePredictionStrategy (TimePredictionStrategy)
    - The logic behind the task scheduling. By default, the tasks are sorted by the number of data points, which works well in most cases. During our experiments, we found that the entropy in the given node also plays a significant role in the final training time of the nodes, so in our in-house implementation, we use a linear regressor combining both task size and entropy (see [thesis](#benchmark)).
- localTrainingAlgorithm (LocalTrainingAlgorithm)
    - Implementation of the local decision tree training. Default is an implementation by Siddharth Murching ([smurching](https://github.com/smurching), [SPARK-3162](https://github.com/apache/spark/pull/19433)) which is based on the Yggdrasil algorithm. In the current state, this implementation is probably not the most efficient solution, because it doesn't fully utilize the advantages of the columnar format, but still requires the data to be transformed into it.
- maxTasksPerBin (Int)
    - This parameter can be used to limit the total number of tasks packed into one bin (the batch of training tasks sent to a single executor). By default, the amount of tasks is not limited, and the algorithm tries to make the bins as large as possible.
- customSplits (Array[Array[Double]])
    - The default discretization logic that is hardcoded into the current random forest implementation can work poorly on some datasets (i.e., when classes are highly imbalanced), so this allows the users to pass in their own precomputed threshold values for individual features.

## Notable differences between ORaF and algorithm in Apache Spark MLlib (2.4.0)

### Removal of ImpurityStats from final models

We have decided to remove the ImpurityStats objects in the finalized version of the tree model. In classification, the final predicted value is the majority class in the appropriate leaf node, and we don't compute the individual class probabilities. In most cases, this does not have any significant impact on the classification performance [1] but helped us mitigate some of the memory management issues we've encountered with larger datasets.

[1] L. Breiman. Bagging predictors. Technical Report 421, University of California Berkeley, 1994.

### Removal of tree depth limit 

As the trees are now eventually trained locally on one executor core, we no longer need to have a globally unique index for every node. Therefore, we can theoretically train the entire subtree for every node, although this would probably be too time intensive for large datasets.

Because the improved algorithm allows training trees deeper than 30 levels which cannot be represented in the 1.x version of the MLlib decision tree models, the old MLlib interface also returns the new ml models, which include a convenience predict method for the old MLlib Vectors. (see [mllib example](#old-mllib-interface-example))

### NodeIdCache enabled by default

Additionally, our method relies heavily on the presence of NodeIdCache, which is used to pair data points with their respective tree nodes quickly. We have decided to enable it by default, as it provides a significant speedup by sacrificing some memory.

## Authors

* Radek Starosta (rstarost@cisco.com, github: @rstarosta)
* Jan Brabec (janbrabe@cisco.com, github: @BrabecJan, twitter: @BrabecJan91)
