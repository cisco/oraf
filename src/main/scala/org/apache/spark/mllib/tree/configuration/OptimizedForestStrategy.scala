/*
 * Modified work Copyright (C) 2019 Cisco Systems
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.tree.configuration

import org.apache.spark.annotation.Since
import org.apache.spark.ml.tree.LocalTrainingAlgorithm
import org.apache.spark.ml.tree.impl.LocalDecisionTree
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.impurity.{Gini, Impurity, Variance}

import scala.beans.BeanProperty
import scala.collection.JavaConverters._

/**
 * Stores all the configuration options for tree construction
 * @param algo  Learning goal.  Supported:
 *              `org.apache.spark.mllib.tree.configuration.Algo.Classification`,
 *              `org.apache.spark.mllib.tree.configuration.Algo.Regression`
 * @param impurity Criterion used for information gain calculation.
 *                 Supported for Classification: [[org.apache.spark.mllib.tree.impurity.Gini]],
 *                  [[org.apache.spark.mllib.tree.impurity.Entropy]].
 *                 Supported for Regression: [[org.apache.spark.mllib.tree.impurity.Variance]].
 * @param maxDepth Maximum depth of the tree (e.g. depth 0 means 1 leaf node, depth 1 means
 *                 1 internal node + 2 leaf nodes).
 * @param numClasses Number of classes for classification.
 *                                    (Ignored for regression.)
 *                                    Default value is 2 (binary classification).
 * @param maxBins Maximum number of bins used for discretizing continuous features and
 *                for choosing how to split on features at each node.
 *                More bins give higher granularity.
 * @param quantileCalculationStrategy Algorithm for calculating quantiles.  Supported:
 *                             `org.apache.spark.mllib.tree.configuration.QuantileStrategy.Sort`
 * @param categoricalFeaturesInfo A map storing information about the categorical variables and the
 *                                number of discrete values they take. An entry (n to k)
 *                                indicates that feature n is categorical with k categories
 *                                indexed from 0: {0, 1, ..., k-1}.
 * @param minInstancesPerNode Minimum number of instances each child must have after split.
 *                            Default value is 1. If a split cause left or right child
 *                            to have less than minInstancesPerNode,
 *                            this split will not be considered as a valid split.
 * @param minInfoGain Minimum information gain a split must get. Default value is 0.0.
 *                    If a split has less information gain than minInfoGain,
 *                    this split will not be considered as a valid split.
 * @param maxMemoryInMB Maximum memory in MB allocated to histogram aggregation. Default value is
 *                      256 MB.  If too small, then 1 node will be split per iteration, and
 *                      its aggregates may exceed this size.
 * @param subsamplingRate Fraction of the training data used for learning decision tree.
 * @param useNodeIdCache If this is true, instead of passing trees to executors, the algorithm will
 *                       maintain a separate RDD of node Id cache for each row.
 * @param checkpointInterval How often to checkpoint when the node Id cache gets updated.
 *                           E.g. 10 means that the cache will get checkpointed every 10 updates. If
 *                           the checkpoint directory is not set in
 *                           [[org.apache.spark.SparkContext]], this setting is ignored.
 */
@Since("1.0.0")
class OptimizedForestStrategy @Since("1.3.0")(
    algo: Algo,
    impurity: Impurity,
    maxDepth: Int,
    numClasses: Int = 2,
    maxBins: Int = 32,
    quantileCalculationStrategy: QuantileStrategy = Sort,
    categoricalFeaturesInfo: Map[Int, Int]
     = Map[Int, Int](),
    minInstancesPerNode: Int = 1,
    minInfoGain: Double = 0.0,
    maxMemoryInMB: Int = 256,
    subsamplingRate: Double = 1,
    useNodeIdCache: Boolean = true,
    checkpointInterval: Int = 10,
    @Since("2.1.2") @BeanProperty var maxMemoryMultiplier: Double = 4.0,
    @Since("2.1.2") @BeanProperty var timePredictionStrategy: TimePredictionStrategy = new DefaultTimePredictionStrategy,
    @Since("2.1.2") @BeanProperty var maxTasksPerBin: Int = Int.MaxValue,
    @Since("2.3.1") @BeanProperty var localTrainingAlgorithm: LocalTrainingAlgorithm = new LocalDecisionTree,
    @Since("2.3.1") @BeanProperty var customSplits: Option[Array[Array[Double]]] = None)
      extends Strategy(algo, impurity, maxDepth, numClasses, maxBins, quantileCalculationStrategy,
        categoricalFeaturesInfo, minInstancesPerNode, minInfoGain, maxMemoryInMB, subsamplingRate,
        useNodeIdCache, checkpointInterval) with Serializable {

  /**
   * Java-friendly constructor for [[org.apache.spark.mllib.tree.configuration.Strategy]]
   */
  @Since("1.1.0")
  def this(
      algo: Algo,
      impurity: Impurity,
      maxDepth: Int,
      numClasses: Int,
      maxBins: Int,
      categoricalFeaturesInfo: java.util.Map[java.lang.Integer, java.lang.Integer]) {
    this(algo, impurity, maxDepth, numClasses, maxBins, Sort,
      categoricalFeaturesInfo.asInstanceOf[java.util.Map[Int, Int]].asScala.toMap)
  }
  /**
   * Check validity of parameters.
   * Throws exception if invalid.
   */
  private[spark] override def assertValid(): Unit = {
    super.assertValid()
    require(maxMemoryMultiplier > 0,
      s"DecisionTree Strategy requires maxMemoryMultiplier > 0, but was given " +
        s"$maxMemoryMultiplier")
    require(maxTasksPerBin > 0,
      s"DecisionTree Strategy requires maxTasksPerBin > 0, but was given " +
        s"$maxMemoryMultiplier")
  }

  /**
   * Returns a shallow copy of this instance.
   */
  @Since("1.2.0")
  override def copy: OptimizedForestStrategy = {
    new OptimizedForestStrategy(algo, impurity, maxDepth, numClasses, maxBins,
      quantileCalculationStrategy, categoricalFeaturesInfo, minInstancesPerNode, minInfoGain,
      maxMemoryInMB, subsamplingRate, useNodeIdCache, checkpointInterval,
      maxMemoryMultiplier, timePredictionStrategy, maxTasksPerBin, localTrainingAlgorithm, customSplits)
  }
}

@Since("1.2.0")
object OptimizedForestStrategy {

  /**
   * Construct a default set of parameters for [[org.apache.spark.mllib.tree.DecisionTree]]
   * @param algo  "Classification" or "Regression"
   */
  @Since("1.2.0")
  def defaultStrategy(algo: String): OptimizedForestStrategy = {
    defaultStrategy(Algo.fromString(algo))
  }

  /**
   * Construct a default set of parameters for [[org.apache.spark.mllib.tree.DecisionTree]]
   * @param algo Algo.Classification or Algo.Regression
   */
  @Since("1.3.0")
  def defaultStrategy(algo: Algo): OptimizedForestStrategy = algo match {
    case Algo.Classification =>
      new OptimizedForestStrategy(algo = Classification, impurity = Gini, maxDepth = 10,
        numClasses = 2)
    case Algo.Regression =>
      new OptimizedForestStrategy(algo = Regression, impurity = Variance, maxDepth = 10,
        numClasses = 0)
  }

}
