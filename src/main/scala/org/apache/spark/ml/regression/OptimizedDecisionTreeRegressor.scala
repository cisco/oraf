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

package org.apache.spark.ml.regression

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree.OptimizedDecisionTreeModelReadWrite.NodeData
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.OptimizedRandomForest
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.mllib.tree.configuration.{TimePredictionStrategy, Algo => OldAlgo, OptimizedForestStrategy => OldStrategy}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel => OldDecisionTreeModel}
import org.apache.spark.mllib.linalg.{Vector => OldVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.json4s.JsonDSL._
import org.json4s.{DefaultFormats, JObject}


/**
 * <a href="http://en.wikipedia.org/wiki/Decision_tree_learning">Decision tree</a>
 * learning algorithm for regression.
 * It supports both continuous and categorical features.
 *
 * TODO: Add maxPartitions setter
 */
@Since("1.4.0")
class OptimizedDecisionTreeRegressor @Since("1.4.0") (@Since("1.4.0") override val uid: String)
  extends Predictor[Vector, OptimizedDecisionTreeRegressor, OptimizedDecisionTreeRegressionModel]
    with OptimizedDecisionTreeRegressorParams with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("odtr"))

  // Override parameter setters from parent trait for Java API compatibility.
  /** @group setParam */
  @Since("1.4.0")
  override def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  override def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

  /**
   * Specifies how often to checkpoint the cached node IDs.
   * E.g. 10 means that the cache will get checkpointed every 10 iterations.
   * This is only used if cacheNodeIds is true and if the checkpoint directory is set in
   * [[org.apache.spark.SparkContext]].
   * Must be at least 1.
   * (default = 10)
   * @group setParam
   */
  @Since("1.4.0")
  override def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  @Since("1.4.0")
  override def setImpurity(value: String): this.type = set(impurity, value)

  /** @group setParam */
  @Since("1.6.0")
  override def setSeed(value: Long): this.type = set(seed, value)

  /** @group setParam */
  @Since("2.0.0")
  override def setMaxMemoryMultiplier(value: Double): this.type = set(maxMemoryMultiplier, value)

  /** @group setParam */
  @Since("2.0.0")
  override def setTimePredictionStrategy(value: TimePredictionStrategy) = timePredictionStrategy = value

  /** @group setParam */
  @Since("2.0.0")
  override def setMaxTasksPerBin(value: Int): this.type
  = set(maxTasksPerBin, value)

  /** @group setParam */
  @Since("2.0.0")
  override def setCustomSplits(value: Option[Array[Array[Double]]]) = customSplits = value

  /** @group setParam */
  @Since("2.0.0")
  override def setLocalTrainingAlgorithm(value: LocalTrainingAlgorithm) = localTrainingAlgorithm = value

  override protected def train(dataset: Dataset[_]): OptimizedDecisionTreeRegressionModel = instrumented { instr =>
    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset)
    val strategy = getOldStrategy(categoricalFeatures)

    instr.logParams(this, params: _*)

    val trees = OptimizedRandomForest.run(oldDataset, strategy, numTrees = 1, featureSubsetStrategy = "all",
      seed = $(seed), instr = Some(instr), parentUID = Some(uid))._1

    trees.head.asInstanceOf[OptimizedDecisionTreeRegressionModel]
  }

  private[ml] def train(data: RDD[LabeledPoint],
                        oldStrategy: OldStrategy): OptimizedDecisionTreeRegressionModel = instrumented { instr =>
    instr.logPipelineStage(this)
    instr.logDataset(data)
    instr.logParams(this, params: _*)

    val trees = OptimizedRandomForest.run(data, oldStrategy, numTrees = 1, featureSubsetStrategy = "all",
      seed = 0L, instr = Some(instr), parentUID = Some(uid))._1

    trees.head.asInstanceOf[OptimizedDecisionTreeRegressionModel]
  }

  /** (private[ml]) Create a Strategy instance to use with the old API. */
  private[ml] def getOldStrategy(categoricalFeatures: Map[Int, Int]): OldStrategy = {
    super.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, getOldImpurity,
      subsamplingRate = 1.0)
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): OptimizedDecisionTreeRegressor = defaultCopy(extra)
}

@Since("1.4.0")
object OptimizedDecisionTreeRegressor
  extends DefaultParamsReadable[OptimizedDecisionTreeRegressor] {
  /** Accessor for supported impurities: variance */
  final val supportedImpurities: Array[String] = TreeRegressorParams.supportedImpurities

  @Since("2.0.0")
  override def load(path: String): OptimizedDecisionTreeRegressor = super.load(path)
}

/**
  * <a href="http://en.wikipedia.org/wiki/Decision_tree_learning">
  * Decision tree (Wikipedia)</a> model for regression.
  * It supports both continuous and categorical features.
  * @param rootNode  Root of the decision tree
  */
@Since("1.4.0")
class OptimizedDecisionTreeRegressionModel private[ml] (
                                                override val uid: String,
                                                override val rootNode: OptimizedNode,
                                                override val numFeatures: Int)
  extends PredictionModel[Vector, OptimizedDecisionTreeRegressionModel]
    with OptimizedDecisionTreeModel with OptimizedDecisionTreeRegressorParams with MLWritable with Serializable {

  require(rootNode != null,
    "DecisionTreeRegressionModel given null rootNode, but it requires a non-null rootNode.")

  /**
    * Construct a decision tree regression model.
    * @param rootNode  Root node of tree, with other nodes attached.
    */
  private[ml] def this(rootNode: OptimizedNode, numFeatures: Int) =
    this(Identifiable.randomUID("dtr"), rootNode, numFeatures)

  override def predict(features: Vector): Double = {
    rootNode.predictImpl(features).prediction
  }

  def predict(features: OldVector): Double = {
    predict(Vectors.dense(features.toArray))
  }

  def oldPredict(vector: OldVector): Double = {
    makePredictionForOldVector(rootNode, vector)
  }

  private def makePredictionForOldVector(topNode: OptimizedNode, features: OldVector): Double = {
    topNode match {
      case node: OptimizedLeafNode =>
        node.prediction
      case node: OptimizedInternalNode =>
        val shouldGoLeft = node.split match {
          case split: ContinuousSplit =>
            features(split.featureIndex) <= split.threshold

          case split: CategoricalSplit =>
            // leftCategories will sort every time, rather use copied ml.Vector?
            split.leftCategories.contains(features(split.featureIndex))
        }

        if (shouldGoLeft) {
          makePredictionForOldVector(node.leftChild, features)
        } else {
          makePredictionForOldVector(node.rightChild, features)
        }

      case _ => throw new RuntimeException("Unexpected error in OptimizedDecisionTreeRegressionModel, unknown Node type.")
    }
  }

  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    transformImpl(dataset)
  }

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val predictUDF = udf { (features: Vector) => predict(features) }
    var output = dataset.toDF()
    if ($(predictionCol).nonEmpty) {
      output = output.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
    }
    output
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): OptimizedDecisionTreeRegressionModel = {
    copyValues(new OptimizedDecisionTreeRegressionModel(uid, rootNode, numFeatures), extra).setParent(parent)
  }

  @Since("1.4.0")
  override def toString: String = {
    s"DecisionTreeRegressionModel (uid=$uid) of depth $depth with $numNodes nodes"
  }

  /** Convert to spark.mllib DecisionTreeModel (losing some information) */
  override private[spark] def toOld: OldDecisionTreeModel = {
    new OldDecisionTreeModel(rootNode.toOld(1), OldAlgo.Regression)
  }

  @Since("2.0.0")
  override def write: MLWriter =
    new OptimizedDecisionTreeRegressionModel.DecisionTreeRegressionModelWriter(this)
}

@Since("2.0.0")
object OptimizedDecisionTreeRegressionModel extends MLReadable[OptimizedDecisionTreeRegressionModel] {

  @Since("2.0.0")
  override def read: MLReader[OptimizedDecisionTreeRegressionModel] =
    new DecisionTreeRegressionModelReader

  @Since("2.0.0")
  override def load(path: String): OptimizedDecisionTreeRegressionModel = super.load(path)

  private[OptimizedDecisionTreeRegressionModel]
  class DecisionTreeRegressionModelWriter(instance: OptimizedDecisionTreeRegressionModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      val extraMetadata: JObject = Map(
        "numFeatures" -> instance.numFeatures)
      DefaultParamsWriter.saveMetadata(instance, path, sc, Some(extraMetadata))
      val (nodeData, _) = NodeData.build(instance.rootNode, 0)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(nodeData).write.parquet(dataPath)
    }
  }

  private class DecisionTreeRegressionModelReader
    extends MLReader[OptimizedDecisionTreeRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[OptimizedDecisionTreeRegressionModel].getName

    override def load(path: String): OptimizedDecisionTreeRegressionModel = {
      implicit val format = DefaultFormats
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val root = OptimizedDecisionTreeModelReadWrite.loadTreeNodes(path, metadata, sparkSession)
      val model = new OptimizedDecisionTreeRegressionModel(metadata.uid, root, numFeatures)
      metadata.getAndSetParams(model)
      model
    }
  }

  /** Convert a model from the old API */
  private[ml] def fromOld(
                           oldModel: OldDecisionTreeModel,
                           parent: OptimizedDecisionTreeRegressor,
                           categoricalFeatures: Map[Int, Int],
                           numFeatures: Int = -1): OptimizedDecisionTreeRegressionModel = {
    require(oldModel.algo == OldAlgo.Regression,
      s"Cannot convert non-regression DecisionTreeModel (old API) to" +
        s" DecisionTreeRegressionModel (new API).  Algo is: ${oldModel.algo}")
    val rootNode = OptimizedNode.fromOld(oldModel.topNode, categoricalFeatures)
    val uid = if (parent != null) parent.uid else Identifiable.randomUID("dtr")
    new OptimizedDecisionTreeRegressionModel(uid, rootNode, numFeatures)
  }
}

