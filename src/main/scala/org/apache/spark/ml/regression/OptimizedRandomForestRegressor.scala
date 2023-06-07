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

import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.OptimizedRandomForest
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.ml.{PredictionModel, Predictor}
import org.apache.spark.mllib.tree.configuration.{TimePredictionStrategy, Algo => OldAlgo}
import org.apache.spark.mllib.tree.model.{RandomForestModel => OldRandomForestModel}
import org.apache.spark.mllib.linalg.{Vector => OldVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.apache.spark.sql.functions._
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._


/**
 * <a href="http://en.wikipedia.org/wiki/Random_forest">Random Forest</a>
 * learning algorithm for regression.
 * It supports both continuous and categorical features.
 *
 */
@Since("1.4.0")
class OptimizedRandomForestRegressor @Since("1.4.0") (@Since("1.4.0") override val uid: String)
  extends Predictor[Vector, OptimizedRandomForestRegressor, OptimizedRandomForestRegressionModel]
    with OptimizedRandomForestRegressorParams with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("orfr"))

  // Override parameter setters from parent trait for Java API compatibility.

  // Parameters from TreeRegressorParams:

  /** @group setParam */
  @Since("1.4.0")
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMaxBins(value: Int): this.type = set(maxBins, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  /** @group setParam */
  @Since("1.4.0")
  def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  /** @group expertSetParam */
  @Since("1.4.0")
  def setCacheNodeIds(value: Boolean): this.type = set(cacheNodeIds, value)

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
  def setCheckpointInterval(value: Int): this.type = set(checkpointInterval, value)

  /** @group setParam */
  @Since("1.4.0")
  def setImpurity(value: String): this.type = set(impurity, value)

  // Parameters from TreeEnsembleParams:

  /** @group setParam */
  @Since("1.4.0")
  def setSubsamplingRate(value: Double): this.type = set(subsamplingRate, value)

  /** @group setParam */
  @Since("1.4.0")
  def setSeed(value: Long): this.type = set(seed, value)

  // Parameters from RandomForestParams:

  /** @group setParam */
  @Since("1.4.0")
  def setNumTrees(value: Int): this.type = set(numTrees, value)

  /** @group setParam */
  @Since("1.4.0")
  def setFeatureSubsetStrategy(value: String): this.type =
    set(featureSubsetStrategy, value)

  /** @group setParam */
  @Since("2.0.0")
  override def setMaxMemoryMultiplier(value: Double): this.type = set(maxMemoryMultiplier, value)

  /** @group setParam */
  @Since("2.0.0")
  override def setTimePredictionStrategy(value: TimePredictionStrategy): this.type= {
    timePredictionStrategy = value
    this
  }

  /** @group setParam */
  @Since("2.0.0")
  override def setMaxTasksPerBin(value: Int): this.type
  = set(maxTasksPerBin, value)

  /** @group setParam */
  @Since("2.0.0")
  override def setCustomSplits(value: Option[Array[Array[Double]]]): this.type = {
    customSplits = value
    this
  }

  /** @group setParam */
  @Since("2.0.0")
  override def setLocalTrainingAlgorithm(value: LocalTrainingAlgorithm): this.type = {
    localTrainingAlgorithm = value
    this
  }

  override protected def train(dataset: Dataset[_]): OptimizedRandomForestRegressionModel = instrumented { instr =>
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val oldDataset: RDD[Instance] = dataset.select(col($(labelCol)), col($(featuresCol)), col($(weightCol))).rdd.map {
      case Row(label: Double, features: Vector, weight: Double) => Instance(label, weight, features)
    }

    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, getOldImpurity)

    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    instr.logParams(this, labelCol, featuresCol, predictionCol, impurity, numTrees,
      featureSubsetStrategy, maxDepth, maxBins, maxMemoryInMB, minInfoGain,
      minInstancesPerNode, seed, subsamplingRate, cacheNodeIds, checkpointInterval)

    val trees = OptimizedRandomForest
      .run(oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))._1
      .map(_.asInstanceOf[OptimizedDecisionTreeRegressionModel])

    val numFeatures = oldDataset.first().features.size
    instr.logNamedValue(Instrumentation.loggerTags.numFeatures, numFeatures)
    new OptimizedRandomForestRegressionModel(uid, trees, numFeatures)
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): OptimizedRandomForestRegressor = defaultCopy(extra)
}

@Since("1.4.0")
object OptimizedRandomForestRegressor extends DefaultParamsReadable[OptimizedRandomForestRegressor]{
  /** Accessor for supported impurity settings: variance */
  @Since("1.4.0")
  final val supportedImpurities: Array[String] = Array("variance")

  /** Accessor for supported featureSubsetStrategy settings: auto, all, onethird, sqrt, log2 */
  @Since("1.4.0")
  final val supportedFeatureSubsetStrategies: Array[String] =
  TreeEnsembleParams.supportedFeatureSubsetStrategies

  @Since("2.0.0")
  override def load(path: String): OptimizedRandomForestRegressor = super.load(path)
}

/**
  * <a href="http://en.wikipedia.org/wiki/Random_forest">Random Forest</a> model for regression.
  * It supports both continuous and categorical features.
  *
  * @param _trees  Decision trees in the ensemble.
  * @param numFeatures  Number of features used by this model
  */
@Since("1.4.0")
class OptimizedRandomForestRegressionModel(
                                           override val uid: String,
                                           private val _trees: Array[OptimizedDecisionTreeRegressionModel],
                                           override val numFeatures: Int)
  extends PredictionModel[Vector, OptimizedRandomForestRegressionModel]
    with OptimizedRandomForestRegressorParams with OptimizedTreeEnsembleModel[OptimizedDecisionTreeRegressionModel]
    with MLWritable with Serializable {

  require(_trees.nonEmpty, "RandomForestRegressionModel requires at least 1 tree.")

  /**
    * Construct a random forest regression model, with all trees weighted equally.
    *
    * @param trees  Component trees
    */
  private[ml] def this(trees: Array[OptimizedDecisionTreeRegressionModel], numFeatures: Int) =
    this(Identifiable.randomUID("rfr"), trees, numFeatures)

  @Since("1.4.0")
  override def trees: Array[OptimizedDecisionTreeRegressionModel] = _trees

  // Note: We may add support for weights (based on tree performance) later on.
  private lazy val _treeWeights: Array[Double] = Array.fill[Double](_trees.length)(1.0)

  @Since("1.4.0")
  override def treeWeights: Array[Double] = _treeWeights

  override protected def transformImpl(dataset: Dataset[_]): DataFrame = {
    val bcastModel = dataset.sparkSession.sparkContext.broadcast(this)
    val predictUDF = udf { (features: Any) =>
      bcastModel.value.predict(features.asInstanceOf[Vector])
    }
    dataset.withColumn($(predictionCol), predictUDF(col($(featuresCol))))
  }

  override def predict(features: Vector): Double = {
    // TODO: When we add a generic Bagging class, handle transform there.  SPARK-7128
    // Predict average of tree predictions.
    // Ignore the weights since all are 1.0 for now.
    _trees.map(_.rootNode.predictImpl(features).prediction).sum / getNumTrees
  }

  def predict(features: OldVector): Double = {
    predict(Vectors.dense(features.toArray))
  }

  def oldPredict(vector: OldVector): Double = {
    _trees.map(_.oldPredict(vector)).sum / getNumTrees
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): OptimizedRandomForestRegressionModel = {
    copyValues(new OptimizedRandomForestRegressionModel(uid, _trees, numFeatures), extra).setParent(parent)
  }

  @Since("1.4.0")
  override def toString: String = {
    s"RandomForestRegressionModel (uid=$uid) with $getNumTrees trees"
  }

  /** (private[ml]) Convert to a model in the old API */
  private[ml] def toOld: OldRandomForestModel = {
    new OldRandomForestModel(OldAlgo.Regression, _trees.map(_.toOld))
  }

  @Since("2.0.0")
  override def write: MLWriter =
    new OptimizedRandomForestRegressionModelSerializer(this)
}

private
class OptimizedRandomForestRegressionModelWriter(instance: OptimizedRandomForestRegressionModel)
  extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    val extraMetadata: JObject = Map(
      "numFeatures" -> instance.numFeatures,
      "numTrees" -> instance.getNumTrees)
    OptimizedEnsembleModelReadWrite.saveImpl(instance, path, sparkSession, extraMetadata)
  }
}

private
class OptimizedRandomForestRegressionModelSerializer(instance: OptimizedRandomForestRegressionModel)
  extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    OptimizedEnsembleModelSerialization.saveImpl[OptimizedRandomForestRegressionModel](instance, path, sparkSession)
  }
}

@Since("2.0.0")
object OptimizedRandomForestRegressionModel extends MLReadable[OptimizedRandomForestRegressionModel] {

  @Since("2.0.0")
  override def read: MLReader[OptimizedRandomForestRegressionModel] = new RandomForestRegressionModelDeserializer

  @Since("2.0.0")
  override def load(path: String): OptimizedRandomForestRegressionModel = super.load(path)


  private class RandomForestRegressionModelReader extends MLReader[OptimizedRandomForestRegressionModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[OptimizedRandomForestRegressionModel].getName
    private val treeClassName = classOf[OptimizedDecisionTreeRegressionModel].getName

    override def load(path: String): OptimizedRandomForestRegressionModel = {
      implicit val format = DefaultFormats
      val (metadata: Metadata, treesData: Array[(Metadata, OptimizedNode)], treeWeights: Array[Double]) =
        OptimizedEnsembleModelReadWrite.loadImpl(path, sparkSession, className, treeClassName)
      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val numTrees = (metadata.metadata \ "numTrees").extract[Int]

      val trees: Array[OptimizedDecisionTreeRegressionModel] = treesData.map { case (treeMetadata, root) =>
        val tree =
          new OptimizedDecisionTreeRegressionModel(treeMetadata.uid, root, numFeatures)
        treeMetadata.getAndSetParams(tree)
        tree
      }
      require(numTrees == trees.length, s"RandomForestRegressionModel.load expected $numTrees" +
        s" trees based on metadata but found ${trees.length} trees.")

      val model = new OptimizedRandomForestRegressionModel(metadata.uid, trees, numFeatures)
      metadata.getAndSetParams(model)
      model
    }
  }

  private class RandomForestRegressionModelDeserializer extends MLReader[OptimizedRandomForestRegressionModel] {

    override def load(path: String): OptimizedRandomForestRegressionModel = {
      OptimizedEnsembleModelSerialization.loadImpl[OptimizedRandomForestRegressionModel](path, sparkSession)
    }
  }

  /** Convert a model from the old API */
  def fromOld(
                           oldModel: OldRandomForestModel,
                           parent: OptimizedRandomForestRegressor,
                           categoricalFeatures: Map[Int, Int],
                           numFeatures: Int = -1): OptimizedRandomForestRegressionModel = {
    require(oldModel.algo == OldAlgo.Regression, "Cannot convert RandomForestModel" +
      s" with algo=${oldModel.algo} (old API) to RandomForestRegressionModel (new API).")
    val newTrees = oldModel.trees.map { tree =>
      // parent for each tree is null since there is no good way to set this.
      OptimizedDecisionTreeRegressionModel.fromOld(tree, null, categoricalFeatures)
    }
    val uid = if (parent != null) parent.uid else Identifiable.randomUID("rfr")
    new OptimizedRandomForestRegressionModel(uid, newTrees, numFeatures)
  }
}
