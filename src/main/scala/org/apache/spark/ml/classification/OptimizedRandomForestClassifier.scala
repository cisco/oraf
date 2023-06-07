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

package org.apache.spark.ml.classification

import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree.OptimizedEnsembleModelSerialization
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.OptimizedRandomForest
import org.apache.spark.ml.util.DefaultParamsReader.Metadata
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.mllib.tree.configuration.{TimePredictionStrategy, Algo => OldAlgo}
import org.apache.spark.mllib.tree.model.{RandomForestModel => OldRandomForestModel}
import org.apache.spark.mllib.linalg.{Vector => OldVector}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Dataset, Row}
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.apache.spark.sql.functions._


/**
 * <a href="http://en.wikipedia.org/wiki/Random_forest">Random Forest</a> learning algorithm for
 * classification.
 * It supports both binary and multiclass labels, as well as both continuous and categorical
 * features.
 */
@Since("1.4.0")
class OptimizedRandomForestClassifier @Since("1.4.0") (
                                                        @Since("1.4.0") override val uid: String)
  extends ProbabilisticClassifier[Vector, OptimizedRandomForestClassifier,
    OptimizedRandomForestClassificationModel]
    with OptimizedRandomForestClassifierParams with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("orfc"))

  // Override parameter setters from parent trait for Java API compatibility.

  // Parameters from TreeClassifierParams:

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

  override protected def train(dataset: Dataset[_]): OptimizedRandomForestClassificationModel = instrumented { instr =>
    instr.logPipelineStage(this)
    instr.logDataset(dataset)
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
    val numClasses: Int = getNumClasses(dataset)

    if (isDefined(thresholds)) {
      require($(thresholds).length == numClasses, this.getClass.getSimpleName +
        ".train() called with non-matching numClasses and thresholds.length." +
        s" numClasses=$numClasses, but thresholds has length ${$(thresholds).length}")
    }

    require(numClasses > 0, s"Classifier (in extractLabeledPoints) found numClasses =" +
      s" $numClasses, but requires numClasses > 0.")
    //    val oldDataset: RDD[Instance] = extractLabeledPoints(dataset, numClasses)
    val oldDataset: RDD[Instance] = dataset.select(col($(labelCol)), col($(featuresCol)), col($(weightCol))).rdd.map {
      case Row(label: Double, features: Vector, weight: Double) =>
        require(label % 1 == 0 && label >= 0 && label < numClasses, s"Classifier was given" +
          s" dataset with invalid label $label.  Labels must be integers in range" +
          s" [0, $numClasses).")
        Instance(label, weight, features)
    }
    val strategy =
      super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity)

    instr.logParams(this, labelCol, featuresCol, predictionCol, probabilityCol, rawPredictionCol,
      impurity, numTrees, featureSubsetStrategy, maxDepth, maxBins, maxMemoryInMB, minInfoGain,
      minInstancesPerNode, seed, subsamplingRate, thresholds, cacheNodeIds, checkpointInterval)

    val trees = OptimizedRandomForest
      .run(oldDataset, strategy, getNumTrees, getFeatureSubsetStrategy, getSeed, Some(instr))._1
      .map(_.asInstanceOf[OptimizedDecisionTreeClassificationModel])

    val numFeatures = oldDataset.first().features.size
    instr.logNumClasses(numClasses)
    instr.logNumFeatures(numFeatures)
    new OptimizedRandomForestClassificationModel(uid, trees, numFeatures, numClasses)
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): OptimizedRandomForestClassifier = defaultCopy(extra)
}

@Since("1.4.0")
object OptimizedRandomForestClassifier
  extends DefaultParamsReadable[OptimizedRandomForestClassifier] {
  /** Accessor for supported impurity settings: entropy, gini */
  @Since("1.4.0")
  final val supportedImpurities: Array[String] = TreeClassifierParams.supportedImpurities

  /** Accessor for supported featureSubsetStrategy settings: auto, all, onethird, sqrt, log2 */
  @Since("1.4.0")
  final val supportedFeatureSubsetStrategies: Array[String] =
  TreeEnsembleParams.supportedFeatureSubsetStrategies

  @Since("2.0.0")
  override def load(path: String): OptimizedRandomForestClassifier = super.load(path)
}

/**
  * <a href="http://en.wikipedia.org/wiki/Random_forest">Random Forest</a> model for classification.
  * It supports both binary and multiclass labels, as well as both continuous and categorical
  * features.
  *
  * @param _trees  Decision trees in the ensemble.
  *                Warning: These have null parents.
  */
@Since("1.4.0")
class OptimizedRandomForestClassificationModel(
                                               @Since("1.5.0") override val uid: String,
                                               private val _trees: Array[OptimizedDecisionTreeClassificationModel],
                                               @Since("1.6.0") override val numFeatures: Int,
                                               @Since("1.5.0") override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, OptimizedRandomForestClassificationModel]
    with OptimizedRandomForestClassifierParams with OptimizedTreeEnsembleModel[OptimizedDecisionTreeClassificationModel]
    with MLWritable with Serializable {

  require(_trees.nonEmpty, "RandomForestClassificationModel requires at least 1 tree.")

  /**
    * Construct a random forest classification model, with all trees weighted equally.
    *
    * @param trees  Component trees
    */
  def this(
                        trees: Array[OptimizedDecisionTreeClassificationModel],
                        numFeatures: Int,
                        numClasses: Int) =
    this(Identifiable.randomUID("rfc"), trees, numFeatures, numClasses)

  @Since("1.4.0")
  override def trees: Array[OptimizedDecisionTreeClassificationModel] = _trees

  // Note: We may add support for weights (based on tree performance) later on.
  private lazy val _treeWeights: Array[Double] = Array.fill[Double](_trees.length)(1.0)

  @Since("1.4.0")
  override def treeWeights: Array[Double] = _treeWeights

  override def predictRaw(features: Vector): Vector = {
    // TODO: When we add a generic Bagging class, handle transform there: SPARK-7128
    // Classifies using majority votes.
    // Ignore the tree weights since all are 1.0 for now.
    val votes = Array.fill[Double](numClasses)(0.0)
    _trees.view.foreach { tree =>
      votes(tree.rootNode.predict(features).toInt) += 1
    }
    Vectors.dense(votes)
  }

  def predict(vector: OldVector): Double = {
    predict(Vectors.dense(vector.toArray))
  }

  def oldPredict(vector: OldVector): Double = {
    val predictions = _trees.map(_.oldPredict(vector))
    // Find most prevalent value
    predictions.groupBy(identity).maxBy(_._2.length)._1
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in RandomForestClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): OptimizedRandomForestClassificationModel = {
    copyValues(new OptimizedRandomForestClassificationModel(uid, _trees, numFeatures, numClasses), extra)
      .setParent(parent)
  }

  @Since("1.4.0")
  override def toString: String = {
    s"RandomForestClassificationModel (uid=$uid) with $getNumTrees trees"
  }

  /** (private[ml]) Convert to a model in the old API */
  private[ml] def toOld: OldRandomForestModel = {
    new OldRandomForestModel(OldAlgo.Classification, _trees.map(_.toOld))
  }

  @Since("2.0.0")
  override def write: MLWriter =
    new OptimizedRandomForestClassificationModelSerializer(this)
}

private
class OptimizedRandomForestClassificationModelWriter(instance: OptimizedRandomForestClassificationModel)
  extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    // Note: numTrees is not currently used, but could be nice to store for fast querying.
    val extraMetadata: JObject = Map(
      "numFeatures" -> instance.numFeatures,
      "numClasses" -> instance.numClasses,
      "numTrees" -> instance.getNumTrees)
    OptimizedEnsembleModelReadWrite.saveImpl(instance, path, sparkSession, extraMetadata)
  }
}

private
class OptimizedRandomForestClassificationModelSerializer(instance: OptimizedRandomForestClassificationModel)
  extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    OptimizedEnsembleModelSerialization.saveImpl[OptimizedRandomForestClassificationModel](instance, path, sparkSession)
  }
}

@Since("2.0.0")
object OptimizedRandomForestClassificationModel extends MLReadable[OptimizedRandomForestClassificationModel] {

  @Since("2.0.0")
  override def read: MLReader[OptimizedRandomForestClassificationModel] =
    new RandomForestClassificationModelDeserializer

  @Since("2.0.0")
  override def load(path: String): OptimizedRandomForestClassificationModel = super.load(path)


  private class RandomForestClassificationModelReader
    extends MLReader[OptimizedRandomForestClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[OptimizedRandomForestClassificationModel].getName
    private val treeClassName = classOf[OptimizedDecisionTreeClassificationModel].getName

    override def load(path: String): OptimizedRandomForestClassificationModel = {
      implicit val format = DefaultFormats
      val (metadata: Metadata, treesData: Array[(Metadata, OptimizedNode)], _) =
        OptimizedEnsembleModelReadWrite.loadImpl(path, sparkSession, className, treeClassName)
      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val numTrees = (metadata.metadata \ "numTrees").extract[Int]

      val trees: Array[OptimizedDecisionTreeClassificationModel] = treesData.map {
        case (treeMetadata, root) =>
          val tree =
            new OptimizedDecisionTreeClassificationModel(treeMetadata.uid, root, numFeatures, numClasses)
          treeMetadata.getAndSetParams(tree)
          tree
      }
      require(numTrees == trees.length, s"RandomForestClassificationModel.load expected $numTrees" +
        s" trees based on metadata but found ${trees.length} trees.")

      val model = new OptimizedRandomForestClassificationModel(metadata.uid, trees, numFeatures, numClasses)
      metadata.getAndSetParams(model)
      model
    }
  }

  private class RandomForestClassificationModelDeserializer
    extends MLReader[OptimizedRandomForestClassificationModel] {

    override def load(path: String): OptimizedRandomForestClassificationModel = {
      OptimizedEnsembleModelSerialization.loadImpl[OptimizedRandomForestClassificationModel](path, sparkSession)
    }
  }

  /** Convert a model from the old API */
  private[ml] def fromOld(
                           oldModel: OldRandomForestModel,
                           parent: OptimizedRandomForestClassifier,
                           categoricalFeatures: Map[Int, Int],
                           numClasses: Int,
                           numFeatures: Int = -1): OptimizedRandomForestClassificationModel = {
    require(oldModel.algo == OldAlgo.Classification, "Cannot convert RandomForestModel" +
      s" with algo=${oldModel.algo} (old API) to RandomForestClassificationModel (new API).")
    val newTrees = oldModel.trees.map { tree =>
      // parent for each tree is null since there is no good way to set this.
      OptimizedDecisionTreeClassificationModel.fromOld(tree, null, categoricalFeatures)
    }
    val uid = if (parent != null) parent.uid else Identifiable.randomUID("rfc")
    new OptimizedRandomForestClassificationModel(uid, newTrees, numFeatures, numClasses)
  }
}
