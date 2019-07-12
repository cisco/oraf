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

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.linalg.{DenseVector, SparseVector, Vector, Vectors}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tree.OptimizedDecisionTreeModelReadWrite.NodeData
import org.apache.spark.ml.tree._
import org.apache.spark.ml.tree.impl.OptimizedRandomForest
import org.apache.spark.ml.util.Instrumentation.instrumented
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.{Vector => OldVector}
import org.apache.spark.mllib.tree.configuration.{TimePredictionStrategy, Algo => OldAlgo, OptimizedForestStrategy => OldStrategy}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel => OldDecisionTreeModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.Dataset
import org.json4s.JsonDSL._
import org.json4s.{DefaultFormats, JObject}


/**
 *  Decision tree learning algorithm (http://en.wikipedia.org/wiki/Decision_tree_learning)
 *  for classification.
 *  It supports both binary and multiclass labels, as well as both continuous and categorical
 *  features.
 */
@Since("1.4.0")
class OptimizedDecisionTreeClassifier @Since("1.4.0") (
                                               @Since("1.4.0") override val uid: String)
  extends ProbabilisticClassifier[Vector, OptimizedDecisionTreeClassifier,
    OptimizedDecisionTreeClassificationModel]
    with OptimizedDecisionTreeClassifierParams with DefaultParamsWritable {

  @Since("1.4.0")
  def this() = this(Identifiable.randomUID("odtc"))

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
  override def setTimePredictionStrategy(value: TimePredictionStrategy): this.type = {
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

  override protected def train(dataset: Dataset[_]): OptimizedDecisionTreeClassificationModel = instrumented { instr =>
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

    val oldDataset: RDD[LabeledPoint] = extractLabeledPoints(dataset, numClasses)
    val strategy =
      getOldStrategy(categoricalFeatures, numClasses)

    instr.logParams(this, params: _*)

    val trees = OptimizedRandomForest.run(oldDataset, strategy, numTrees = 1, featureSubsetStrategy = "all",
      seed = $(seed), instr = Some(instr), parentUID = Some(uid))._1

    trees.head.asInstanceOf[OptimizedDecisionTreeClassificationModel]
  }

  private[ml] def train(data: RDD[LabeledPoint],
      oldStrategy: OldStrategy): OptimizedDecisionTreeClassificationModel = instrumented { instr =>
    instr.logPipelineStage(this)
    instr.logDataset(data)
    instr.logParams(this, params: _*)

    val trees = OptimizedRandomForest.run(data, oldStrategy, numTrees = 1, featureSubsetStrategy = "all",
      seed = 0L, instr = Some(instr), parentUID = Some(uid))._1

    trees.head.asInstanceOf[OptimizedDecisionTreeClassificationModel]
  }

  /** (private[ml]) Create a Strategy instance to use with the old API. */
  private[ml] def getOldStrategy(
                                  categoricalFeatures: Map[Int, Int],
                                  numClasses: Int): OldStrategy = {
    super.getOldStrategy(categoricalFeatures, numClasses, OldAlgo.Classification, getOldImpurity,
      subsamplingRate = 1.0)
  }

  @Since("1.4.1")
  override def copy(extra: ParamMap): OptimizedDecisionTreeClassifier = defaultCopy(extra)
}

@Since("1.4.0")
object OptimizedDecisionTreeClassifier
  extends DefaultParamsReadable[OptimizedDecisionTreeClassifier] {
  /** Accessor for supported impurities: entropy, gini */
  @Since("1.4.0")
  final val supportedImpurities: Array[String] = TreeClassifierParams.supportedImpurities

  @Since("2.0.0")
  override def load(path: String): OptimizedDecisionTreeClassifier = super.load(path)
}

/**
  * Decision tree model (http://en.wikipedia.org/wiki/Decision_tree_learning) for classification.
  * It supports both binary and multiclass labels, as well as both continuous and categorical
  * features.
  */
@Since("1.4.0")
class OptimizedDecisionTreeClassificationModel(
                                               @Since("1.4.0")override val uid: String,
                                               @Since("1.4.0")override val rootNode: OptimizedNode,
                                               @Since("1.6.0")override val numFeatures: Int,
                                               @Since("1.5.0")override val numClasses: Int)
  extends ProbabilisticClassificationModel[Vector, OptimizedDecisionTreeClassificationModel]
    with OptimizedDecisionTreeModel with OptimizedDecisionTreeClassifierParams with MLWritable with Serializable {

  require(rootNode != null,
    "DecisionTreeClassificationModel given null rootNode, but it requires a non-null rootNode.")

  /**
    * Construct a decision tree classification model.
    * @param rootNode  Root node of tree, with other nodes attached.
    */
  def this(rootNode: OptimizedNode, numFeatures: Int, numClasses: Int) =
    this(Identifiable.randomUID("dtc"), rootNode, numFeatures, numClasses)

  override def predict(features: Vector): Double = {
    rootNode.predict(features)
  }

  def predict(features: OldVector): Double = {
    rootNode.predict(Vectors.dense(features.toArray))
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
            // leftCategories will sort every time, rather use
            split.leftCategories.contains(features(split.featureIndex))
        }

        if (shouldGoLeft) {
          makePredictionForOldVector(node.leftChild, features)
        } else {
          makePredictionForOldVector(node.rightChild, features)
        }

      case _ => throw new RuntimeException("Unexpected error in OptimizedDecisionTreeClassificationModel, unknown Node type.")
    }
  }

  // TODO: Make sure this is correct
  override protected def predictRaw(features: Vector): Vector = {
    val predictions = Array.fill[Double](numClasses)(0.0)
    predictions(rootNode.predict(features).toInt) = 1.0

    Vectors.dense(predictions)
  }

  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector = {
    rawPrediction match {
      case dv: DenseVector =>
        ProbabilisticClassificationModel.normalizeToProbabilitiesInPlace(dv)
        dv
      case sv: SparseVector =>
        throw new RuntimeException("Unexpected error in DecisionTreeClassificationModel:" +
          " raw2probabilityInPlace encountered SparseVector")
    }
  }

  @Since("1.4.0")
  override def copy(extra: ParamMap): OptimizedDecisionTreeClassificationModel = {
    copyValues(new OptimizedDecisionTreeClassificationModel(uid, rootNode, numFeatures, numClasses), extra)
      .setParent(parent)
  }

  @Since("1.4.0")
  override def toString: String = {
    s"DecisionTreeClassificationModel (uid=$uid) of depth $depth with $numNodes nodes"
  }

  /** Convert to spark.mllib DecisionTreeModel (losing some information) */
  override private[spark] def toOld: OldDecisionTreeModel = {
    new OldDecisionTreeModel(rootNode.toOld(1), OldAlgo.Classification)
  }

  @Since("2.0.0")
  override def write: MLWriter =
    new OptimizedDecisionTreeClassificationModel.DecisionTreeClassificationModelSerializer(this)
}

@Since("2.0.0")
object OptimizedDecisionTreeClassificationModel extends MLReadable[OptimizedDecisionTreeClassificationModel] {

  @Since("2.0.0")
  override def read: MLReader[OptimizedDecisionTreeClassificationModel] =
    new DecisionTreeClassificationModelDeserializer

  @Since("2.0.0")
  override def load(path: String): OptimizedDecisionTreeClassificationModel = super.load(path)

  private[OptimizedDecisionTreeClassificationModel]
  class DecisionTreeClassificationModelWriter(instance: OptimizedDecisionTreeClassificationModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      val extraMetadata: JObject = Map(
        "numFeatures" -> instance.numFeatures,
        "numClasses" -> instance.numClasses)
      DefaultParamsWriter.saveMetadata(instance, path, sc, Some(extraMetadata))
      val (nodeData, _) = NodeData.build(instance.rootNode, 0)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(nodeData).write.parquet(dataPath)
    }
  }

  private[OptimizedDecisionTreeClassificationModel]
  class DecisionTreeClassificationModelSerializer(instance: OptimizedDecisionTreeClassificationModel)
    extends MLWriter {

    override protected def saveImpl(path: String): Unit = {
      OptimizedEnsembleModelSerialization.saveImpl[OptimizedDecisionTreeClassificationModel](instance, path, sparkSession)
    }
  }

  private class DecisionTreeClassificationModelReader
    extends MLReader[OptimizedDecisionTreeClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[OptimizedDecisionTreeClassificationModel].getName

    override def load(path: String): OptimizedDecisionTreeClassificationModel = {
      implicit val format = DefaultFormats
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val numFeatures = (metadata.metadata \ "numFeatures").extract[Int]
      val numClasses = (metadata.metadata \ "numClasses").extract[Int]
      val root = OptimizedDecisionTreeModelReadWrite.loadTreeNodes(path, metadata, sparkSession)
      val model = new OptimizedDecisionTreeClassificationModel(metadata.uid, root, numFeatures, numClasses)
      metadata.getAndSetParams(model)
      model
    }
  }

  private class DecisionTreeClassificationModelDeserializer
    extends MLReader[OptimizedDecisionTreeClassificationModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[OptimizedDecisionTreeClassificationModel].getName

    override def load(path: String): OptimizedDecisionTreeClassificationModel = {
      OptimizedEnsembleModelSerialization.loadImpl[OptimizedDecisionTreeClassificationModel](path, sparkSession)
    }
  }

  /** Convert a model from the old API */
  private[ml] def fromOld(
                           oldModel: OldDecisionTreeModel,
                           parent: OptimizedDecisionTreeClassifier,
                           categoricalFeatures: Map[Int, Int],
                           numFeatures: Int = -1): OptimizedDecisionTreeClassificationModel = {
    require(oldModel.algo == OldAlgo.Classification,
      s"Cannot convert non-classification DecisionTreeModel (old API) to" +
        s" DecisionTreeClassificationModel (new API).  Algo is: ${oldModel.algo}")
    val rootNode = OptimizedNode.fromOld(oldModel.topNode, categoricalFeatures)
    val uid = if (parent != null) parent.uid else Identifiable.randomUID("dtc")
    // Can't infer number of features from old model, so default to -1
    new OptimizedDecisionTreeClassificationModel(uid, rootNode, numFeatures, -1)
  }
}
