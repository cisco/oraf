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

package org.apache.spark.ml.tree.impl

import org.apache.spark.Partitioner
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification.OptimizedDecisionTreeClassificationModel
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.regression.OptimizedDecisionTreeRegressionModel
import org.apache.spark.ml.tree._
import org.apache.spark.ml.util.Instrumentation
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo, OptimizedForestStrategy => OldStrategy}
import org.apache.spark.mllib.tree.impurity.ImpurityCalculator
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.random.{SamplingUtils, XORShiftRandom}

import java.io.IOException
import scala.collection.{SeqView, mutable}
import scala.util.{Random, Try}


/**
 * ALGORITHM
 *
 * This is a sketch of the algorithm to help new developers.
 *
 * The algorithm partitions data by instances (rows).
 * On each iteration, the algorithm splits a set of nodes.  In order to choose the best split
 * for a given node, sufficient statistics are collected from the distributed data.
 * For each node, the statistics are collected to some worker node, and that worker selects
 * the best split.
 *
 * This setup requires discretization of continuous features.  This binning is done in the
 * findSplits() method during initialization, after which each continuous feature becomes
 * an ordered discretized feature with at most maxBins possible values.
 *
 * The main loop in the algorithm operates on a queue of nodes (nodeStack).  These nodes
 * lie at the periphery of the tree being trained.  If multiple trees are being trained at once,
 * then this queue contains nodes from all of them.  Each iteration works roughly as follows:
 *   On the master node:
 *     - Some number of nodes are pulled off of the queue (based on the amount of memory
 *       required for their sufficient statistics).
 *     - For random forests, if featureSubsetStrategy is not "all," then a subset of candidate
 *       features are chosen for each node.  See method selectNodesToSplit().
 *   On worker nodes, via method findBestSplits():
 *     - The worker makes one pass over its subset of instances.
 *     - For each (tree, node, feature, split) tuple, the worker collects statistics about
 *       splitting.  Note that the set of (tree, node) pairs is limited to the nodes selected
 *       from the queue for this iteration.  The set of features considered can also be limited
 *       based on featureSubsetStrategy.
 *     - For each node, the statistics for that node are aggregated to a particular worker
 *       via reduceByKey().  The designated worker chooses the best (feature, split) pair,
 *       or chooses to stop splitting if the stopping criteria are met.
 *   On the master node:
 *     - The master collects all decisions about splitting nodes and updates the model.
 *     - The updated model is passed to the workers on the next iteration.
 * This process continues until the node queue is empty.
 *
 * Most of the methods in this implementation support the statistics aggregation, which is
 * the heaviest part of the computation.  In general, this implementation is bound by either
 * the cost of statistics computation on workers or by communicating the sufficient statistics.
 */
private[spark] object OptimizedRandomForest extends Logging {

  def runWithLabeledPoints(
                            oldInput: RDD[LabeledPoint],
                            strategy: OldStrategy,
                            numTrees: Int,
                            featureSubsetStrategy: String,
                            seed: Long,
                            instr: Option[Instrumentation],
                            prune: Boolean = true,
                            parentUID: Option[String] = None,
                            computeStatistics: Boolean = false,
                            sampleWeights: Option[RDD[Double]] = None): (Array[OptimizedDecisionTreeModel], Option[TrainingStatistics]) = {
    val input = sampleWeights.map(weightsRdd => {
      oldInput.zip(weightsRdd).map {
        case (labeledPoint, weight) => Instance(labeledPoint.label, weight, labeledPoint.features)
      }
    }).getOrElse {
      oldInput.map(labeledPoint => Instance(labeledPoint.label, 1.0, labeledPoint.features))
    }

    run(input, strategy, numTrees, featureSubsetStrategy, seed, instr, prune, parentUID, computeStatistics)
  }

  /**
   * Train a random forest.
   *
   * @param input Training data: RDD of `LabeledPoint`
   * @return an unweighted set of trees
   */
  def run(
      input: RDD[Instance],
      strategy: OldStrategy,
      numTrees: Int,
      featureSubsetStrategy: String,
      seed: Long,
      instr: Option[Instrumentation],
      prune: Boolean = true,
      parentUID: Option[String] = None,
      computeStatistics: Boolean = false)
      : (Array[OptimizedDecisionTreeModel], Option[TrainingStatistics]) = {

    val timer = new TimeTracker()

    timer.start("total")

    timer.start("init")

    val retaggedInput = input.retag(classOf[Instance])
    val metadata =
      OptimizedDecisionTreeMetadata.buildMetadata(retaggedInput, strategy, numTrees, featureSubsetStrategy)
    instr match {
      case Some(instrumentation) =>
        instrumentation.logNumFeatures(metadata.numFeatures)
        instrumentation.logNumClasses(metadata.numClasses)
      case None =>
        logInfo("numFeatures: " + metadata.numFeatures)
        logInfo("numClasses: " + metadata.numClasses)
    }

    val timePredictionStrategy = strategy.getTimePredictionStrategy
    val localTrainingAlgorithm: LocalTrainingAlgorithm = strategy.getLocalTrainingAlgorithm

    // Find the splits and the corresponding bins (interval between the splits) using a sample
    // of the input data.
    timer.start("findSplits")

    val splits = strategy.customSplits.map(splits => {
      if(metadata.numFeatures != splits.length) {
        throw new IllegalArgumentException("strategy.customSplits have wrong size: metadata.numFeatures= " +
          s"${metadata.numFeatures} while customSplits.length= ${splits.length}")
      }
      splits.zipWithIndex.map { case (feature, idx) =>
        // Set metadata:
        metadata.setNumSplits(idx, feature.length)
        // Convert Array[Array[Double]] into Array[Array[Split]]
        feature.map(threshold => new ContinuousSplit(idx, threshold).asInstanceOf[Split])
      }
    }).getOrElse(findSplits(retaggedInput, metadata, seed))

    timer.stop("findSplits")
    logDebug("numBins: feature: number of bins")
    logDebug(Range(0, metadata.numFeatures).map { featureIndex =>
      s"\t$featureIndex\t${metadata.numBins(featureIndex)}"
    }.mkString("\n"))

    // Bin feature values (TreePoint representation).
    // Cache input RDD for speedup during multiple passes.
    val treeInput = OptimizedTreePoint.convertToTreeRDD(retaggedInput, splits, metadata)

    val withReplacement = numTrees > 1

    val baggedInput = BaggedPoint
      .convertToBaggedRDD(treeInput, strategy.subsamplingRate, numTrees, withReplacement, (treePoint: TreePoint) => treePoint.weight, seed = seed)
      .persist(StorageLevel.MEMORY_AND_DISK)

    val distributedMaxDepth = Math.min(strategy.maxDepth, 30)

    // Max memory usage for aggregates
    // TODO: Calculate memory usage more precisely.
    val maxMemoryUsage: Long = strategy.maxMemoryInMB * 1024L * 1024L
    logDebug("max memory usage for aggregates = " + maxMemoryUsage + " bytes.")

    /*
     * The main idea here is to perform group-wise training of the decision tree nodes thus
     * reducing the passes over the data from (# nodes) to (# nodes / maxNumberOfNodesPerGroup).
     * Each data sample is handled by a particular node (or it reaches a leaf and is not used
     * in lower levels).
     */

    // Create an RDD of node Id cache.
    // At first, all the rows belong to the root nodes (node Id == 1).
    val nodeIdCache = if (strategy.useNodeIdCache) {
      Some(OptimizedNodeIdCache.init(
        data = baggedInput,
        numTrees = numTrees,
        checkpointInterval = strategy.checkpointInterval,
        initVal = 1))
    } else {
      None
    }

    /*
     * Stack of nodes to train: (treeIndex, node)
     * The reason this is a stack is that we train many trees at once, but we want to focus on
     * completing trees, rather than training all simultaneously.  If we are splitting nodes from
     * 1 tree, then the new nodes to split will be put at the top of this stack, so we will continue
     * training the same tree in the next iteration.  This focus allows us to send fewer trees to
     * workers on each iteration; see topNodesForGroup below.
     */
    val nodeStack = new mutable.ArrayStack[(Int, OptimizedLearningNode)]
    val localTrainingStack = new mutable.ListBuffer[LocalTrainingTask]

    val rng = new Random()
    rng.setSeed(seed)

    // Allocate and queue root nodes.
    val topNodes =
      Array.fill[OptimizedLearningNode](numTrees)(OptimizedLearningNode.emptyNode(nodeIndex = 1))
    Range(0, numTrees).foreach(treeIndex => nodeStack.push((treeIndex, topNodes(treeIndex))))

    timer.stop("init")

    timer.start("distributedTraining")

    // Calculate the local training threshold (the number of data points which fit onto a single executor core).
    // Attempts to determine this value dynamically from the cluster setup.

    val numCores = baggedInput.context.getConf.getInt("spark.executor.cores", 1)

    val maxExecutorMemory = Try(
      baggedInput.sparkContext.getExecutorMemoryStatus.head match {
        case (executorId, (usedMemory, maxMemory)) => maxMemory / numCores
      }
    ).getOrElse(maxMemoryUsage)

    val nodeMemUsage = OptimizedRandomForest.aggregateSizeForNode(metadata, None) * 8L
    val featuresMem = (metadata.numFeatures + metadata.numTrees + 1) * 8L

    val localTrainingThreshold =
      ((maxExecutorMemory - nodeMemUsage) / (strategy.maxMemoryMultiplier * featuresMem)).toInt

    val trainingLimits = TrainingLimits(localTrainingThreshold, distributedMaxDepth)

    while (nodeStack.nonEmpty) {
      // Collect some nodes to split, and choose features for each node (if subsampling).
      // Each group of nodes may come from one or multiple trees, and at multiple levels.
      val (nodesForGroup, treeToNodeToIndexInfo) =
        OptimizedRandomForest.selectNodesToSplit(nodeStack, maxMemoryUsage, metadata, rng)
      // Sanity check (should never occur):
      assert(nodesForGroup.nonEmpty,
        s"OptimizedRandomForest selected empty nodesForGroup.  Error for unknown reason.")

      // Only send trees to worker if they contain nodes being split this iteration.
      val topNodesForGroup: Map[Int, OptimizedLearningNode] =
        nodesForGroup.keys.map(treeIdx => treeIdx -> topNodes(treeIdx)).toMap

      // Choose node splits, and enqueue new nodes as needed.
      timer.start("findBestSplits")
      OptimizedRandomForest.findBestSplits(baggedInput, metadata, topNodesForGroup, nodesForGroup,
        treeToNodeToIndexInfo, splits,
        (nodeStack, localTrainingStack),
        trainingLimits,
        timer, nodeIdCache)
      timer.stop("findBestSplits")
    }
    timer.stop("distributedTraining")

    timer.start("localTraining")

    val nodeStats = mutable.ListBuffer.empty[NodeStatistics]
    val numExecutors = Math.max(baggedInput.context.getExecutorMemoryStatus.size - 1, 1)

    val numPartitions = numExecutors * numCores

    /**
     * Pack smaller nodes together using first-fit decreasing bin-packing and then sort
     * the resulting bins by their predicted duration (implicitly in decreasing order).
     *
     * @return List[LocalTrainingBin] sorted in decreasing order
     */
    def scheduleLocalTrainingTasks: Seq[LocalTrainingBin] = {
      val trainingPlan = new LocalTrainingPlan(localTrainingThreshold,
        timePredictionStrategy,
        strategy.maxTasksPerBin)

      localTrainingStack.sorted.foreach(task => trainingPlan.scheduleTask(task))
      trainingPlan.bins.sorted.toList
    }

    /**
     * Group all nodes in the current batch of local training tasks by tree.
     *
     * @param batch
     * @return (treeId, nodeList) tuples for all trees in the batch
     */
    def getNodesForTrees(batch: Seq[LocalTrainingBin]): Map[Int, Seq[Int]] = {
      batch.flatMap(bin => {
        bin.tasks.map(task => (task.treeIndex, task.node.id))
      }).groupBy {
        case (treeId, _) => treeId
      }.map {
        case (treeId, nodes) => (treeId, nodes.map { case (_, nodeId) => nodeId })
      }
    }

    /**
     * Determine which node subset the input BaggedPoint belongs to for every tree, either
     * using NodeIdCache or by evaluating it in the current model.
     *
     * @return (baggedPoint, nodeIdArray)
     */
    def getNodeIdsForPoints: RDD[(BaggedPoint[OptimizedTreePoint], Array[Int])] = {
      if (nodeIdCache.nonEmpty) {
        baggedInput.zip(nodeIdCache.get.nodeIdsForInstances)
      } else {
        baggedInput.map(point =>
          (point,
            Range(0, numTrees)
              .map(treeId => topNodes(treeId).predictImpl(point.datum.binnedFeatures, splits))
              .toArray)
        )
      }
    }

    /**
     * Filter the points used in nodes in the current batch and duplicate them if they are
     * used in multiple trees.
     *
     * @return RDD((treeId, nodeId), (treePoint, sampleWeight))
     */
    def filterDataInBatch(batch: Seq[LocalTrainingBin],
                          pointsWithNodeIds: RDD[(BaggedPoint[OptimizedTreePoint], Array[Int])]) = {
      val nodeSets: Map[Int, Seq[Int]] = getNodesForTrees(batch)
      val nodeSetsBc = baggedInput.sparkContext.broadcast(nodeSets)

      pointsWithNodeIds.flatMap {
        case (baggedPoint, nodeIdsForTree) =>
          nodeSetsBc.value.keys
            .filter(treeId => baggedPoint.subsampleCounts(treeId) > 0)
            .map(treeId => (treeId, nodeIdsForTree(treeId)))
            .filter { case (treeId, nodeId) => nodeSetsBc.value(treeId).contains(nodeId) }
            .map { case (treeId, nodeId) =>
              ((treeId, nodeId), (baggedPoint.datum, baggedPoint.subsampleCounts(treeId) * baggedPoint.datum.sampleWeight))
            }
      }
    }

    /**
     * Partition the data so that each bin is processed on one executor.
     *
     * @return partitioned data
     */
    def partitionByBin(batch: Seq[LocalTrainingBin],
                       filtered: RDD[((Int, Int), (OptimizedTreePoint, Double))]) = {
      val treeNodeMapping = batch.zipWithIndex.flatMap {
        case (bin, partitionIndex) =>
          bin.tasks.map(task => ((task.treeIndex, task.node.id), partitionIndex))
      }.toMap

      filtered.partitionBy(new NodeIdPartitioner(batch.length, treeNodeMapping))
    }

    /**
     * In each partition, group points that belong to the same node and train the nodes
     * using a local training algorithm.
     *
     * @return
     */
    def runLocalTraining(partitioned: RDD[((Int, Int), (OptimizedTreePoint, Double))]) = {
      partitioned
        .mapPartitions(partition => {
          partition.toSeq
            .groupBy { case (nodeIds, _) => nodeIds }
            .values
            .map(pointsWithIndices =>
              (pointsWithIndices.head._1, pointsWithIndices.map { case (_, point) => point }))
            .map { case ((treeIndex, nodeIndex), points) =>
              trainNodeLocally(treeIndex, nodeIndex, points)
            }.toIterator
        }).collect()
    }

    /**
     * Run local training and collect statistics about the training duration and data size.
     * @return
     */
    def trainNodeLocally(treeIndex: Int, nodeIndex: Int, points: Seq[(OptimizedTreePoint, Double)]) = {
      val startTime = System.nanoTime()
      val pointArray = points.map(_._1).toArray
      val instanceWeights = points.map(_._2).toArray
      val node = OptimizedLearningNode.emptyNode(nodeIndex)

      val currentLevel = LearningNode.indexToLevel(nodeIndex)
      val localMaxDepth = metadata.maxDepth - currentLevel

      val tree = localTrainingAlgorithm.fitNode(pointArray, instanceWeights, node,
        metadata, splits, Some(localMaxDepth), prune)

      val time = (System.nanoTime() - startTime) / 1e9
      (time, tree, points.length, treeIndex, nodeIndex)
    }

    /**
     * Update the main model on driver with a locally trained subtree.
     */
    def updateModelWithSubtree(learningNode: OptimizedLearningNode, treeIndex: Int): Unit = {
      val parent = OptimizedLearningNode.getNode(
        OptimizedLearningNode.parentIndex(learningNode.id), topNodes(treeIndex))
      if (OptimizedLearningNode.isLeftChild(learningNode.id)) {
        parent.leftChild = Some(learningNode)
      } else {
        parent.rightChild = Some(learningNode)
      }
    }


    timer.start("localTrainingScheduling")
    val trainingPlan = scheduleLocalTrainingTasks
    timer.stop("localTrainingScheduling")

    val pointsWithNodeIds = getNodeIdsForPoints.cache()

    val finishedNodeMap = trainingPlan.grouped(numPartitions).flatMap(batch => {
      timer.start("localTrainingFitting")

      val filtered = filterDataInBatch(batch, pointsWithNodeIds)
      val partitioned = partitionByBin(batch, filtered)
      val finished = runLocalTraining(partitioned)

      val nodesMap = finished.map { case (time, node, rows, treeIndex, nodeIndex) =>
        if (computeStatistics) {
          nodeStats += NodeStatistics(nodeIndex, rows, node.impurity, time)
        }

        ((treeIndex, nodeIndex), node)
      }

      timer.stop("localTrainingFitting")
      nodesMap
    }).toMap

    timer.stop("localTraining")

    baggedInput.unpersist()

    timer.stop("total")


    logInfo("Internal timing for DecisionTree:")
    logInfo(s"$timer")

    // Delete any remaining checkpoints used for node Id cache.
    if (nodeIdCache.nonEmpty) {
      try {
        nodeIdCache.get.deleteAllCheckpoints()
      } catch {
        case e: IOException =>
          logWarning(s"delete all checkpoints failed. Error reason: ${e.getMessage}")
      }
    }

    val numFeatures = metadata.numFeatures

    val model: Array[OptimizedDecisionTreeModel] = parentUID match {
      case Some(uid) =>
        if (strategy.algo == OldAlgo.Classification) {
          topNodes.zipWithIndex.map { case (rootNode, treeIndex) =>
            new OptimizedDecisionTreeClassificationModel(uid,
              rootNode.toNodeWithLocalNodesMap(finishedNodeMap, treeIndex, prune),
              numFeatures, strategy.getNumClasses)
          }
        } else {
          topNodes.zipWithIndex.map { case (rootNode, treeIndex) =>
            new OptimizedDecisionTreeRegressionModel(uid,
              rootNode.toNodeWithLocalNodesMap(finishedNodeMap, treeIndex, prune),
              numFeatures)
          }
        }
      case None =>
        if (strategy.algo == OldAlgo.Classification) {
          topNodes.zipWithIndex.map { case (rootNode, treeIndex) =>
            new OptimizedDecisionTreeClassificationModel(rootNode.toNodeWithLocalNodesMap(finishedNodeMap, treeIndex, prune),
              numFeatures, strategy.getNumClasses)
          }
        } else {
          topNodes.zipWithIndex.map { case (rootNode, treeIndex) =>
            new OptimizedDecisionTreeRegressionModel(rootNode.toNodeWithLocalNodesMap(finishedNodeMap, treeIndex, prune),
              numFeatures)
          }
        }
    }

    if (computeStatistics) {
      return (model, Some(TrainingStatistics(timer, nodeStats.toList)))
    }

    (model, None)
  }

  /**
   * Helper for binSeqOp, for data which can contain a mix of ordered and unordered features.
   *
   * For ordered features, a single bin is updated.
   * For unordered features, bins correspond to subsets of categories; either the left or right bin
   * for each subset is updated.
   *
   * @param agg  Array storing aggregate calculation, with a set of sufficient statistics for
   *             each (feature, bin).
   * @param treePoint  Data point being aggregated.
   * @param splits possible splits indexed (numFeatures)(numSplits)
   * @param unorderedFeatures  Set of indices of unordered features.
   * @param instanceWeight  Weight (importance) of instance in dataset.
   */
  private def mixedBinSeqOp(
      agg: OptimizedDTStatsAggregator,
      treePoint: OptimizedTreePoint,
      splits: Array[Array[Split]],
      unorderedFeatures: Set[Int],
      instanceWeight: Double,
      featuresForNode: Option[Array[Int]]): Unit = {
    val numFeaturesPerNode = if (featuresForNode.nonEmpty) {
      // Use subsampled features
      featuresForNode.get.length
    } else {
      // Use all features
      agg.metadata.numFeatures
    }
    // Iterate over features.
    var featureIndexIdx = 0
    while (featureIndexIdx < numFeaturesPerNode) {
      val featureIndex = if (featuresForNode.nonEmpty) {
        featuresForNode.get.apply(featureIndexIdx)
      } else {
        featureIndexIdx
      }
      if (unorderedFeatures.contains(featureIndex)) {
        AggUpdateUtils.updateUnorderedFeature(agg,
          featureValue = treePoint.binnedFeatures(featureIndex), label = treePoint.label,
          featureIndex = featureIndex, featureIndexIdx = featureIndexIdx,
          featureSplits = splits(featureIndex), instanceWeight = instanceWeight)
      } else {
        AggUpdateUtils.updateOrderedFeature(agg,
          featureValue = treePoint.binnedFeatures(featureIndex), label = treePoint.label,
          featureIndexIdx = featureIndexIdx, instanceWeight = instanceWeight)
      }
      featureIndexIdx += 1
    }
  }

  /**
   * Helper for binSeqOp, for regression and for classification with only ordered features.
   *
   * For each feature, the sufficient statistics of one bin are updated.
   *
   * @param agg  Array storing aggregate calculation, with a set of sufficient statistics for
   *             each (feature, bin).
   * @param treePoint  Data point being aggregated.
   * @param instanceWeight  Weight (importance) of instance in dataset.
   */
  private def orderedBinSeqOp(
      agg: OptimizedDTStatsAggregator,
      treePoint: OptimizedTreePoint,
      instanceWeight: Double,
      featuresForNode: Option[Array[Int]]): Unit = {
    val label = treePoint.label

    // Iterate over features.
    if (featuresForNode.nonEmpty) {
      // Use subsampled features
      var featureIndexIdx = 0
      while (featureIndexIdx < featuresForNode.get.length) {
        val binIndex = treePoint.binnedFeatures(featuresForNode.get.apply(featureIndexIdx))
        agg.update(featureIndexIdx, binIndex, label, instanceWeight)
        featureIndexIdx += 1
      }
    } else {
      // Use all features
      val numFeatures = agg.metadata.numFeatures
      var featureIndex = 0
      while (featureIndex < numFeatures) {
        val binIndex = treePoint.binnedFeatures(featureIndex)
        agg.update(featureIndex, binIndex, label, instanceWeight)
        featureIndex += 1
      }
    }
  }

  /**
   * Given a group of nodes, this finds the best split for each node.
   *
   * @param input Training data: RDD of [[TreePoint]]
   * @param metadata Learning and dataset metadata
   * @param topNodesForGroup For each tree in group, tree index -> root node.
   *                         Used for matching instances with nodes.
   * @param nodesForGroup Mapping: treeIndex --> nodes to be split in tree
   * @param treeToNodeToIndexInfo Mapping: treeIndex --> nodeIndex --> nodeIndexInfo,
   *                              where nodeIndexInfo stores the index in the group and the
   *                              feature subsets (if using feature subsets).
   * @param splits possible splits for all features, indexed (numFeatures)(numSplits)
   * @param stacks  Queue of nodes to split, with values (treeIndex, node).
   *                   Updated with new non-leaf nodes which are created.
   * @param nodeIdCache Node Id cache containing an RDD of Array[Int] where
   *                    each value in the array is the data point's node Id
   *                    for a corresponding tree. This is used to prevent the need
   *                    to pass the entire tree to the executors during
   *                    the node stat aggregation phase.
   */
  private[tree] def findBestSplits(
                                    input: RDD[BaggedPoint[OptimizedTreePoint]],
                                    metadata: OptimizedDecisionTreeMetadata,
                                    topNodesForGroup: Map[Int, OptimizedLearningNode],
                                    nodesForGroup: Map[Int, Array[OptimizedLearningNode]],
                                    treeToNodeToIndexInfo: Map[Int, Map[Int, NodeIndexInfo]],
                                    splits: Array[Array[Split]],
                                    stacks: (mutable.ArrayStack[(Int, OptimizedLearningNode)],
                                             mutable.ListBuffer[LocalTrainingTask]),
                                    limits: TrainingLimits,
                                    timer: TimeTracker = new TimeTracker,
                                    nodeIdCache: Option[OptimizedNodeIdCache] = None): Unit = {

    /*
     * The high-level descriptions of the best split optimizations are noted here.
     *
     * *Group-wise training*
     * We perform bin calculations for groups of nodes to reduce the number of
     * passes over the data.  Each iteration requires more computation and storage,
     * but saves several iterations over the data.
     *
     * *Bin-wise computation*
     * We use a bin-wise best split computation strategy instead of a straightforward best split
     * computation strategy. Instead of analyzing each sample for contribution to the left/right
     * child node impurity of every split, we first categorize each feature of a sample into a
     * bin. We exploit this structure to calculate aggregates for bins and then use these aggregates
     * to calculate information gain for each split.
     *
     * *Aggregation over partitions*
     * Instead of performing a flatMap/reduceByKey operation, we exploit the fact that we know
     * the number of splits in advance. Thus, we store the aggregates (at the appropriate
     * indices) in a single array for all bins and rely upon the RDD aggregate method to
     * drastically reduce the communication overhead.
     */

    // numNodes:  Number of nodes in this group
    val numNodes = nodesForGroup.values.map(_.length).sum
    logDebug("numNodes = " + numNodes)
    logDebug("numFeatures = " + metadata.numFeatures)
    logDebug("numClasses = " + metadata.numClasses)
    logDebug("isMulticlass = " + metadata.isMulticlass)
    logDebug("isMulticlassWithCategoricalFeatures = " +
      metadata.isMulticlassWithCategoricalFeatures)
    logDebug("using nodeIdCache = " + nodeIdCache.nonEmpty.toString)

    val (nodeStack, localTrainingStack) = stacks

    /**
     * Performs a sequential aggregation over a partition for a particular tree and node.
     *
     * For each feature, the aggregate sufficient statistics are updated for the relevant
     * bins.
     *
     * @param treeIndex Index of the tree that we want to perform aggregation for.
     * @param nodeInfo The node info for the tree node.
     * @param agg Array storing aggregate calculation, with a set of sufficient statistics
     *            for each (node, feature, bin).
     * @param baggedPoint Data point being aggregated.
     */
    def nodeBinSeqOp(
        treeIndex: Int,
        nodeInfo: NodeIndexInfo,
        agg: Array[OptimizedDTStatsAggregator],
        baggedPoint: BaggedPoint[OptimizedTreePoint]): Unit = {
      if (nodeInfo != null) {
        val aggNodeIndex = nodeInfo.nodeIndexInGroup
        val featuresForNode = nodeInfo.featureSubset
        val instanceWeight = baggedPoint.subsampleCounts(treeIndex) * baggedPoint.datum.sampleWeight
        if (metadata.unorderedFeatures.isEmpty) {
          orderedBinSeqOp(agg(aggNodeIndex), baggedPoint.datum, instanceWeight, featuresForNode)
        } else {
          mixedBinSeqOp(agg(aggNodeIndex), baggedPoint.datum, splits,
            metadata.unorderedFeatures, instanceWeight, featuresForNode)
        }
        agg(aggNodeIndex).updateParent(baggedPoint.datum.label, instanceWeight)
      }
    }

    /**
     * Performs a sequential aggregation over a partition.
     *
     * Each data point contributes to one node. For each feature,
     * the aggregate sufficient statistics are updated for the relevant bins.
     *
     * @param agg  Array storing aggregate calculation, with a set of sufficient statistics for
     *             each (node, feature, bin).
     * @param baggedPoint   Data point being aggregated.
     * @return  agg
     */
    def binSeqOp(
        agg: Array[OptimizedDTStatsAggregator],
        baggedPoint: BaggedPoint[OptimizedTreePoint]): Array[OptimizedDTStatsAggregator] = {
      treeToNodeToIndexInfo.foreach { case (treeIndex, nodeIndexToInfo) =>
        val nodeIndex =
          topNodesForGroup(treeIndex).predictImpl(baggedPoint.datum.binnedFeatures, splits)
        nodeBinSeqOp(treeIndex, nodeIndexToInfo.getOrElse(nodeIndex, null), agg, baggedPoint)
      }
      agg
    }

    /**
     * Do the same thing as binSeqOp, but with nodeIdCache.
     */
    def binSeqOpWithNodeIdCache(
        agg: Array[OptimizedDTStatsAggregator],
        dataPoint: (BaggedPoint[OptimizedTreePoint], Array[Int])): Array[OptimizedDTStatsAggregator] = {
      treeToNodeToIndexInfo.foreach { case (treeIndex, nodeIndexToInfo) =>
        val baggedPoint = dataPoint._1
        val nodeIdCache = dataPoint._2
        val nodeIndex = nodeIdCache(treeIndex)
        nodeBinSeqOp(treeIndex, nodeIndexToInfo.getOrElse(nodeIndex, null), agg, baggedPoint)
      }

      agg
    }

    /**
     * Get node index in group --> features indices map,
     * which is a short cut to find feature indices for a node given node index in group.
     */
    def getNodeToFeatures(
        treeToNodeToIndexInfo: Map[Int, Map[Int, NodeIndexInfo]]): Option[Map[Int, Array[Int]]] = {
      if (!metadata.subsamplingFeatures) {
        None
      } else {
        val mutableNodeToFeatures = new mutable.HashMap[Int, Array[Int]]()
        treeToNodeToIndexInfo.values.foreach { nodeIdToNodeInfo =>
          nodeIdToNodeInfo.values.foreach { nodeIndexInfo =>
            assert(nodeIndexInfo.featureSubset.isDefined)
            mutableNodeToFeatures(nodeIndexInfo.nodeIndexInGroup) = nodeIndexInfo.featureSubset.get
          }
        }
        Some(mutableNodeToFeatures.toMap)
      }
    }

    def addTrainingTask(node: OptimizedLearningNode,
                        treeIndex: Int,
                        rows: Long,
                        nodeLevel: Int,
                        impurity: Double) = {
      if (rows < limits.localTrainingThreshold) {
        val task = new LocalTrainingTask(node, treeIndex, rows, impurity)
        localTrainingStack += task
      } else {
        nodeStack.push((treeIndex, node))
      }
    }

    // array of nodes to train indexed by node index in group
    val nodes = new Array[OptimizedLearningNode](numNodes)
    nodesForGroup.foreach { case (treeIndex, nodesForTree) =>
      nodesForTree.foreach { node =>
        nodes(treeToNodeToIndexInfo(treeIndex)(node.id).nodeIndexInGroup) = node
      }
    }

    // Calculate best splits for all nodes in the group
    timer.start("chooseSplits")

    // In each partition, iterate all instances and compute aggregate stats for each node,
    // yield a (nodeIndex, nodeAggregateStats) pair for each node.
    // After a `reduceByKey` operation,
    // stats of a node will be shuffled to a particular partition and be combined together,
    // then best splits for nodes are found there.
    // Finally, only best Splits for nodes are collected to driver to construct decision tree.
    val nodeToFeatures = getNodeToFeatures(treeToNodeToIndexInfo)
    val nodeToFeaturesBc = input.sparkContext.broadcast(nodeToFeatures)

    val partitionAggregates: RDD[(Int, OptimizedDTStatsAggregator)] = if (nodeIdCache.nonEmpty) {
      input.zip(nodeIdCache.get.nodeIdsForInstances).mapPartitions { points =>
        // Construct a nodeStatsAggregators array to hold node aggregate stats,
        // each node will have a nodeStatsAggregator
        val nodeStatsAggregators = Array.tabulate(numNodes) { nodeIndex =>
          val featuresForNode = nodeToFeaturesBc.value.map { nodeToFeatures =>
            nodeToFeatures(nodeIndex)
          }
          new OptimizedDTStatsAggregator(metadata, featuresForNode)
        }

        // iterator all instances in current partition and update aggregate stats
        points.foreach(binSeqOpWithNodeIdCache(nodeStatsAggregators, _))

        // transform nodeStatsAggregators array to (nodeIndex, nodeAggregateStats) pairs,
        // which can be combined with other partition using `reduceByKey`
        val indexSeq: SeqView[(OptimizedDTStatsAggregator, Int), Array[(OptimizedDTStatsAggregator, Int)]] = nodeStatsAggregators.view.zipWithIndex
        indexSeq.map(_.swap).iterator
      }
    } else {
      input.mapPartitions { points =>
        // Construct a nodeStatsAggregators array to hold node aggregate stats,
        // each node will have a nodeStatsAggregator
        val nodeStatsAggregators = Array.tabulate(numNodes) { nodeIndex =>
          val featuresForNode = nodeToFeaturesBc.value.flatMap { nodeToFeatures =>
            Some(nodeToFeatures(nodeIndex))
          }
          new OptimizedDTStatsAggregator(metadata, featuresForNode)
        }

        // iterator all instances in current partition and update aggregate stats
        points.foreach(binSeqOp(nodeStatsAggregators, _))

        // transform nodeStatsAggregators array to (nodeIndex, nodeAggregateStats) pairs,
        // which can be combined with other partition using `reduceByKey`
        val indexSeq: SeqView[(OptimizedDTStatsAggregator, Int), Array[(OptimizedDTStatsAggregator, Int)]] = nodeStatsAggregators.view.zipWithIndex
        indexSeq.map(_.swap).iterator
      }
    }

    // Aggregate sufficient stats by node, then find best splits
    val nodeToBestSplits = partitionAggregates.reduceByKey((a, b) => a.merge(b)).map {
      case (nodeIndex, aggStats) =>
        val featuresForNode = nodeToFeaturesBc.value.flatMap { nodeToFeatures =>
          Some(nodeToFeatures(nodeIndex))
        }

        // find best split for each node
        val (split: Split, stats: ImpurityStats) =
          OptimizedRandomForest.binsToBestSplit(aggStats, splits, featuresForNode, nodes(nodeIndex))
        (nodeIndex, (split, stats))
    }.collectAsMap()

    timer.stop("chooseSplits")

    // Perform splits
    val nodeIdUpdaters = if (nodeIdCache.nonEmpty) {
      Array.fill[mutable.Map[Int, NodeIndexUpdater]](
        metadata.numTrees)(mutable.Map[Int, NodeIndexUpdater]())
    } else {
      null
    }
    // Iterate over all nodes in this group.
    nodesForGroup.foreach { case (treeIndex, nodesForTree) =>
      nodesForTree.foreach { node =>
        val nodeIndex = node.id
        val nodeLevel = LearningNode.indexToLevel(nodeIndex)
        val nodeInfo = treeToNodeToIndexInfo(treeIndex)(nodeIndex)
        val aggNodeIndex = nodeInfo.nodeIndexInGroup
        val (split: Split, stats: ImpurityStats) =
          nodeToBestSplits(aggNodeIndex)
        logDebug("best split = " + split)

        // Extract info for this node.  Create children if not leaf.
        val isLeaf =
          (stats.gain <= 0) || (nodeLevel == limits.distributedMaxDepth)
        node.isLeaf = isLeaf
        node.stats = stats
        logDebug("Node = " + node)

        if (!isLeaf) {
          node.split = Some(split)
          val childIsLeaf = (nodeLevel + 1) == limits.distributedMaxDepth
          val leftChildIsLeaf = childIsLeaf || (stats.leftImpurity == 0.0)
          val rightChildIsLeaf = childIsLeaf || (stats.rightImpurity == 0.0)
          node.leftChild = Some(OptimizedLearningNode(
            OptimizedLearningNode.leftChildIndex(nodeIndex),
            leftChildIsLeaf, ImpurityStats.getEmptyImpurityStats(stats.leftImpurityCalculator)))
          node.rightChild = Some(OptimizedLearningNode(
            LearningNode.rightChildIndex(nodeIndex),
            rightChildIsLeaf, ImpurityStats.getEmptyImpurityStats(stats.rightImpurityCalculator)))

          if (nodeIdCache.nonEmpty) {
            val nodeIndexUpdater = NodeIndexUpdater(
              split = split,
              nodeIndex = nodeIndex)
            nodeIdUpdaters(treeIndex).put(nodeIndex, nodeIndexUpdater)
          }

          // enqueue left child and right child if they are not leaves
          if (!leftChildIsLeaf) {
            addTrainingTask(node.leftChild.get, treeIndex, stats.leftImpurityCalculator.count.toLong,
              nodeLevel, stats.leftImpurity)
          }
          if (!rightChildIsLeaf) {
            addTrainingTask(node.rightChild.get, treeIndex, stats.rightImpurityCalculator.count.toLong,
              nodeLevel, stats.rightImpurity)
          }

          logDebug("leftChildIndex = " + node.leftChild.get.id +
            ", impurity = " + stats.leftImpurity)
          logDebug("rightChildIndex = " + node.rightChild.get.id +
            ", impurity = " + stats.rightImpurity)
        }
      }
    }

    if (nodeIdCache.nonEmpty) {
      // Update the cache if needed.
      nodeIdCache.get.updateNodeIndices(input, nodeIdUpdaters, splits)
    }
  }


  /**
   * Return a list of pairs (featureIndexIdx, featureIndex) where featureIndex is the global
   * (across all trees) index of a feature and featureIndexIdx is the index of a feature within the
   * list of features for a given node. Filters out features known to be constant
   * (features with 0 splits)
   */
  private[impl] def getFeaturesWithSplits(
      metadata: OptimizedDecisionTreeMetadata,
      featuresForNode: Option[Array[Int]]): SeqView[(Int, Int), Seq[_]] = {
    Range(0, metadata.numFeaturesPerNode).view.map { featureIndexIdx =>
      featuresForNode.map(features => (featureIndexIdx, features(featureIndexIdx)))
        .getOrElse((featureIndexIdx, featureIndexIdx))
    }.withFilter { case (_, featureIndex) =>
      metadata.numSplits(featureIndex) != 0
    }
  }

  private[impl] def getBestSplitByGain(
      parentImpurityCalculator: ImpurityCalculator,
      metadata: OptimizedDecisionTreeMetadata,
      featuresForNode: Option[Array[Int]],
      splitsAndImpurityInfo: Seq[(Split, ImpurityStats)]): (Split, ImpurityStats) = {
    val (bestSplit, bestSplitStats) =
      if (splitsAndImpurityInfo.isEmpty) {
        // If no valid splits for features, then this split is invalid,
        // return invalid information gain stats.  Take any split and continue.
        // Splits is empty, so arbitrarily choose to split on any threshold
        val dummyFeatureIndex = featuresForNode.map(_.head).getOrElse(0)
        if (metadata.isContinuous(dummyFeatureIndex)) {
          (new ContinuousSplit(dummyFeatureIndex, 0),
            ImpurityStats.getInvalidImpurityStats(parentImpurityCalculator))
        } else {
          val numCategories = metadata.featureArity(dummyFeatureIndex)
          (new CategoricalSplit(dummyFeatureIndex, Array(), numCategories),
            ImpurityStats.getInvalidImpurityStats(parentImpurityCalculator))
        }
      } else {
        splitsAndImpurityInfo.maxBy(_._2.gain)
      }
    (bestSplit, bestSplitStats)
  }

  /**
   * Find the best split for a node.
   *
   * @param binAggregates Bin statistics.
   * @return tuple for best split: (Split, information gain, prediction at node)
   */
  private[tree] def binsToBestSplit(
      binAggregates: OptimizedDTStatsAggregator,
      splits: Array[Array[Split]],
      featuresForNode: Option[Array[Int]],
      node: OptimizedLearningNode): (Split, ImpurityStats) = {
    val validFeatureSplits = getFeaturesWithSplits(binAggregates.metadata, featuresForNode)
    // For each (feature, split), calculate the gain, and select the best (feature, split).
    val parentImpurityCalc = if (node.stats == null) None else Some(node.stats.impurityCalculator)
    val splitsAndImpurityInfo =
      validFeatureSplits.map { case (featureIndexIdx, featureIndex) =>
        SplitUtils.chooseSplit(binAggregates, featureIndex, featureIndexIdx, splits(featureIndex),
          parentImpurityCalc)
      }
    getBestSplitByGain(binAggregates.getParentImpurityCalculator(), binAggregates.metadata,
      featuresForNode, splitsAndImpurityInfo)
  }

  private[impl] def findUnorderedSplits(
      metadata: OptimizedDecisionTreeMetadata,
      featureIndex: Int): Array[Split] = {
    // Unordered features
    // 2^(maxFeatureValue - 1) - 1 combinations
    val featureArity = metadata.featureArity(featureIndex)
    Array.tabulate[Split](metadata.numSplits(featureIndex)) { splitIndex =>
      val categories = extractMultiClassCategories(splitIndex + 1, featureArity)
      new CategoricalSplit(featureIndex, categories.toArray, featureArity)
    }
  }

  /**
   * Returns splits for decision tree calculation.
   * Continuous and categorical features are handled differently.
   *
   * Continuous features:
   *   For each feature, there are numBins - 1 possible splits representing the possible binary
   *   decisions at each node in the tree.
   *   This finds locations (feature values) for splits using a subsample of the data.
   *
   * Categorical features:
   *   For each feature, there is 1 bin per split.
   *   Splits and bins are handled in 2 ways:
   *   (a) "unordered features"
   *       For multiclass classification with a low-arity feature
   *       (i.e., if isMulticlass && isSpaceSufficientForAllCategoricalSplits),
   *       the feature is split based on subsets of categories.
   *   (b) "ordered features"
   *       For regression and binary classification,
   *       and for multiclass classification with a high-arity feature,
   *       there is one bin per category.
   *
   * @param input Training data: RDD of [[LabeledPoint]]
   * @param metadata Learning and dataset metadata
   * @param seed random seed
   * @return Splits, an Array of [[Split]]
   *          of size (numFeatures, numSplits)
   */
  protected[tree] def findSplits(
      input: RDD[Instance],
      metadata: OptimizedDecisionTreeMetadata,
      seed: Long): Array[Array[Split]] = {

    logDebug("isMulticlass = " + metadata.isMulticlass)

    val numFeatures = metadata.numFeatures

    // Sample the input only if there are continuous features.
    val continuousFeatures = Range(0, numFeatures).filter(metadata.isContinuous)
    val sampledInput = if (continuousFeatures.nonEmpty) {
      // Calculate the number of samples for approximate quantile calculation.
      val requiredSamples = math.max(metadata.maxBins * metadata.maxBins, 10000)
      val fraction = if (requiredSamples < metadata.numExamples) {
        requiredSamples.toDouble / metadata.numExamples
      } else {
        1.0
      }
      logDebug("fraction of data used for calculating quantiles = " + fraction)
      input.sample(withReplacement = false, fraction, new XORShiftRandom(seed).nextInt())
    } else {
      input.sparkContext.emptyRDD[Instance]
    }

    findSplitsBySorting(sampledInput, metadata, continuousFeatures)
  }

  private def findSplitsBySorting(
      input: RDD[Instance],
      metadata: OptimizedDecisionTreeMetadata,
      continuousFeatures: IndexedSeq[Int]): Array[Array[Split]] = {

    val continuousSplits: scala.collection.Map[Int, Array[Split]] = {
      // reduce the parallelism for split computations when there are less
      // continuous features than input partitions. this prevents tasks from
      // being spun up that will definitely do no work.
      val numPartitions = math.min(continuousFeatures.length, input.partitions.length)

      input
        .flatMap(point => continuousFeatures.map(idx => (idx, point.features(idx))))
        .groupByKey(numPartitions)
        .map { case (idx, samples) =>
          val thresholds = findSplitsForContinuousFeature(samples, metadata, idx)
          val splits: Array[Split] = thresholds.map(thresh => new ContinuousSplit(idx, thresh))
          logDebug(s"featureIndex = $idx, numSplits = ${splits.length}")
          (idx, splits)
        }.collectAsMap()
    }

    val numFeatures = metadata.numFeatures
    val splits: Array[Array[Split]] = Array.tabulate(numFeatures) {
      case i if metadata.isContinuous(i) =>
        val split = continuousSplits(i)
        metadata.setNumSplits(i, split.length)
        split

      case i if metadata.isCategorical(i) && metadata.isUnordered(i) =>
        findUnorderedSplits(metadata, i)

      case i if metadata.isCategorical(i) =>
        // Ordered features
        //   Splits are constructed as needed during training.
        Array.empty[Split]
    }
    splits
  }

  /**
   * Nested method to extract list of eligible categories given an index. It extracts the
   * position of ones in a binary representation of the input. If binary
   * representation of an number is 01101 (13), the output list should (3.0, 2.0,
   * 0.0). The maxFeatureValue depict the number of rightmost digits that will be tested for ones.
   */
  private[tree] def extractMultiClassCategories(
      input: Int,
      maxFeatureValue: Int): List[Double] = {
    var categories = List[Double]()
    var j = 0
    var bitShiftedInput = input
    while (j < maxFeatureValue) {
      if (bitShiftedInput % 2 != 0) {
        // updating the list of categories.
        categories = j.toDouble :: categories
      }
      // Right shift by one
      bitShiftedInput = bitShiftedInput >> 1
      j += 1
    }
    categories
  }

  /**
   * Find splits for a continuous feature
   * NOTE: Returned number of splits is set based on `featureSamples` and
   *       could be different from the specified `numSplits`.
   *       The `numSplits` attribute in the `DecisionTreeMetadata` class will be set accordingly.
   *
   * @param featureSamples feature values of each sample
   * @param metadata decision tree metadata
   *                 NOTE: `metadata.numbins` will be changed accordingly
   *                       if there are not enough splits to be found
   * @param featureIndex feature index to find splits
   * @return array of split thresholds
   */
  private[tree] def findSplitsForContinuousFeature(
      featureSamples: Iterable[Double],
      metadata: OptimizedDecisionTreeMetadata,
      featureIndex: Int): Array[Double] = {
    require(metadata.isContinuous(featureIndex),
      "findSplitsForContinuousFeature can only be used to find splits for a continuous feature.")

    val splits: Array[Double] = if (featureSamples.isEmpty) {
      Array.empty[Double]
    } else {
      val numSplits = metadata.numSplits(featureIndex)

      // get count for each distinct value
      val (valueCountMap, numSamples) = featureSamples.foldLeft((Map.empty[Double, Int], 0)) {
        case ((m, cnt), x) =>
          (m + ((x, m.getOrElse(x, 0) + 1)), cnt + 1)
      }
      // sort distinct values
      val valueCounts = valueCountMap.toSeq.sortBy(_._1).toArray

      val possibleSplits = valueCounts.length - 1
      if (possibleSplits == 0) {
        // constant feature
        Array.empty[Double]
      } else if (possibleSplits <= numSplits) {
        // if possible splits is not enough or just enough, just return all possible splits
        (1 to possibleSplits)
          .map(index => (valueCounts(index - 1)._1 + valueCounts(index)._1) / 2.0)
          .toArray
      } else {
        // stride between splits
        val stride: Double = numSamples.toDouble / (numSplits + 1)
        logDebug("stride = " + stride)

        // iterate `valueCount` to find splits
        val splitsBuilder = mutable.ArrayBuilder.make[Double]
        var index = 1
        // currentCount: sum of counts of values that have been visited
        var currentCount = valueCounts(0)._2
        // targetCount: target value for `currentCount`.
        // If `currentCount` is closest value to `targetCount`,
        // then current value is a split threshold.
        // After finding a split threshold, `targetCount` is added by stride.
        var targetCount = stride
        while (index < valueCounts.length) {
          val previousCount = currentCount
          currentCount += valueCounts(index)._2
          val previousGap = math.abs(previousCount - targetCount)
          val currentGap = math.abs(currentCount - targetCount)
          // If adding count of current value to currentCount
          // makes the gap between currentCount and targetCount smaller,
          // previous value is a split threshold.
          if (previousGap < currentGap) {
            splitsBuilder += (valueCounts(index - 1)._1 + valueCounts(index)._1) / 2.0
            targetCount += stride
          }
          index += 1
        }

        splitsBuilder.result()
      }
    }
    splits
  }

  private[tree] class NodeIndexInfo(
      val nodeIndexInGroup: Int,
      val featureSubset: Option[Array[Int]]) extends Serializable

  /**
   * Pull nodes off of the queue, and collect a group of nodes to be split on this iteration.
   * This tracks the memory usage for aggregates and stops adding nodes when too much memory
   * will be needed; this allows an adaptive number of nodes since different nodes may require
   * different amounts of memory (if featureSubsetStrategy is not "all").
   *
   * @param nodeStack  Queue of nodes to split.
   * @param maxMemoryUsage  Bound on size of aggregate statistics.
   * @return  (nodesForGroup, treeToNodeToIndexInfo).
   *          nodesForGroup holds the nodes to split: treeIndex --> nodes in tree.
   *
   *          treeToNodeToIndexInfo holds indices selected features for each node:
   *            treeIndex --> (global) node index --> (node index in group, feature indices).
   *          The (global) node index is the index in the tree; the node index in group is the
   *           index in [0, numNodesInGroup) of the node in this group.
   *          The feature indices are None if not subsampling features.
   */
  private[tree] def selectNodesToSplit(
                                        nodeStack: mutable.ArrayStack[(Int, OptimizedLearningNode)],
                                        maxMemoryUsage: Long,
                                        metadata: OptimizedDecisionTreeMetadata,
                                        rng: Random)
    : (Map[Int, Array[OptimizedLearningNode]], Map[Int, Map[Int, NodeIndexInfo]]) = {
    // Collect some nodes to split:
    //  nodesForGroup(treeIndex) = nodes to split
    val mutableNodesForGroup =
      new mutable.HashMap[Int, mutable.ArrayBuffer[OptimizedLearningNode]]()
    val mutableTreeToNodeToIndexInfo =
      new mutable.HashMap[Int, mutable.HashMap[Int, NodeIndexInfo]]()
    var memUsage: Long = 0L
    var numNodesInGroup = 0
    // If maxMemoryInMB is set very small, we want to still try to split 1 node,
    // so we allow one iteration if memUsage == 0.
    var groupDone = false
    while (nodeStack.nonEmpty && !groupDone) {
      val (treeIndex, node) = nodeStack.top
      // Choose subset of features for node (if subsampling).
      val featureSubset: Option[Array[Int]] = if (metadata.subsamplingFeatures) {
        Some(SamplingUtils.reservoirSampleAndCount(Range(0,
          metadata.numFeatures).iterator, metadata.numFeaturesPerNode, rng.nextLong())._1)
      } else {
        None
      }
      // Check if enough memory remains to add this node to the group.
      val nodeMemUsage = OptimizedRandomForest.aggregateSizeForNode(metadata, featureSubset) * 8L
      if (memUsage + nodeMemUsage <= maxMemoryUsage || memUsage == 0) {
        nodeStack.pop()
        mutableNodesForGroup.getOrElseUpdate(treeIndex,
          new mutable.ArrayBuffer[OptimizedLearningNode]()) += node
        mutableTreeToNodeToIndexInfo
          .getOrElseUpdate(treeIndex, new mutable.HashMap[Int, NodeIndexInfo]())(node.id)
          = new NodeIndexInfo(numNodesInGroup, featureSubset)
        numNodesInGroup += 1
        memUsage += nodeMemUsage
      } else {
        groupDone = true
      }
    }
    if (memUsage > maxMemoryUsage) {
      // If maxMemoryUsage is 0, we should still allow splitting 1 node.
      logWarning(s"Tree learning is using approximately $memUsage bytes per iteration, which" +
        s" exceeds requested limit maxMemoryUsage=$maxMemoryUsage. This allows splitting" +
        s" $numNodesInGroup nodes in this iteration.")
    }
    // Convert mutable maps to immutable ones.
    val nodesForGroup: Map[Int, Array[OptimizedLearningNode]] =
      mutableNodesForGroup.mapValues(_.toArray).toMap
    val treeToNodeToIndexInfo = mutableTreeToNodeToIndexInfo.mapValues(_.toMap).toMap
    (nodesForGroup, treeToNodeToIndexInfo)
  }

  /**
   * Get the number of values to be stored for this node in the bin aggregates.
   *
   * @param featureSubset  Indices of features which may be split at this node.
   *                       If None, then use all features.
   */
  private def aggregateSizeForNode(
      metadata: OptimizedDecisionTreeMetadata,
      featureSubset: Option[Array[Int]]): Long = {
    val totalBins = if (featureSubset.nonEmpty) {
      featureSubset.get.map(featureIndex => metadata.numBins(featureIndex).toLong).sum
    } else {
      metadata.numBins.map(_.toLong).sum
    }
    if (metadata.isClassification) {
      metadata.numClasses * totalBins
    } else {
      3 * totalBins
    }
  }
}

private class NodeIdPartitioner(override val numPartitions: Int,
                                   val nodeIdPartitionMapping: Map[(Int, Int), Int])
  extends Partitioner {

  def getPartition(key: Any): Int = {
    val k = key.asInstanceOf[(Int, Int)]

    // orElse part should never happen
    nodeIdPartitionMapping.getOrElse(k, 1)
  }
}

private case class TrainingLimits(localTrainingThreshold: Int,
                                  distributedMaxDepth: Int)
