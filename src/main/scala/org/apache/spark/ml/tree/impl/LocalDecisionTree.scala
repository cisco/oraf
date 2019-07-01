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

import scala.util.Random

import org.apache.spark.ml.tree._
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.util.random.SamplingUtils


/** Object exposing methods for local training of decision trees */
class LocalDecisionTree extends LocalTrainingAlgorithm {

  /**
   * Fully splits the passed-in node on the provided local dataset, returning the finalized Internal / Leaf Node
   * with its fully trained descendants.
   *
   * @param node LearningNode to use as the root of the subtree fit on the passed-in dataset
   * @param metadata learning and dataset metadata for DecisionTree
   * @param splits splits(i) = array of splits for feature i
   */
  def fitNode(
                           input: Array[OptimizedTreePoint],
                           instanceWeights: Array[Double],
                           node: OptimizedLearningNode,
                           metadata: OptimizedDecisionTreeMetadata,
                           splits: Array[Array[Split]],
                           maxDepthOverride: Option[Int] = None,
                           prune: Boolean = true): OptimizedNode = {

    // The case with 1 node (depth = 0) is handled separately.
    // This allows all iterations in the depth > 0 case to use the same code.
    // TODO: Check that learning works when maxDepth > 0 but learning stops at 1 node (because of
    //       other parameters).
    val maxDepth = maxDepthOverride.getOrElse(metadata.maxDepth)

    if (maxDepth == 0) {
      return node.toNode
    }

    val labels = input.map(_.label)

    // Prepare column store.
    //   Note: rowToColumnStoreDense checks to make sure numRows < Int.MaxValue.
    val colStoreInit: Array[Array[Int]] = LocalDecisionTreeUtils
      .rowToColumnStoreDense(input.map(_.binnedFeatures))

    // Fit a decision tree model on the dataset
    val learningNode = trainDecisionTree(node, colStoreInit, instanceWeights, labels,
      metadata, splits, maxDepth)

    // Create the finalized InternalNode and prune the tree
    learningNode.toNode(prune)
  }

  /**
   * Locally fits a decision tree model.
   *
   * @param rootNode Node to use as root of the tree fit on the passed-in dataset
   * @param colStoreInit Array of columns of training data
   * @param instanceWeights Array of weights for each training example
   * @param metadata learning and dataset metadata for DecisionTree
   * @param splits splits(i) = Array of possible splits for feature i
   * @return rootNode with its completely trained subtree
   */
  private[ml] def trainDecisionTree(
                                  rootNode: OptimizedLearningNode,
                                  colStoreInit: Array[Array[Int]],
                                  instanceWeights: Array[Double],
                                  labels: Array[Double],
                                  metadata: OptimizedDecisionTreeMetadata,
                                  splits: Array[Array[Split]],
                                  maxDepth: Int): OptimizedLearningNode = {

    // Sort each column by decision tree node.
    val colStore: Array[FeatureColumn] = colStoreInit.zipWithIndex.map { case (col, featureIndex) =>
      val featureArity: Int = metadata.featureArity.getOrElse(featureIndex, 0)
      FeatureColumn(featureIndex, col)
    }

    val numRows = colStore.headOption match {
      case None => 0
      case Some(column) => column.values.length
    }

    // Create a new TrainingInfo describing the status of our partially-trained subtree
    // at each iteration of training
    var trainingInfo: TrainingInfo = TrainingInfo(colStore,
      nodeOffsets = Array[(Int, Int)]((0, numRows)), currentLevelActiveNodes = Array(rootNode))

    // Iteratively learn, one level of the tree at a time.
    // Note: We do not use node IDs.
    var currentLevel = 0
    var doneLearning = false
    val rng = new Random()

    while (currentLevel < maxDepth && !doneLearning) {
      // Splits each active node if possible, returning an array of new active nodes
      val nextLevelNodes: Array[OptimizedLearningNode] =
        computeBestSplits(trainingInfo, instanceWeights, labels, metadata, splits, rng)
      // Count number of non-leaf nodes in the next level
      val estimatedRemainingActive = nextLevelNodes.count(!_.isLeaf)
      // TODO: Check to make sure we split something, and stop otherwise.
      doneLearning = currentLevel + 1 >= maxDepth || estimatedRemainingActive == 0
      if (!doneLearning) {
        // Obtain a new trainingInfo instance describing our current training status
        trainingInfo = trainingInfo.update(splits, nextLevelNodes)
      }
      currentLevel += 1
    }

    // Done with learning
    rootNode
  }

  /**
   * Iterate over feature values and labels for a specific (node, feature), updating stats
   * aggregator for the current node.
   */ private[impl] def updateAggregator( statsAggregator: OptimizedDTStatsAggregator, col: FeatureColumn, indices: Array[Int], instanceWeights: Array[Double],
      labels: Array[Double],
      from: Int,
      to: Int,
      featureIndexIdx: Int,
      featureSplits: Array[Split]): Unit = {
    val metadata = statsAggregator.metadata
    if (metadata.isUnordered(col.featureIndex)) {
      from.until(to).foreach { idx =>
        val rowIndex = indices(idx)
        AggUpdateUtils.updateUnorderedFeature(statsAggregator, col.values(idx), labels(rowIndex),
          featureIndex = col.featureIndex, featureIndexIdx, featureSplits,
          instanceWeight = instanceWeights(rowIndex))
      }
    } else {
      from.until(to).foreach { idx =>
        val rowIndex = indices(idx)
        AggUpdateUtils.updateOrderedFeature(statsAggregator, col.values(idx), labels(rowIndex),
          featureIndexIdx, instanceWeight = instanceWeights(rowIndex))
      }
    }
  }

  /**
   * Find the best splits for all active nodes
   *
   * @param trainingInfo Contains node offset info for current set of active nodes
   * @return  Array of new active nodes formed by splitting the current set of active nodes.
   */
  private def computeBestSplits(
      trainingInfo: TrainingInfo,
      instanceWeights: Array[Double],
      labels: Array[Double],
      metadata: OptimizedDecisionTreeMetadata,
      splits: Array[Array[Split]],
      rng: Random): Array[OptimizedLearningNode] = {
    // For each node, select the best split across all features
    trainingInfo match {
      case TrainingInfo(columns: Array[FeatureColumn], nodeOffsets: Array[(Int, Int)],
        currentLevelActiveNodes: Array[OptimizedLearningNode], _) => {
        // Filter out leaf nodes from the previous iteration
        val activeNonLeafs = currentLevelActiveNodes.zipWithIndex.filterNot(_._1.isLeaf)
        // Iterate over the active nodes in the current level.
        activeNonLeafs.flatMap { case (node: OptimizedLearningNode, nodeIndex: Int) =>
          // Features for the current node start at fromOffset and end at toOffset
          val (from, to) = nodeOffsets(nodeIndex)
          // Get impurityCalculator containing label stats for all data points at the current node
          val parentImpurityCalc = ImpurityUtils.getParentImpurityCalculator(metadata,
            trainingInfo.indices, from, to, instanceWeights, labels)

          // Randomly select a subset of features
          val featureSubset = if (metadata.subsamplingFeatures) {
            Some(SamplingUtils.reservoirSampleAndCount(Range(0, metadata.numFeatures).iterator,
              metadata.numFeaturesPerNode, rng.nextLong())._1)
          } else {
            None
          }

          val validFeatureSplits = OptimizedRandomForest.getFeaturesWithSplits(metadata,
            featuresForNode = featureSubset)
          // Find the best split for each feature for the current node
          val splitsAndImpurityInfo = validFeatureSplits.map { case (_, featureIndex) =>
            val col = columns(featureIndex)
            // Create a DTStatsAggregator to hold label statistics for each bin of the current
            // feature & compute said label statistics
            val statsAggregator = new OptimizedDTStatsAggregator(metadata, Some(Array(featureIndex)))
            updateAggregator(statsAggregator, col, trainingInfo.indices, instanceWeights,
              labels, from, to, featureIndexIdx = 0, splits(col.featureIndex))
            // Choose best split for current feature based on label statistics
            SplitUtils.chooseSplit(statsAggregator, featureIndex, featureIndexIdx = 0,
              splits(featureIndex), Some(parentImpurityCalc))
          }
          // Find the best split overall (across all features) for the current node
          val (bestSplit, bestStats) = OptimizedRandomForest.getBestSplitByGain(parentImpurityCalc,
            metadata, featuresForNode = None, splitsAndImpurityInfo)
          // Split current node, get an iterator over its children
          splitIfPossible(node, metadata, bestStats, bestSplit)
        }
      }
    }
  }

  /**
   * Splits the passed-in node if permitted by the parameters of the learning algorithm,
   * returning an iterator over its children. Returns an empty array if node could not be split.
   *
   * @param metadata learning and dataset metadata for DecisionTree
   * @param stats Label impurity stats associated with the current node
   */
  private[impl] def splitIfPossible(
                                     node: OptimizedLearningNode,
                                     metadata: OptimizedDecisionTreeMetadata,
                                     stats: ImpurityStats,
                                     split: Split): Iterator[OptimizedLearningNode] = {
    if (stats.valid) {
      // Split node and return an iterator over its children; we filter out leaf nodes later
      doSplit(node, split, stats)
      Iterator(node.leftChild.get, node.rightChild.get)
    } else {
      node.stats = stats
      node.isLeaf = true
      Iterator()
    }
  }

  /**
   * Splits the passed-in node. This method returns nothing, but modifies the passed-in node
   * by updating its split and stats members.
   *
   * @param split Split to associate with the passed-in node
   * @param stats Label impurity statistics to associate with the passed-in node
   */
  private[impl] def doSplit(
                             node: OptimizedLearningNode,
                             split: Split,
                             stats: ImpurityStats): Unit = {
    val leftChildIsLeaf = stats.leftImpurity == 0
    node.leftChild = Some(OptimizedLearningNode(id = OptimizedLearningNode.leftChildIndex(node.id),
      isLeaf = leftChildIsLeaf,
      ImpurityStats.getEmptyImpurityStats(stats.leftImpurityCalculator)))
    val rightChildIsLeaf = stats.rightImpurity == 0
    node.rightChild = Some(OptimizedLearningNode(
      id = OptimizedLearningNode.rightChildIndex(node.id),
      isLeaf = rightChildIsLeaf,
      ImpurityStats.getEmptyImpurityStats(stats.rightImpurityCalculator)
    ))
    node.split = Some(split)
    node.isLeaf = false
    node.stats = stats
  }

}
