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

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.tree._
import org.apache.spark.ml.util.TestingUtils._
import org.apache.spark.mllib.tree.configuration.{QuantileStrategy, Algo => OldAlgo, OptimizedForestStrategy => OldStrategy}
import org.apache.spark.mllib.tree.impurity.{Entropy, Gini, Variance}
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.util.collection.OpenHashMap

import scala.collection.JavaConverters._
import scala.collection.mutable

/**
  * Test suite for [[RandomForest]].
  */
class OptimizedRandomForestSuite extends SparkFunSuite with MLlibTestSparkContext {

  import OptimizedRandomForestSuite.mapToVec

  private val seed = 42


  /////////////////////////////////////////////////////////////////////////////
  // Tests for split calculation
  /////////////////////////////////////////////////////////////////////////////

  test("Binary classification with continuous features: split calculation") {
    val arr = OptimizedRandomForestSuite.generateOrderedInstancesWithLabel1()
    assert(arr.length === 1000)
    val rdd = sc.parallelize(arr)
    val strategy = new OldStrategy(OldAlgo.Classification, Gini, 3, 2, 100)
    val metadata = OptimizedDecisionTreeMetadata.buildMetadata(rdd, strategy)
    assert(!metadata.isUnordered(featureIndex = 0))
    val splits = OptimizedRandomForest.findSplits(rdd, metadata, seed = 42)
    assert(splits.length === 2)
    assert(splits(0).length === 99)
  }

  test("Binary classification with binary (ordered) categorical features: split calculation") {
    val arr = OptimizedRandomForestSuite.generateCategoricalInstances()
    assert(arr.length === 1000)
    val rdd = sc.parallelize(arr)
    val strategy = new OldStrategy(OldAlgo.Classification, Gini, maxDepth = 2, numClasses = 2,
      maxBins = 100, categoricalFeaturesInfo = Map(0 -> 2, 1 -> 2))

    val metadata = OptimizedDecisionTreeMetadata.buildMetadata(rdd, strategy)
    val splits = OptimizedRandomForest.findSplits(rdd, metadata, seed = 42)
    assert(!metadata.isUnordered(featureIndex = 0))
    assert(!metadata.isUnordered(featureIndex = 1))
    assert(splits.length === 2)
    // no splits pre-computed for ordered categorical features
    assert(splits(0).length === 0)
  }

  test("Binary classification with 3-ary (ordered) categorical features," +
    " with no samples for one category: split calculation") {
    val arr = OptimizedRandomForestSuite.generateCategoricalInstances()
    assert(arr.length === 1000)
    val rdd = sc.parallelize(arr)
    val strategy = new OldStrategy(OldAlgo.Classification, Gini, maxDepth = 2, numClasses = 2,
      maxBins = 100, categoricalFeaturesInfo = Map(0 -> 3, 1 -> 3))

    val metadata = OptimizedDecisionTreeMetadata.buildMetadata(rdd, strategy)
    assert(!metadata.isUnordered(featureIndex = 0))
    assert(!metadata.isUnordered(featureIndex = 1))
    val splits = OptimizedRandomForest.findSplits(rdd, metadata, seed = 42)
    assert(splits.length === 2)
    // no splits pre-computed for ordered categorical features
    assert(splits(0).length === 0)
  }

  test("find splits for a continuous feature") {
    // find splits for normal case
    {
      val fakeMetadata = new OptimizedDecisionTreeMetadata(1, 200000, 200000, 0, 0,
        Map(), Set(),
        Array(6), Gini, QuantileStrategy.Sort,
        0, 0, 0.0, 0.0, 0, 0
      )
      val featureSamples = Array.fill(10000)(math.random).filter(_ != 0.0)
      val splits = OptimizedRandomForest.findSplitsForContinuousFeature(featureSamples, fakeMetadata, 0)
      assert(splits.length === 5)
      assert(fakeMetadata.numSplits(0) === 5)
      assert(fakeMetadata.numBins(0) === 6)
      // check returned splits are distinct
      assert(splits.distinct.length === splits.length)
    }

    // SPARK-16957: Use midpoints for split values.
    {
      val fakeMetadata = new OptimizedDecisionTreeMetadata(1, 8, 8, 0, 0,
        Map(), Set(),
        Array(3), Gini, QuantileStrategy.Sort,
        0, 0, 0.0, 0.0, 0, 0
      )

      // TODO: Why doesn't this work after filtering 0.0?
      // possibleSplits <= numSplits
      {
        val featureSamples = Array(0, 1, 0, 0, 1, 0, 1, 1).map(_.toDouble)
        val splits = OptimizedRandomForest.findSplitsForContinuousFeature(featureSamples, fakeMetadata, 0)
        val expectedSplits = Array((0.0 + 1.0) / 2)
        assert(splits === expectedSplits)
      }

      // possibleSplits > numSplits
      {
        val featureSamples = Array(0, 0, 1, 1, 2, 2, 3, 3).map(_.toDouble)
        val splits = OptimizedRandomForest.findSplitsForContinuousFeature(featureSamples, fakeMetadata, 0)
        val expectedSplits = Array((0.0 + 1.0) / 2, (2.0 + 3.0) / 2)
        assert(splits === expectedSplits)
      }
    }

    // find splits should not return identical splits
    // when there are not enough split candidates, reduce the number of splits in metadata
    {
      val fakeMetadata = new OptimizedDecisionTreeMetadata(1, 12, 12, 0, 0,
        Map(), Set(),
        Array(5), Gini, QuantileStrategy.Sort,
        0, 0, 0.0, 0.0, 0, 0
      )
      val featureSamples = Array(1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3).map(_.toDouble)
      val splits = OptimizedRandomForest.findSplitsForContinuousFeature(featureSamples, fakeMetadata, 0)
      val expectedSplits = Array((1.0 + 2.0) / 2, (2.0 + 3.0) / 2)
      assert(splits === expectedSplits)
      // check returned splits are distinct
      assert(splits.distinct.length === splits.length)
    }

    // find splits when most samples close to the minimum
    {
      val fakeMetadata = new OptimizedDecisionTreeMetadata(1, 18, 18, 0, 0,
        Map(), Set(),
        Array(3), Gini, QuantileStrategy.Sort,
        0, 0, 0.0, 0.0, 0, 0
      )
      val featureSamples = Array(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 4, 5)
        .map(_.toDouble)
      val splits = OptimizedRandomForest.findSplitsForContinuousFeature(featureSamples, fakeMetadata, 0)
      val expectedSplits = Array((2.0 + 3.0) / 2, (3.0 + 4.0) / 2)
      assert(splits === expectedSplits)
    }

    // find splits when most samples close to the maximum
    {
      val fakeMetadata = new OptimizedDecisionTreeMetadata(1, 17, 17, 0, 0,
        Map(), Set(),
        Array(2), Gini, QuantileStrategy.Sort,
        0, 0, 0.0, 0.0, 0, 0
      )
      val featureSamples = Array(0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2)
        .map(_.toDouble).filter(_ != 0.0)
      val splits = OptimizedRandomForest.findSplitsForContinuousFeature(featureSamples, fakeMetadata, 0)
      val expectedSplits = Array((1.0 + 2.0) / 2)
      assert(splits === expectedSplits)
    }

    // find splits for constant feature
    {
      val fakeMetadata = new OptimizedDecisionTreeMetadata(1, 3, 3, 0, 0,
        Map(), Set(),
        Array(3), Gini, QuantileStrategy.Sort,
        0, 0, 0.0, 0.0, 0, 0
      )
      val featureSamples = Array(0, 0, 0).map(_.toDouble).filter(_ != 0.0)
      val featureSamplesEmpty = Array.empty[Double]
      val splits = OptimizedRandomForest.findSplitsForContinuousFeature(featureSamples, fakeMetadata, 0)
      assert(splits === Array.empty[Double])
      val splitsEmpty =
        OptimizedRandomForest.findSplitsForContinuousFeature(featureSamplesEmpty, fakeMetadata, 0)
      assert(splitsEmpty === Array.empty[Double])
    }
  }

  test("train with empty arrays") {
    val lp = Instance(1.0, 1.0, Vectors.dense(Array.empty[Double]))
    val data = Array.fill(5)(lp)
    val rdd = sc.parallelize(data)

    val strategy = new OldStrategy(OldAlgo.Regression, Gini, maxDepth = 2,
      maxBins = 5)
    withClue("DecisionTree requires number of features > 0," +
      " but was given an empty features vector") {
      intercept[IllegalArgumentException] {
        OptimizedRandomForest.run(rdd, strategy, 1, "all", 42L, instr = None)._1
      }
    }
  }

  test("train with constant features") {
    val lp = Instance(1.0, 1.0, Vectors.dense(0.0, 0.0, 0.0))
    val data = Array.fill(5)(lp)
    val rdd = sc.parallelize(data)
    val strategy = new OldStrategy(
      OldAlgo.Classification,
      Gini,
      maxDepth = 2,
      numClasses = 2,
      maxBins = 5,
      categoricalFeaturesInfo = Map(0 -> 1, 1 -> 5))
    val Array(tree) = OptimizedRandomForest.run(rdd, strategy, 1, "all", 42L, instr = None)._1
    assert(tree.rootNode.impurity === -1.0)
    assert(tree.depth === 0)
    assert(tree.rootNode.prediction === lp.label)

    // Test with no categorical features
    val strategy2 = new OldStrategy(
      OldAlgo.Regression,
      Variance,
      maxDepth = 2,
      maxBins = 5)
    val Array(tree2) = OptimizedRandomForest.run(rdd, strategy2, 1, "all", 42L, instr = None)._1
    assert(tree2.rootNode.impurity === -1.0)
    assert(tree2.depth === 0)
    assert(tree2.rootNode.prediction === lp.label)
  }

  test("Multiclass classification with unordered categorical features: split calculations") {
    val arr = OptimizedRandomForestSuite.generateCategoricalInstances()
    assert(arr.length === 1000)
    val rdd = sc.parallelize(arr)
    val strategy = new OldStrategy(
      OldAlgo.Classification,
      Gini,
      maxDepth = 2,
      numClasses = 100,
      maxBins = 100,
      categoricalFeaturesInfo = Map(0 -> 3, 1 -> 3))

    val metadata = OptimizedDecisionTreeMetadata.buildMetadata(rdd, strategy)
    assert(metadata.isUnordered(featureIndex = 0))
    assert(metadata.isUnordered(featureIndex = 1))
    val splits = OptimizedRandomForest.findSplits(rdd, metadata, seed = 42)
    assert(splits.length === 2)
    assert(splits(0).length === 3)
    assert(metadata.numSplits(0) === 3)
    assert(metadata.numBins(0) === 3)
    assert(metadata.numSplits(1) === 3)
    assert(metadata.numBins(1) === 3)

    // Expecting 2^2 - 1 = 3 splits per feature
    def checkCategoricalSplit(s: Split, featureIndex: Int, leftCategories: Array[Double]): Unit = {
      assert(s.featureIndex === featureIndex)
      assert(s.isInstanceOf[CategoricalSplit])
      val s0 = s.asInstanceOf[CategoricalSplit]
      assert(s0.leftCategories === leftCategories)
      assert(s0.numCategories === 3) // for this unit test
    }
    // Feature 0
    checkCategoricalSplit(splits(0)(0), 0, Array(0.0))
    checkCategoricalSplit(splits(0)(1), 0, Array(1.0))
    checkCategoricalSplit(splits(0)(2), 0, Array(0.0, 1.0))
    // Feature 1
    checkCategoricalSplit(splits(1)(0), 1, Array(0.0))
    checkCategoricalSplit(splits(1)(1), 1, Array(1.0))
    checkCategoricalSplit(splits(1)(2), 1, Array(0.0, 1.0))
  }

  test("Multiclass classification with ordered categorical features: split calculations") {
    val arr = OptimizedRandomForestSuite.generateCategoricalInstancesForMulticlassForOrderedFeatures()
    assert(arr.length === 3000)
    val rdd = sc.parallelize(arr)
    val strategy = new OldStrategy(OldAlgo.Classification, Gini, maxDepth = 2, numClasses = 100,
      maxBins = 100, categoricalFeaturesInfo = Map(0 -> 10, 1 -> 10))
    // 2^(10-1) - 1 > 100, so categorical features will be ordered

    val metadata = OptimizedDecisionTreeMetadata.buildMetadata(rdd, strategy)
    assert(!metadata.isUnordered(featureIndex = 0))
    assert(!metadata.isUnordered(featureIndex = 1))
    val splits = OptimizedRandomForest.findSplits(rdd, metadata, seed = 42)
    assert(splits.length === 2)
    // no splits pre-computed for ordered categorical features
    assert(splits(0).length === 0)
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of other algorithm internals
  /////////////////////////////////////////////////////////////////////////////

  test("extract categories from a number for multiclass classification") {
    val l = OptimizedRandomForest.extractMultiClassCategories(13, 10)
    assert(l.length === 3)
    assert(Seq(3.0, 2.0, 0.0) === l)
  }

  test("Avoid aggregation on the last level") {
    val arr = Array(
      Instance(0.0, 1.0, Vectors.dense(1.0, 0.0, 0.0)),
      Instance(1.0, 1.0, Vectors.dense(0.0, 1.0, 1.0)),
      Instance(0.0, 1.0, Vectors.dense(2.0, 0.0, 0.0)),
      Instance(1.0, 1.0, Vectors.dense(0.0, 2.0, 1.0)))
    val input = sc.parallelize(arr)

    val strategy = new OldStrategy(algo = OldAlgo.Classification, impurity = Gini, maxDepth = 1,
      numClasses = 2, categoricalFeaturesInfo = Map(0 -> 3))
    val metadata = OptimizedDecisionTreeMetadata.buildMetadata(input, strategy)
    val splits = OptimizedRandomForest.findSplits(input, metadata, seed = 42)

    val treeInput = OptimizedTreePoint.convertToTreeRDD(input, splits, metadata)
    val baggedInput = BaggedPoint.convertToBaggedRDD(treeInput, 1.0, 1, withReplacement = false)

    val topNode = OptimizedLearningNode.emptyNode(nodeIndex = 1)
    assert(topNode.isLeaf === false)
    assert(topNode.stats === null)

    val nodesForGroup = Map(0 -> Array(topNode))
    val treeToNodeToIndexInfo = Map(0 -> Map(
      topNode.id -> new OptimizedRandomForest.NodeIndexInfo(0, None)
    ))
    val nodeStack = new mutable.ArrayStack[(Int, OptimizedLearningNode)]
    val localTrainingStack = new mutable.ListBuffer[LocalTrainingTask]
    val maxMemoryUsage = 100 * 1024L * 1024L
    val maxMemoryMultiplier = 4.0
    OptimizedRandomForest.findBestSplits(baggedInput, metadata, Map(0 -> topNode),
      nodesForGroup, treeToNodeToIndexInfo, splits, (nodeStack, localTrainingStack), TrainingLimits(100000, metadata.maxDepth))

    // don't enqueue leaf nodes into node queue
    assert(nodeStack.isEmpty)

    // set impurity and predict for topNode
    assert(topNode.stats !== null)
    assert(topNode.stats.impurity > 0.0)

    // set impurity and predict for child nodes
    assert(topNode.leftChild.get.toNode.prediction === 0.0)
    assert(topNode.rightChild.get.toNode.prediction === 1.0)
    assert(topNode.leftChild.get.stats.impurity === 0.0)
    assert(topNode.rightChild.get.stats.impurity === 0.0)
  }

  test("Avoid aggregation if impurity is 0.0") {
    val arr = Array(
      Instance(0.0, 1.0, Vectors.dense(1.0, 0.0, 0.0)),
      Instance(1.0, 1.0, Vectors.dense(0.0, 1.0, 1.0)),
      Instance(0.0, 1.0, Vectors.dense(2.0, 0.0, 0.0)),
      Instance(1.0, 1.0, Vectors.dense(0.0, 2.0, 1.0)))
    val input = sc.parallelize(arr)

    val strategy = new OldStrategy(algo = OldAlgo.Classification, impurity = Gini, maxDepth = 5,
      numClasses = 2, categoricalFeaturesInfo = Map(0 -> 3))
    val metadata = OptimizedDecisionTreeMetadata.buildMetadata(input, strategy)
    val splits = OptimizedRandomForest.findSplits(input, metadata, seed = 42)

    val treeInput = OptimizedTreePoint.convertToTreeRDD(input, splits, metadata)
    val baggedInput = BaggedPoint.convertToBaggedRDD(treeInput, 1.0, 1, withReplacement = false)

    val topNode = OptimizedLearningNode.emptyNode(nodeIndex = 1)
    assert(topNode.isLeaf === false)
    assert(topNode.stats === null)

    val nodesForGroup = Map(0 -> Array(topNode))
    val treeToNodeToIndexInfo = Map(0 -> Map(
      topNode.id -> new OptimizedRandomForest.NodeIndexInfo(0, None)
    ))
    val nodeStack = new mutable.ArrayStack[(Int, OptimizedLearningNode)]
    val localTrainingStack = new mutable.ListBuffer[LocalTrainingTask]
    val maxMemoryUsage = 100 * 1024L * 1024L
    val maxMemoryMultiplier = 4.0
    OptimizedRandomForest.findBestSplits(baggedInput, metadata, Map(0 -> topNode),
      nodesForGroup, treeToNodeToIndexInfo, splits, (nodeStack, localTrainingStack), TrainingLimits(100000, metadata.maxDepth))

    // don't enqueue a node into node queue if its impurity is 0.0
    assert(nodeStack.isEmpty)

    // set impurity and predict for topNode
    assert(topNode.stats !== null)
    assert(topNode.stats.impurity > 0.0)

    // set impurity and predict for child nodes
    assert(topNode.leftChild.get.toNode.prediction === 0.0)
    assert(topNode.rightChild.get.toNode.prediction === 1.0)
    assert(topNode.leftChild.get.stats.impurity === 0.0)
    assert(topNode.rightChild.get.stats.impurity === 0.0)
  }

  test("Use soft prediction for binary classification with ordered categorical features") {
    // The following dataset is set up such that the best split is {1} vs. {0, 2}.
    // If the hard prediction is used to order the categories, then {0} vs. {1, 2} is chosen.
    val arr = Array(
      Instance(0.0, 1.0, Vectors.dense(0.0)),
      Instance(0.0, 1.0, Vectors.dense(0.0)),
      Instance(0.0, 1.0, Vectors.dense(0.0)),
      Instance(1.0, 1.0, Vectors.dense(0.0)),
      Instance(0.0, 1.0, Vectors.dense(1.0)),
      Instance(0.0, 1.0, Vectors.dense(1.0)),
      Instance(0.0, 1.0, Vectors.dense(1.0)),
      Instance(0.0, 1.0, Vectors.dense(1.0)),
      Instance(0.0, 1.0, Vectors.dense(2.0)),
      Instance(0.0, 1.0, Vectors.dense(2.0)),
      Instance(0.0, 1.0, Vectors.dense(2.0)),
      Instance(1.0, 1.0, Vectors.dense(2.0)))
    val input = sc.parallelize(arr)

    // Must set maxBins s.t. the feature will be treated as an ordered categorical feature.
    val strategy = new OldStrategy(algo = OldAlgo.Classification, impurity = Gini, maxDepth = 1,
      numClasses = 2, categoricalFeaturesInfo = Map(0 -> 3), maxBins = 3)

    val model = OptimizedRandomForest.run(input, strategy, numTrees = 1, featureSubsetStrategy = "all",
      seed = 42, instr = None, prune = false)._1.head

    model.rootNode match {
      case n: OptimizedInternalNode => n.split match {
        case s: CategoricalSplit =>
          assert(s.leftCategories === Array(1.0))
        case _ => fail("model.rootNode.split was not a CategoricalSplit")
      }
      case _ => fail("model.rootNode was not an InternalNode")
    }
  }

  test("Second level node building with vs. without groups") {
    val arr = OptimizedRandomForestSuite.generateOrderedInstances()
    assert(arr.length === 1000)
    val rdd = sc.parallelize(arr)
    // For tree with 1 group
    val strategy1 =
      new OldStrategy(OldAlgo.Classification, Entropy, 3, 2, 100, maxMemoryInMB = 1000)
    // For tree with multiple groups
    val strategy2 =
      new OldStrategy(OldAlgo.Classification, Entropy, 3, 2, 100, maxMemoryInMB = 0)

    val tree1 = OptimizedRandomForest.run(rdd, strategy1, numTrees = 1, featureSubsetStrategy = "all",
      seed = 42, instr = None)._1.head
    val tree2 = OptimizedRandomForest.run(rdd, strategy2, numTrees = 1, featureSubsetStrategy = "all",
      seed = 42, instr = None)._1.head

    def getChildren(rootNode: OptimizedNode): Array[OptimizedInternalNode] = rootNode match {
      case n: OptimizedInternalNode =>
        assert(n.leftChild.isInstanceOf[OptimizedInternalNode])
        assert(n.rightChild.isInstanceOf[OptimizedInternalNode])
        Array(n.leftChild.asInstanceOf[OptimizedInternalNode], n.rightChild.asInstanceOf[OptimizedInternalNode])
      case _ => fail("rootNode was not an InternalNode")
    }

    // Single group second level tree construction.
    val children1 = getChildren(tree1.rootNode)
    val children2 = getChildren(tree2.rootNode)

    // Verify whether the splits obtained using single group and multiple group level
    // construction strategies are the same.
    for (i <- 0 until 2) {
      assert(children1(i).gain > 0)
      assert(children2(i).gain > 0)
      assert(children1(i).split === children2(i).split)
      assert(children1(i).impurity === children2(i).impurity)
      assert(children1(i).leftChild.impurity === children2(i).leftChild.impurity)
      assert(children1(i).rightChild.impurity === children2(i).rightChild.impurity)
      assert(children1(i).prediction === children2(i).prediction)
    }
  }

  def binaryClassificationTestWithContinuousFeaturesAndSubsampledFeatures(strategy: OldStrategy) {
    val numFeatures = 50
    val arr = OptimizedRandomForestSuite.generateOrderedInstances(numFeatures, 1000)
    val rdd = sc.parallelize(arr)

    // Select feature subset for top nodes.  Return true if OK.
    def checkFeatureSubsetStrategy(
                                    numTrees: Int,
                                    featureSubsetStrategy: String,
                                    numFeaturesPerNode: Int): Unit = {
      val seeds = Array(123, 5354, 230, 349867, 23987)
      val maxMemoryUsage: Long = 128 * 1024L * 1024L
      val metadata =
        OptimizedDecisionTreeMetadata.buildMetadata(rdd, strategy, numTrees, featureSubsetStrategy)
      seeds.foreach { seed =>
        val failString = s"Failed on test with:" +
          s"numTrees=$numTrees, featureSubsetStrategy=$featureSubsetStrategy," +
          s" numFeaturesPerNode=$numFeaturesPerNode, seed=$seed"
        val nodeStack = new mutable.ArrayStack[(Int, OptimizedLearningNode)]
        val topNodes: Array[OptimizedLearningNode] = new Array[OptimizedLearningNode](numTrees)
        Range(0, numTrees).foreach { treeIndex =>
          topNodes(treeIndex) = OptimizedLearningNode.emptyNode(nodeIndex = 1)
          nodeStack.push((treeIndex, topNodes(treeIndex)))
        }
        val rng = new scala.util.Random(seed = seed)
        val (nodesForGroup: Map[Int, Array[OptimizedLearningNode]],
        treeToNodeToIndexInfo: Map[Int, Map[Int, OptimizedRandomForest.NodeIndexInfo]]) =
          OptimizedRandomForest.selectNodesToSplit(nodeStack, maxMemoryUsage, metadata, rng)

        assert(nodesForGroup.size === numTrees, failString)
        assert(nodesForGroup.values.forall(_.length == 1), failString) // 1 node per tree

        if (numFeaturesPerNode == numFeatures) {
          // featureSubset values should all be None
          assert(treeToNodeToIndexInfo.values.forall(_.values.forall(_.featureSubset.isEmpty)),
            failString)
        } else {
          // Check number of features.
          assert(treeToNodeToIndexInfo.values.forall(_.values.forall(
            _.featureSubset.get.length === numFeaturesPerNode)), failString)
        }
      }
    }

    checkFeatureSubsetStrategy(numTrees = 1, "auto", numFeatures)
    checkFeatureSubsetStrategy(numTrees = 1, "all", numFeatures)
    checkFeatureSubsetStrategy(numTrees = 1, "sqrt", math.sqrt(numFeatures).ceil.toInt)
    checkFeatureSubsetStrategy(numTrees = 1, "log2",
      (math.log(numFeatures) / math.log(2)).ceil.toInt)
    checkFeatureSubsetStrategy(numTrees = 1, "onethird", (numFeatures / 3.0).ceil.toInt)

    val realStrategies = Array(".1", ".10", "0.10", "0.1", "0.9", "1.0")
    for (strategy <- realStrategies) {
      val expected = (strategy.toDouble * numFeatures).ceil.toInt
      checkFeatureSubsetStrategy(numTrees = 1, strategy, expected)
    }

    val integerStrategies = Array("1", "10", "100", "1000", "10000")
    for (strategy <- integerStrategies) {
      val expected = if (strategy.toInt < numFeatures) strategy.toInt else numFeatures
      checkFeatureSubsetStrategy(numTrees = 1, strategy, expected)
    }

    val invalidStrategies = Array("-.1", "-.10", "-0.10", ".0", "0.0", "1.1", "0")
    for (invalidStrategy <- invalidStrategies) {
      intercept[IllegalArgumentException] {
        val metadata =
          OptimizedDecisionTreeMetadata.buildMetadata(rdd, strategy, numTrees = 1, invalidStrategy)
      }
    }

    checkFeatureSubsetStrategy(numTrees = 2, "all", numFeatures)
    checkFeatureSubsetStrategy(numTrees = 2, "auto", math.sqrt(numFeatures).ceil.toInt)
    checkFeatureSubsetStrategy(numTrees = 2, "sqrt", math.sqrt(numFeatures).ceil.toInt)
    checkFeatureSubsetStrategy(numTrees = 2, "log2",
      (math.log(numFeatures) / math.log(2)).ceil.toInt)
    checkFeatureSubsetStrategy(numTrees = 2, "onethird", (numFeatures / 3.0).ceil.toInt)

    for (strategy <- realStrategies) {
      val expected = (strategy.toDouble * numFeatures).ceil.toInt
      checkFeatureSubsetStrategy(numTrees = 2, strategy, expected)
    }

    for (strategy <- integerStrategies) {
      val expected = if (strategy.toInt < numFeatures) strategy.toInt else numFeatures
      checkFeatureSubsetStrategy(numTrees = 2, strategy, expected)
    }
    for (invalidStrategy <- invalidStrategies) {
      intercept[IllegalArgumentException] {
        val metadata =
          OptimizedDecisionTreeMetadata.buildMetadata(rdd, strategy, numTrees = 2, invalidStrategy)
      }
    }
  }

  test("Binary classification with continuous features: subsampling features") {
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val strategy = new OldStrategy(algo = OldAlgo.Classification, impurity = Gini, maxDepth = 2,
      numClasses = 2, categoricalFeaturesInfo = categoricalFeaturesInfo)
    binaryClassificationTestWithContinuousFeaturesAndSubsampledFeatures(strategy)
  }

  test("Binary classification with continuous features and node Id cache: subsampling features") {
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val strategy = new OldStrategy(algo = OldAlgo.Classification, impurity = Gini, maxDepth = 2,
      numClasses = 2, categoricalFeaturesInfo = categoricalFeaturesInfo,
      useNodeIdCache = true)
    binaryClassificationTestWithContinuousFeaturesAndSubsampledFeatures(strategy)
  }

  test("normalizeMapValues") {
    val map = new OpenHashMap[Int, Double]()
    map(0) = 1.0
    map(2) = 2.0
    TreeEnsembleModel.normalizeMapValues(map)
    val expected = Map(0 -> 1.0 / 3.0, 2 -> 2.0 / 3.0)
    assert(mapToVec(map.toMap) ~== mapToVec(expected) relTol 0.01)
  }

    ///////////////////////////////////////////////////////////////////////////////
    // Tests for pruning of redundant subtrees (generated by a split improving the
    // impurity measure, but always leading to the same prediction).
    ///////////////////////////////////////////////////////////////////////////////

  test("SPARK-3159 tree model redundancy - classification") {
    // The following dataset is set up such that splitting over feature_1 for points having
    // feature_0 = 0 improves the impurity measure, despite the prediction will always be 0
    // in both branches.
    val arr = Array(
      Instance(0.0, 1.0, Vectors.dense(0.0, 1.0)),
      Instance(1.0, 1.0, Vectors.dense(0.0, 1.0)),
      Instance(0.0, 1.0, Vectors.dense(0.0, 0.0)),
      Instance(1.0, 1.0, Vectors.dense(1.0, 0.0)),
      Instance(0.0, 1.0, Vectors.dense(1.0, 0.0)),
      Instance(1.0, 1.0, Vectors.dense(1.0, 1.0))
    )
    val rdd = sc.parallelize(arr)

    val numClasses = 2
    val strategy = new OldStrategy(algo = OldAlgo.Classification, impurity = Gini, maxDepth = 4,
      numClasses = numClasses, maxBins = 32)

    val prunedTree = OptimizedRandomForest.run(rdd, strategy, numTrees = 1, featureSubsetStrategy = "auto",
      seed = 42, instr = None)._1.head

    val unprunedTree = OptimizedRandomForest.run(rdd, strategy, numTrees = 1, featureSubsetStrategy = "auto",
      seed = 42, instr = None, prune = false)._1.head

    assert(prunedTree.numNodes === 5)
    assert(unprunedTree.numNodes === 7)
  }

  test("SPARK-3159 tree model redundancy - regression") {
    // The following dataset is set up such that splitting over feature_0 for points having
    // feature_1 = 1 improves the impurity measure, despite the prediction will always be 0.5
    // in both branches.
    val arr = Array(
      Instance(0.0, 1.0, Vectors.dense(0.0, 1.0)),
      Instance(1.0, 1.0, Vectors.dense(0.0, 1.0)),
      Instance(0.0, 1.0, Vectors.dense(0.0, 0.0)),
      Instance(0.0, 1.0, Vectors.dense(1.0, 0.0)),
      Instance(1.0, 1.0, Vectors.dense(1.0, 1.0)),
      Instance(0.0, 1.0, Vectors.dense(1.0, 1.0)),
      Instance(0.5, 1.0, Vectors.dense(1.0, 1.0))
    )
    val rdd = sc.parallelize(arr)

    val strategy = new OldStrategy(algo = OldAlgo.Regression, impurity = Variance, maxDepth = 4,
      numClasses = 0, maxBins = 32)

    val prunedTree = OptimizedRandomForest.run(rdd, strategy, numTrees = 1, featureSubsetStrategy = "auto",
      seed = 42, instr = None)._1.head

    val unprunedTree = OptimizedRandomForest.run(rdd, strategy, numTrees = 1, featureSubsetStrategy = "auto",
      seed = 42, instr = None, prune = false)._1.head

    assert(prunedTree.numNodes === 3)
    assert(unprunedTree.numNodes === 5)
  }
}

object OptimizedRandomForestSuite {
  def mapToVec(map: Map[Int, Double]): Vector = {
    val size = (map.keys.toSeq :+ 0).max + 1
    val (indices, values) = map.toSeq.sortBy(_._1).unzip
    Vectors.sparse(size, indices.toArray, values.toArray)
  }

  def generateOrderedInstances(numFeatures: Int, numInstances: Int): Array[Instance] = {
    val arr = new Array[Instance](numInstances)
    for (i <- 0 until numInstances) {
      val label = if (i < numInstances / 10) {
        0.0
      } else if (i < numInstances / 2) {
        1.0
      } else if (i < numInstances * 0.9) {
        0.0
      } else {
        1.0
      }
      val features = Array.fill[Double](numFeatures)(i.toDouble)
      arr(i) = Instance(label, 1.0, Vectors.dense(features))
    }
    arr
  }

  def generateOrderedInstancesWithLabel1(): Array[Instance] = {
    val arr = new Array[Instance](1000)
    for (i <- 0 until 1000) {
      val lp = Instance(1.0, 2.0, Vectors.dense(i.toDouble, 999.0 - i))
      arr(i) = lp
    }
    arr
  }

  def generateOrderedInstancesWithLabel0(): Array[Instance] = {
    val arr = new Array[Instance](1000)
    for (i <- 0 until 1000) {
      val lp = Instance(0.0, 2.0, Vectors.dense(i.toDouble, 1000.0 - i))
      arr(i) = lp
    }
    arr
  }

  def generateOrderedInstances(): Array[Instance] = {
    val arr = new Array[Instance](1000)
    for (i <- 0 until 1000) {
      val label = if (i < 100) {
        0.0
      } else if (i < 500) {
        1.0
      } else if (i < 900) {
        0.0
      } else {
        1.0
      }
      arr(i) = Instance(label, 1.0, Vectors.dense(i.toDouble, 1000.0 - i))
    }
    arr
  }

  def generateCategoricalInstances(): Array[Instance] = {
    val arr = new Array[Instance](1000)
    for (i <- 0 until 1000) {
      if (i < 600) {
        arr(i) = Instance(1.0, 1.0, Vectors.dense(0.0, 1.0))
      } else {
        arr(i) = Instance(0.0, 1.0, Vectors.dense(1.0, 0.0))
      }
    }
    arr
  }

  def generateCategoricalInstancesAsJavaList(): java.util.List[Instance] = {
    generateCategoricalInstances().toList.asJava
  }

  def generateCategoricalInstancesForMulticlass(): Array[Instance] = {
    val arr = new Array[Instance](3000)
    for (i <- 0 until 3000) {
      if (i < 1000) {
        arr(i) = Instance(2.0, 1.0, Vectors.dense(2.0, 2.0))
      } else if (i < 2000) {
        arr(i) = Instance(1.0, 1.0, Vectors.dense(1.0, 2.0))
      } else {
        arr(i) = Instance(2.0, 1.0, Vectors.dense(2.0, 2.0))
      }
    }
    arr
  }

  def generateContinuousInstancesForMulticlass(): Array[Instance] = {
    val arr = new Array[Instance](3000)
    for (i <- 0 until 3000) {
      if (i < 2000) {
        arr(i) = Instance(2.0, 1.0, Vectors.dense(2.0, i))
      } else {
        arr(i) = Instance(1.0, 1.0, Vectors.dense(2.0, i))
      }
    }
    arr
  }

  def generateCategoricalInstancesForMulticlassForOrderedFeatures():
  Array[Instance] = {
    val arr = new Array[Instance](3000)
    for (i <- 0 until 3000) {
      if (i < 1001) {
        arr(i) = Instance(2.0, 1.0, Vectors.dense(2.0, 2.0))
      } else if (i < 2000) {
        arr(i) = Instance(1.0, 1.0, Vectors.dense(1.0, 2.0))
      } else {
        arr(i) = Instance(1.0, 1.0, Vectors.dense(2.0, 2.0))
      }
    }
    arr
  }
}
