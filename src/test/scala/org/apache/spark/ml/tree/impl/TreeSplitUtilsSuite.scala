/*
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
import org.apache.spark.ml.tree.{CategoricalSplit, ContinuousSplit, Split}
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.tree.impurity.{Entropy, Impurity}
import org.apache.spark.mllib.tree.model.ImpurityStats
import org.apache.spark.mllib.util.MLlibTestSparkContext

/** Suite exercising helper methods for making split decisions during decision tree training. */
class TreeSplitUtilsSuite
  extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  /**
   * Get a DTStatsAggregator for sufficient stat collection/impurity calculation populated
   * with the data from the specified training points.
   */
  private def getAggregator(
      metadata: DecisionTreeMetadata,
      col: FeatureColumn,
      from: Int,
      to: Int,
      labels: Array[Double],
      featureSplits: Array[Split]): DTStatsAggregator = {

    val statsAggregator = new DTStatsAggregator(metadata, featureSubset = None)
    val instanceWeights = Array.fill[Double](col.values.length)(1.0)
    val indices = col.values.indices.toArray
    AggUpdateUtils.updateParentImpurity(statsAggregator, indices, from, to, instanceWeights, labels)
    (new LocalDecisionTree).updateAggregator(statsAggregator, col, indices, instanceWeights, labels,
      from, to, col.featureIndex, featureSplits)
    statsAggregator
  }

  /** Check that left/right impurities match what we'd expect for a split. */
  private def validateImpurityStats(
      impurity: Impurity,
      labels: Array[Double],
      stats: ImpurityStats,
      expectedLeftStats: Array[Double],
      expectedRightStats: Array[Double]): Unit = {
    // Verify that impurity stats were computed correctly for split
    val numClasses = (labels.max + 1).toInt
    val fullImpurityStatsArray
      = Array.tabulate[Double](numClasses)((label: Int) => labels.count(_ == label).toDouble)
    val fullImpurity = Entropy.calculate(fullImpurityStatsArray, labels.length)
    assert(stats.impurityCalculator.stats === fullImpurityStatsArray)
    assert(stats.impurity === fullImpurity)
    assert(stats.leftImpurityCalculator.stats === expectedLeftStats)
    assert(stats.rightImpurityCalculator.stats === expectedRightStats)
    assert(stats.valid)
  }

  /* * * * * * * * * * * Choosing Splits  * * * * * * * * * * */

  test("chooseSplit: choose correct type of split (continuous split)") {
    // Construct (binned) continuous data
    val labels = Array(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
    val col = FeatureColumn(featureIndex = 0, values = Array(8, 1, 1, 2, 3, 5, 6))
    // Get an array of continuous splits corresponding to values in our binned data
    val splits = LocalTreeTests.getContinuousSplits(1.to(8).toArray, featureIndex = 0)
    // Construct DTStatsAggregator, compute sufficient stats
    val metadata = OptimizedTreeTests.getMetadata(numExamples = 7,
      numFeatures = 1, numClasses = 2, Map.empty)
    val statsAggregator = getAggregator(metadata, col, from = 1, to = 4, labels, splits)
    // Choose split, check that it's a valid ContinuousSplit
    val (split1, stats1) = SplitUtils.chooseSplit(statsAggregator, col.featureIndex,
      col.featureIndex, splits)
    assert(stats1.valid && split1.isInstanceOf[ContinuousSplit])
  }

  test("chooseOrderedCategoricalSplit: basic case") {
    // Helper method for testing ordered categorical split
    def testHelper(
        values: Array[Int],
        labels: Array[Double],
        expectedLeftCategories: Array[Double],
        expectedLeftStats: Array[Double],
        expectedRightStats: Array[Double]): Unit = {
      val featureIndex = 0
      // Construct FeatureVector to store categorical data
      val featureArity = values.max + 1
      val arityMap = Map[Int, Int](featureIndex -> featureArity)
      val col = FeatureColumn(featureIndex = 0, values = values)
      // Construct DTStatsAggregator, compute sufficient stats
      val metadata = OptimizedTreeTests.getMetadata(numExamples = values.length, numFeatures = 1,
        numClasses = 2, arityMap, unorderedFeatures = Some(Set.empty))
      val statsAggregator = getAggregator(metadata, col, from = 0, to = values.length,
        labels, featureSplits = Array.empty)
      // Choose split
      val (split, stats) =
        SplitUtils.chooseOrderedCategoricalSplit(statsAggregator, col.featureIndex,
          col.featureIndex)
      // Verify that split has the expected left-side/right-side categories
      val expectedRightCategories = Range(0, featureArity)
        .filter(c => !expectedLeftCategories.contains(c)).map(_.toDouble).toArray
      split match {
        case s: CategoricalSplit =>
          assert(s.featureIndex === featureIndex)
          assert(s.leftCategories === expectedLeftCategories)
          assert(s.rightCategories === expectedRightCategories)
        case _ =>
          throw new AssertionError(
            s"Expected CategoricalSplit but got ${split.getClass.getSimpleName}")
      }
      validateImpurityStats(Entropy, labels, stats, expectedLeftStats, expectedRightStats)
    }

    val values = Array(0, 0, 1, 2, 2, 2, 2)
    val labels1 = Array(0, 0, 1, 1, 1, 1, 1).map(_.toDouble)
    testHelper(values, labels1, Array(0.0), Array(2.0, 0.0), Array(0.0, 5.0))

    val labels2 = Array(0, 0, 0, 1, 1, 1, 1).map(_.toDouble)
    testHelper(values, labels2, Array(0.0, 1.0), Array(3.0, 0.0), Array(0.0, 4.0))
  }

  test("chooseContinuousSplit: basic case") {
    // Construct data for continuous feature
    val featureIndex = 0
    val thresholds = Array(0, 1, 2, 3)
    val values = thresholds.indices.toArray
    val labels = Array(0.0, 0.0, 1.0, 1.0)
    val col = FeatureColumn(featureIndex = featureIndex, values = values)

    // Construct DTStatsAggregator, compute sufficient stats
    val splits = LocalTreeTests.getContinuousSplits(thresholds, featureIndex)
    val metadata = OptimizedTreeTests.getMetadata(numExamples = values.length, numFeatures = 1,
      numClasses = 2, Map.empty)
    val statsAggregator = getAggregator(metadata, col, from = 0, to = values.length, labels, splits)

    // Choose split, verify that it has expected threshold
    val (split, stats) = SplitUtils.chooseContinuousSplit(statsAggregator, featureIndex,
      featureIndex, splits)
    split match {
      case s: ContinuousSplit =>
        assert(s.featureIndex === featureIndex)
        assert(s.threshold === 1)
      case _ =>
        throw new AssertionError(
          s"Expected ContinuousSplit but got ${split.getClass.getSimpleName}")
    }
    // Verify impurity stats of split
    validateImpurityStats(Entropy, labels, stats, expectedLeftStats = Array(2.0, 0.0),
      expectedRightStats = Array(0.0, 2.0))
  }
}
