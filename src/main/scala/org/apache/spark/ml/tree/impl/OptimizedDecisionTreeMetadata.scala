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

import scala.collection.mutable
import scala.util.Try

import org.apache.spark.internal.Logging
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.TreeEnsembleParams
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.configuration.QuantileStrategy._
import org.apache.spark.mllib.tree.configuration.Strategy
import org.apache.spark.mllib.tree.impurity.Impurity
import org.apache.spark.rdd.RDD

/**
  * Learning and dataset metadata for DecisionTree.
  *
  * @param numClasses    For classification: labels can take values {0, ..., numClasses - 1}.
  *                      For regression: fixed at 0 (no meaning).
  * @param maxBins  Maximum number of bins, for all features.
  * @param featureArity  Map: categorical feature index to arity.
  *                      I.e., the feature takes values in {0, ..., arity - 1}.
  * @param numBins  Number of bins for each feature.
  */
class OptimizedDecisionTreeMetadata(
                                     numFeatures: Int,
                                     numExamples: Long,
                                     numClasses: Int,
                                     maxBins: Int,
                                     featureArity: Map[Int, Int],
                                     unorderedFeatures: Set[Int],
                                     numBins: Array[Int],
                                     impurity: Impurity,
                                     quantileStrategy: QuantileStrategy,
                                     maxDepth: Int,
                                     minInstancesPerNode: Int,
                                     minInfoGain: Double,
                                     numTrees: Int,
                                     numFeaturesPerNode: Int) extends DecisionTreeMetadata(numFeatures,
  numExamples, numClasses, maxBins, featureArity, unorderedFeatures, numBins, impurity, quantileStrategy, maxDepth, minInstancesPerNode, minInfoGain, numTrees, numFeaturesPerNode) with Serializable {
}

object OptimizedDecisionTreeMetadata extends Logging {

  /**
    * Construct a [[DecisionTreeMetadata]] instance for this dataset and parameters.
    * This computes which categorical features will be ordered vs. unordered,
    * as well as the number of splits and bins for each feature.
    */
  def buildMetadata(
                     input: RDD[LabeledPoint],
                     strategy: Strategy,
                     numTrees: Int,
                     featureSubsetStrategy: String): OptimizedDecisionTreeMetadata = {

    val numFeatures = input.map(_.features.size).take(1).headOption.getOrElse {
      throw new IllegalArgumentException(s"DecisionTree requires size of input RDD > 0, " +
        s"but was given by empty one.")
    }
    require(numFeatures > 0, s"DecisionTree requires number of features > 0, " +
      s"but was given an empty features vector")
    val numExamples = input.count()
    val numClasses = strategy.algo match {
      case Classification => strategy.numClasses
      case Regression => 0
    }

    val maxPossibleBins = math.min(strategy.maxBins, numExamples).toInt
    if (maxPossibleBins < strategy.maxBins) {
      logWarning(s"DecisionTree reducing maxBins from ${strategy.maxBins} to $maxPossibleBins" +
        s" (= number of training instances)")
    }

    // We check the number of bins here against maxPossibleBins.
    // This needs to be checked here instead of in Strategy since maxPossibleBins can be modified
    // based on the number of training examples.
    if (strategy.categoricalFeaturesInfo.nonEmpty) {
      val maxCategoriesPerFeature = strategy.categoricalFeaturesInfo.values.max
      val maxCategory =
        strategy.categoricalFeaturesInfo.find(_._2 == maxCategoriesPerFeature).get._1
      require(maxCategoriesPerFeature <= maxPossibleBins,
        s"DecisionTree requires maxBins (= $maxPossibleBins) to be at least as large as the " +
          s"number of values in each categorical feature, but categorical feature $maxCategory " +
          s"has $maxCategoriesPerFeature values. Considering remove this and other categorical " +
          "features with a large number of values, or add more training examples.")
    }

    val unorderedFeatures = new mutable.HashSet[Int]()
    val numBins = Array.fill[Int](numFeatures)(maxPossibleBins)
    if (numClasses > 2) {
      // Multiclass classification
      val maxCategoriesForUnorderedFeature =
        ((math.log(maxPossibleBins / 2 + 1) / math.log(2.0)) + 1).floor.toInt
      strategy.categoricalFeaturesInfo.foreach { case (featureIndex, numCategories) =>
        // Hack: If a categorical feature has only 1 category, we treat it as continuous.
        // TODO(SPARK-9957): Handle this properly by filtering out those features.
        if (numCategories > 1) {
          // Decide if some categorical features should be treated as unordered features,
          //  which require 2 * ((1 << numCategories - 1) - 1) bins.
          // We do this check with log values to prevent overflows in case numCategories is large.
          // The next check is equivalent to: 2 * ((1 << numCategories - 1) - 1) <= maxBins
          if (numCategories <= maxCategoriesForUnorderedFeature) {
            unorderedFeatures.add(featureIndex)
            numBins(featureIndex) = numUnorderedBins(numCategories)
          } else {
            numBins(featureIndex) = numCategories
          }
        }
      }
    } else {
      // Binary classification or regression
      strategy.categoricalFeaturesInfo.foreach { case (featureIndex, numCategories) =>
        // If a categorical feature has only 1 category, we treat it as continuous: SPARK-9957
        if (numCategories > 1) {
          numBins(featureIndex) = numCategories
        }
      }
    }

    // Set number of features to use per node (for random forests).
    val _featureSubsetStrategy = featureSubsetStrategy match {
      case "auto" =>
        if (numTrees == 1) {
          "all"
        } else {
          if (strategy.algo == Classification) {
            "sqrt"
          } else {
            "onethird"
          }
        }
      case _ => featureSubsetStrategy
    }

    val numFeaturesPerNode: Int = _featureSubsetStrategy match {
      case "all" => numFeatures
      case "sqrt" => math.sqrt(numFeatures).ceil.toInt
      case "log2" => math.max(1, (math.log(numFeatures) / math.log(2)).ceil.toInt)
      case "onethird" => (numFeatures / 3.0).ceil.toInt
      case _ =>
        Try(_featureSubsetStrategy.toInt).filter(_ > 0).toOption match {
          case Some(value) => math.min(value, numFeatures)
          case None =>
            Try(_featureSubsetStrategy.toDouble).filter(_ > 0).filter(_ <= 1.0).toOption match {
              case Some(value) => math.ceil(value * numFeatures).toInt
              case _ => throw new IllegalArgumentException(s"Supported values:" +
                s" ${TreeEnsembleParams.supportedFeatureSubsetStrategies.mkString(", ")}," +
                s" (0.0-1.0], [1-n].")
            }
        }
    }

    new OptimizedDecisionTreeMetadata(numFeatures, numExamples, numClasses, numBins.max,
      strategy.categoricalFeaturesInfo, unorderedFeatures.toSet, numBins,
      strategy.impurity, strategy.quantileCalculationStrategy, strategy.maxDepth,
      strategy.minInstancesPerNode, strategy.minInfoGain, numTrees, numFeaturesPerNode)
  }

  /**
    * Version of [[DecisionTreeMetadata#buildMetadata]] for DecisionTree.
    */
  def buildMetadata(
                     input: RDD[LabeledPoint],
                     strategy: Strategy): OptimizedDecisionTreeMetadata = {
    buildMetadata(input, strategy, numTrees = 1, featureSubsetStrategy = "all")
  }

  /**
    * Given the arity of a categorical feature (arity = number of categories),
    * return the number of bins for the feature if it is to be treated as an unordered feature.
    * There is 1 split for every partitioning of categories into 2 disjoint, non-empty sets;
    * there are math.pow(2, arity - 1) - 1 such splits.
    * Each split has 2 corresponding bins.
    */
  def numUnorderedBins(arity: Int): Int = (1 << arity - 1) - 1

}
