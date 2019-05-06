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

package org.apache.spark.ml.tree

import java.util.Locale

import org.apache.spark.ml.param._
import org.apache.spark.ml.tree.impl.LocalDecisionTree
import org.apache.spark.mllib.tree.configuration.{DefaultTimePredictionStrategy, OptimizedForestStrategy, TimePredictionStrategy, Algo => OldAlgo}
import org.apache.spark.mllib.tree.impurity.{Impurity => OldImpurity}


private[ml] trait OptimizedDecisionTreeParams extends DecisionTreeParams {

  final val maxMemoryMultiplier: DoubleParam = new DoubleParam(this, "maxMemoryMultiplier", "",
    ParamValidators.gt(0.0))

  var timePredictionStrategy: TimePredictionStrategy = new DefaultTimePredictionStrategy

  final val maxTasksPerBin: IntParam
  = new IntParam (this, "maxTasksPerBin", "", ParamValidators.gt(0))

  var customSplits: Option[Array[Array[Double]]] = None

  var localTrainingAlgorithm: LocalTrainingAlgorithm = new LocalDecisionTree

  setDefault(maxDepth -> 5, maxBins -> 32, minInstancesPerNode -> 1, minInfoGain -> 0.0,
    maxMemoryInMB -> 256, cacheNodeIds -> true, checkpointInterval -> 10,
    maxMemoryMultiplier -> 4, maxTasksPerBin -> Int.MaxValue)

  @deprecated("This method is deprecated and will be removed in 3.0.0.", "2.1.0")
  def setMaxMemoryMultiplier(value: Double): this.type = set(maxMemoryMultiplier, value)

  /** @group expertGetParam */
  final def getMaxMemoryMultiplier: Double = $(maxMemoryMultiplier)

  @deprecated("This method is deprecated and will be removed in 3.0.0.", "2.1.0")
  def setTimePredictionStrategy(value: TimePredictionStrategy) = timePredictionStrategy = value

  final def getTimePredictionStrategy: TimePredictionStrategy = timePredictionStrategy

  @deprecated("This method is deprecated and will be removed in 3.0.0.", "2.1.0")
  def setMaxTasksPerBin(value: Int): this.type = set(maxTasksPerBin, value)

  /** @group expertGetParam */
  final def getMaxTasksPerBin: Int = $(maxTasksPerBin)

  @deprecated("This method is deprecated and will be removed in 3.0.0.", "2.1.0")
  def setCustomSplits(value: Option[Array[Array[Double]]]) = customSplits = value

  /** @group expertGetParam */
  final def getCustomSplits: Option[Array[Array[Double]]] = customSplits

  @deprecated("This method is deprecated and will be removed in 3.0.0.", "2.1.0")
  def setLocalTrainingAlgorithm(value: LocalTrainingAlgorithm) = this.localTrainingAlgorithm = value

  /** @group expertGetParam */
  final def getLocalTrainingAlgorithm: LocalTrainingAlgorithm = localTrainingAlgorithm

  /** (private[ml]) Create a Strategy instance to use with the old API. */
  private[ml] override def getOldStrategy(
                                           categoricalFeatures: Map[Int, Int],
                                           numClasses: Int,
                                           oldAlgo: OldAlgo.Algo,
                                           oldImpurity: OldImpurity,
                                           subsamplingRate: Double): OptimizedForestStrategy = {
    val strategy = OptimizedForestStrategy.defaultStrategy(oldAlgo)
    strategy.impurity = oldImpurity
    strategy.checkpointInterval = getCheckpointInterval
    strategy.maxBins = getMaxBins
    strategy.maxDepth = getMaxDepth
    strategy.maxMemoryInMB = getMaxMemoryInMB
    strategy.minInfoGain = getMinInfoGain
    strategy.minInstancesPerNode = getMinInstancesPerNode
    strategy.useNodeIdCache = getCacheNodeIds
    strategy.numClasses = numClasses
    strategy.categoricalFeaturesInfo = categoricalFeatures
    strategy.subsamplingRate = subsamplingRate
    strategy.maxMemoryMultiplier = getMaxMemoryMultiplier
    strategy.timePredictionStrategy = getTimePredictionStrategy
    strategy.localTrainingAlgorithm = getLocalTrainingAlgorithm
    strategy
  }
}

private[spark] object OptimizedTreeEnsembleParams {
  // These options should be lowercase.
  final val supportedTimePredictionStrategies: Array[String] =
    Array("size").map(_.toLowerCase(Locale.ROOT))

  final val supportedLocalTrainingAlgorithms: Array[String] =
    Array("yggdrasil").map(_.toLowerCase(Locale.ROOT))
}

private[ml] trait OptimizedDecisionTreeClassifierParams
  extends OptimizedDecisionTreeParams with TreeClassifierParams

private[ml] trait OptimizedDecisionTreeRegressorParams
  extends OptimizedDecisionTreeParams with TreeRegressorParams

private[ml] trait OptimizedTreeEnsembleParams extends TreeEnsembleParams
  with OptimizedDecisionTreeParams {
  private[ml] override def getOldStrategy(
                                  categoricalFeatures: Map[Int, Int],
                                  numClasses: Int,
                                  oldAlgo: OldAlgo.Algo,
                                  oldImpurity: OldImpurity): OptimizedForestStrategy = {
    super.getOldStrategy(categoricalFeatures, numClasses, oldAlgo, oldImpurity, getSubsamplingRate)
  }
}

private[ml] trait OptimizedRandomForestParams extends RandomForestParams
  with OptimizedTreeEnsembleParams

private[ml] trait OptimizedRandomForestClassifierParams
  extends OptimizedRandomForestParams with TreeClassifierParams

private[ml] trait OptimizedRandomForestRegressorParams
  extends OptimizedRandomForestParams with TreeRegressorParams
