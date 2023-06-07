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

import org.apache.spark.ml.Predictor
import org.apache.spark.ml.feature.Instance
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.regression.OptimizedDecisionTreeRegressionModel
import org.apache.spark.ml.tree.{OptimizedDecisionTreeParams, TreeRegressorParams}
import org.apache.spark.ml.util.{Identifiable, MetadataUtils}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo, Strategy => OldStrategy}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Dataset, Row}
import org.apache.spark.sql.functions._

/**
 * Test-only class for fitting a decision tree regressor on a dataset small enough to fit on a
 * single machine.
 */
private[impl] final class LocalDecisionTreeRegressor(override val uid: String)
  extends Predictor[Vector, LocalDecisionTreeRegressor, OptimizedDecisionTreeRegressionModel]
    with OptimizedDecisionTreeParams with TreeRegressorParams {

  def this() = this(Identifiable.randomUID("local_dtr"))

  // Override parameter setters from parent trait for Java API compatibility.
  def setMaxDepth(value: Int): this.type = set(maxDepth, value)

  def setMaxBins(value: Int): this.type = set(maxBins, value)

  def setMinInstancesPerNode(value: Int): this.type = set(minInstancesPerNode, value)

  def setMinInfoGain(value: Double): this.type = set(minInfoGain, value)

  def setMaxMemoryInMB(value: Int): this.type = set(maxMemoryInMB, value)

  def setImpurity(value: String): this.type = set(impurity, value)

  def setSeed(value: Long): this.type = set(seed, value)

  def copy(extra: ParamMap): LocalDecisionTreeRegressor = defaultCopy(extra)

  override protected def train(dataset: Dataset[_]): OptimizedDecisionTreeRegressionModel = {
    val categoricalFeatures: Map[Int, Int] =
      MetadataUtils.getCategoricalFeatures(dataset.schema($(featuresCol)))
//    val oldDataset: RDD[Instance] = extractLabeledPoints(dataset)
    val oldDataset: RDD[Instance] = dataset.select(col($(labelCol)), col($(featuresCol)), col($(weightCol))).rdd.map {
      case Row(label: Double, features: Vector, weight: Double) => Instance(label, weight, features)
    }

    val strategy = getOldStrategy(categoricalFeatures)
    val model = LocalTreeTests.train(oldDataset, strategy, parentUID = Some(uid),
      seed = getSeed)
    model.asInstanceOf[OptimizedDecisionTreeRegressionModel]
  }

  /** Create a Strategy instance to use with the old API. */
  private[impl] def getOldStrategy(categoricalFeatures: Map[Int, Int]): OldStrategy = {
    super.getOldStrategy(categoricalFeatures, numClasses = 0, OldAlgo.Regression, getOldImpurity,
      subsamplingRate = 1.0)
  }
}
