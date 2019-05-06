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

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.LabeledPoint
import org.apache.spark.ml.tree.impl.{OptimizedTreeTests, TreeTests}
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTest, MLTestingUtils}
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.mllib.tree.{DecisionTree => OldDecisionTree, DecisionTreeSuite => OldDecisionTreeSuite}
import org.apache.spark.mllib.util.LinearDataGenerator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

class OptimizedDecisionTreeRegressorSuite extends MLTest with DefaultReadWriteTest {

  import OptimizedDecisionTreeRegressorSuite.compareAPIs
  import testImplicits._

  private var categoricalDataPointsRDD: RDD[LabeledPoint] = _
  private var linearRegressionData: DataFrame = _

  private val seed = 42

  override def beforeAll() {
    super.beforeAll()
    categoricalDataPointsRDD =
      sc.parallelize(OldDecisionTreeSuite.generateCategoricalDataPoints().map(_.asML))
    linearRegressionData = sc.parallelize(LinearDataGenerator.generateLinearInput(
      intercept = 6.3, weights = Array(4.7, 7.2), xMean = Array(0.9, -1.3),
      xVariance = Array(0.7, 1.2), nPoints = 1000, seed, eps = 0.5), 2).map(_.asML).toDF()
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////

  test("Regression stump with 3-ary (ordered) categorical features") {
    val dt = new DecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(100)
      .setSeed(1)
    val odt = new OptimizedDecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(100)
      .setSeed(1)
    val categoricalFeatures = Map(0 -> 3, 1 -> 3)
    compareAPIs(categoricalDataPointsRDD, dt, odt, categoricalFeatures)
  }

  test("Regression stump with binary (ordered) categorical features") {
    val dt = new DecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(100)
    val odt = new OptimizedDecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(100)
    val categoricalFeatures = Map(0 -> 2, 1 -> 2)
    compareAPIs(categoricalDataPointsRDD, dt, odt, categoricalFeatures)
  }

  test("copied model must have the same parent") {
    val categoricalFeatures = Map(0 -> 2, 1 -> 2)
    val df = TreeTests.setMetadata(categoricalDataPointsRDD, categoricalFeatures, numClasses = 0)
    val dtr = new OptimizedDecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(8)
    val model = dtr.fit(df)
    MLTestingUtils.checkCopyAndUids(dtr, model)
  }

  test("prediction on single instance") {
    val dt = new OptimizedDecisionTreeRegressor()
      .setImpurity("variance")
      .setMaxDepth(3)
      .setSeed(123)

    // In this data, feature 1 is very important.
    val data: RDD[LabeledPoint] = OptimizedTreeTests.featureImportanceData(sc)
    val categoricalFeatures = Map.empty[Int, Int]
    val df: DataFrame = OptimizedTreeTests.setMetadata(data, categoricalFeatures, 0)

    val model = dt.fit(df)
    testPredictionModelSinglePrediction(model, df)
  }

  test("should support all NumericType labels and not support other types") {
    val dt = new OptimizedDecisionTreeRegressor().setMaxDepth(1)
    MLTestingUtils.checkNumericTypes[OptimizedDecisionTreeRegressionModel, OptimizedDecisionTreeRegressor](
      dt, spark, isClassification = false) { (expected, actual) =>
      OptimizedTreeTests.checkEqual(expected, actual)
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of model save/load
  /////////////////////////////////////////////////////////////////////////////

  test("read/write") {
    def checkModelData(
                        model: OptimizedDecisionTreeRegressionModel,
                        model2: OptimizedDecisionTreeRegressionModel): Unit = {
      OptimizedTreeTests.checkEqual(model, model2)
      assert(model.numFeatures === model2.numFeatures)
    }

    val dt = new OptimizedDecisionTreeRegressor()
    val rdd = OptimizedTreeTests.getTreeReadWriteData(sc)

    // Categorical splits with tree depth 2
    val categoricalData: DataFrame =
      OptimizedTreeTests.setMetadata(rdd, Map(0 -> 2, 1 -> 3), numClasses = 0)
    testEstimatorAndModelReadWrite(dt, categoricalData,
      OptimizedTreeTests.allParamSettings, OptimizedTreeTests.allParamSettings, checkModelData)

    // Continuous splits with tree depth 2
    val continuousData: DataFrame =
      OptimizedTreeTests.setMetadata(rdd, Map.empty[Int, Int], numClasses = 0)
    testEstimatorAndModelReadWrite(dt, continuousData,
      OptimizedTreeTests.allParamSettings, OptimizedTreeTests.allParamSettings, checkModelData)

    // Continuous splits with tree depth 0
    testEstimatorAndModelReadWrite(dt, continuousData,
      OptimizedTreeTests.allParamSettings ++ Map("maxDepth" -> 0),
      OptimizedTreeTests.allParamSettings ++ Map("maxDepth" -> 0), checkModelData)
  }
}

private[ml] object OptimizedDecisionTreeRegressorSuite extends SparkFunSuite {

  /**
    * Train 2 decision trees on the given dataset, one using the old API and one using the new API.
    * Convert the old tree to the new format, compare them, and fail if they are not exactly equal.
    */
  def compareAPIs(
                   data: RDD[LabeledPoint],
                   dt: DecisionTreeRegressor,
                   odt: OptimizedDecisionTreeRegressor,
                   categoricalFeatures: Map[Int, Int]): Unit = {
    val numFeatures = data.first().features.size
    val newData: DataFrame = OptimizedTreeTests.setMetadata(data, categoricalFeatures, numClasses = 0)
    val newTree = dt.fit(newData)
    val optimizedTree = odt.fit(newData)
    // Use parent from newTree since this is not checked anyways.
    OptimizedTreeTests.checkEqual(newTree, optimizedTree)
    assert(optimizedTree.numFeatures === numFeatures)
  }
}