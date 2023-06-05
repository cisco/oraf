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
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.tree.impl.{OptimizedRandomForestSuite, OptimizedTreeTests}
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTest, MLTestingUtils}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

/**
  * Test suite for [[OptimizedRandomForestRegressor]].
  */
class OptimizedRandomForestRegressorSuite extends MLTest with DefaultReadWriteTest{

  import OptimizedRandomForestRegressorSuite.compareAPIs
  import testImplicits._

  private var orderedInstances50_1000: RDD[Instance] = _

  override def beforeAll() {
    super.beforeAll()
    orderedInstances50_1000 =
      sc.parallelize(OptimizedRandomForestSuite.generateOrderedInstances(numFeatures = 50, 1000))
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////

  def regressionTestWithContinuousFeatures(rf: RandomForestRegressor, orf: OptimizedRandomForestRegressor): Unit = {
    val categoricalFeaturesInfo = Map.empty[Int, Int]
    val newRF = rf
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(10)
      .setNumTrees(1)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)
    val optimizedRF = orf
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(10)
      .setNumTrees(1)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)
    compareAPIs(orderedInstances50_1000, newRF, optimizedRF, categoricalFeaturesInfo)
  }

  // Fixed
  test("Regression with continuous features:" +
    " comparing DecisionTree vs. RandomForest(numTrees = 1)") {
    val rf = new RandomForestRegressor()
    val orf = new OptimizedRandomForestRegressor()
    regressionTestWithContinuousFeatures(rf, orf)
  }

  test("Regression with continuous features and node Id cache :" +
    " comparing DecisionTree vs. RandomForest(numTrees = 1)") {
    val rf = new RandomForestRegressor()
      .setCacheNodeIds(true)
    val orf = new OptimizedRandomForestRegressor()
      .setCacheNodeIds(true)
    regressionTestWithContinuousFeatures(rf, orf)
  }

  test("prediction on single instance") {
    val rf = new OptimizedRandomForestRegressor()
      .setImpurity("variance")
      .setMaxDepth(2)
      .setMaxBins(10)
      .setNumTrees(1)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)

    val df = orderedInstances50_1000.toDF()
    val model = rf.fit(df)
    testPredictionModelSinglePrediction(model, df)
  }


  test("should support all NumericType labels and not support other types") {
    val rf = new OptimizedRandomForestRegressor().setMaxDepth(1)
    MLTestingUtils.checkNumericTypes[OptimizedRandomForestRegressionModel, OptimizedRandomForestRegressor](
      rf, spark, isClassification = false) { (expected, actual) =>
      OptimizedTreeTests.checkEqual(expected, actual)
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of model save/load
  /////////////////////////////////////////////////////////////////////////////

  test("read/write") {
    def checkModelData(
                        model: OptimizedRandomForestRegressionModel,
                        model2: OptimizedRandomForestRegressionModel): Unit = {
      OptimizedTreeTests.checkEqual(model, model2)
      assert(model.numFeatures === model2.numFeatures)
    }

    val rf = new OptimizedRandomForestRegressor().setNumTrees(2)
    val rdd = OptimizedTreeTests.getTreeReadWriteData(sc)

    val allParamSettings = OptimizedTreeTests.allParamSettings ++ Map("impurity" -> "variance")

    val continuousData: DataFrame =
      OptimizedTreeTests.setMetadata(rdd, Map.empty[Int, Int], numClasses = 0)
    testEstimatorAndModelReadWrite(rf, continuousData, allParamSettings,
      allParamSettings, checkModelData)
  }
}

private object OptimizedRandomForestRegressorSuite extends SparkFunSuite {

  /**
    * Train 2 models on the given dataset, one using the old API and one using the new API.
    * Convert the old model to the new format, compare them, and fail if they are not exactly equal.
    */
  def compareAPIs(
                   data: RDD[Instance],
                   rf: RandomForestRegressor,
                   orf: OptimizedRandomForestRegressor,
                   categoricalFeatures: Map[Int, Int]): Unit = {
    val numFeatures = data.first().features.size
    val oldPoints = data.map(i => LabeledPoint(i.label, i.features))

    val newData: DataFrame = OptimizedTreeTests.setMetadata(data, categoricalFeatures, numClasses = 0)
    val oldData: DataFrame = OptimizedTreeTests.setMetadataForLabeledPoints(oldPoints, categoricalFeatures, numClasses = 0)

    val oldModel = rf.fit(oldData)
    val optimizedModel = orf.fit(newData)
    // Use parent from newTree since this is not checked anyways.
    OptimizedTreeTests.checkEqualOldRegression(oldModel, optimizedModel)
    assert(oldModel.numFeatures === numFeatures)
  }
}