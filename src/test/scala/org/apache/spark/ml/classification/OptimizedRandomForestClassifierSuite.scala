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

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.tree.impl.{OptimizedRandomForestSuite, OptimizedTreeTests, TreeTests}
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTest, MLTestingUtils}
import org.apache.spark.mllib.regression.{LabeledPoint => OldLabeledPoint}
import org.apache.spark.mllib.tree.configuration.{Algo => OldAlgo}
import org.apache.spark.mllib.tree.{EnsembleTestHelper, RandomForest => OldRandomForest}
import org.apache.spark.mllib.util.TestingUtils._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

/**
  * Test suite for [[OptimizedRandomForestClassifier]].
  */
class OptimizedRandomForestClassifierSuite extends MLTest with DefaultReadWriteTest {

  import OptimizedRandomForestClassifierSuite.compareAPIs
  import testImplicits._

  private var orderedInstances50_1000: RDD[Instance] = _
  private var orderedInstances5_20: RDD[Instance] = _

  override def beforeAll() {
    super.beforeAll()
    orderedInstances50_1000 =
      sc.parallelize(OptimizedRandomForestSuite.generateOrderedInstances(numFeatures = 50, 1000))
    orderedInstances5_20 =
      sc.parallelize(OptimizedRandomForestSuite.generateOrderedInstances(numFeatures = 5, 20))
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////

  def binaryClassificationTestWithContinuousFeatures(rf: RandomForestClassifier, orf: OptimizedRandomForestClassifier) {
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2
    val newRF = rf
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setNumTrees(1)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)
    val optimizedRF = orf
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setNumTrees(1)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)
    compareAPIs(orderedInstances50_1000, newRF, optimizedRF, categoricalFeatures, numClasses)
  }

  test("Binary classification with continuous features:" +
    " comparing DecisionTree vs. RandomForest(numTrees = 1)") {
    val rf = new RandomForestClassifier()
    val orf = new OptimizedRandomForestClassifier()
    binaryClassificationTestWithContinuousFeatures(rf, orf)
  }

  test("Binary classification with continuous features and node Id cache:" +
    " comparing DecisionTree vs. RandomForest(numTrees = 1)") {
    val rf = new RandomForestClassifier()
      .setCacheNodeIds(true)
    val orf = new OptimizedRandomForestClassifier()
      .setCacheNodeIds(true)
    binaryClassificationTestWithContinuousFeatures(rf, orf)
  }

  test("alternating categorical and continuous features with multiclass labels to test indexing") {
    val arr = Array(
      Instance(0.0, 1.0, Vectors.dense(1.0, 0.0, 0.0, 3.0, 1.0)),
      Instance(1.0, 1.0, Vectors.dense(0.0, 1.0, 1.0, 1.0, 2.0)),
      Instance(0.0, 1.0, Vectors.dense(2.0, 0.0, 0.0, 6.0, 3.0)),
      Instance(2.0, 1.0, Vectors.dense(0.0, 2.0, 1.0, 3.0, 2.0))
    )
    val rdd = sc.parallelize(arr)
    val categoricalFeatures = Map(0 -> 3, 2 -> 2, 4 -> 4)
    val numClasses = 3

    val rf = new RandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(5)
      .setNumTrees(2)
      .setFeatureSubsetStrategy("all")
      .setSeed(12345)
    val orf = new OptimizedRandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(5)
      .setNumTrees(2)
      .setFeatureSubsetStrategy("all")
      .setSeed(12345)
    compareAPIs(rdd, rf, orf, categoricalFeatures, numClasses)
  }

  // Skip test: Different random generators are created during local training
  ignore("subsampling rate in RandomForest") {
    val rdd = orderedInstances5_20
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2

    val rf1 = new RandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setCacheNodeIds(true)
      .setNumTrees(3)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)
    val orf1 = new OptimizedRandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setCacheNodeIds(true)
      .setNumTrees(3)
      .setFeatureSubsetStrategy("auto")
      .setSeed(123)
    compareAPIs(rdd, rf1, orf1, categoricalFeatures, numClasses)

    val rf2 = rf1.setSubsamplingRate(0.5)
    val orf2 = orf1.setSubsamplingRate(0.5)
    compareAPIs(rdd, rf2, orf2, categoricalFeatures, numClasses)
  }

  test("predictRaw and predictProbability") {
    val rdd = orderedInstances5_20
    val rf = new OptimizedRandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(3)
      .setNumTrees(3)
      .setSeed(123)
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2

    val df: DataFrame = OptimizedTreeTests.setMetadata(rdd, categoricalFeatures, numClasses)
    val model = rf.fit(df)

    MLTestingUtils.checkCopyAndUids(rf, model)

    testTransformer[(Vector, Double, Double)](df, model, "prediction", "rawPrediction") {
      case Row(pred: Double, rawPred: Vector) =>
      assert(pred === rawPred.argmax,
        s"Expected prediction $pred but calculated ${rawPred.argmax} from rawPrediction.")
    }

    ProbabilisticClassifierSuite.testPredictMethods[
      Vector, OptimizedRandomForestClassificationModel](this, model, df)
  }

  test("prediction on single instance") {
    val rdd = orderedInstances5_20
    val rf = new OptimizedRandomForestClassifier()
      .setImpurity("Gini")
      .setMaxDepth(3)
      .setNumTrees(3)
      .setSeed(123)
    val categoricalFeatures = Map.empty[Int, Int]
    val numClasses = 2

    val df: DataFrame = OptimizedTreeTests.setMetadata(rdd, categoricalFeatures, numClasses)
    val model = rf.fit(df)

    testPredictionModelSinglePrediction(model, df)
  }

  test("Fitting without numClasses in metadata") {
    val df: DataFrame = OptimizedTreeTests.featureImportanceData(sc).toDF()
    val rf = new OptimizedRandomForestClassifier().setMaxDepth(1).setNumTrees(1)
    rf.fit(df)
  }

  test("should support all NumericType labels and not support other types") {
    val rf = new OptimizedRandomForestClassifier().setMaxDepth(1)
    MLTestingUtils.checkNumericTypes[OptimizedRandomForestClassificationModel, OptimizedRandomForestClassifier](
      rf, spark) { (expected, actual) =>
      OptimizedTreeTests.checkEqual(expected, actual)
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of model save/load
  /////////////////////////////////////////////////////////////////////////////

  test("read/write") {
    def checkModelData(
                        model: OptimizedRandomForestClassificationModel,
                        model2: OptimizedRandomForestClassificationModel): Unit = {
      OptimizedTreeTests.checkEqual(model, model2)
      assert(model.numFeatures === model2.numFeatures)
      assert(model.numClasses === model2.numClasses)
    }

    val rf = new OptimizedRandomForestClassifier().setNumTrees(2)
    val rdd = OptimizedTreeTests.getTreeReadWriteData(sc)

    val allParamSettings = OptimizedTreeTests.allParamSettings ++ Map("impurity" -> "entropy")

    val continuousData: DataFrame =
      OptimizedTreeTests.setMetadata(rdd, Map.empty[Int, Int], numClasses = 2)
    testEstimatorAndModelReadWrite(rf, continuousData, allParamSettings,
      allParamSettings, checkModelData)
  }
}

private object OptimizedRandomForestClassifierSuite extends SparkFunSuite {

  /**
    * Train 2 models on the given dataset, one using the old API and one using the new API.
    * Convert the old model to the new format, compare them, and fail if they are not exactly equal.
    */
  def compareAPIs(
                   data: RDD[Instance],
                   rf: RandomForestClassifier,
                   orf: OptimizedRandomForestClassifier,
                   categoricalFeatures: Map[Int, Int],
                   numClasses: Int): Unit = {
    val numFeatures = data.first().features.size
    val oldPoints = data.map(i => LabeledPoint(i.label, i.features))

    val newData: DataFrame = OptimizedTreeTests.setMetadata(data, categoricalFeatures, numClasses)
    val oldData: DataFrame = OptimizedTreeTests.setMetadataForLabeledPoints(oldPoints, categoricalFeatures, numClasses)

    val oldModel = rf.fit(oldData)
    val optimizedModel = orf.fit(newData)

    // Use parent from newTree since this is not checked anyways.
    OptimizedTreeTests.checkEqualOldClassification(oldModel, optimizedModel)
    assert(optimizedModel.hasParent)
    assert(!optimizedModel.trees.head.asInstanceOf[OptimizedDecisionTreeClassificationModel].hasParent)
    assert(optimizedModel.numClasses === numClasses)
    assert(optimizedModel.numFeatures === numFeatures)
  }
}