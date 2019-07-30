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
import org.apache.spark.ml.param.ParamsSuite
import org.apache.spark.ml.tree.OptimizedLeafNode
import org.apache.spark.ml.tree.impl.OptimizedTreeTests
import org.apache.spark.ml.util.{DefaultReadWriteTest, MLTest, MLTestingUtils}
import org.apache.spark.mllib.tree.{DecisionTreeSuite => OldDecisionTreeSuite}
import org.apache.spark.ml.tree.impl.OptimizedRandomForestSuite
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row}

class OptimizedDecisionTreeClassifierSuite extends MLTest with DefaultReadWriteTest {

  import OptimizedDecisionTreeClassifierSuite.compareAPIs
  import testImplicits._

  private var categoricalDataPointsRDD: RDD[Instance] = _
  private var orderedLabeledPointsWithLabel0RDD: RDD[Instance] = _
  private var orderedLabeledPointsWithLabel1RDD: RDD[Instance] = _
  private var categoricalDataPointsForMulticlassRDD: RDD[Instance] = _
  private var continuousDataPointsForMulticlassRDD: RDD[Instance] = _
  private var categoricalDataPointsForMulticlassForOrderedFeaturesRDD: RDD[Instance] = _

  override def beforeAll() {
    super.beforeAll()
    categoricalDataPointsRDD =
      sc.parallelize(OptimizedRandomForestSuite.generateCategoricalInstances())
    orderedLabeledPointsWithLabel0RDD =
      sc.parallelize(OptimizedRandomForestSuite.generateOrderedInstancesWithLabel0())
    orderedLabeledPointsWithLabel1RDD =
      sc.parallelize(OptimizedRandomForestSuite.generateOrderedInstancesWithLabel1())
    categoricalDataPointsForMulticlassRDD =
      sc.parallelize(OptimizedRandomForestSuite.generateCategoricalInstancesForMulticlass())
    continuousDataPointsForMulticlassRDD =
      sc.parallelize(OptimizedRandomForestSuite.generateContinuousInstancesForMulticlass())
    categoricalDataPointsForMulticlassForOrderedFeaturesRDD = sc.parallelize(
      OptimizedRandomForestSuite.generateCategoricalInstancesForMulticlassForOrderedFeatures())
  }

  test("params") {
    ParamsSuite.checkParams(new OptimizedDecisionTreeClassifier)
    val model = new OptimizedDecisionTreeClassificationModel("dtc", new OptimizedLeafNode(0.0, 0.0), 1, 2)
    ParamsSuite.checkParams(model)
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests calling train()
  /////////////////////////////////////////////////////////////////////////////

  test("Binary classification stump with ordered categorical features") {
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("gini")
      .setMaxDepth(2)
      .setMaxBins(100)
      .setSeed(1)
    val dt = new DecisionTreeClassifier()
      .setImpurity("gini")
      .setMaxDepth(2)
      .setMaxBins(100)
      .setSeed(1)
    val categoricalFeatures = Map(0 -> 3, 1 -> 3)
    val numClasses = 2
    compareAPIs(categoricalDataPointsRDD, dt, odt, categoricalFeatures, numClasses)
  }

  test("Binary classification stump with fixed labels 0,1 for Entropy,Gini") {
    val dt = new DecisionTreeClassifier()
      .setMaxDepth(3)
      .setMaxBins(100)
    val odt = new OptimizedDecisionTreeClassifier()
      .setMaxDepth(3)
      .setMaxBins(100)
    val numClasses = 2
    Array(orderedLabeledPointsWithLabel0RDD, orderedLabeledPointsWithLabel1RDD).foreach { rdd =>
      OptimizedDecisionTreeClassifier.supportedImpurities.foreach { impurity =>
        dt.setImpurity(impurity)
        odt.setImpurity(impurity)
        compareAPIs(rdd, dt, odt, categoricalFeatures = Map.empty[Int, Int], numClasses)
      }
    }
  }

  test("Multiclass classification stump with 3-ary (unordered) categorical features") {
    val rdd = categoricalDataPointsForMulticlassRDD
    val dt = new DecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
    val numClasses = 3
    val categoricalFeatures = Map(0 -> 3, 1 -> 3)
    compareAPIs(rdd, dt, odt, categoricalFeatures, numClasses)
  }

  test("Binary classification stump with 1 continuous feature, to check off-by-1 error") {
    val arr = Array(
      Instance(0.0, 1.0, Vectors.dense(0.0)),
      Instance(1.0, 1.0, Vectors.dense(1.0)),
      Instance(1.0, 1.0, Vectors.dense(2.0)),
      Instance(1.0, 1.0, Vectors.dense(3.0)))
    val rdd = sc.parallelize(arr)
    val dt = new DecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
    val numClasses = 2
    compareAPIs(rdd, dt, odt, categoricalFeatures = Map.empty[Int, Int], numClasses)
  }

  test("Binary classification stump with 2 continuous features") {
    val arr = Array(
      Instance(0.0, 1.0, Vectors.sparse(2, Seq((0, 0.0)))),
      Instance(1.0, 1.0, Vectors.sparse(2, Seq((1, 1.0)))),
      Instance(0.0, 1.0, Vectors.sparse(2, Seq((0, 0.0)))),
      Instance(1.0, 1.0, Vectors.sparse(2, Seq((1, 2.0)))))
    val rdd = sc.parallelize(arr)
    val dt = new DecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
    val numClasses = 2
    compareAPIs(rdd, dt, odt, categoricalFeatures = Map.empty[Int, Int], numClasses)
  }

  test("Multiclass classification stump with unordered categorical features," +
    " with just enough bins") {
    val maxBins = 2 * (math.pow(2, 3 - 1).toInt - 1) // just enough bins to allow unordered features
    val rdd = categoricalDataPointsForMulticlassRDD
    val dt = new DecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(maxBins)
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(maxBins)
    val categoricalFeatures = Map(0 -> 3, 1 -> 3)
    val numClasses = 3
    compareAPIs(rdd, dt, odt, categoricalFeatures, numClasses)
  }

  test("Multiclass classification stump with continuous features") {
    val rdd = continuousDataPointsForMulticlassRDD
    val dt = new DecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(100)
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(100)
    val numClasses = 3
    compareAPIs(rdd, dt, odt, categoricalFeatures = Map.empty[Int, Int], numClasses)
  }

  test("Multiclass classification stump with continuous + unordered categorical features") {
    val rdd = continuousDataPointsForMulticlassRDD
    val dt = new DecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(100)
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(100)
    val categoricalFeatures = Map(0 -> 3)
    val numClasses = 3
    compareAPIs(rdd, dt, odt, categoricalFeatures, numClasses)
  }

  test("Multiclass classification stump with 10-ary (ordered) categorical features") {
    val rdd = categoricalDataPointsForMulticlassForOrderedFeaturesRDD
    val dt = new DecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(100)
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(100)
    val categoricalFeatures = Map(0 -> 10, 1 -> 10)
    val numClasses = 3
    compareAPIs(rdd, dt, odt, categoricalFeatures, numClasses)
  }

  test("Multiclass classification tree with 10-ary (ordered) categorical features," +
    " with just enough bins") {
    val rdd = categoricalDataPointsForMulticlassForOrderedFeaturesRDD
    val dt = new DecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(10)
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(10)
    val categoricalFeatures = Map(0 -> 10, 1 -> 10)
    val numClasses = 3
    compareAPIs(rdd, dt, odt, categoricalFeatures, numClasses)
  }

  test("split must satisfy min instances per node requirements") {
    val arr = Array(
      Instance(0.0, 1.0, Vectors.sparse(2, Seq((0, 0.0)))),
      Instance(1.0, 1.0, Vectors.sparse(2, Seq((1, 1.0)))),
      Instance(0.0, 1.0, Vectors.sparse(2, Seq((0, 1.0)))))
    val rdd = sc.parallelize(arr)
    val dt = new DecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setMinInstancesPerNode(2)
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setMinInstancesPerNode(2)
    val numClasses = 2
    compareAPIs(rdd, dt, odt, categoricalFeatures = Map.empty[Int, Int], numClasses)
  }

  test("do not choose split that does not satisfy min instance per node requirements") {
    // if a split does not satisfy min instances per node requirements,
    // this split is invalid, even though the information gain of split is large.
    val arr = Array(
      Instance(0.0, 1.0, Vectors.dense(0.0, 1.0)),
      Instance(1.0, 1.0, Vectors.dense(1.0, 1.0)),
      Instance(0.0, 1.0, Vectors.dense(0.0, 0.0)),
      Instance(0.0, 1.0, Vectors.dense(0.0, 0.0)))
    val rdd = sc.parallelize(arr)
    val dt = new DecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxBins(2)
      .setMaxDepth(2)
      .setMinInstancesPerNode(2)
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxBins(2)
      .setMaxDepth(2)
      .setMinInstancesPerNode(2)
    val categoricalFeatures = Map(0 -> 2, 1 -> 2)
    val numClasses = 2
    compareAPIs(rdd, dt, odt, categoricalFeatures, numClasses)
  }

  test("split must satisfy min info gain requirements") {
    val arr = Array(
      Instance(0.0, 1.0, Vectors.sparse(2, Seq((0, 0.0)))),
      Instance(1.0, 1.0, Vectors.sparse(2, Seq((1, 1.0)))),
      Instance(0.0, 1.0, Vectors.sparse(2, Seq((0, 1.0)))))
    val rdd = sc.parallelize(arr)

    val dt = new DecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setMinInfoGain(1.0)
    val odt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(2)
      .setMinInfoGain(1.0)
    val numClasses = 2
    compareAPIs(rdd, dt, odt, categoricalFeatures = Map.empty[Int, Int], numClasses)
  }

  test("predictRaw and predictProbability") {
    val rdd = continuousDataPointsForMulticlassRDD
    val dt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(100)
    val categoricalFeatures = Map(0 -> 3)
    val numClasses = 3

    val newData: DataFrame = OptimizedTreeTests.setMetadata(rdd, categoricalFeatures, numClasses)
    val newTree = dt.fit(newData)

    MLTestingUtils.checkCopyAndUids(dt, newTree)

    testTransformer[(Vector, Double, Double)](newData, newTree,
      "prediction", "rawPrediction", "probability") {
      case Row(pred: Double, rawPred: Vector, probPred: Vector) =>
        assert(pred === rawPred.argmax,
          s"Expected prediction $pred but calculated ${rawPred.argmax} from rawPrediction.")
        val sum = rawPred.toArray.sum
        assert(Vectors.dense(rawPred.toArray.map(_ / sum)) === probPred,
          "probability prediction mismatch")
    }

    ProbabilisticClassifierSuite.testPredictMethods[
      Vector, OptimizedDecisionTreeClassificationModel](this, newTree, newData)
  }

  test("prediction on single instance") {
    val rdd = continuousDataPointsForMulticlassRDD
    val dt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(4)
      .setMaxBins(100)
    val categoricalFeatures = Map(0 -> 3)
    val numClasses = 3

    val newData: DataFrame = OptimizedTreeTests.setMetadata(rdd, categoricalFeatures, numClasses)
    val newTree = dt.fit(newData)

    testPredictionModelSinglePrediction(newTree, newData)
  }

  test("training with 1-category categorical feature") {
    val data = sc.parallelize(Seq(
      Instance(0, 1.0, Vectors.dense(0, 2, 3)),
      Instance(1, 1.0, Vectors.dense(0, 3, 1)),
      Instance(0, 1.0, Vectors.dense(0, 2, 2)),
      Instance(1, 1.0, Vectors.dense(0, 3, 9)),
      Instance(0, 1.0, Vectors.dense(0, 2, 6))
    ))
    val df = OptimizedTreeTests.setMetadata(data, Map(0 -> 1), 2)
    val dt = new OptimizedDecisionTreeClassifier().setMaxDepth(3)
    dt.fit(df)
  }

  test("should support all NumericType labels and not support other types") {
    val dt = new OptimizedDecisionTreeClassifier().setMaxDepth(1)
    MLTestingUtils.checkNumericTypes[OptimizedDecisionTreeClassificationModel, OptimizedDecisionTreeClassifier](
      dt, spark) { (expected, actual) =>
        OptimizedTreeTests.checkEqual(expected, actual)
      }
  }

  test("Fitting without numClasses in metadata") {
    val df: DataFrame = OptimizedTreeTests.featureImportanceData(sc).toDF()
    val dt = new OptimizedDecisionTreeClassifier().setMaxDepth(1)
    dt.fit(df)
  }

  /////////////////////////////////////////////////////////////////////////////
  // Tests of model save/load
  /////////////////////////////////////////////////////////////////////////////

  test("read/write") {
    def checkModelData(
        model: OptimizedDecisionTreeClassificationModel,
        model2: OptimizedDecisionTreeClassificationModel): Unit = {
      OptimizedTreeTests.checkEqual(model, model2)
      assert(model.numFeatures === model2.numFeatures)
      assert(model.numClasses === model2.numClasses)
    }

    val dt = new OptimizedDecisionTreeClassifier()
    val rdd = OptimizedTreeTests.getTreeReadWriteData(sc)

    val allParamSettings = OptimizedTreeTests.allParamSettings ++ Map("impurity" -> "entropy")

    // Categorical splits with tree depth 2
    val categoricalData: DataFrame =
      OptimizedTreeTests.setMetadata(rdd, Map(0 -> 2, 1 -> 3), numClasses = 2)
    testEstimatorAndModelReadWrite(dt, categoricalData, allParamSettings,
      allParamSettings, checkModelData)

    // Continuous splits with tree depth 2
    val continuousData: DataFrame =
      OptimizedTreeTests.setMetadata(rdd, Map.empty[Int, Int], numClasses = 2)
    testEstimatorAndModelReadWrite(dt, continuousData, allParamSettings,
      allParamSettings, checkModelData)

    // Continuous splits with tree depth 0
    testEstimatorAndModelReadWrite(dt, continuousData, allParamSettings ++ Map("maxDepth" -> 0),
      allParamSettings ++ Map("maxDepth" -> 0), checkModelData)
  }

  test("SPARK-20043: " +
       "ImpurityCalculator builder fails for uppercase impurity type Gini in model read/write") {
    val rdd = OptimizedTreeTests.getTreeReadWriteData(sc)
    val data: DataFrame =
      OptimizedTreeTests.setMetadata(rdd, Map.empty[Int, Int], numClasses = 2)

    val dt = new OptimizedDecisionTreeClassifier()
      .setImpurity("Gini")
      .setMaxDepth(2)
    val model = dt.fit(data)

    testDefaultReadWrite(model)
  }
}

private[ml] object OptimizedDecisionTreeClassifierSuite extends SparkFunSuite {

  /**
   * Train 2 decision trees on the given dataset, one using the old API and one using the new API.
   * Convert the old tree to the new format, compare them, and fail if they are not exactly equal.
   */
  def compareAPIs(
      instances: RDD[Instance],
      dt: DecisionTreeClassifier,
      odt: OptimizedDecisionTreeClassifier,
      categoricalFeatures: Map[Int, Int],
      numClasses: Int): Unit = {
    val numFeatures = instances.first().features.size
    val oldDataPoints = instances.map(p => LabeledPoint(p.label, p.features))

    val newData: DataFrame = OptimizedTreeTests.setMetadataForLabeledPoints(oldDataPoints, categoricalFeatures, numClasses)
    val optimizedData: DataFrame = OptimizedTreeTests.setMetadata(instances, categoricalFeatures, numClasses)

    val newTree = dt.fit(newData)
    val optimizedTree = odt.fit(optimizedData)
    // Use parent from newTree since this is not checked anyways.
    OptimizedTreeTests.checkEqual(newTree, optimizedTree)
    assert(newTree.numFeatures === numFeatures)
    assert(optimizedTree.numFeatures === numFeatures)
  }
}
