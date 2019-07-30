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
import org.apache.spark.ml.feature.{Instance, LabeledPoint}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.tree._
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext

/** Unit tests for helper classes/methods specific to local tree training */
class LocalTreeUnitSuite
  extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest {

  test("Fit a single decision tree regressor on constant features") {
    // Generate constant, continuous data
    val data = sc.parallelize(Range(0, 8).map(_ => Instance(1, 1.0, Vectors.dense(1))))
    val df = spark.sqlContext.createDataFrame(data)
    // Initialize estimator
    val dt = new LocalDecisionTreeRegressor()
      .setFeaturesCol("features") // indexedFeatures
      .setLabelCol("label")
      .setMaxDepth(3)
    // Fit model
    val model = dt.fit(df)
    assert(model.rootNode.isInstanceOf[OptimizedLeafNode])
    val root = model.rootNode.asInstanceOf[OptimizedLeafNode]
    assert(root.prediction == 1)
  }

  test("Fit a single decision tree regressor on some continuous features") {
    // Generate continuous data
    val data = sc.parallelize(Range(0, 8).map(x => Instance(x, 1.0, Vectors.dense(x))))
    val df = spark.createDataFrame(data)
    // Initialize estimator
    val dt = new LocalDecisionTreeRegressor()
      .setFeaturesCol("features") // indexedFeatures
      .setLabelCol("label")
      .setMaxDepth(3)
    // Fit model
    val model = dt.fit(df)

    // Check that model is of depth 3 (the specified max depth) and that leaf/internal nodes have
    // the correct class.
    // Validate root
    assert(model.rootNode.isInstanceOf[OptimizedInternalNode])
    // Validate first level of tree (nodes with depth = 1)
    val root = model.rootNode.asInstanceOf[OptimizedInternalNode]
    assert(root.leftChild.isInstanceOf[OptimizedInternalNode] && root.rightChild.isInstanceOf[OptimizedInternalNode])
    // Validate second and third levels of tree (nodes with depth = 2 or 3)
    val left = root.leftChild.asInstanceOf[OptimizedInternalNode]
    val right = root.rightChild.asInstanceOf[OptimizedInternalNode]
    val grandkids = Array(left.leftChild, left.rightChild, right.leftChild, right.rightChild)
    grandkids.foreach { grandkid =>
      assert(grandkid.isInstanceOf[OptimizedInternalNode])
      val grandkidNode = grandkid.asInstanceOf[OptimizedInternalNode]
      assert(grandkidNode.leftChild.isInstanceOf[OptimizedLeafNode])
      assert(grandkidNode.rightChild.isInstanceOf[OptimizedLeafNode])
    }
  }

  test("Fit deep local trees") {

    /**
     * Deep tree test. Tries to fit tree on synthetic data designed to force tree
     * to split to specified depth.
     */
    def deepTreeTest(depth: Int): Unit = {
      val deepTreeData = OptimizedTreeTests.deepTreeData(sc, depth)
      val df = spark.createDataFrame(deepTreeData)
      // Construct estimators; single-tree random forest & decision tree regressor.
      val localTree = new LocalDecisionTreeRegressor()
        .setFeaturesCol("features") // indexedFeatures
        .setLabelCol("label")
        .setMaxDepth(depth)
        .setMinInfoGain(0.0)

      // Fit model, check depth...
      val localModel = localTree.fit(df)
      assert(localModel.rootNode.subtreeDepth == depth)
    }

    // Test small depth tree
    deepTreeTest(10)
    // Test medium depth tree
    deepTreeTest(40)
    // Test high depth tree
    deepTreeTest(200)
  }

}
