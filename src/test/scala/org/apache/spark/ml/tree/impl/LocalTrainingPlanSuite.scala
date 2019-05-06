/*
 * Copyright (C) 2019 Cisco Systems
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.tree.impl

import org.apache.spark.SparkFunSuite
import org.apache.spark.ml.tree.OptimizedLearningNode
import org.apache.spark.mllib.tree.configuration.{DefaultTimePredictionStrategy, TimePredictionStrategy}
import org.apache.spark.mllib.util.MLlibTestSparkContext

class LocalTrainingPlanSuite extends SparkFunSuite with MLlibTestSparkContext {

  val timePredictonStrategy: TimePredictionStrategy = new DefaultTimePredictionStrategy

  test("memory restriction") {
    val plan = new LocalTrainingPlan(10, timePredictonStrategy, Int.MaxValue)

    plan.scheduleTask(new LocalTrainingTask(node = OptimizedLearningNode.emptyNode(1),
      treeIndex = 1, rows = 2, impurity = 1.0))
    plan.scheduleTask(new LocalTrainingTask(node = OptimizedLearningNode.emptyNode(1),
      treeIndex = 1, rows = 2, impurity = 1.0))
    plan.scheduleTask(new LocalTrainingTask(node = OptimizedLearningNode.emptyNode(1),
      treeIndex = 1, rows = 2, impurity = 1.0))

    plan.scheduleTask(new LocalTrainingTask(node = OptimizedLearningNode.emptyNode(1),
      treeIndex = 1, rows = 9, impurity = 1.0))

    assert(plan.bins.length == 2)
    assert(plan.bins.head.tasks.length == 3)
    assert(plan.bins(1).tasks.length == 1)
  }

  test("count restriction") {
    val plan = new LocalTrainingPlan(10, timePredictonStrategy, 2)

    plan.scheduleTask(new LocalTrainingTask(node = OptimizedLearningNode.emptyNode(1),
      treeIndex = 1, rows = 2, impurity = 1.0))
    plan.scheduleTask(new LocalTrainingTask(node = OptimizedLearningNode.emptyNode(1),
      treeIndex = 1, rows = 2, impurity = 1.0))
    plan.scheduleTask(new LocalTrainingTask(node = OptimizedLearningNode.emptyNode(1),
      treeIndex = 1, rows = 2, impurity = 1.0))

    assert(plan.bins.length == 2)
    assert(plan.bins.head.tasks.length == 2)
    assert(plan.bins(1).tasks.length == 1)
  }

  test("task implicit ordering by memory usage descending") {
    val l = List(new LocalTrainingTask(node = OptimizedLearningNode.emptyNode(1),
      treeIndex = 1, rows = 1, impurity = 1.0),
      new LocalTrainingTask(node = OptimizedLearningNode.emptyNode(1),
      treeIndex = 2, rows = 5, impurity = 1.0),
      new LocalTrainingTask(node = OptimizedLearningNode.emptyNode(1),
      treeIndex = 3, rows = 3, impurity = 1.0)
    )

    val sorted = l.sorted

    assert(sorted.head.treeIndex == 2)
  }
}
