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

import org.apache.spark.ml.tree.OptimizedLearningNode
import org.apache.spark.mllib.tree.configuration.TimePredictionStrategy

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Represents a single decision tree Node selected to be split locally.
  *
  * @param node OptimizedLearningNode to split
  * @param treeIndex index of its corresponding tree in the ensemble
  * @param rows number of data points in the data subset corresponding to the node
  * @param impurity the impurity value in the current data subset corresponding to the node
  */
case class LocalTrainingTask(node: OptimizedLearningNode,
                             treeIndex: Int,
                             rows: Long,
                             impurity: Double) {

  /**
    * Computes the expected time requirement of the task based on a given time prediction strategy.
    * The computed time is a value relative to other tasks, not the actual time in seconds.
    *
    * @param timePredictionStrategy strategy to calculate expected time requirements
    * @return Double time prediction (relative value)
    */
  private[impl] def computeTimePrediction(timePredictionStrategy: TimePredictionStrategy)
  : Double = {
    timePredictionStrategy.predict(rows, impurity)
  }
}
object LocalTrainingTask {
  // Implicit ordering of the local training tasks in decreasing order based on its data size --
  // we to pack the largest tasks first (greedy first-fit descending bin-packing).
  implicit val orderingByRows: Ordering[LocalTrainingTask] =
    Ordering.by((task: LocalTrainingTask) => task.rows).reverse
}

/**
  * Represents a set of LocalTrainingTasks to be processed together on one executor.
  * (i.e. the total memory requirements of all tasks in the bin is below the local training threshold)
  *
  * @param maxRows the maximum number of data points which fit in this bin
  * @param timePredictionStrategy strategy to calculate expected time requirements
  */
class LocalTrainingBin(val maxRows: Long,
                       timePredictionStrategy: TimePredictionStrategy) {
  var currentRows: Long = 0
  var tasks: ListBuffer[LocalTrainingTask] = mutable.ListBuffer[LocalTrainingTask]()
  var totalTimePrediction: Double = 0

  /**
    * Attempts to add the task into the LocalTrainingBin and returns whether the action succeeded.
    * @param task LocalTrainingTask
    * @return true if task was succesfully added / false if the task couldn't fit anymore
    */
  def fitTask(task: LocalTrainingTask): Boolean = {
    if (currentRows + task.rows <= maxRows) {
      tasks += task
      currentRows += task.rows
      totalTimePrediction += task.computeTimePrediction(timePredictionStrategy)
      return true
    }
    false
  }
}

object LocalTrainingBin {
  // Implicit ordering of the bins -- we want to process the bins that are expected to take
  // the longest time during the earliest batches of the local training process.
  implicit val orderingByTimePrediction: Ordering[LocalTrainingBin] =
    Ordering.by((bin: LocalTrainingBin) => bin.totalTimePrediction).reverse
}

/**
  *
  * @param maxBinRows
  * @param timePredictionStrategy
  * @param maxTasksPerBin
  */
class LocalTrainingPlan(val maxBinRows: Long,
                        val timePredictionStrategy: TimePredictionStrategy,
                        val maxTasksPerBin: Int) {
  var bins: mutable.ListBuffer[LocalTrainingBin] = mutable.ListBuffer[LocalTrainingBin]()

  /**
    * Schedules the LocalTrainingTask into the first available LocalTrainingBin, or creates a new
    * one if it doesn't fit into any of them.
    *
    * @param task LocalTrainingTask
    */
  def scheduleTask(task: LocalTrainingTask): Unit = {
    bins.find(bin => bin.tasks.size < maxTasksPerBin && bin.fitTask(task)).getOrElse {
      val newBin = new LocalTrainingBin(maxBinRows, timePredictionStrategy)
      newBin.fitTask(task)
      bins += newBin
    }
  }
}
