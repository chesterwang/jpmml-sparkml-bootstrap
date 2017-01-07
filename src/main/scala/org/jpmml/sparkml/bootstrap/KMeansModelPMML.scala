package org.jpmml.sparkml.bootstrap

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

/**
  * Created by chester on 16-9-15.
  */
object KMeansModelPMML extends SparkRuntimePrepare{
  def main(args: Array[String]) {
//    val sparkconf = new SparkConf().setMaster("local[*]").setAppName("asdf")
//    val sc = new SparkContext(sparkconf)

    val data = sc.textFile("data/kmeans_data.txt")
    val parsedData = data.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
    val numClusters = 2
    val numIterations = 20
    val clusters = KMeans.train(parsedData, numClusters, numIterations)


    // Export to PMML to a String in PMML format
//    println("PMML Model:\n" + clusters.toPMML)

    // Export the model to a local file in PMML format
    clusters.toPMML("data/kmeans.xml") //todo:好的处理方法，两个工具类函数,一个local一个hdfs

    // Export the model to a directory on a distributed file system in PMML format
//    clusters.toPMML(sc, "/tmp/kmeans")

    // Export the model to the OutputStream in PMML format
//    clusters.toPMML(System.out)



  }

}
