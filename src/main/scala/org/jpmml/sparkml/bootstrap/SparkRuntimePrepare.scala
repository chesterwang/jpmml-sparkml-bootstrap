package org.jpmml.sparkml.bootstrap

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/** 准备sparkcontext sqlcontext 运行环境
  * Created by chester on 16-9-15.
  */
trait SparkRuntimePrepare {

  val sparkconf = new SparkConf().setMaster("local[*]").setAppName("pmml")
  val sc = new SparkContext(sparkconf)
  sc.setLogLevel("ERROR")
  val sqlContext = new SQLContext(sc)
}
