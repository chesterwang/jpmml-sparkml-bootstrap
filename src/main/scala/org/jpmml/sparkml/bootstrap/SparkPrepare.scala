package org.jpmml.sparkml.bootstrap

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by chester on 16-9-15.
  */
trait SparkPrepare {

  val sparkconf = new SparkConf().setMaster("local[*]").setAppName("pmml")
  val sc = new SparkContext(sparkconf)
  val sqlContext = new SQLContext(sc)

}
