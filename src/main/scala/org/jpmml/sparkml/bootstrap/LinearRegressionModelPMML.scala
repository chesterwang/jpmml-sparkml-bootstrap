package org.jpmml.sparkml.bootstrap

import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, SVMModel, SVMWithSGD}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LassoWithSGD, LinearRegressionWithSGD, RidgeRegressionWithSGD}

/**
  * Created by chester on 16-9-15.
  */
object LinearRegressionModelPMML extends SparkRuntimePrepare {
  def main(args: Array[String]) {

    val lp = Array[LabeledPoint](
      LabeledPoint(3, Vectors.dense(Array(3.0))),
      LabeledPoint(1, Vectors.dense(Array(1.0))),
      LabeledPoint(0, Vectors.dense(Array(0.2))),
      LabeledPoint(-1, Vectors.dense(Array(-1.0))),
      LabeledPoint(-5, Vectors.dense(Array(-4.0)))
    )

    val testRDD = sc.parallelize(lp)

    val linReg = new LinearRegressionWithSGD().setIntercept(true)
    linReg.optimizer.setNumIterations(1000).setStepSize(1.0)

    val model = linReg.run(testRDD)

    model.toPMML("data/LinearRegression.xml")

    val model2 = new RidgeRegressionWithSGD().setIntercept(true).run(testRDD)
    model2.toPMML("data/RidgeRegression.xml")

    val model3 = new LassoWithSGD().setIntercept(true).run(testRDD)
    model3.toPMML("data/LassoRegression.xml")


    val testData2 = sc.parallelize(
      Array[LabeledPoint](
        LabeledPoint(1, Vectors.dense(Array(3.0))),
        LabeledPoint(1, Vectors.dense(Array(1.0))),
        LabeledPoint(1, Vectors.dense(Array(0.2))),
        LabeledPoint(0, Vectors.dense(Array(-1.0))),
        LabeledPoint(0, Vectors.dense(Array(-4.0)))
      )
    )

    val model4 = new SVMWithSGD().setIntercept(true).run(testData2)
    model4.toPMML("data/svm.xml")

    val lrsgd = new LogisticRegressionWithSGD().setIntercept(true)
    lrsgd.optimizer
      .setStepSize(10.0)
      .setRegParam(0.0)
      .setNumIterations(10)

    val model5 = lrsgd.run(testData2)
    model5.toPMML("data/logistic.xml")

  }

}
