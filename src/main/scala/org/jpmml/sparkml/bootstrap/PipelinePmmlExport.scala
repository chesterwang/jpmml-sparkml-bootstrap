package org.jpmml.sparkml.bootstrap

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.mllib.linalg.Vectors
import org.jpmml.model.MetroJAXBUtil
import org.jpmml.sparkml.ConverterUtil

/**
  * Created by chester on 16-9-16.
  */
object PipelinePmmlExport extends SparkPrepare {

  def main(args: Array[String]) {
    val data = sqlContext.createDataFrame(
      Seq(
        (1.0, 2.0, 3.0,"xiao",1.0),
        (4.0, 6.0, 6.0,"xiao",1.0),
        (7.0, 8.0, 9.0,"tuo",0.0),
        (5.0, 9.0, 10.0,"xxx",0.0)
      )
    )
    val df = data.toDF("a","b","c","d","e")


    val buck = new Bucketizer()
      .setInputCol("a")
      .setSplits(Array(-10,0,2,5,6,9,10))
      .setOutputCol("buck_a")

    val bin = new Binarizer()
      .setInputCol("b")
      .setThreshold(5.0)
      .setOutputCol("bin_b")

    val sim = new StringIndexerModel(Array("xiao","tuo","xxx"))
      .setInputCol("d")
      .setOutputCol("si_d")

    val assembler = new VectorAssembler()
      .setInputCols(Array("buck_a","bin_b","si_d"))
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setFitIntercept(true)
      .setLabelCol("e")
      .setFeaturesCol("features")

    val pipeline = new Pipeline()
      .setStages(Array(buck,bin,sim,assembler, lr))


    val pipelineModel = pipeline.fit(df);

    val pmml = ConverterUtil.toPMML(df.schema, pipelineModel);
    println(pmml.toString)
    MetroJAXBUtil.marshalPMML(pmml, System.out)

    ConverterUtil.createFeatureConverter()




//    sqlContext.createDataFrame()

  }

}
