package org.jpmml.sparkml.bootstrap

import java.io.{File, FileOutputStream}

import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.{Pipeline }
import org.apache.spark.ml.feature._
import org.jpmml.model.{JAXBUtil, MetroJAXBUtil}
import org.jpmml.sparkml.ConverterUtil
import javax.xml.bind.JAXBContext
import javax.xml.transform.stream.StreamSource

import org.dmg.pmml.PMML

import scala.collection.JavaConversions._
import scala.collection.JavaConverters._

/**
  * pipeline的转化过程导出为pmml
  * Created by chester on 16-9-16.
  */
object PipelinePmmlExport extends SparkRuntimePrepare {

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

    val pipeline= getSamplePipeline()
    val pipelineModel = pipeline.fit(df)

    val pmml = ConverterUtil.toPMML(df.schema, pipelineModel);
    MetroJAXBUtil.marshalPMML(pmml, new FileOutputStream("data/lr_pipeline.xml"))
    val context = JAXBContext.newInstance(classOf[PMML])
    val unm = context.createUnmarshaller()
    val parsedPmml = unm.unmarshal(new File("data/lr_pipeline.xml")).asInstanceOf[PMML]
    println(
      s"""模型描述
        |copyright:${parsedPmml.getHeader.getCopyright}
        |description: ${parsedPmml.getHeader.getDescription}
      """.stripMargin)

    println("---------pmml中的数据字典")
    parsedPmml.getDataDictionary.getDataFields.asScala
      .foreach{
        x =>
          println(x.getName + " " + x.getDataType.value())
      }
    parsedPmml.getTransformationDictionary()


    println("---------模型列表描述")
    parsedPmml.getModels.asScala
      .foreach{
        x =>
          println(s"算法名字: ${x.getAlgorithmName}")
          println(s"函数名字: ${x.getFunctionName.value()}")
          println(s"模型类名: ${x.getClass.getCanonicalName}")
      }

    val hhh = JAXBUtil.unmarshalPMML(new StreamSource(new File("data/lr_pipeline.xml")))
    hhh.getModels.asScala
      .foreach{
        x =>
          println(s"模型名字:${x.getModelName}")
          println(s"算法名字:${x.getAlgorithmName}")
          val fields = x.getMiningSchema.getMiningFields.map{y => y.getName.toString}
          println(s"字典名字列表:$fields")
      }
  }

  def getSamplePipeline()  ={

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
    pipeline


  }

}
