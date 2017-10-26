package de.mpii.spark.task

import java.io.{ByteArrayInputStream, DataInputStream}

import de.mpii.mr.input.clueweb.ClueWebWarcRecord
import de.mpii.mr.utils.CleanContentTxt
import de.mpii.GlobalConstants
import de.mpii.spark.{DocVec, VectorizeDoc}


import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{BytesWritable, IntWritable}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.jsoup.Jsoup

/**
  * Created by khui on 10/26/15.
  */
object ExtractCWTxt {
  private var inputdir: String = ""
  private var outputdir: String = ""
  private var parNumber: Int = 32
  private var cwyear: String = "cw6y"

  def parse(args: List[String]): Unit = args match {
    case ("-i") :: dir :: tail =>
      inputdir = dir
      parse(tail)
    case ("-o") :: file :: tail =>
      outputdir = file
      parse(tail)
    case ("-c") :: coreNum :: tail =>
      parNumber = coreNum.toInt
      parse(tail)
    case ("-y") :: year :: tail =>
      cwyear = year
      parse(tail)
    case (Nil) =>
  }


  def main(args: Array[String]) {
    parse(args.toList)
    val conf: SparkConf = new SparkConf()
      .setAppName("Extract cwid-doccontent for wt")
      .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .set("spark.local.dir", GlobalConstants.TMP_DIR)
      .set("spark.akka.frameSize", "256")
      .set("spark.kryoserializer.buffer.max.mb", "1024")
      .registerKryoClasses(
        Array(classOf[String],
          classOf[(String, String)],
          classOf[(Int, String)],
          classOf[(String, Double)],
          classOf[Map[String, (Map[String, (Int, Int, Int)], Int, String)]],
          classOf[(Int, Int, Int)],
          classOf[Map[String, (Int, Int, Int)]],
          classOf[Seq[(Int, String, String)]],
          classOf[VectorizeDoc],
          classOf[(Int, String, String)]
        ))
    val sc: SparkContext = new SparkContext(conf)

    // check existence of output directory
    val fs: FileSystem = FileSystem
      .get(sc.hadoopConfiguration)


    if (fs.exists(new Path(outputdir))) {
      println(outputdir + " is being removed")
      fs.delete(new Path(outputdir), true)
    }


    val pathList: List[String] = fs
      .listStatus(new Path(inputdir))
      .filter(f => !f.getPath.getName.startsWith("_"))
      .map(f => f.getPath.toString)
      .toList

    val qids =
      cwyear match {
        case "cw09" => 101 to 200
        case "cw12" => 201 to 300
        case "cw6y" => 1 to 300
        case "cwtest" => 101 to 105
        case "cw4y" => 101 to 300
      }


    val years =
      cwyear match {
        case "cw09" => Array("cw09")
        case "cw12" => Array("cw12")
        case "cwtest" => Array("cw09")
        case "cw4y" => Array("cw09", "cw12")
        case "cw6y" => Array("cw09", "cw12")
      }


    val psf: VectorizeDoc = new VectorizeDoc()
    val cleanpipeline: CleanContentTxt = new CleanContentTxt(64)


    pathList
      .par
      .map(p => (p, psf.path2qid(p)))
      .filter(t2 => qids.toSet.contains(t2._2))
      .foreach {
        pathQid => {
          val path = pathQid._1
          val qid = pathQid._2
          val (year, collen,docnum, cwrecord) = psf.qid2yearrecord(qid)
          val parNumber = this.parNumber

          val cwidTerms: Array[(String, Array[String])] =
            sc
              // read in for each query
              .sequenceFile(path, classOf[IntWritable], classOf[BytesWritable], parNumber)
              .map { kv =>
                val cwdoc: ClueWebWarcRecord = Class.forName(cwrecord).newInstance().asInstanceOf[ClueWebWarcRecord]
                cwdoc.readFields(new DataInputStream(new ByteArrayInputStream(kv._2.getBytes)))
                (kv._1.get(), cwdoc.getDocid, cwdoc.getContent)
              }
              .filter(v => v._3 != null)
              .collect()
              .map { case (q, cwid, doccontent) => (q, cwid, Jsoup.parse(doccontent)) }
              .filter(v => v._3.body() != null)
              .filter(v => v._3.body().text().length < GlobalConstants.MAX_DOC_LENGTH)
              .map { case (q, cwid, htmlDoc) => (q, cwid, cleanpipeline.clean2tokens(htmlDoc, cwid, true)) }
              .filter(v => v._3 != null)
              .map(t3 => t3._2 -> t3._3.toArray[String](Array.empty[String]))
              .map { case (cwid, terms) =>
                cwid ->
                  terms
                    .filter(term => term.length < GlobalConstants.MAX_TOKEN_LENGTH)
              }

            sc
              .parallelize(cwidTerms, 1)
              .map{case(cwid, terms) => terms.+:(cwid).mkString(" ")}
              .saveAsTextFile(outputdir + "/" + qid)


        }
      }
  }
}


