package de.mpii.spark.task

import java.io.{ByteArrayInputStream, DataInputStream}

import de.mpii.mr.input.clueweb.ClueWebWarcRecord
import de.mpii.spark.{DocVec, VectorizeDoc}
import de.mpii.GlobalConstants

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.hadoop.io.{BytesWritable, IntWritable}
import org.apache.spark.{SparkConf, SparkContext}


object ExtractCWHtmls {
  private var inputdir: String = ""
  private var outputdir: String = ""
  private var parNumber: Int = 32
  private var cwyear: String = "cw6y"
  private var task: String = "adhoc" // diversity

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
    case ("-s") :: session :: tail =>
      task = session
      parse(tail)
    case ("-y") :: year :: tail =>
      cwyear = year
      parse(tail)
    case (Nil) =>
  }


  def main(args: Array[String]) {
    parse(args.toList)
    val conf: SparkConf = new SparkConf()
      .setAppName("Extract html from clueweb: " + cwyear + " " + task)
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

    val pathList: List[String] = fs
      .listStatus(new Path(inputdir))
      .filter(f => !f.getPath.getName.startsWith("_"))
      .map(f => f.getPath.toString)
      .toList

    val qids =
      cwyear match {
        case "cw09" => 101 to 200
        case "cw12" => 201 to 300
        case "cw4y" => 101 to 300
        case "cw6y" => 1 to 300
        case "cwtest" => 101 to 105
      }

    val psf: VectorizeDoc = new VectorizeDoc()


    pathList
      .par
      .map(p => (p, psf.path2qid(p)))
      .filter(t2 => qids.toSet.contains(t2._2))
      .foreach{
        pathQid => {
          val path = pathQid._1
          val qid = pathQid._2
          val (year, collen, docnum, cwrecord) = psf.qid2yearrecord(qid)
          val qidcwidhtml =
          sc
            // read in for each query
            .sequenceFile(path, classOf[IntWritable], classOf[BytesWritable], parNumber)
            .map { kv =>
              val cwdoc: ClueWebWarcRecord = Class.forName(cwrecord).newInstance().asInstanceOf[ClueWebWarcRecord]
              cwdoc.readFields(new DataInputStream(new ByteArrayInputStream(kv._2.getBytes)))
              (kv._1.get(), cwdoc.getDocid, cwdoc.getContent)
            }
            .filter(v => v._3 != null)
          val qidcwids = qidcwidhtml.map{case(qid, cwid, html)=>(qid, cwid)}.collect()
          qidcwids
            .par
            .foreach{case(queryid, docid) =>
              qidcwidhtml
                .filter{case(q, d, html)=> queryid==q && docid==d}
                .map(v => v._3)
                .coalesce(1)
                .saveAsTextFile(outputdir + "/" + queryid + "/" + docid)
            }
          System.out.println("Finished:" + year +  "_"  + qid + "_" + qidcwids.length)
        }

      }
  }
}

