package de.mpii.spark

import java.io.{ByteArrayInputStream, DataInputStream}
import java.util

import de.mpii.GlobalConstants
import de.mpii.mr.input.clueweb.ClueWebWarcRecord
import de.mpii.mr.utils.CleanContentTxt

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.SparkContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory
import org.jsoup.Jsoup
import org.jsoup.nodes.Document

import scala.collection.immutable.Iterable
import scala.collection.mutable


/**
  * Created by khui on 04/11/15.
  */
class VectorizeDoc extends java.io.Serializable{


  def docstr2Termids(doccontent: String, collectionLen: Int, termIdxDfCf: Map[String, (Int, Int, Int)], tokenizer: Option[TokenizerFactory]=None): Array[Int] ={
    val docterms = if (tokenizer.isEmpty) doccontent.split(" ") else tokenizer.get.create(doccontent).getTokens.toArray(Array[String]())
    val termids = docterms
      .filter(termIdxDfCf.contains)
      .map(v => termIdxDfCf(v)._1)
      .distinct
    termids
  }

  def path2qid(path: String) = {
    new Path(path).getName.split("-")(0).toInt
  }

  def qid2year(qid: Int): (String, Int) = {
    qid match {
      case x if 1 until 201 contains x => ("cw09", GlobalConstants.CARDINALITY_CW09)
      case x if 201 until 301 contains x => ("cw12", GlobalConstants.CARDINALITY_CW12)
    }
  }
  def qid2yearrecord(qid: Int): (String, Int, Int, String) = {
    qid match {
      case x if 1 to 200 contains x => ("cw09",
        GlobalConstants.CARDINALITY_CW09,
        GlobalConstants.DOCFREQUENCY_CW09,
        "de.mpii.mr.input.clueweb.ClueWeb09WarcRecord")
      case x if 201 to 300 contains x => ("cw12",
        GlobalConstants.CARDINALITY_CW12,
        GlobalConstants.DOCFREQUENCY_CW12,
        "de.mpii.mr.input.clueweb.ClueWeb12WarcRecord")
    }
  }

}

class ParseCwdoc{

  val cleanpipeline: CleanContentTxt = new CleanContentTxt(64)

  def bytes2Docstr(queryid: Int, doc: Array[Byte], cwrecord: String): (Int, String, String) = {
    val cwdoc: ClueWebWarcRecord = Class.forName(cwrecord).newInstance().asInstanceOf[ClueWebWarcRecord]
    cwdoc.readFields(new DataInputStream(new ByteArrayInputStream(doc)))
    println(queryid + " " + cwdoc.getDocid)
    val webpage: Document = Jsoup.parse(cwdoc.getContent)
    if (webpage.body != null) {
      (queryid, cwdoc.getDocid, cleanpipeline.cleanTxtStr(webpage, cwdoc.getDocid, true))
    } else {
      (queryid, cwdoc.getDocid, null)
    }

  }

}


trait DocVec{


  def mkstring(cwid: String, termtfidf: Map[Int, Double]): String = {
    termtfidf
      .toArray
      .map(t2 => t2._1->Math.round(t2._2 * 1E8) * 1E-8 )
      .filter(t2 => t2._2 != 0)
      .sortBy(t2 => t2._1)
      .map(t2 => t2._1 + ":" + "%.4E".format(t2._2))
      .+:(cwid)
      .mkString(" ")
  }

  def recursiveListFiles(f: String): Array[String] = {
    val fs = FileSystem.get(new Configuration())
    val status_list: Array[String] = fs.listStatus(new Path(f))
      .map(s => s.getPath)
      .filter(p => !p.getName.startsWith("_"))
      .map(_.toString)
    status_list
  }

}
