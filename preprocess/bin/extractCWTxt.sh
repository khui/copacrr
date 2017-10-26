#!/usr/bin/env bash


libdir=$(readlink -f ../java/extractcwdoc/dependency)
javaworkspace=$(readlink -f ../java/extractcwdoc/target)
jdk=$JAVA_HOME
tmpdir=$TMPDIR
year=cw6y #cw09 cw12

cwdocdir=/user/data/ExtractCwdocs/Extracted/cwbinary_ql_qrel_runs/cw6y
outputdir=/user/data/ExtractCwdocs/Extracted/cwtxt_ql_qrel_runs

jarsClasspath=`find $libdir -name "*.jar"|tr '\n' ','`
classname="de.mpii.spark.task.ExtractCWTxt"


START=$(date +%s.%N)

islocal=$1
if ! $islocal
then
	echo run on the cluster
	# for cluster
	spark-submit \
	--class $classname \
	--executor-memory 20G \
	--driver-memory 40G \
	--driver-java-options "-Djava.io.tmpdir=$tmpdir" \
	--executor-cores 8 \
	--num-executors 16 \
	--master yarn://139.19.52.105 \
	--jars $jarsClasspath \
	$javaworkspace/extractcwdoc.jar \
	-i $cwdocdir \
	-o $outputdir \
	-y $year \
	-c 16
else
	echo run locally
	# for local
	spark-submit \
	--driver-memory 40G \
	--driver-java-options "-Djava.io.tmpdir=$tmpdir" \
	--class $classname \
	--master local[*] \
	--jars $jarsClasspath \
	$javaworkspace/extractcwdoc.jar \
	-i $cwdocdir \
	-o $outputdir \
	-y $year \
	-c 8
fi
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $expid finished within $DIFF


