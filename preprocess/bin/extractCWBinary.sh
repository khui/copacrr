#!/usr/bin/env bash
year=$1 # cw09 or cw12
timestamp=`date +"%m%d%M%H"`


javaworkspace=$(readlink -f ../java/extractcwdoc/target)
libdir=$(readlink -f ../java/extractcwdoc/dependency)
jdk=$JAVA_HOME
filterfile=/user/data/ExtractCwdocs/Extracted/cwidfilter_ql_qrel_runs.cwid_qid
outdirs=/user/data/ExtractCwdocs/Extracted/cwbinary_ql_qrel_runs

if [ "$year" == "cw09" ]; then
	cols=/data/clueweb09/ClueWeb09_English_1:/data/clueweb09/ClueWeb09_English_2:/data/clueweb09/ClueWeb09_English_3:/data/clueweb09/ClueWeb09_English_4:/data/clueweb09/ClueWeb09_English_5:/data/clueweb09/ClueWeb09_English_6:/data/clueweb09/ClueWeb09_English_7:/data/clueweb09/ClueWeb09_English_8:/data/clueweb09/ClueWeb09_English_9:/data/clueweb09/ClueWeb09_English_10
elif [ "$year" == "cw12" ]; then
	cols=/data/clueweb12/ClueWeb12_00:/data/clueweb12/ClueWeb12_01:/data/clueweb12/ClueWeb12_02:/data/clueweb12/ClueWeb12_03:/data/clueweb12/ClueWeb12_04:/data/clueweb12/ClueWeb12_05:/data/clueweb12/ClueWeb12_06:/data/clueweb12/ClueWeb12_07:/data/clueweb12/ClueWeb12_08:/data/clueweb12/ClueWeb12_09:/data/clueweb12/ClueWeb12_10:/data/clueweb12/ClueWeb12_11:/data/clueweb12/ClueWeb12_12:/data/clueweb12/ClueWeb12_13:/data/clueweb12/ClueWeb12_14:/data/clueweb12/ClueWeb12_15:/data/clueweb12/ClueWeb12_16:/data/clueweb12/ClueWeb12_17:/data/clueweb12/ClueWeb12_18:/data/clueweb12/ClueWeb12_19
fi



export HADOOP_CLASSPATH="find $libdir -name "*.jar" | tr '\n' ':'"
export JAVA_HOME=${jdk}

START=$(date +%s.%N)
hadoop jar $javaworkspace/extractcwdoc.jar \
	de.mpii.mr.tasks.ExtractDocBinary \
     	-libjars `find $libdir -name "*.jar"|tr '\n' ','` \
	-o $outdirs \
	-c $cols \
	-f $filterfile \
	-n $year
END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $expid finished within $DIFF

