#!/usr/bin/env bash
timestamp=`date +"%m%d%M%H"`
id=$timestamp.`hostname`.$BASHPID
echo $id

log4jconf=/home/khui/workspace/javaworkspace/log4j.xml
tmpdir=/GW/D5data-2/khui/cw-docvector-termdf/tmp

export CUDA_VISIBLE_DEVICES=''   #0,1

source /home/khui/workspace/pyenv/startpy35.sh
pythondir=$PWD

START=$(date +%s.%N)
islocal=true
#if ! $islocal
#then
#	echo run on the cluster
#	# for cluster
#	spark-submit \
#	--master yarn://139.19.52.105 \
#	--driver-memory 40G \
#	--executor-memory 10G \
#	--executor-cores 8 \
#	--num-executors 8 \
#	--driver-java-options "-Djava.io.tmpdir=$tmpdir" \
#	$pythondir/train_dumpweight_sdrmm.py\
#	-d simdim\
#	-w winlen
#else
spark-submit \
--master local[*] \
--driver-memory 90G \
--driver-java-options "-Djava.io.tmpdir=$tmpdir" \
$pythondir/dump_doc_mat.py
#$pythondir/dump_per_qeury_terms.py




END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $expid finished within $DIFF


