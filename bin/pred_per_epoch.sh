#!/usr/bin/env bash
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

source $CUR_DIR/set_env.sh

START=$(date +%s.%N)

expname=pacrrpub
train_years=wt12_13
test_year=wt10
numneg=6
batch=32
winlen=3
kmaxpool=3 
binmat=False
context=False
combine=16
iterations=10
shuffle=False
parentdir=/GW/D5data-2/khui/results/pacrrpub
outdir=$parentdir

python -m pred_per_epoch with\
	expname=$expname \
	train_years=$train_years \
	test_year=$test_year \
	numneg=$numneg \
	batch=$batch \
	winlen=$winlen \
	kmaxpool=$kmaxpool \
	binmat=$binmat \
	context=$context \
	combine=$combine \
	shuffle=$shuffle \
	parentdir=$parentdir \
	epochs=$iterations \
	outdir=$outdir

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)
echo $id finished within $DIFF


