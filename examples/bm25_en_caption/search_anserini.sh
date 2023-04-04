export ANSERINI_PATH="<path>/<to>/<anserini>"
export QREL_DIR="<path>/<to>/<qrel>"
export PATH_TREC_EVAL="<path>/<to>/<trec_eval>"

SPLIT=$1
SETTING=$2

if [[ ${SETTING} == "easy" ]];
then
	postfix=${SETTING}.${SPLIT}
else
	postfix=${SETTING}
fi

text_index=indexes/lucene-index.atomic.text.flat.${postfix}
image_index=indexes/lucene-index.atomic.image.flat.${postfix}

t2i_run=runs/run.${SPLIT}.bm25-anserini-default.t2i.${SETTING}.trec
i2t_run=runs/run.${SPLIT}.bm25-anserini-default.i2t.${SETTING}.trec

${ANSERINI_PATH}/target/appassembler/bin/SearchCollection \
  -index ${text_index} \
  -topics image-collection.${postfix}/${SPLIT}.image-caption.jsonl \
  -topicreader JsonString \
  -topicfield contents \
  -output ${i2t_run} \
  -bm25 -hits 1000 -parallelism 64 -threads 64

${ANSERINI_PATH}/target/appassembler/bin/SearchCollection \
  -index ${image_index} \
  -topics text-collection.${postfix}/${SPLIT}.text.jsonl \
  -topicreader JsonString \
  -topicfield contents \
  -output ${t2i_run} \
  -bm25 -hits 1000 -parallelism 64 -threads 64

echo "==== Text2Image ===="
t2i_qrel="${QREL_DIR}/${SPLIT}.qrels.t2i.projected.trec"
$PATH_TREC_EVAL -c -m recip_rank -M 10 ${t2i_qrel} ${t2i_run}
$PATH_TREC_EVAL -c -m recall.10,1000 ${t2i_qrel} ${t2i_run} 

echo "==== Image2Text ===="
i2t_qrel="${QREL_DIR}/${SPLIT}.qrels.i2t.projected.trec"
$PATH_TREC_EVAL -c -m recip_rank -M 10 ${i2t_qrel} ${i2t_run}
$PATH_TREC_EVAL -c -m recall.10,1000 ${i2t_qrel} ${i2t_run}
