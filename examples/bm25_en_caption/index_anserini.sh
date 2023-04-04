export ANSERINI_PATH="<path>/<to>/<anserini>"


DEFAULT_DIR=$(pwd)
MODE=$1
SETTING=$2


if [[ "${MODE}" == "encode" ]]; then
    python convert_jsonl.py --split train --encode_field image_caption
    python convert_jsonl.py --split validation --encode_field image_caption
    python convert_jsonl.py --split test --encode_field image_caption
    python convert_jsonl.py --split other --encode_field image_caption

    python convert_jsonl.py --split train --encode_field text
    python convert_jsonl.py --split validation --encode_field text
    python convert_jsonl.py --split test --encode_field text
    python convert_jsonl.py --split other --encode_field text
fi

if [[ "${MODE}" == "index" ]]; then
    if [[ "${SETTING}" == "easy" ]]; then

        for SPLIT in "validation" "test";
        do
            text_folder=text-collection.${SETTING}.${SPLIT}
            mkdir ${text_folder}
            cd ${text_folder}
            ln -s ../text-collection/${SPLIT}*.jsonl .
            cd ${DEFAULT_DIR}

            ${ANSERINI_PATH}/target/appassembler/bin/IndexCollection \
                -collection JsonCollection \
                -input ${text_folder} \
                -index indexes/lucene-index.atomic.text.flat.${SETTING}.${SPLIT} \
                -generator DefaultLuceneDocumentGenerator \
                -threads 8 -storePositions -storeDocvectors -storeRaw

            image_folder=image-collection.${SETTING}.${SPLIT}
            mkdir ${image_folder}
            cd ${image_folder}
            ln -s ../image-collection/${SPLIT}*.jsonl .
            cd ${DEFAULT_DIR}

            echo "Indexing ... ${SETTING}.${SPLIT}"
            ${ANSERINI_PATH}/target/appassembler/bin/IndexCollection \
                -collection JsonCollection \
                -input ${image_folder} \
                -index indexes/lucene-index.atomic.image.flat.${SETTING}.${SPLIT} \
                -generator DefaultLuceneDocumentGenerator \
                -threads 8 -storePositions -storeDocvectors -storeRaw
        done

    elif [[ "${SETTING}" == "medium" ]]; then

        text_folder=text-collection.${SETTING}
        mkdir ${text_folder}
        cd ${text_folder}
        ln -s ../text-collection/train*.jsonl .
        ln -s ../text-collection/validation*.jsonl .
        ln -s ../text-collection/test*.jsonl .
        cd ${DEFAULT_DIR}

        ${ANSERINI_PATH}/target/appassembler/bin/IndexCollection \
            -collection JsonCollection \
            -input ${text_folder} \
            -index indexes/lucene-index.atomic.text.flat.${SETTING} \
            -generator DefaultLuceneDocumentGenerator \
            -threads 8 -storePositions -storeDocvectors -storeRaw

        image_folder=image-collection.${SETTING}
        mkdir ${image_folder}
        cd ${image_folder}
        ln -s ../image-collection/${SPLIT}*.jsonl .
        cd ${DEFAULT_DIR}

        echo "Indexing ..."
        ${ANSERINI_PATH}/target/appassembler/bin/IndexCollection \
            -collection JsonCollection \
            -input ${image_folder} \
            -index indexes/lucene-index.atomic.image.flat.${SETTING} \
            -generator DefaultLuceneDocumentGenerator \
            -threads 8 -storePositions -storeDocvectors -storeRaw

    elif [[ "${SETTING}" == "hard" ]]; then

        text_folder=text-collection.${SETTING}
        mkdir ${text_folder}
        cd ${text_folder}
        ln -s ../text-collection/*.jsonl .
        cd ${DEFAULT_DIR}

        ${ANSERINI_PATH}/target/appassembler/bin/IndexCollection \
            -collection JsonCollection \
            -input ${text_folder} \
            -index indexes/lucene-index.atomic.text.flat.${SETTING} \
            -generator DefaultLuceneDocumentGenerator \
            -threads 8 -storePositions -storeDocvectors -storeRaw

        image_folder=image-collection.${SETTING}
        mkdir ${image_folder}
        cd ${image_folder}
        ln -s ../image-collection/${SPLIT}*.jsonl .
        cd ${DEFAULT_DIR}

        echo "Indexing ..."
        ${ANSERINI_PATH}/target/appassembler/bin/IndexCollection \
            -collection JsonCollection \
            -input ${image_folder} \
            -index indexes/lucene-index.atomic.image.flat.${SETTING} \
            -generator DefaultLuceneDocumentGenerator \
            -threads 8 -storePositions -storeDocvectors -storeRaw
    fi
fi
