#!/bin/bash

if [ "$#" -lt 3 ] || [ "$#" -gt 6 ]; then
    echo "Error: 3 to 5 arguments required."
    echo "Usage: $0 <dataset_type> <codec_type> <model_name> [qp_list] [slice_num] [frames]"
    echo ""
    echo "Arguments:"
    echo "  dataset_type : SNU, UCF101, vimeo90k"
    echo "  codec_type   : hevc, avc, av1, vp9, vvc"
    echo "  model_name   : TVRN, EMA, GIMM, GIMM+VQE, CVRS, CVRS_finetuned, IFRNet"
    echo "  qp_list      : (Optional) Comma-separated list of QPs."
    echo "  slice_num    : (Optional) Number of parallel slices (Default: 8)."
    echo "  frames    : (Optional) dataset frmnum, septuplet or 65frames."
    exit 1
fi

DATASET_TYPE=$1
CODEC_TYPE=$2
MODEL_NAME=$3
USER_QP_INPUT=${4:-""}
SLICE_NUM=${5:-8}
DATASET_FRMNUM=${6:-""}  

OPT_FILE="/code/codes/options/test_septuplet/test_TVRN_compress_temporal_epm_old_restoration_unified.yml"
STAA_OPT="/code/codes/options/test/test_STAA.yml"
MODE="cqp"
MAIN_CHECKPOINT="/model/fengxm/VRN/MIMO_VRN/MIMO_TVRN_codec_restoration_unified_restoration_adaptor/60000_G.pth"

FORCE_QP=""
USE_USER_QP=false

if [ "$MODEL_NAME" == "CVRS_finetuned" ]; then
    OPT_FILE="/code/codes/options/test_septuplet/test_TVRN_compress_cstvr_surrogate.yml"
fi

if [ -n "$USER_QP_INPUT" ]; then
    USE_USER_QP=true
    echo "Custom QP list provided: '$USER_QP_INPUT'. Ignoring default model/dataset QP logic."
fi

case "$DATASET_TYPE" in
    SNU)
        PARALLEL_SCRIPT_NAME="benchmark_parallel_test.py"
        SCRIPT_NAME="benchmark_SNUFILM.py"
        DATA_PATH="/data/fengxm/vimeo90k/snufilm_test/test/"
        DATASET="snufilm"
        ;;
    UCF101)
        PARALLEL_SCRIPT_NAME="benchmark_parallel_test.py"
        SCRIPT_NAME="benchmark_ucf101.py"
        DATA_PATH="/data/fengxm/vimeo90k/UCF101_septuplet/"
        DATASET="ucf101"
        ;;
    vimeo90k)
        PARALLEL_SCRIPT_NAME="benchmark_parallel_test.py"
        SCRIPT_NAME="benchmark_vimeo.py"
        DATA_PATH="/data/fengxm/vimeo90k/vimeo_septuplet_test/"
        DATASET="vimeo90k"
        ;;
    *)
        echo "Error: Unknown dataset_type '$DATASET_TYPE'."
        exit 1
        ;;
esac

QPS=()

if [ "$USE_USER_QP" = true ]; then
    IFS=',' read -ra QPS <<< "$USER_QP_INPUT"
    
    if [ ${#QPS[@]} -eq 0 ]; then
        echo "Error: Invalid QP list format."
        exit 1
    fi
    
    CLEAN_QPS=()
    for val in "${QPS[@]}"; do
        val=$(echo $val | xargs)
        if ! [[ "$val" =~ ^[0-9]+$ ]]; then
            echo "Error: QP value '$val' is not a valid integer."
            exit 1
        fi
        CLEAN_QPS+=("$val")
    done
    QPS=("${CLEAN_QPS[@]}")

else
    if [ -n "$FORCE_QP" ]; then
        QPS=("$FORCE_QP")
    else
        if [[ "$CODEC_TYPE" == "hevc" || "$CODEC_TYPE" == "avc" || "$CODEC_TYPE" == "vvc" ]]; then
            QPS=(18 22 27 32 37) 
        elif [[ "$CODEC_TYPE" == "av1" || "$CODEC_TYPE" == "vp9" ]]; then
            QPS=(20 26 32 38 44)
        else
            echo "Error: Unknown codec_type '$CODEC_TYPE'."
            exit 1
        fi

        if [ "$MODEL_NAME" == "TVRN" ]; then
            echo "Model is TVRN. Applying +1 offset to default QP values..."
            NEW_QPS=()
            for qp in "${QPS[@]}"; do
                NEW_QPS+=($((qp + 1)))
            done
            QPS=("${NEW_QPS[@]}")
        fi
    fi
fi

CHECKPOINT_ARG=""
HAS_CHECKPOINT=true

case "$MODEL_NAME" in
    TVRN|EMA|GIMM|"GIMM+VQE")
        CHECKPOINT_ARG="--checkpoints $MAIN_CHECKPOINT"
        HAS_CHECKPOINT=true
        ;;
    IFRNet|CVRS|CVRS_finetuned)
        HAS_CHECKPOINT=false
        echo "Note: Model $MODEL_NAME will not use --checkpoints argument."
        ;;
    *)
        CHECKPOINT_ARG="--checkpoints $MAIN_CHECKPOINT"
        echo "Warning: Unknown model '$MODEL_NAME'. Using default checkpoint path."
        ;;
esac

echo "=========================================="
echo "Starting Benchmark:"
echo "Dataset: $DATASET_TYPE"
echo "Codec: $CODEC_TYPE"
echo "Model: $MODEL_NAME"
echo "Config: $OPT_FILE"
echo "QP List: ${QPS[*]}"
echo "Total Slices: $SLICE_NUM"
if [ "$HAS_CHECKPOINT" = true ]; then
    echo "Checkpoint: $MAIN_CHECKPOINT"
else
    echo "Checkpoint: None"
fi
echo "=========================================="

for qp in "${QPS[@]}"; do
    echo "----------------------------------------"
    echo "Running: QP=$qp ..."
    
    CMD=(
        python "/code/codes/$PARALLEL_SCRIPT_NAME"
        -opt "$OPT_FILE"
        -staa_opt "$STAA_OPT"
        -path "$DATA_PATH"
        -mode "$MODE"
        -model "$MODEL_NAME"
        -qp "$qp"
        --codec_type "$CODEC_TYPE"
        --total_slices "$SLICE_NUM"
        --script_name $SCRIPT_NAME
        --dataset $DATASET
    )

    if [ "$HAS_CHECKPOINT" = true ]; then
        CMD+=($CHECKPOINT_ARG)
    fi

    if [ -n "$DATASET_FRMNUM" ]; then
        CMD+=(--dataset_type "$DATASET_FRMNUM")
    fi

    echo "[CMD] ${CMD[*]}"

    "${CMD[@]}"
    
    if [ $? -ne 0 ]; then
        echo "Error: Command failed (QP=$qp). Stopping."
        exit 1
    fi
done

echo "=========================================="
echo "All tests completed!"