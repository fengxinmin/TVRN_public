# ----------------------------------------Septuplet----------------------------------------#
bash run_benchmark.sh SNU hevc codec_reference  17,22,27,32,37 1
bash run_benchmark.sh SNU hevc TVRN  20,23,28,33,38  1
bash run_benchmark.sh SNU hevc TVRN_S  20,23,28,33,38  1
bash run_benchmark.sh SNU hevc EMA  17,22,27,32,37  1
bash run_benchmark.sh SNU hevc GIMM  17,22,27,32,37  1
bash run_benchmark.sh SNU hevc STAA  17,22,27,32,37  1
bash run_benchmark.sh SNU hevc IFRNet  17,22,27,32,37 1
bash run_benchmark.sh SNU hevc RIFE  17,22,27,32,37  1


bash run_benchmark.sh SNU vvc codec_reference  19,23,28,33,38 8
bash run_benchmark.sh SNU vvc TVRN  19,23,28,33,38  8
bash run_benchmark.sh SNU vvc EMA  18,22,27,32,37  8
bash run_benchmark.sh SNU vvc GIMM  18,22,27,32,37  8
bash run_benchmark.sh SNU vvc STAA  18,22,27,32,37  8
bash run_benchmark.sh SNU vvc IFRNet  18,22,27,32,37  8

bash run_benchmark.sh SNU av1 codec_reference  20,26,32,38,44 8
bash run_benchmark.sh SNU av1 TVRN  21,27,33,39,45  8
bash run_benchmark.sh SNU av1 EMA  20,26,32,38,44  8
bash run_benchmark.sh SNU av1 GIMM  20,26,32,38,44  8
bash run_benchmark.sh SNU av1 STAA  20,26,32,38,44  8
bash run_benchmark.sh SNU av1 IFRNet  20,26,32,38,44  8
bash run_benchmark.sh SNU av1 RIFE  20,26,32,38,44  8

# ----------------------------------------65 frames----------------------------------------#

bash run_benchmark.sh SNU hevc codec_reference  17,22,27,32 8 65frames
bash run_benchmark.sh SNU hevc TVRN  20,23,28,33  8 65frames
bash run_benchmark.sh SNU hevc GIMM  17,22,27,32  8 65frames
bash run_benchmark.sh SNU hevc EMA  17,22,27,32  8 65frames
bash run_benchmark.sh SNU hevc IFRNet  17,22,27,32  8 65frames
bash run_benchmark.sh SNU hevc RIFE  17,22,27,32  8 65frames
bash run_benchmark.sh SNU hevc STAA  17,22,27,32  8 65frames


bash run_benchmark.sh SNU vvc codec_reference  19,23,28,33 8  65frames
bash run_benchmark.sh SNU vvc TVRN  19,23,28,33  8  65frames
bash run_benchmark.sh SNU vvc GIMM  18,22,27,32  8  65frames
bash run_benchmark.sh SNU vvc EMA  18,22,27,32  8  65frames
bash run_benchmark.sh SNU vvc IFRNet  18,22,27,32  8  65frames
bash run_benchmark.sh SNU vvc RIFE  18,22,27,32  8  65frames
bash run_benchmark.sh SNU vvc STAA  18,22,27,32  8  65frames

bash run_benchmark.sh SNU av1 codec_reference  20,26,32,38 8  65frames
bash run_benchmark.sh SNU av1 TVRN  21,27,33,39  8  65frames
bash run_benchmark.sh SNU av1 GIMM  20,26,32,38  8  65frames
bash run_benchmark.sh SNU av1 EMA  20,26,32,38  8  65frames
bash run_benchmark.sh SNU av1 IFRNet  20,26,32,38  8  65frames
bash run_benchmark.sh SNU av1 RIFE  20,26,32,38  8  65frames
bash run_benchmark.sh SNU av1 STAA  20,26,32,38  8  65frames
