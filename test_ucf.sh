################################# single QP test ################################
bash run_benchmark.sh UCF101 hevc TVRN  20,23,28,33,38  8
bash run_benchmark.sh UCF101 hevc EMA  18,22,27,32,37  8
bash run_benchmark.sh UCF101 hevc GIMM  18,22,27,32,37  8
bash run_benchmark.sh UCF101 hevc CVRS   18,22,27,32,37  8
bash run_benchmark.sh UCF101 hevc CVRS_finetuned   18,22,27,32,37  8
bash run_benchmark.sh UCF101 hevc STAA  18,22,27,32,37  8
bash run_benchmark.sh UCF101 hevc IFRNet  18,22,27,32,37  8
bash run_benchmark.sh UCF101 hevc RIFE  18,22,27,32,37  8
bash run_benchmark.sh UCF101 hevc RIFE  18,22,27,32,37  1
bash run_benchmark.sh UCF101 hevc codec_reference  18,22,27,32,37  1


bash run_benchmark.sh UCF101 vvc TVRN  19,23,28,33,38  8
bash run_benchmark.sh UCF101 vvc EMA  18,22,27,32,37  8
bash run_benchmark.sh UCF101 vvc GIMM  18,22,27,32,37  8
bash run_benchmark.sh UCF101 vvc GIMM+VQE  18,22,27,32,37  8
bash run_benchmark.sh UCF101 vvc CVRS   18,22,27,32,37  8
bash run_benchmark.sh UCF101 vvc CVRS_finetuned   18,22,27,32,37  8
bash run_benchmark.sh UCF101 vvc STAA  18,22,27,32,37  8
bash run_benchmark.sh UCF101 vvc IFRNet  18,22,27,32,37  8
bash run_benchmark.sh UCF101 vvc RIFE  18,22,27,32,37  8
bash run_benchmark.sh UCF101 vvc codec_reference  18,22,27,32,37  8


bash run_benchmark.sh UCF101 av1 TVRN  21,27,33,39,45  8
bash run_benchmark.sh UCF101 av1 EMA  20,26,32,38,44  8
bash run_benchmark.sh UCF101 av1 GIMM  20,26,32,38,44  8
bash run_benchmark.sh UCF101 av1 GIMM+VQE  20,26,32,38,44  8
bash run_benchmark.sh UCF101 av1 CVRS   20,26,32,38,44  8
bash run_benchmark.sh UCF101 av1 CVRS_finetuned   20,26,32,38,44  8
bash run_benchmark.sh UCF101 av1 STAA  20,26,32,38,44  8
bash run_benchmark.sh UCF101 av1 IFRNet  20,26,32,38,44  8
bash run_benchmark.sh UCF101 av1 RIFE  20,26,32,38,44  8
bash run_benchmark.sh UCF101 av1 codec_reference  20,26,32,38,44  8



