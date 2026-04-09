################################# single QP test ################################
bash run_benchmark.sh vimeo90k hevc TVRN  20,23,28,33,38  1
bash run_benchmark.sh vimeo90k hevc EMA  17,22,27,32,37  1
bash run_benchmark.sh vimeo90k hevc GIMM  17,22,27,32,37  1
bash run_benchmark.sh vimeo90k hevc GIMM+VQE  17,22,27,32,37  1
bash run_benchmark.sh vimeo90k hevc CVRS   17,22,27,32,37  1
bash run_benchmark.sh vimeo90k hevc CVRS_finetuned   17,22,27,32,37  1
bash run_benchmark.sh vimeo90k hevc STAA  17,22,27,32,37  1
bash run_benchmark.sh vimeo90k hevc IFRNet  17,22,27,32,37  1
bash run_benchmark.sh vimeo90k hevc RIFE  17,22,27,32,37  1
bash run_benchmark.sh vimeo90k hevc codec_reference  18,22,27,32,37  1


bash run_benchmark.sh vimeo90k av1 TVRN 21,27,33,39,45  8
bash run_benchmark.sh vimeo90k av1 EMA 20,26,32,38,44  8
bash run_benchmark.sh vimeo90k av1 GIMM 20,26,32,38,44  8
bash run_benchmark.sh vimeo90k av1 GIMM+VQE 20,26,32,38,44  8
bash run_benchmark.sh vimeo90k av1 CVRS  20,26,32,38,44  8
bash run_benchmark.sh vimeo90k av1 CVRS_finetuned  20,26,32,38,44  8
bash run_benchmark.sh vimeo90k av1 STAA 20,26,32,38,44  8
bash run_benchmark.sh vimeo90k av1 IFRNet 20,26,32,38,44  8
bash run_benchmark.sh vimeo90k av1 RIFE 20,26,32,38,44  8
bash run_benchmark.sh vimeo90k av1 codec_reference  20,26,32,38,44  8


bash run_benchmark.sh vimeo90k vvc TVRN  19,23,28,33,38  8
bash run_benchmark.sh vimeo90k vvc EMA  18,22,27,32,37  8
bash run_benchmark.sh vimeo90k vvc GIMM  18,22,27,32,37  8
bash run_benchmark.sh vimeo90k vvc GIMM+VQE  18,22,27,32,37  8
bash run_benchmark.sh vimeo90k vvc CVRS   18,22,27,32,37  8
bash run_benchmark.sh vimeo90k vvc CVRS_finetuned   18,22,27,32,37  8
bash run_benchmark.sh vimeo90k vvc STAA  18,22,27,32,37  8
bash run_benchmark.sh vimeo90k vvc IFRNet  18,22,27,32,37  8
bash run_benchmark.sh vimeo90k vvc RIFE  18,22,27,32,37  8
bash run_benchmark.sh vimeo90k vvc codec_reference  18,22,27,32,37  8

