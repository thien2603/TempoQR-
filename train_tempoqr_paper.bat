@echo off
echo Training TempoQR with Paper Settings
echo =====================================

python train_qa_model.py ^
    --model tempoqr ^
    --dataset_name wikidata_big ^
    --tkbc_model_file tcomplex.ckpt ^
    --tkg_file full.txt ^
    --supervision soft ^
    --max_epochs 20 ^
    --batch_size 32 ^
    --lr 2e-4 ^
    --valid_freq 1 ^
    --eval_k 10 ^
    --frozen 1 ^
    --lm_frozen 1 ^
    --fuse add ^
    --extra_entities False ^
    --corrupt_hard 0.0 ^
    --mode train ^
    --save_to tempoqr_paper_replicate

echo Training completed!
pause
