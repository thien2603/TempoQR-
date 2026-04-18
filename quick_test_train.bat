@echo off
echo =====================================
echo   TEMPOQR QUICK TEST TRAINING
echo =====================================
echo.
echo Training with small dataset:
echo - Train samples: 100
echo - Test samples: 50  
echo - Epochs: 2
echo - Batch size: 4
echo.

python train_qa_model_quick_test.py ^
    --model tempoqr ^
    --dataset_name wikidata_big ^
    --tkbc_model_file tcomplex.ckpt ^
    --tkg_file full.txt ^
    --supervision soft ^
    --max_epochs 2 ^
    --batch_size 4 ^
    --lr 2e-4 ^
    --valid_freq 1 ^
    --eval_k 10 ^
    --frozen 1 ^
    --lm_frozen 1 ^
    --fuse add ^
    --extra_entities False ^
    --corrupt_hard 0.0 ^
    --mode train ^
    --save_to tempoqr_quick_test ^
    --quick_test True ^
    --train_samples 100 ^
    --test_samples 50

echo.
echo =====================================
echo   QUICK TEST COMPLETED!
echo =====================================
echo Check results at: results/wikidata_big/tempoqr_quick_test_quick_test.log
echo Model saved at: models/wikidata_big/qa_models/tempoqr_quick_test.ckpt
echo.
pause
