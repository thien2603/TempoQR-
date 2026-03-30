# TempoQR
This is the code for the manuscript "TempoQR: Temporal Question Reasoning over Knowledge Graphs" (AAAI2022).
Paper: https://arxiv.org/abs/2112.05785

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/tempoqr-temporal-question-reasoning-over/question-answering-on-cronquestions)](https://paperswithcode.com/sota/question-answering-on-cronquestions?p=tempoqr-temporal-question-reasoning-over)

![image](https://user-images.githubusercontent.com/43810712/151389025-c783018e-b5cb-4585-bd24-a6cb9d015f9f.png)

## Installation

Clone and create a conda environment
``` 
git clone https://github.com/thien2603/TempoQR-.git
cd TempoQR
conda create --prefix ./tempoqr_env python=3.7
conda activate ./tempoqr_env
```

The implementation is based on CronKGQA in [Question Answering over Temporal Knowledge Graphs](https://arxiv.org/abs/2106.01515) and their code from https://github.com/apoorvumang/CronKGQA. You can find more installation details there.
We use TComplEx KG Embeddings as implemented in https://github.com/facebookresearch/tkbc.

Install TempoQR requirements
```
conda install --file requirements.txt -c conda-forge
```

## Dataset and pretrained models download

Download and unzip ``data.zip`` and ``models.zip`` in the root directory.

Drive: https://drive.google.com/drive/folders/1aS2s5sZ0qlDpGZ9rdR7HcHym23N3pUea?usp=sharing.

## Running the code

### Original TempoQR:
```
python ./train_qa_model.py --model tempoqr --supervision soft
python ./train_qa_model.py --model tempoqr --supervision hard
```

### Evaluation and Prediction Scripts:
```
# Model evaluation with detailed output
python evaluate_model.py

# Full dataset evaluation (30,000 questions)
python evaluate_full_dataset.py

# Sample predictions (50-3000 samples)
python predict_10_samples.py

# Detailed per-question analysis
python evaluate_model_detailed.py
```

Other models: "entityqr" and "cronkgqa" with hard and soft supervisions.

To use a corrupted TKG change to "--tkg_file train_corXX.txt" and "--tkbc_model_file tcomplex_corXX.ckpt", where XX=20,33,50.

To evaluate on unseen complex questions change to "--test test_bef_and_aft" or "--test test_fir_las_bef_aft".

Please explore more argument options in train_qa_model.py.

Minor Note: Not all modules have been tested after the code merging.

## Features Added

- ✅ **Model Evaluation Scripts**: Complete evaluation with accuracy metrics
- ✅ **Prediction Analysis**: Detailed sample-by-sample analysis
- ✅ **Entity Name Mapping**: Convert Q*** to readable names
- ✅ **Question Type Classification**: Time vs Entity questions
- ✅ **Full Dataset Testing**: 30,000 question evaluation
- ✅ **Results Export**: Save detailed results to files

## Cite

If you find our method, code, or experimental setups useful, please cite our paper:
```
@misc{mavromatis2021tempoqr,
      title={TempoQR: Temporal Question Reasoning over Knowledge Graphs}, 
      author={Costas Mavromatis and Prasanna Lakkur Subramanyam and Vassilis N. Ioannidis and Soji Adeshina and Phillip R. Howard and Tetiana Grinberg and Nagib Hakim and George Karypis},
      year={2021},
      eprint={2112.05785},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
