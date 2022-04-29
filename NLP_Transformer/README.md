# Code Reference

**The NLP_Transformer is from https://github.com/jadore801120/attention-is-all-you-need-pytorch by Yu-Hsiang Huang. On the train.py is modified to be able to test two optimizers: AdaBelief and Aida.**



# Usage

## WMT'16 Multimodal Translation: de-en

An example of training for the WMT'16 Multimodal Translation task (http://www.statmt.org/wmt16/multimodal-task.html).

### 0) Download the spacy language model.
```bash
# conda install -c conda-forge spacy 
python -m spacy download en
python -m spacy download de
```

### 1) Preprocess the data with torchtext and spacy.
```bash
python preprocess.py -lang_src de -lang_trg en -share_vocab -save_data m30k_deen_shr.pkl
```

### 2) Train the model
```bash
execute run.sh
```

### 3) Test the model
```bash
python translate.py -data_pkl m30k_deen_shr.pkl -model trained.chkpt -output prediction.txt
```


