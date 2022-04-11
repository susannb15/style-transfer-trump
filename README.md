# style-transfer-trump
Project for the seminar "Controllable Neural Text Generation" (WS21/22)

This is an implementation of the paper: Style Transfer Through Back-Translation. Shrimai Prabhumoye, Yulia Tsvetkov, Ruslan Salakhutdinov, Alan W Black.

The goal is to train a translation system that "trumpifies" Biden's tweets through back-translation. 

##  Dependencies

This code runs on Python 3.6 and torch 1.10. You can install all dependencies needed to run the code with the requirements.txt:

```
pip install -r requirements.txt
```

or via conda:

```
conda env create --file style_transfer.yml
conda activate style_transfer
```

## Data

The method requires parallel data to train a translation system and two datasets of different styles, as well as a classifier that is pre-trained to distinguish between those styles. 

This project uses the Europarl corpus for neural machine translation training. It has been reduced to 15k training and 5k validation sentences to make it easier to test the code. You can find the full Europarl corpus here: https://www.statmt.org/europarl/

The distinctive styles in this project are Trump and Biden tweets from https://www.kaggle.com/rohanrao/joe-biden-tweets and https://www.kaggle.com/kingburrito666/better-donald-trump-tweets

The data for this project has already been preprocessed and is ready to be used for training. 

If you want to use your own data, you need to build a vocabulary using OpenNMT. This requires a configuration YAML file that specifies the data that should be used. The config_transl.yaml was used to produce the vocab for this coding project. You can use that for orientation or look up OpenNMT's quickstart at (https://github.com/OpenNMT/OpenNMT-py).

```python
onmt_build_vocab -config config_transl.yaml -n_sample 10000
```

You can download the trained translation models and the trump-generator of this project here:

https://unisaarlandde-my.sharepoint.com/:u:/g/personal/s8suboyy_uni-saarland_de/ETRmUx_HqSRMqpEyUdlTluoBHVi-svd-I5rycnH2-YqZDw?e=1Nr5Up

## Model Training
Model parameters are specified in a configuration YAML file. You will need 3 systems: a translation model, a back-translation model and the style decoder (= the trump-generator). 

### Translation models

Train the translation model (en-fr):
```python
python train_nmt.py -config config_transl.yaml
```
and the back-translation model (fr-en):
```python
python train_nmt.py -config config_backtransl.yaml
```
You can modify the configuration files if you want to use different parameters or want to use you own data. For a full documentation of possible training arguments please look at the documentation of OpenNMT (https://github.com/OpenNMT/OpenNMT-py).

### Preparing data for the style decoder

The style decoder is trained on parallel data of the style we want it to learn. That means we translate our style relevant data
(the Trump tweets) into French by using the en-fr translation model:
```python
onmt_translate -model models/en-fr/en-fr.pt -src data/trump-train.csv -output data/trump-train.fr -gpu 0
```
You also want to translate a portion of Biden's tweets to French to "trumpify" them later with the trump-generator:
```python
onmt_translate -model models/en-fr/en-fr.pt -src data/biden-test.csv -output data/biden-test.fr -gpu 0
```
### Style Decoder Training
The configuration file for the style decoder must specify the encoder of a previously trained back-translation model, as well as a the path to a pre-trained style classifier and the target style (the style classifier was trained on 0 = Biden and 1 = Trump, so setting this argument to 1 will train a trump-generator, setting it to 0 will train a biden-generator). The style decoder/trump-generator is then trained by running:
```python
python train_style_decoder.py -config config_style.yaml
```

### Transfer the style of Trump to the Biden test set
```python
onmt_translate -model models/trump_generator/trump_generator.pt -src data/biden-test.fr -output data/biden-test.trump -gpu 0
```
## REFERENCES

Klein, G., Kim, Y., Deng, Y., Senellart, J., & Rush, A. M. (2017, July). OpenNMT: Open-Source Toolkit for Neural Machine Translation. In Proceedings of ACL 2017, System Demonstrations (pp. 67-72).

Koehn, Philipp. "Europarl: A parallel corpus for statistical machine translation." Proceedings of machine translation summit x: papers. 2005.

Prabhumoye, S., Tsvetkov, Y., Salakhutdinov, R., & Black, A. W. (2018, July). Style Transfer Through Back-Translation. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) (pp. 866-876).
