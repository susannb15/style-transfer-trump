# style-transfer-trump
Project for the seminar "Controllable Neural Text Generation" (WS21/22)

##  SETUP AND REQUIREMENTS

This code runs on Python 3.6 and torch 1.10.
If you use conda you can create the needed environment like this:

conda env create --file style_transfer.yml
conda activate style_transfer

## DATA

The data for this project has already been preprocessed with the prep_data.py and can be found in the data folder. 
For testing, the europarl corpus has been reduced to 15k training and 5k validation sentences. You can find the full 
europarl corpus here:
https://www.statmt.org/europarl/

Biden and Trump tweets are from:
https://www.kaggle.com/rohanrao/joe-biden-tweets
https://www.kaggle.com/kingburrito666/better-donald-trump-tweets

## MODEL TRAINING

### Translation models

Train the translation model (en-fr):

python train_nmt.py -config config_transl.yaml

and the "backtranslation" model (fr-en):

python train_nmt.py -config config_backtransl.yaml

The config files contain all information that the trainer needs. You can modify the config files if you want to use 
different parameters or want to use you own data. For a full documentation of arguments that can be included in the 
config files please look at the documentation of onmt (https://github.com/OpenNMT/OpenNMT-py)

### Preparing data for the style decoder

The style decoder is trained on parallel data of the style we want it to learn. That means we translate our style relevant data
(the Trump tweets) into French by using the en-fr translation model:

onmt_translate -model models/en-fr/en-fr.pt -src data/trump-train.csv -output data/trump-train.fr -gpu 0

You also want to translate a portion of Biden's tweets to French to "trumpify" them later with the style transfer model:

onmt_translate -model models/en-fr/en-fr.pt -src data/biden-test.csv -output data/biden-test.fr -gpu 0

### Style Decoder Training

The style decoder is trained by running:

python train_style_decoder.py -config config_style.yaml

The config file must specify a model (the backtranslation model) - this will be the encoder for the style transfer model. 

### Transfer the style of Trump's tweets to Biden's tweets

onmt_translate -model models/trump_generator/trump_generator.pt -src data/biden-test.fr -output data/biden-test.trump -gpu 0
