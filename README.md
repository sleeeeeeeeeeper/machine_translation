# LSTM Machine Translation

Practice using LSTM to translate English to Chinese

## Datasets

The source language is TED2020.en-zh_cn.en, and the target language is TED2020.en-zh_cn.zh_cn.

## Installation

First, clone the project locally.

``` bash
git clone https://github.com/sleeeeeeeeeeper/machine_translation.git
cd machine_translation
```

After that, create the environment and install the dependencies

``` bash
conda create -n MT python=3.11 -y
conda activate MT
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### 1. preproccess

Convert the original dataset into a csv file containing pairs of Chinese and English sentences.

``` bash
python preprocess.py
```

### 2. build dataset

Build a thesaurus and turn the contrasting sentences into a numerical index to generate a data set for training.Both thesaurus and data sets are placed under the "data" folder.

``` bash
python dataset.py
```

### 3. download the pre-trained word vector

Here we use fasttext's pre-trained word vector. We first download the original word vector in 300 dimensions, then reduce it to 128 dimensions and save it.

``` bash
python models.py
```

### 4. train

Train the Seq2Seq model to implement translation.The trained model and training log are placed in the "results" folder.

``` bash
python train.py
```

### 5. Web UI

A simple web application was made with streamlit for translation.

``` bash
streamlit run translate.py
```
