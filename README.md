# Overview
This is the Python implementation of End-to-End Multi-view Lipreading tested on the OuluVS2 dataset. If you use this package in your research, please kindly cite this paper:

[1] End-to-End Multi-View Lipreading, S. Petridis, Y. Wang, Z. Li, M. Pantic. British Machine Vision Conference. London, September 2017. 

## Dependencies
To run the codes, the following dependencies are required:
- miniconda2 
- matplotlib 
- pydotplus 
- tabulate 
- scikit-learn
- ipython 
- pillow 
- theano (cpu)
- lasagne 
- nolearn 

It is suggested that you use miniconda to manage your python environment. Miniconda can be downloaded from http://conda.pydata.org/miniconda.html. No CUDA installation is required.

The code is tested on:
- Ubuntu 16.04, Python 2.7.13, Theano 0.9.0, Lasagne 0.2.dev1. 

## Dataset
The OuluVS2 audiovisual database was collected at the Center of Machine Vision Research, Department of Computer Science and Engineering, University of Oulu, Finland. It was designed to facilitate research on visual speech recognition, sometimes also referred to as automatic lip-reading.
You need to sign a license agreement before you can use this dataset. Details can be found on: 
http://www.ee.oulu.fi/research/imag/OuluVS2/index.html
After you have downloaded the dataset successfully, you can use the provided scripts to pre-process the dataset. 
Instructions on how to pre-process the dataset can be found on `preprocessOulu.pdf`

## Usage
To use this package, make sure you have:

1. Installed all the necessary dependencies using Miniconda2 in an environment of your selection (for example, `avsr`)
2. Have successfully preprocessed the OuluVS2 dataset. Check the preprocessOulu.pdf for more information regarding how to pre-process the data and pre-train the encoder with RBMs (if needed). Weights for the pre-trained encodes can be found at https://ibug.doc.ic.ac.uk/resources/EndToEndLipreading/.
 For pre-training you will need the following toolbox https://github.com/stavros99/DeepLearningToolbox_Matlab

Let's assume `$ROOT` is the root folder of this package (e.g. `$ROOT=/home/user_name/end-to-end-multiview-lipreading`).

First activate the environment of dependencies in your terminal (replace `avsr` with your environment's name if needed), then go into `runners` folder:
```
source activate avsr
cd $ROOT/runners
```

Then run the single-view experiments:
```
./run_experiments.oulu_1stream.sh ./experiments/oulu_1stream_experiments.txt
```

This will run the single-view lip-reading experiments for five different views (frontal, 30 degrees, 45 degrees, 60 degrees and profile). Each experment will be repeated 10 times. The results will be saved to: `$ROOT/oulu/results/1stream`. 

To run multi-view experiments, you need to fully complete the running of single-view experiment, as those resulting models are used as the starting point in multi-view experiments. These models are automatically saved in: `$ROOT/oulu/results/1stream/best_models`. 

To extract the weights out of single-view models, go to the following folder and run scripts:
```
cd $ROOT/oulu/extract_weights
python extract_encoder_from_1stream_final.py 
python extract_lstm_from_1stream_final.py
```
The extracted weights (for Encoder and for LSTM) are saved in `$ROOT/oulu/models/final_1stream_models`.

Then you can run the multi-view experiments. Simply go back to `runners` folder, and run corresponding scripts. 
```
cd $ROOT/runners
./run_experiments.oulu_2stream.sh ./experiments/oulu_2stream_experiments.txt
./run_experiments.oulu_3stream.sh ./experiments/oulu_3stream_experiments.txt
./run_experiments.oulu_3stream.sh ./experiments/oulu_3stream_experiments.txt
./run_experiments.oulu_4stream.sh ./experiments/oulu_4stream_experiments.txt
./run_experiments.oulu_5stream.sh ./experiments/oulu_5stream_experiments.txt
```

## Config Files:
Experiment settings are controlled by Config files. You can find all the Config files in: `$ROOT/oulu/config`. The meaning of some important options is explained below.

- [streamX]:   the setting for the Xth-stream data
- data: the path of the input data for this stream
- model: the path of the pre-trained encoder model
- lstm_model: the path of the pre-trained lstm model
- imagesize: size of the mouth ROI image, e.g. 29,50
- input_dimensions: the dimensions of the mouth image, e.g. 1450
- shape: the number of hidden units in different layers of encoders, e.g. 2000,1000,500,50

- [lstm_classifier]:  options for lstm classifiers
- windowsize:  the size of windows to calculate delta and delta delta features
- use_blstm: use Bi-directional LSTM or not
- lstm_size: number of hidden units used in the LSTM classifiers
- output_classes: number of output classes
- fusiontype: how to fuse the data from different views 

- [training]:  options for training process
- learning_rate: learning rate to use train the model
- num_epoch: the number of maximum training epoch 


## Best models
We have also released the best-performing models for single-view, 2-view and 3-view experiments. Those models have achieved the current state-of-the-art accuracies on the OuluVS2 dataset, as reported in [1].
You can find those models at https://ibug.doc.ic.ac.uk/resources/EndToEndLipreading/.

Please note that in order to use the pre-trained models you need to subtract the mean image of each video (i.e., you should compute the mean image of the video and remove it from all frames in that video) and then z-normalise each image, i.e., remove the mean pixel value and divide by the standard deviation of all pixels in that image. Check `preTrainEncoderWithRBMs.m` for an example.

