# extract the encoder models out of the 1-stream pre-trained models
# the pre-trained models will be placed in 'ip-avsr-release/oulu/results/1stream/best_models' after running 1stream experiments
# usage: python extract_encoder_from_1stream_final.py

from __future__ import print_function
import sys
sys.path.insert(0, '../')
sys.path.insert(0, '../../')
import numpy as np
import theano.tensor as T
import argparse
import os
from modelzoo import deltanet_majority_vote
from utils.io import save_mat
from custom.nonlinearities import select_nonlinearity


def parse_options(rt,modelName,outName,input_dim):
    options = dict()
    options['input'] = '../results/1stream/best_models/'+modelName+'.'+str(rt)+'.pkl'
    output_folder = '../models/final_1stream_models/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    options['output'] = output_folder+outName+'.'+str(rt)
    options['input_dim'] = input_dim
    options['shape'] = '2000,1000,500,50'
    options['nonlinearities'] = 'rectify,rectify,rectify,linear'
    options['lstm_size'] = 450
    options['output_classes'] = 10
    options['use_blstm'] = True
    return options


def main():

    model = ['1stream_test',
             '1stream_test30',
             '1stream_test45',
             '1stream_test60',
             '1stream_test90']

    out = ['1stream_encoder_model_0',
               '1stream_encoder_model_30',
               '1stream_encoder_model_45',
               '1stream_encoder_model_60',
               '1stream_encoder_model_90']

    dim = [1450,
           1276,
           1247,
           1540,
           1320]

    runTime = 10

    for n in range(len(model)):

        modelName = model[n]
        outName = out[n]
        input_dim = dim[n]

        for rt in range(1,runTime+1):

            options = parse_options(rt,modelName,outName,input_dim)

            print('Current options:')
            print(options)
            print(' ')

            window = T.iscalar('theta')
            inputs1 = T.tensor3('inputs1', dtype='float32')
            mask = T.matrix('mask', dtype='uint8')
            shape = [int(i) for i in options['shape'].split(',')]
            nonlinearities = [select_nonlinearity(s) for s in options['nonlinearities'].split(',')]
            network = deltanet_majority_vote.load_saved_model(options['input'],
                                                              (shape, nonlinearities),
                                                              (None, None, options['input_dim']), inputs1, (None, None), mask,
                                                              options['lstm_size'], window, options['output_classes'],
                                                              use_blstm=options['use_blstm'])
            d = deltanet_majority_vote.extract_encoder_weights(network, ['fc1', 'fc2', 'fc3', 'bottleneck'],
                                                               [('w1', 'b1'), ('w2', 'b2'), ('w3', 'b3'), ('w4', 'b4')])
            expected_keys = ['w1', 'w2', 'w3', 'w4', 'b1', 'b2', 'b3', 'b4']
            keys = d.keys()
            for k in keys:
                assert k in expected_keys
                assert type(d[k]) == np.ndarray
            if 'output' in options:
                print('save extracted weights to {}'.format(options['output']))
                save_mat(d, options['output'])


if __name__ == '__main__':
    main()
