from keras.callbacks import ModelCheckpoint,Callback,LearningRateScheduler,TensorBoard
from keras.models import load_model
import random
import numpy as np
from scipy import misc
import gc
from keras.optimizers import Adam
from imageio import imread
from datetime import datetime
import os
import json
import models
from utils import DataLoader, LrPolicy
from config import Config
import argparse
import tensorflow as tf
from keras import backend as K

_EPSILON = tf.keras.backend.epsilon()
ALPHA = 1
GAMMA =5


def get_parser():
    
    parser = argparse.ArgumentParser('train')
    parser.add_argument('--configPath', '-c', required=True)
    return parser

def train(args=None):
    parser = get_parser()
    args = parser.parse_args(args)
    conf=Config()
    conf.load(args.configPath)
    time=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    trainString="%s_%s_%s_%s" % (conf.model,conf.optimizer,str(conf.lr),time)
    os.makedirs(conf.logPath+"/"+trainString)
    conf.save(conf.logPath+"/"+trainString+'/config.json')
    print('Compiling model...')
    model_checkpoint = ModelCheckpoint(conf.logPath+"/"+trainString+'/Checkpoint-{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=False, save_weights_only=True)
    change_lr = LearningRateScheduler(LrPolicy(conf.lr).stepDecay)
    tbCallBack=TensorBoard(log_dir=conf.logPath+"/"+trainString+'/logs', histogram_freq=0,  write_graph=True, write_images=True)
    with tf.device('/gpu:1'):
        model=models.modelCreator(conf.model,conf.inputShape,conf.classes,conf.pretrainedModel)
        if conf.loss == "focal_loss":
            print("Conf loss : cross_focal_loss")
            model.compile(optimizer = conf.optimizer, loss = focal_loss_reg)
        else:
            model.compile(optimizer = conf.optimizer, loss = conf.loss)
        data = [conf.trainDataPath+"/"+f for f in os.listdir(conf.trainDataPath) if '.jpg' in f]
        random.shuffle(data)
        thr=int(len(data)*conf.validationSplit)
        trainData=data[thr:]
        valData=data[:thr]
        trainDataLoader=DataLoader(conf.batchSize,conf.inputShape,trainData,conf.guaMaxValue)
        validationDataLoader=DataLoader(conf.batchSize,conf.inputShape,valData,conf.guaMaxValue)
        print('Fitting model...')
        model.fit_generator(generator=trainDataLoader.generator(),
                        validation_data=validationDataLoader.generator(),
                        steps_per_epoch=len(trainData)//conf.batchSize,
                        validation_steps=len(valData)//conf.batchSize,
                        epochs=conf.epoches,
                        verbose=1,
                        initial_epoch=0,
                        callbacks = [model_checkpoint, change_lr,tbCallBack]
                        )

####################################################
# No improvment measure with focal losses @mathiane
####################################################
# def cross_focal_loss(y_true, y_pred):
#     # Define epsilon so that the backpropagation will not result in NaN
#     # for 0 divisor case
#     epsilon = K.epsilon()
#     # Add the epsilon to prediction value
#     #y_pred = y_pred + epsilon
#     # Clip the prediciton value
#     y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
#     # Calculate p_t
#     # Go back to a binary ground truth
#     num_cls = 2
#     p_t = tf.where(K.equal(y_true, 1), y_pred, 1-y_pred)
#     # Calculate alpha_t
#     alpha_factor = K.ones_like(y_true)*ALPHA
#     alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1-alpha_factor)
#     y_pred =  tf.add(y_pred, epsilon)
#     CCE = tf.multiply(y_true, -tf.log(y_pred))
#     weight = alpha_t * K.pow((1-p_t), GAMMA)
#     print("weight  ", weight)
#     # Calculate focal loss
#     #loss = weight * cross_entropy
#     loss = weight * CCE
#     # Sum the losses in mini_batch
#     loss = K.sum(loss, axis=1)
#     return loss


# # TO DO Try with binary input

# def focal_loss(gamma=3, alpha=1.):

#     gamma = float(gamma)
#     alpha = float(alpha)

#     def focal_loss_fixed(y_true, y_pred):
#         """Focal loss for multi-classification
#         FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
#         Notice: y_pred is probability after softmax
#         gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
#         d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
#         Focal Loss for Dense Object Detection
#         https://arxiv.org/abs/1708.02002
#         Arguments:
#             y_true {tensor} -- ground truth labels, shape of [batch_size, num_cls]
#             y_pred {tensor} -- model's output, shape of [batch_size, num_cls]
#         Keyword Arguments:
#             gamma {float} -- (default: {2.0})
#             alpha {float} -- (default: {4.0})
#         Returns:
#             [tensor] -- loss.
#         # From https://medium.com/swlh/multi-class-classification-with-focal-loss-for-imbalanced-datasets-c478700e65f5
#         """
#         epsilon = 1.e-9
#         y_true = tf.convert_to_tensor(y_true, tf.float32)
#         print("y_true  ", y_true)
#         y_pred = tf.convert_to_tensor(y_pred, tf.float32)
#         y_pred = tf.clip_by_value(tf.math.sigmoid(y_pred), clip_value_min=_EPSILON, clip_value_max=1-_EPSILON)
#         print("y_pred  ", y_pred)

#         model_out = tf.add(y_pred, epsilon)
#         ce = tf.multiply(y_true, -tf.log(model_out))
#         weight = tf.multiply(y_true, tf.pow(tf.subtract(1., model_out), gamma))
#         fl = tf.multiply(alpha, tf.multiply(weight, ce))
#         reduced_fl = tf.reduce_max(fl, axis=1)
#         return tf.reduce_mean(reduced_fl)
#     return focal_loss_fixed

# def focal_loss_reg(y_true, y_pred):
#     print("LOSS FUNCTION = focal_loss_reg")
#     # Note : https://openaccess.thecvf.com/content_ECCV_2018/papers/Xiankai_Lu_Deep_Regression_Tracking_ECCV_2018_paper.pdf
#     y_true = tf.convert_to_tensor(y_true, tf.float32)
#     y_pred = tf.convert_to_tensor(y_pred, tf.float32)
#     l2 = tf.pow(tf.subtract(y_true, y_pred), 2)
#     l_gamma = tf.pow(tf.subtract(y_true, y_pred), GAMMA)
#     return tf.multiply(l2,l_gamma)

if __name__ == "__main__":
    train()
