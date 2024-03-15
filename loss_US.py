# -*- coding: utf-8 -*-
"""
Custom Loss Function
Using subclass to build loss function

Modified based on DeepCGH's public code
- M. Hossein Eybposh, Nicholas W. Caira, Mathew Atisa, Praneeth Chakravarthula, and Nicolas C. Pégard, "DeepCGH: 3D computer-generated holography using deep learning," Opt. Express 28, 26636-26650 (2020)

"""

import tensorflow as tf
from tensorflow.keras.losses import Loss
from us_holo_pkg.asm_TF import asm_propagator, physical_constraint, phase_expand_func, target_resize_func


class CustomLoss(Loss):
    def __init__(self,prop_param,expand_ratio=1,loss_type='cosine_similarity',intensity_lamda=0.1,name="custom_loss"):
        super().__init__(name=name)
        self.prop_param=prop_param
        self.shape=prop_param.input_shape
        self.zs = prop_param.prop_distance
        self.f0 = prop_param.Freq
        self.medium = prop_param.medium
        self.ps = prop_param.grid_spacing
        self.txdr_output=prop_param.txdr_output
        self.txdr_point_spec=self.txdr_output.points
        self.physical_limit=physical_constraint(prop_param, cal_transmission=False)
        self.expand_ratio=expand_ratio

        self.loss_type=loss_type
        self.intensity_lamda=intensity_lamda

    def call(self,y_true, y_pred):
        # apply phisical limit (txdr element)
        y_pred = self.physical_limit(y_pred)
        y_pred = phase_expand_func(y_pred, self.expand_ratio)

        # asm propagation
        y_pred_prop =asm_propagator(y_pred, self.prop_param,expand_ratio=self.expand_ratio) #y_pred(input) is phase map

        y_true= target_resize_func(y_true,self.expand_ratio, self.prop_param)

        on_target_mean, off_target_mean = self.efficiency(y_true,y_pred_prop)

        intensity_lamda=self.intensity_lamda

        if self.loss_type == 'cosine_similarity':
            return self.accuracy(y_true, y_pred_prop)
        elif self.loss_type == 'cosine_similarity+intensity':
            return self.accuracy(y_true, y_pred_prop)+tf.math.log(1/on_target_mean+off_target_mean)*intensity_lamda


    def accuracy(self,y_true, y_pred): # cosine similarity based loss function
        denom = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), axis=[1, 2, 3])*tf.reduce_sum(tf.pow(y_true, 2), axis=[1, 2, 3]))
        return 1-tf.reduce_mean((tf.reduce_sum(y_pred * y_true, axis=[1, 2, 3])+0.001)/(denom+0.001), axis = 0)


    def efficiency(self,y_true, y_pred):
        # Calculation for on-target efficiency
        zero_mask = tf.math.not_equal(y_true, 0)
        # Adding a small value to avoid division by zero
        on_target_mean = tf.reduce_mean(tf.boolean_mask(y_pred, zero_mask)) 
        
        # Calculation for off-target efficiency
        non_zero_mask = tf.math.equal(y_true, 0)
        off_target_mean = tf.reduce_mean(tf.boolean_mask(y_pred, non_zero_mask)) 
        
        return on_target_mean, off_target_mean

    def uniformity(self,true,pred):
        on_target = tf.math.multiply(true, pred)
        zero_mask = tf.math.not_equal(on_target, 0)
        on_target_mean = tf.reduce_mean(tf.boolean_mask(on_target, zero_mask))
        on_target_std = tf.math.reduce_std(tf.boolean_mask(on_target,zero_mask))
        return 1-on_target_std/on_target_mean
