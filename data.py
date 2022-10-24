# -*- coding: utf-8 -*-
"""
Generate datasets consisting of Disk, line, dots, mixed and mnist.

Modified based on DeepCGH's public code
- M. Hossein Eybposh, Nicholas W. Caira, Mathew Atisa, Praneeth Chakravarthula, and Nicolas C. Pégard, "DeepCGH: 3D computer-generated holography using deep learning," Opt. Express 28, 26636-26650 (2020)

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
from tqdm import tqdm
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import datasets
from skimage.draw import circle, line_aa
from skimage import morphology
import cv2



class DeepHUG_Datasets(object):
    '''
    Class for the Dataset object used in DeepCGH algorithm.
    Inputs:
        num_iter   int, determines the number of iterations of the GS algorithm
        input_shape   tuple of shape (height, width)
    Returns:
        Instance of the object
    '''
    def __init__(self, params):
        try:
            assert params['object_type'] in ['Disk', 'Line', 'Dot','mixed','mnist'], 'Object type not supported'
            self.path=params['path']
            self.shape = params['shape']
            self.N = params['N']
            self.ratio=params['train_ratio']
            self.object_size = params['object_size']
            self.intensity = params['intensity']
            self.object_count = params['object_count']
            self.name = params['name']
            self.object_type = params['object_type']
            self.centralized = params['centralized']
            self.FoV = params['FoV']
            self.normalize = params['normalize']
            self.compression=params['compression']
            self.dilation_factor=int(params['dilation_factor']/2)
        except:
            assert False, 'Not all parameters are provided!'

        self.__check_availability()
            
    def __get_line(self, shape, start, end):
        img = np.zeros(shape, dtype=np.float32)
        rr, cc, val = line_aa(start[0], start[1], end[0], end[1])
        img[rr, cc] = val * 1
        dilated_img = morphology.dilation(img,morphology.disk(radius=self.dilation_factor))
        return dilated_img
    
    def get_circle(self, shape, radius, location):
        """Creates a single circle.
    
        Parameters
        ----------
        shape : tuple of ints
            Shape of the output image
        radius : int
            Radius of the circle.
        location : tuple of ints
            location (x,y) in the image
    
        Returns
        -------
        img
            a binary 2D image with a circle inside
        rr2, cc2
            the indices for a circle twice the size of the circle. This is will determine where we should not create circles
        """
        img = np.zeros(shape, dtype=np.float32)
        rr, cc = circle(location[0], location[1], radius, shape=img.shape)
        img[rr, cc] = 1
        # get the indices that are forbidden and return it
        rr2, cc2 = circle(location[0], location[1], 2*radius, shape=img.shape)
        return img, rr2, cc2

    def __get_forbidFoV(self):
        shape=self.shape
        allow_x = set(range(shape[0]))
        allow_y = set(range(shape[1]))

        center=round(shape[0]/2)

        allow_FoV =list(range(center-round(self.FoV/2),center+round(self.FoV/2)))

        forbid_x,forbid_y = self.__get_allowables(allow_x,allow_y,allow_FoV,allow_FoV)
        return forbid_x,forbid_y

    def __get_allowables(self, allow_x, allow_y, forbid_x, forbid_y):
        '''
        Remove the coords in forbid_x and forbid_y from the sets of points in
        allow_x and allow_y.
        '''
        for i in forbid_x:
            try:
                allow_x.remove(i)
            except:
                continue
        for i in forbid_y:
            try:
                allow_y.remove(i)
            except:
                continue
        return allow_x, allow_y
    
    def __get_randomCenter(self, allow_x, allow_y):
        list_x = list(allow_x)
        list_y = list(allow_y)
        ind_x = np.random.randint(0,len(list_x))
        ind_y = np.random.randint(0,len(list_y))
        return list_x[ind_x], list_y[ind_y]
    
    def __get_randomStartEnd(self, shape):
        start = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
        end = (np.random.randint(0, shape[0]), np.random.randint(0, shape[1]))
        return start, end

    #% there shouldn't be any overlap between the two circles 
    def __get_RandDots(self, shape, maxnum = [10, 20]):
        '''
        returns a single sample (2D image) with random dots
        '''
        # number of random lines
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        xs = list(np.random.randint(0, shape[0], (n,)))
        ys = list(np.random.randint(0, shape[1], (n,)))
        
        for x, y in zip(xs, ys):
            image[x, y] = 1
        image=morphology.dilation(image,morphology.disk(self.dilation_factor))
        return image

    #% there shouldn't be any overlap between the two circles 
    def __get_RandLines(self, shape, maxnum = [10, 20]):
        '''
        returns a single sample (2D image) with random lines
        '''
        # number of random lines
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        for i in range(n):
            # generate centers
            start, end = self.__get_randomStartEnd(shape)
            
            # get circle
            img = self.__get_line(shape, start, end)
            image += img
        image[image>0] = 1
        image -= image.min()
        image /= image.max()
        return image
    
    #% there shouldn't be any overlap between the two circles 
    def __get_RandBlobs(self, shape, maxnum = [10,12], radius = 5, intensity = 1):
        '''
        returns a single sample (2D image) with random blobs
        '''
        # random number of blobs to be generated
        n = 0
        while n == 0:
            n = np.random.randint(int(maxnum[0]), int(maxnum[1]))
        image = np.zeros(shape)
        
        try: # in case the radius of the blobs is variable, get the largest diameter
            r = radius[-1]
        except:
            r = radius
        
        # define sets for storing the values
        allow_x = set(range(shape[0]))

        allow_y = set(range(shape[1]))
        if not self.centralized:
            forbid_x = set(list(range(r)) + list(range(shape[0]-r, shape[0])))
            forbid_y = set(list(range(r)) + list(range(shape[1]-r, shape[1])))
        else:
            forbid_x,forbid_y=self.__get_forbidFoV()
        
        allow_x, allow_y = self.__get_allowables(allow_x, allow_y, forbid_x, forbid_y)
        count = 0
        # else
        for i in range(n):
            # generate centers
            x, y = self.__get_randomCenter(allow_x, allow_y)
            
            if isinstance(radius, list):
                r = int(np.random.randint(radius[0], radius[1]))
            else:
                r = radius
            
            if isinstance(intensity, list):
                int_4_this = int(np.random.randint(np.round(intensity[0]*100), np.round(intensity[1]*100)))
                int_4_this /= 100.
            else:
                int_4_this = intensity
            
            # get circle
            img, xs, ys = self.get_circle(shape, r, (x,y))
            allow_x, allow_y = self.__get_allowables(allow_x, allow_y, set(xs), set(ys))
            image += img * int_4_this
            count += 1
            if len(allow_x) == 0 or len(allow_y) == 0:
                break
        return image
    
    def coord2image(self, coords):
        num_planes = self.shape[-1]
        
        sample = np.zeros(self.shape)
        
        for plane in range(num_planes):
            canvas = np.zeros(self.shape[:-1], dtype=np.float32)
        
            for i in range(coords.shape[-1]):
                img, _, __ = self.get_circle(self.shape[:-1], self.object_size, [coords[0, i], coords[1, i]])
                canvas += img.astype(np.float32)
            
            sample[:, :, plane] = (canvas>0)*1.
            
            if (num_planes > 1) and (plane != 0 and self.normalize == True):
                sample[:, :, plane] *= np.sqrt(np.sum(sample[:, :, 0]**2)/np.sum(sample[:, :, plane]**2))
            
        sample -= sample.min()
        sample /= sample.max()
        
        return np.expand_dims(sample, axis = 0)
    
    #TODO
    def __make_sample(self):
        
        num_planes = self.shape[-1]
        
        sample = np.zeros(self.shape)
        
        for plane in range(num_planes):
            if self.object_type == 'Disk':
                img = self.__get_RandBlobs(shape = (self.shape[0], self.shape[1]),
                                           maxnum = self.object_count,
                                           radius = self.object_size,
                                           intensity = self.intensity)
            elif self.object_type == 'Line':
                img = self.__get_RandLines((self.shape[0], self.shape[1]),
                                           maxnum = self.object_count)
            elif self.object_type == 'Dot':
                img = self.__get_RandDots(shape = (self.shape[0], self.shape[1]),
                                          maxnum = self.object_count)
            elif self.object_type == 'mixed':
                #ratio=[30,40,30] #dots, disk, lines
                mixed_type=self.mixed_list[self.mixed_index]

                if mixed_type==1: #dots
                    img = self.__get_RandDots(shape=(self.shape[0], self.shape[1]),
                                              maxnum=self.object_count)
                elif mixed_type==2: ##disk
                    img = self.__get_RandBlobs(shape=(self.shape[0], self.shape[1]),
                                               maxnum=self.object_count,
                                               radius=self.object_size,
                                               intensity=self.intensity)
                else: ##lines
                    img = self.__get_RandLines((self.shape[0], self.shape[1]),
                                               maxnum=self.object_count)

                

            sample[:, :, plane] = img
            
            if (num_planes > 1) and (plane != 0 and self.normalize == True):
                sample[:, :, plane] *= np.sqrt(np.sum(sample[:, :, 0]**2)/np.sum(sample[:, :, plane]**2))
        
        sample -= sample.min()
        sample /= sample.max()
        
        return sample


    ######################## generate #####################
    def __check_availability(self):
        print('Current working directory is:')
        print(os.getcwd(),'\n')

        self.filename=self.object_type+'_SHP{}_N{}_SZ{}_INT{}_Crowd{}_CNT{}_Split.tfrecords'.format(self.shape,
                                           self.N,
                                           self.object_size,
                                           self.intensity,
                                           self.object_count,
                                           self.centralized)

        self.absolute_file_path=os.path.join(os.getcwd(),self.path,self.filename)
        if not (os.path.exists(self.absolute_file_path.replace('Split','')) or os.path.exists(self.absolute_file_path.replace('Split','Train'))):
            warnings.warn('File does not exist. New dataset will be generated once getDataset is called')
            print(self.absolute_file_path)
        else:
            print('Data already exists.')

    def __bytes_feature(self,value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def __generate(self):
        '''
        Creates a dataset of randomly located blobs and stores data in an TFRecords file. Each sample (3D image) contains
        a randomly determined number of blobs that are randomly located in individual planes.
        Inputs:
            filename : str
                path to the dataset file
            N: int
                determines the fraction of N that is used as "train". The rest will be the "test" data
            shape: (int, int)
                tuple of integers, shape of the 2D planes
            maxnum: int
                determines the max number of blobs
            radius: int
                determines the radius of the blobs
            intensity: float or [float,float]
                intensity of the blobs. If as scalar, it's a binary blob. If a list, first element is the min intensity and second one is the max intensity.
            normalize: bool
                flag taht determines wheter the 3D data is normalized for fixed energy from plane to plane
        Outputs:
            aa:

            out_dataset:
                numpy.ndarray. Numpy array with shape (samples, x,y)
        '''
        train_size = np.floor(self.ratio * self.N)
        options = tf.io.TFRecordOptions(compression_type=self.compression)

        if self.object_type=='mnist':
            mnist=datasets.mnist
            tr,te=mnist.load_data()
        if self.object_type=='mixed':
            self.mixed_list = np.concatenate(
                (np.ones([int(self.N * 0.2), 1]), np.ones([int(self.N * 0.4), 1]) * 2, np.ones([int(self.N * 0.4), 1]) * 3), axis=None) # dot, disk, line
            np.random.shuffle(self.mixed_list)

        with tf.io.TFRecordWriter(self.absolute_file_path.replace('Split','Train'),options=options) as writer_train:
            with tf.io.TFRecordWriter(self.absolute_file_path.replace('Split','Test'),options=options) as writer_test:
                for i in tqdm(range(self.N)):
                    self.mixed_index = i
                    if self.object_type=='mnist':
                        sample=tr[0][i]
                        sample=cv2.resize(sample,(self.shape[0],self.shape[1]))
                        sample=sample/np.max(sample)
                        sample = np.expand_dims(sample, axis=-1)
                    else:
                        sample=self.__make_sample()
                    image_raw=sample.tostring()
                    feature={'sample':self.__bytes_feature(image_raw)}

                    # 2. Create a tf.train.Features
                    features=tf.train.Features(feature=feature)
                    # 3. Create an example protocol
                    example=tf.train.Example(features=features)
                    # 4. Serialize the Example to string
                    example_to_string=example.SerializeToString()
                    # 5. Write to TFRecord
                    if i < train_size:
                        writer_train.write(example_to_string)
                    else:
                        writer_test.write(example_to_string)

    def getDataset(self):
        if not(os.path.exists(self.absolute_file_path.replace('Split','')) or os.path.exists(self.absolute_file_path.replace('Split','Train'))):
            print('Generating data...')
            folder=os.path.join(os.getcwd(),self.path)

            if not os.path.exists(folder):
                os.makedirs(folder)

            self.__generate()
        self.dataset_paths=[self.absolute_file_path.replace('Split','Train'),self.absolute_file_path.replace('Split','Test')]

    def load_data(self,batch_size,shuffle,epochs=1):
        if isinstance(self.dataset_paths,list) and ('tfrecords' in self.dataset_paths[0]) and ('tfrecords' in self.dataset_paths[1]):
            image_feature_description={'sample':tf.io.FixedLenFeature([],tf.string)}

            def __parse_image_function(example_proto):
                parsed_features=tf.io.parse_single_example(example_proto,image_feature_description)
                img=tf.cast(tf.reshape(tf.io.decode_raw(parsed_features['sample'],tf.float64),self.shape),tf.float32)
                return {'target':img},{'phi':img}

            def val_func():
                validation = tf.data.TFRecordDataset(self.dataset_paths[1],
                                                     compression_type=self.compression,
                                                     buffer_size=None,
                                                     num_parallel_reads=2).map(__parse_image_function).batch(batch_size)
                return validation

            def train_func():
                train=tf.data.TFRecordDataset(self.dataset_paths[0],
                                              compression_type=self.compression,
                                              buffer_size=None,
                                              num_parallel_reads=2).map(__parse_image_function).repeat(epochs).shuffle(shuffle).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
                return train

            return train_func(), val_func()
        else:
            raise('You got a problem in your fline name')

    ######################## make generator for instant training
    def debug_generator(self, batch_size):
        def _generator(N):
            for _ in range(N):
                sample = self.__make_sample()
                yield {'target':sample}, {'phi':sample}
        
        data_types = tf.float64
        data_shape = self.shape
        types = {'target': data_types}, {'phi': data_types}
        shapes = {'target': data_shape}, {'phi': data_shape}
        return tf.data.Dataset.from_generator(
            _generator, args=[self.N],
            output_shapes=shapes, output_types=types).batch(batch_size)


    def debug_sample(self):
        if self.object_type=='mnist':
            mnist=datasets.mnist
            tr,te=mnist.load_data()
            sample=te[0][random.randrange(0,5000)]
            sample = cv2.resize(sample, (self.shape[0], self.shape[1]))
            sample = sample / np.max(sample)
            sample = np.expand_dims(sample,axis=-1)
        else:
            self.mixed_list=[3,3,3]
            self.mixed_index = 0
            sample = self.__make_sample()
        sample = np.expand_dims(sample, axis = 0)
        return sample

    def mnist_test(self,n):
        mnist=datasets.mnist
        tr,te=mnist.load_data()
        sample=te[0][n]
        sample=self.preprocess(sample)
        return sample

    def preprocess(self,target):

        shape = self.shape
        target = np.array(target)
        target[target <= np.max(target) * 0.5] = 0
        pad_size=int((shape[0]-target.shape[0])/2)
        target=np.pad(target,((pad_size,pad_size),(pad_size,pad_size)),'constant',constant_values=0)
        target[target > 0] = 1

        # Expand channels to put SGD and DL: [batch,Ny,Nz,planes]
        target = np.expand_dims(target, axis=-1)
        target = np.expand_dims(target, axis=0)

        return target
