import os
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as pilimg
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from model_HUG import MDHGN, Unet, DenseHUnet , HUnet, SegNet, FusionHUnet, SRResNet,InceptionHUnet
from loss_US import CustomLoss
from us_holo_pkg.makeshape import makeDisc
from us_holo_pkg.asm_TF import asm_propagator,physical_constraint, physical_constraint_real, scale, scale_pil
from data import DeepHUG_Datasets
from datetime import datetime
from scipy.io import savemat
import math
import time

class method():

    def __init__(self,prop_param,dataset_param):
        self.prop_param=prop_param
        self.loss=CustomLoss(prop_param)
        self.physical_lim_func=physical_constraint(prop_param, cal_transmission=False)
        self.dataset_param=dataset_param
        self.dataset = DeepHUG_Datasets(self.dataset_param)


    def make_phase_target(self):
        shape=self.prop_param.input_shape
        target=np.zeros((shape[0],shape[1]))
        pattern_range_axial=[110,170]
        pattern_range_lateral=[142,147]
        dphase=2*math.pi/(pattern_range_axial[1]-pattern_range_axial[0])
        phase=0
        for i in range(pattern_range_axial[0],pattern_range_axial[1]):
            target[i,pattern_range_lateral[0]:pattern_range_lateral[1]]=phase
            phase=phase+dphase
        plt.figure()
        plt.imshow(target)
        plt.title("Target pattern")
        plt.show()

        target=np.expand_dims(target,axis=-1)
        target=np.expand_dims(target,axis=0)
        return target

    def load_target_img(self,target_path,plot=False):
        shape=self.prop_param.input_shape

        target=pilimg.open(target_path)
        #Resize target for cell stimulation exp (FoV: 1.5mm x 1.5mm)
        target = np.array(target)
        target[target<=np.max(target)*0.5]=0
        # target = target // np.max(target)
        target = scale_pil(target,shape[0],shape[1])
        target[target>0]=1


        if plot:
            plt.figure()
            plt.imshow(target)
            plt.title("Target pattern")
            plt.show()

        #Expand channels to put DiffPAT and DL: [batch,Ny,Nz,planes]

        target=np.expand_dims(target,axis=-1)
        target=np.expand_dims(target,axis=0)

        return target

    def test_mnist(self,image_order):
        return self.dataset.mnist_test(image_order)

    def get_phase(self,target):
        pass

    def phase_expand_func(self,phase,expand_ratio):

        phase = tf.repeat(phase, expand_ratio, axis=1)
        phase = tf.repeat(phase, expand_ratio, axis=2)
        return phase

    def target_resize_func(self,img,expand_ratio):
        expand_ratio=self.prop_param.input_shape[1]*expand_ratio
        resized = tf.image.resize(img, [expand_ratio, expand_ratio], method='bicubic')
        resized = tf.math.greater(resized,0.1)
        resized = tf.cast(resized,tf.float32)
        return resized

    def propagate(self,input_matrix,reverse=False,return_complex=False,expand_ratio=1):

        input_matrix=self.phase_expand_func(input_matrix,expand_ratio)
        #input_matrix: complex matrix [batch,Ny,Nz,1]
        prop_result=asm_propagator(input_tf=input_matrix,prop_param=self.prop_param,reverse=reverse,return_complex=return_complex,expand_ratio=expand_ratio)

        return prop_result

    def weight_cal(self, pressure, target):
        on_target = tf.math.multiply(target, pressure)
        zero_mask = tf.math.not_equal(on_target,0)
        on_target_mean=tf.reduce_mean(tf.boolean_mask(on_target,zero_mask))
        weight = tf.math.divide_no_nan(on_target_mean, pressure)
        weight = tf.math.multiply(weight,target)
        return weight


    def visualize(self,target_img,retrieved_phase,propagated_pressure,save=False):
        retrieved_phase = tf.math.angle(retrieved_phase)  # originally, retrieved phase are exp(j*angles).

        fig,(ax1,ax2,ax3)=plt.subplots(nrows=1,ncols=3)
        target=ax1.imshow(target_img[0,:,:,0],cmap='jet')
        ax1.set_title("target")
        phase=ax2.imshow(retrieved_phase.numpy()[0,:,:,0],cmap='jet')
        ax2.set_title("phase")
        propagated=ax3.imshow(propagated_pressure.numpy()[0,:,:,0],cmap='jet')
        ax3.set_title("propagation result")

        fig.colorbar(target,ax=ax1,shrink=0.4)
        fig.colorbar(phase, ax=ax2, shrink=0.4)
        fig.colorbar(propagated, ax=ax3, shrink=0.4)


        if save==True:
            plt.savefig('test.png',dpi=200)

    def assess(self,target_img,propagated_pressure,expand_ratio=1):
        target_img = self.target_resize_func(target_img,expand_ratio)
        def __accuracy(true,pred):
            # pred = tf.divide(pred, tf.reduce_max(pred))
            denom = tf.sqrt(
                tf.reduce_sum(tf.pow(pred, 2), axis=[1, 2, 3]) * tf.reduce_sum(tf.pow(true, 2), axis=[1, 2, 3]))
            return tf.reduce_mean((tf.reduce_sum(pred * true, axis=[1, 2, 3]) + 0.001) / (denom + 0.001), axis=0)

        def __PC(true,pred):
            # pred = tf.divide(pred, tf.reduce_max(pred))
            cov=tf.reduce_sum((true-tf.reduce_mean(true))*(pred-tf.reduce_mean(pred)))
            denom=tf.math.sqrt(tf.reduce_sum(tf.math.square((true-tf.reduce_mean(true)))))*tf.math.sqrt(tf.reduce_sum(tf.math.square((pred-tf.reduce_mean(pred)))))
            return cov/denom

        def __MSE(true,pred):
            pred = tf.divide(pred, tf.reduce_max(pred))
            return 1-tf.math.reduce_mean(tf.math.pow(true-pred,2))

        def __SSIM(true,pred):
            pred = tf.divide(pred, tf.reduce_max(pred))
            # pred = tf.where(pred>0.5,1.0,0.0)
            return tf.reduce_mean(tf.image.ssim(true,pred,max_val=1))

        def __Efficiency(true,pred):
            on_target = tf.math.multiply(true, pred)
            zero_mask = tf.math.not_equal(on_target, 0)
            on_target_mean = tf.reduce_mean(tf.boolean_mask(on_target, zero_mask))
            return on_target_mean

        def __uniformity(true,pred):
            on_target = tf.math.multiply(true, pred)
            zero_mask = tf.math.not_equal(on_target, 0)
            on_target_mean = tf.reduce_mean(tf.boolean_mask(on_target, zero_mask))
            on_target_std = tf.math.reduce_std(tf.boolean_mask(on_target,zero_mask))
            return 1-on_target_std/on_target_mean

        def __SD(true,pred):
            pred = tf.divide(pred, tf.reduce_max(pred))
            on_target = tf.math.multiply(true, pred)
            zero_mask = tf.math.not_equal(on_target, 0)
            on_target_mean = tf.reduce_mean(tf.boolean_mask(on_target, zero_mask))
            nom=tf.math.sqrt(tf.reduce_mean(tf.boolean_mask(tf.math.square((on_target - on_target_mean)),zero_mask)))
            return 100*(nom/on_target_mean)

        def __PSNR(true,pred):
            pred = tf.divide(pred, tf.reduce_max(pred))
            MSE=tf.math.reduce_mean(tf.math.pow(true-pred,2))
            return 10*(tf.math.log(1/MSE)/tf.math.log(10.0))


        #ACC
        ACC=__accuracy(target_img,propagated_pressure).numpy()

        #pearson's correlation coefficient
        PC=__PC(target_img,propagated_pressure).numpy()

        #Uniformity
        Uniformity=__uniformity(target_img,propagated_pressure).numpy()

        #Standard deviation
        SD=__SD(target_img,propagated_pressure).numpy()

        #PSNR
        PSNR=__PSNR(target_img,propagated_pressure).numpy()

        #SSIM
        SSIM=__SSIM(target_img,propagated_pressure).numpy()
        #Focusing efficiency
        Efficiency=__Efficiency(target_img,propagated_pressure).numpy()

        #NMSE
        MSE=__MSE(target_img,propagated_pressure).numpy()

        print("Accuracy:",ACC,"PC:",PC,", Uniformity:", Uniformity,", standard deviation:", SD,", PSNR:", PSNR,", Efficiency:", Efficiency,", SSIM:", SSIM," MSE:", MSE)
        return ACC, PC, Uniformity, SD, PSNR, Efficiency, SSIM, MSE

    def save_matrix(self,retrieved_phase,letter="notdefined",path='./test_result/'):
        retrieved_phase = tf.math.angle(retrieved_phase).numpy()[0,:,:,0]
        # retrieved_phase=retrieved_phase.numpy()[0,:,:,0]
        mat_to_save={"phase_angle":retrieved_phase}
        savemat(path+letter+'_'+'retrieved_phase.mat',mat_to_save)
        print("Phase was successfully saved as mat file")

    def save_result_pressure(self,tensor_matrix,letter="notdefined",path='./test_result/'):
        pressure_numpy=tensor_matrix.numpy()[0,:,:,0]
        pressure_max=np.max(pressure_numpy)
        # pressure_numpy=pressure_numpy-np.min(pressure_numpy)/np.max(pressure_numpy)-np.min(pressure_numpy)
        # pressure_uint=(pressure_numpy*255).astype(np.uint8)
        plt.imsave(path+letter+'_'+str(pressure_max)+'_'+'.jpeg',pressure_numpy,cmap='jet')
        np.savetxt(path+letter+'.csv', pressure_numpy, delimiter=",")


class IASA(method):
    #iterative angular spectrum apporoach
    def __init__(self,prop_param,dataset_param,GS_param):
        super(IASA, self).__init__(prop_param,dataset_param)
        self.GS_param=GS_param
        self.shape=prop_param.input_shape
        # self.physical_lim_func=physical_limit(prop_param,cal_transmission=True)
        self.physical_constraint_real_func = physical_constraint_real(prop_param, cal_transmission=False)

    def __propagate(self,input_matrix,reverse):
        #For IASA, get prop results as complex matrix
        prop_result_complex=asm_propagator(input_tf=input_matrix,prop_param=self.prop_param,reverse=reverse,return_complex=True)
        return prop_result_complex

    def source(self):
        shape=self.prop_param.input_shape
        temp_matrix = makeDisc(shape[0], shape[1], shape[0] / 2-1, shape[1] / 2-1,
                               self.prop_param.txdr_output.D_txdr_point / 2)
        temp_matrix = np.expand_dims(temp_matrix, axis=0)
        temp_matrix = np.expand_dims(temp_matrix, axis=-1)
        temp_matrix = tf.constant(temp_matrix.astype(float))
        return temp_matrix


    def get_phase(self,target,get_computation_time=False):
        # target: [batch, Ny, Nz, planes]
        iteration=self.GS_param['iteration']
        target=tf.convert_to_tensor(target,dtype=tf.float32)


        source=tf.cast(self.source(),tf.complex64)
        # A_angle = tf.random.normal(0, 1, tf.shape(target))  # initial angle distribution on txdr plane
        start=time.time()
        A=self.__propagate(tf.cast(target,tf.complex64),reverse=True) #backward propagte the target to txdr plane #output complex pressure

        for i in range(iteration):
            A = self.physical_constraint_real_func(tf.math.real(A)) + 1j * self.physical_constraint_real_func(tf.math.imag(A))  # apply txdr limit
            # A=tf.where(tf.cast(source,bool),A,0)
            B=source*tf.math.exp(1j*tf.cast(tf.math.angle(A),tf.complex64))
            C=self.__propagate(B,reverse=False)
            C_angle=tf.cast(tf.math.angle(C),tf.complex64)

            if i%20==0:
                print("iteration:", i)
                self.assess(target_img=target,propagated_pressure=tf.cast(tf.abs(C),tf.float32))

            D=tf.cast(target,tf.complex64)*tf.math.exp(1j*C_angle)
            A=self.__propagate(D,reverse=True)

        A = self.physical_constraint_real_func(tf.math.real(A)) + 1j * self.physical_constraint_real_func(tf.math.imag(A))
        # A = tf.where(tf.cast(source, bool), A, 0)
        # A = source*tf.math.exp(1j * tf.cast(tf.math.angle(A), tf.complex64))
        end=time.time()
        retrieved_phase=self.physical_lim_func(tf.math.angle(A))
        computation_time=end-start
        if get_computation_time:
            return retrieved_phase, computation_time
        else:
            return retrieved_phase

    def weight_name_generator(self):
        model_name='IASA'
        Freq=self.prop_param.Freq
        input_size=self.shape[0]

        weight_name=model_name+'_'+str(int(Freq/1e6))+'MHz_'+'_'+str(input_size)+'_'+str(int(self.prop_param.element_size*1e6))+'um'
        return weight_name

class IASA_weighted(IASA): #weighted GS algorithm
    def __init__(self,prop_param,dataset_param,GS_param):
        super(IASA_weighted, self).__init__(prop_param, dataset_param,GS_param)

    def __propagate(self,input_matrix,reverse):
        #For IASA, get prop results as complex matrix
        prop_result_complex=asm_propagator(input_tf=input_matrix,prop_param=self.prop_param,reverse=reverse,return_complex=True)
        return prop_result_complex

    def __weight_initialization(self):
        weight= np.ones((self.shape[0],self.shape[1]))
        weight = np.expand_dims(weight, axis=0)
        weight = np.expand_dims(weight, axis=-1)
        weight = tf.convert_to_tensor(weight,dtype=tf.float32)
        return weight

    def get_phase(self,target,get_computation_time=False):
        # target: [batch, Ny, Nz, planes]
        iteration=self.GS_param['iteration']
        target=tf.convert_to_tensor(target,dtype=tf.float32)


        source=tf.cast(self.source(),tf.complex64)
        weight=self.__weight_initialization()
        # A_angle = tf.random.normal(0, 1, tf.shape(target))  # initial angle distribution on txdr plane
        start=time.time()
        A=self.__propagate(tf.cast(target,tf.complex64),reverse=True) #backward propagte the target to txdr plane #output complex pressure

        for i in range(iteration):
            A = self.physical_constraint_real_func(tf.math.real(A)) + 1j * self.physical_constraint_real_func(tf.math.imag(A))  # apply txdr limit
            # A=tf.where(tf.cast(source,bool),A,0)
            B=source*tf.math.exp(1j*tf.cast(tf.math.angle(A),tf.complex64))
            C=self.__propagate(B,reverse=False)
            C_angle=tf.cast(tf.math.angle(C),tf.complex64)

            #weight calculation
            temp=tf.abs(C)
            weight=weight*self.weight_cal(temp,target)


            if i%20==0:
                print("iteration:", i)
                self.assess(target_img=target,propagated_pressure=tf.cast(tf.abs(C),tf.float32))

            D=tf.cast(weight*target,tf.complex64)*tf.math.exp(1j*C_angle)
            A=self.__propagate(D,reverse=True)

        A = self.physical_constraint_real_func(tf.math.real(A)) + 1j * self.physical_constraint_real_func(tf.math.imag(A))
        # A = tf.where(tf.cast(source, bool), A, 0)
        # A = source*tf.math.exp(1j * tf.cast(tf.math.angle(A), tf.complex64))
        end=time.time()
        retrieved_phase=self.physical_lim_func(tf.math.angle(A))
        computation_time=end-start
        if get_computation_time:
            return retrieved_phase, computation_time
        else:
            return retrieved_phase

    def weight_name_generator(self):
        model_name='IASA_weighted'
        Freq=self.prop_param.Freq
        input_size=self.shape[0]
        weight_name=model_name+'_'+str(int(Freq/1e6))+'MHz_'+'_'+str(input_size)+'_'+str(int(self.prop_param.element_size*1e6))+'um'
        return weight_name



class DiffPAT(method):

    def __init__(self, prop_param, dataset_param, DiffPAT_param):
        super(DiffPAT, self).__init__(prop_param, dataset_param)
        self.DiffPAT_param=DiffPAT_param
        self.shape=prop_param.input_shape
        self.physical_lim_func=physical_constraint(prop_param, cal_transmission=False)
        self.loss=CustomLoss(prop_param, expand_ratio=self.prop_param.prop_resize_ratio, loss_type=DiffPAT_param['loss'], intensity_lamda=float(DiffPAT_param['intensity_lamda']))

    def get_phase(self,target,get_computation_time=False):
        #target: [batch, Ny, Nz, planes]
        phase2opt=tf.Variable(tf.random.normal((1,self.shape[0],self.shape[1],1)))
        target=tf.convert_to_tensor(target,dtype=tf.float32)

        optmizer = Adam(learning_rate=self.DiffPAT_param['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        loss_fn = lambda: self.loss(target, phase2opt)
        start = time.time()
        for i in range(self.DiffPAT_param['iteration']):
            optmizer.minimize(loss_fn,phase2opt)

            if i%10==0:
                print("iteration:", i)
                self.assess(target_img=target,propagated_pressure=self.propagate(self.physical_lim_func(phase2opt)))

        phase2opt=self.physical_lim_func(phase2opt)
        end = time.time()
        computation_time = end - start

        if get_computation_time:
            return phase2opt, computation_time
        else:
            return phase2opt

    def weight_name_generator(self):
        model_name='DiffPAT'+'_'+self.DiffPAT_param['loss']
        Freq=self.prop_param.Freq
        input_size=self.shape[0]

        weight_name=model_name+'_'+str(int(Freq/1e6))+'MHz'+'_'+str(input_size)+'_'+str(int(self.prop_param.element_size*1e6))+'um'
        return weight_name



class DL(method):

    def __init__(self,prop_param,dataset_param,DL_param):
        super(DL, self).__init__(prop_param,dataset_param)
        self.loss=CustomLoss(prop_param,expand_ratio=self.prop_param.prop_resize_ratio,loss_type=DL_param['loss'],intensity_lamda=float(DL_param['intensity_lamda']))
        self.DL_param=DL_param
        self.prop_param=prop_param
        self.dataset_param=dataset_param
        self.model=eval(self.DL_param['model']+"(self.prop_param.input_shape)")
        self.physical_lim_func=physical_constraint(prop_param, cal_transmission=False)



    def __load_dataset(self,dataset_param):
        dataset=self.dataset
        dataset.getDataset()
        trainset,valset=dataset.load_data(batch_size=self.DL_param['batch_size'],shuffle=True)
        return trainset,valset

    def train(self,weights_name,train_on=False):

        name=self.DL_param['weights_path'] + weights_name + '.h5'

        logdir="logs/"+datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback=tf.keras.callbacks.TensorBoard(log_dir=logdir)
        earlystopping=EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
        input_img=self.load_target_img(target_path="Z:/3.Individuals/2019LMH/US_simulation/k-wave-toolbox-version-1.3/Simulation_script/letter_data/resize/"+"u.png",plot=False)
        custom_fun_callback=self.propagate,self.save_matrix,self.save_result_pressure

        # Load dataset to train
        train,val=self.__load_dataset(self.dataset_param)

        print(name,' is name of the weight')
        # trained weigth exists -> load the weights / not exists -> start training
        if train_on == True:
            print("Training starts")
            if os.path.isfile(name):
                self.model.load_weights(name)
                print("There's callable wieghts at", name)
                # input("Press Enter to continue...")
                exit()
            else:
                print("There's no available weights. Start at the beginning")
                # input("Press Enter to continue...")

            self.model.compile(optimizer=Adam(learning_rate=self.DL_param['learning_rate']),
                                   loss=self.loss)
            self.model.summary()
            callbacks_list = [
                ModelCheckpoint(name, save_best_only=True),
                CSVLogger('./LOSS PROGRESS/'+weights_name+"_progress.csv", append=True),tensorboard_callback,earlystopping,
            ]

            self.model.fit(train,
                           validation_data=val,
                           callbacks=callbacks_list,
                           epochs=self.DL_param['epochs']
                           )

            self.model.save(name)
        else:
            if os.path.isfile(name):
                self.model.load_weights(name)
                print("There's callable wieghts at", name)
            else:
                print("There's no callable weights",name)
                # exit()
        self.model.summary()

    def get_phase(self,target,get_computation_time=False):
        start=time.time()
        retrieved_phase=tf.convert_to_tensor(self.model.predict(target)) #angles
        retrieved_phase=self.physical_lim_func(retrieved_phase)
        end=time.time()
        computation_time=end-start

        if get_computation_time:
            return retrieved_phase, computation_time
        else:
            return retrieved_phase

    def weight_name_generator(self):
        model_name=self.DL_param['model']
        Freq=self.prop_param.Freq
        dataset_type=self.dataset_param['object_type']
        input_size=self.dataset_param['shape'][0]
        input_depth=self.dataset_param['shape'][2]
        TXDR_ele_size=int(self.prop_param.element_size*1e6) #[um]


        if self.DL_param['MULTIPLEX']==False:
            # weight_name=model_name+'_'+self.DL_param['loss']+'_'+str(self.DL_param['intensity_lamda'])\
            #             +'_'+str(int(Freq/1e6))+'MHz_'+dataset_type+'_'+str(input_size)
            weight_name=model_name+'_'+self.DL_param['loss']\
                        +'_'+str(int(Freq/1e6))+'MHz_'+dataset_type+'_'+str(input_size)+'_'+str(TXDR_ele_size)+'um'
        else:
            weight_name = model_name + '_' + str(int(Freq / 1e6)) + 'MHz_' + dataset_type + '_' + str(input_size)+'_'+str(input_depth)+'depth'
        return weight_name

