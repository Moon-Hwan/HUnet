import tensorflow as tf
print(tf.__version__)

from Variables import *
from HUG_method import DiffPAT,IASA, IASA_weighted, DL
import string
import csv


######### 1. Check simulation environment ####################################
check_sim_env()


######### 2. Decide method for phase retrieval and initialization ############
    ## If it needs to be trained, training is processed in this step.

Method=3 #
if Method==1:
# -------------- 1 - DiffPAT ----------------------------------------------
# loss='cosine_similarity+intensity'
    loss='cosine_similarity'
    intensity_lamda='0.1'
    DiffPAT_param['loss']=loss
    DiffPAT_param['intensity_lamda']=intensity_lamda
    algorithm=DiffPAT(prop_param, dataset_param, DiffPAT_param)
    weights_name=algorithm.weight_name_generator()

elif Method==2:
# -------------- 2 - Iterative angular spectrum approach ------------------
    algorithm=IASA(prop_param,dataset_param,IASA_param) # IASA_weighted can also be used.
    weights_name=algorithm.weight_name_generator()
elif Method==3:
# -------------- 3 - Deep-learning ---------------------------------------
    model='FusionHUnet' ## Unet, HUnet, InceptionHUnet, DenseHUnet , FusionHUnet, MDHGN, SegNet, SRResNet
    loss='cosine_similarity+intensity' # 'cosine_similarity', 'cosine_similarity+intensity'
    intensity_lamda='0.1'
    element_size='750e-6'

    DL_param['model']=model
    DL_param['loss']=loss
    DL_param['intensity_lamda']=intensity_lamda
    prop_param.element_size=float(element_size)

    algorithm=DL(prop_param,dataset_param,DL_param)
    print(DL_param['model']," will be load")
    weights_name=algorithm.weight_name_generator()
    algorithm.train(weights_name,train_on=False)

#-------------------------------------------------------------------------------

TEST_custom_target=True
if TEST_custom_target:
######## 3. Load custom target image ############################################
    target_path="./test_data/"
    target_name="h.png"
    target_img=algorithm.load_target_img(target_path=target_path+target_name,plot=True) #[batch,Ny,Nz,planes]


######## 4. Phase retrieval (hologram generation) ################################
    retrieved_phase=algorithm.get_phase(target_img)

######## 5. Propagate the phase to target plane   ################################
    expand_ratio=prop_param.prop_resize_ratio
    propagated_pressure=algorithm.propagate(retrieved_phase,return_complex=False,expand_ratio=expand_ratio)

######## 6. Visualize the result & assess the performance ########################
    algorithm.visualize(target_img,retrieved_phase,propagated_pressure)
    algorithm.assess(target_img,propagated_pressure,expand_ratio=expand_ratio)


######## 7. Save the results if it is needed. ####################################
    # algorithm.save_result_pressure(propagated_pressure)
    # algorithm.save_matrix(retrieved_phase)


######## 8. Assess on images of dots, U and H & Save pressure and phase as mat file
TEST_Dot_U_H=False
if TEST_Dot_U_H:
    createFolder('./test_result/'+weights_name)
    expand_ratio=prop_param.prop_resize_ratio
    test_path='./test_result/'+weights_name+'/'
    target_path="./test_data/"
    letter_array=['dots','u','h'] #dots4, u, h

    for letter in letter_array:
        target_name = letter+".png"
        target_img = algorithm.load_target_img(target_path=target_path + target_name,plot=False)  # [batch,Ny,Nz,planes]
        retrieved_phase,computation_time = algorithm.get_phase(target_img,get_computation_time=True)
        propagated_pressure = algorithm.propagate(retrieved_phase, return_complex=False, expand_ratio=expand_ratio)
        assess_result=algorithm.assess(target_img, propagated_pressure, expand_ratio=expand_ratio)
        algorithm.save_result_pressure(propagated_pressure,letter,test_path)
        algorithm.save_matrix(retrieved_phase,letter,test_path)
        algorithm.visualize(target_img,retrieved_phase,propagated_pressure)




######## 9. Test for MNIST data (10000 numbers)  ####################################
TEST_MNIST=False
if TEST_MNIST:

    createFolder('./test_mnist/'+weights_name)
    expand_ratio=prop_param.prop_resize_ratio
    test_path='./test_mnist/'+weights_name+'/'

    f = open(test_path+weights_name+'_test.csv','w',newline='')
    wr=csv.writer(f)
    wr.writerow(['letter','cosine similarity','PC','Uniformity','SD','PSNR','Efficiency','SSIM','MSE','Computation_time'])
    alphabet_upper=string.ascii_uppercase
    for i in range(0,10000):
        target_img=algorithm.test_mnist(i)
        retrieved_phase,computation_time = algorithm.get_phase(target_img,get_computation_time=True)
        propagated_pressure = algorithm.propagate(retrieved_phase, return_complex=False, expand_ratio=expand_ratio)
        assess_result=algorithm.assess(target_img, propagated_pressure, expand_ratio=expand_ratio)
        wr.writerow([i+1, assess_result[0], assess_result[1], assess_result[2], assess_result[3], assess_result[4],
                     assess_result[5], assess_result[6], assess_result[7], computation_time])
    f.close()



