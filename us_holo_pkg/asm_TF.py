import tensorflow as tf
from tensorflow.keras.layers import AveragePooling2D, UpSampling2D, ZeroPadding2D
from PIL import Image
import numpy as np
import math
from us_holo_pkg.makeshape import makeDisc

#####################

class physical_constraint(tf.keras.layers.Layer): # physical constraint layer for phase. Output complex pressure field
    #input: angle in radian [-pi,pi]
    def __init__(self,prop_param,cal_transmission):
        self.prop_param=prop_param
        self.cal_transmission=cal_transmission
        self.padding=self.prop_param.PC_padding
    def __call__(self, phase2Blimited, **kwargs):
        # input: tf.complex64, [batch,N,M,planes] (channels last)
        averaged=self.average_pool_limit(phase2Blimited)
        expanded=self.expand_repeat(averaged)
        return expanded

    def get_config(self):
        config=super(physical_constraint, self).get_config()
        config["prop_param"]=self.prop_param
        return config

    def expand_repeat(self,phase):
        # repeat to expand and make to complex plane (with retreived angles)
        shape=self.prop_param.input_shape
        elem_pos = self.prop_param.txdr_output.ele_pos.astype(int)
        ele_num = max(elem_pos.shape)
        points = self.prop_param.txdr_output.points
        ele_size = points.width

        phase=UpSampling2D((ele_size,ele_size))(phase)
        if self.padding>0:
            # If the size of deep learning-generated hologram is not matched to input image size (due to the network's characteristics),
            # force to match the hologram size to the input size.
            phase = ZeroPadding2D(padding=((self.padding, 0), (self.padding, 0)))(phase)

        complex_plane = tf.math.exp(1j*tf.cast(phase,tf.complex64))


        if self.cal_transmission:
            alpha_t=tf.cast(self.transmission_coeff(phase),tf.complex64)
            complex_plane=tf.math.multiply(alpha_t,complex_plane)
        # make outside of txdr plane to zero

        # apply txdr element limitation (assign same phase in each single element)
        if self.prop_param.txdr_output.single_txdr:
            temp_matrix = makeDisc(shape[0], shape[1], shape[0] / 2-1, shape[1] / 2-1, self.prop_param.txdr_output.D_txdr_point / 2)
            temp_matrix = tf.constant(temp_matrix)
            temp_matrix = tf.expand_dims(temp_matrix,axis=0)
            temp_matrix = tf.expand_dims(temp_matrix, axis=-1)
            return tf.where(temp_matrix, complex_plane, 0)
            # expanded_phase=temp_matrix*complex_plane
        else:
            temp_matrix = np.zeros([shape[0], shape[1]])
            temp_matrix[elem_pos[0, 0].item():elem_pos[ele_num - 1, 0].item() + ele_size,
            elem_pos[0, 1].item():elem_pos[ele_num - 1, 1].item() + ele_size] = 1
            temp_matrix = tf.expand_dims(temp_matrix,axis=0)
            temp_matrix = tf.expand_dims(temp_matrix, axis=-1)
            return tf.where(temp_matrix, complex_plane, 0)



    def average_pool_limit(self,phase):
        txdr_points_spec = self.prop_param.txdr_output.points
        ele_size = txdr_points_spec.width

        if self.padding>0:
            # If the size of deep learning-generated hologram is not matched to input image size (due to the network's characteristics),
            # force to match the hologram size to the input size.
            phase = phase[:, self.padding:, self.padding:, :]
        averaged_phase = AveragePooling2D(pool_size=(ele_size, ele_size), strides=(ele_size, ele_size))(phase)
        return averaged_phase

    def transmission_coeff(self,phase):
        # calculate transmission coefficient
        Freq=self.prop_param.Freq

        Z_lens=self.prop_param.lens_impedance
        Z_medium=self.prop_param.medium.sound_speed*self.prop_param.medium.density
        Z_txdr=self.prop_param.txdr_impedance
        SoS_lens=self.prop_param.lens_SoS
        SoS_water=self.prop_param.medium.sound_speed

        #calculate thickness
        substrate=0.5e-3 # to make the lens rigid
        thickness=tf.math.divide(phase,tf.constant(2*math.pi*Freq*(1/SoS_water-1/SoS_lens)))
        thickness=thickness-tf.reduce_min(thickness)
        thickness=thickness+tf.constant(substrate)

        #calculate transmission coefficient
        k_h=Freq*2*math.pi/SoS_lens #wave vector of lens
        nom=4*Z_txdr*Z_lens**2*Z_medium
        denom=Z_lens**2*(Z_txdr+Z_medium)**2*tf.math.cos(k_h*thickness)**2+(Z_lens**2+Z_txdr*Z_medium)**2*tf.math.sin(k_h*thickness)**2
        alpha_t=tf.math.sqrt(tf.math.divide(nom,denom))
        return alpha_t


class physical_constraint_real(physical_constraint): # physical constraint layer for any real number. Output real number.
    #input: angle in radian [-pi,pi]
    def __init__(self,prop_param,cal_transmission):
        super(physical_constraint_real, self).__init__(prop_param, cal_transmission)

    def __call__(self, phase2Blimited, **kwargs):
        # input: tf.complex64, [batch,N,M,planes] (channels last)
        averaged=self.average_pool_limit(phase2Blimited)
        expanded=self.expand_repeat(averaged)
        return expanded

    def get_config(self):
        config=super(physical_constraint, self).get_config()
        config["prop_param"]=self.prop_param
        return config

    def expand_repeat(self,real_number):
        # repeat to expand and make to complex plane (with retreived angles)
        shape=self.prop_param.input_shape
        elem_pos = self.prop_param.txdr_output.ele_pos.astype(int)
        ele_num = max(elem_pos.shape)
        points = self.prop_param.txdr_output.points
        ele_size = points.width

        real_number = UpSampling2D((ele_size, ele_size))(real_number) # nearest neighborhood upsampling to match the simulation grid size

        if self.padding>0:
            real_number = ZeroPadding2D(padding=((self.padding, 0), (self.padding, 0)))(real_number)
        real_number = tf.cast(real_number, tf.complex64)

        # make outside of txdr plane to zero
        if self.prop_param.txdr_output.single_txdr:
            temp_matrix = makeDisc(shape[0], shape[1], shape[0] / 2-1, shape[1] / 2-1, self.prop_param.txdr_output.D_txdr_point / 2)
            temp_matrix = tf.expand_dims(temp_matrix,axis=0)
            temp_matrix = tf.expand_dims(temp_matrix, axis=-1)
            return tf.where(temp_matrix, real_number, 0)
        else:
            temp_matrix = np.zeros([shape[0], shape[1]])
            temp_matrix[elem_pos[0, 0].item():elem_pos[ele_num - 1, 0].item() + ele_size,
            elem_pos[0, 1].item():elem_pos[ele_num - 1, 1].item() + ele_size] = 1
            temp_matrix = tf.expand_dims(temp_matrix,axis=0)
            temp_matrix = tf.expand_dims(temp_matrix, axis=-1)
            return tf.where(temp_matrix, real_number, 0)


def asm_propagator(input_tf,prop_param,reverse=False,return_complex=False,expand_ratio=1):
    # Referred ASM
    # input_tf=#[1,Ny,Nz,1]
    # set the variables from propagation parameter group
    shape=tuple(ele*expand_ratio for ele in prop_param.input_shape)
    medium=prop_param.medium
    Freq=prop_param.Freq
    Zs=prop_param.prop_distance
    dx=prop_param.grid_spacing/expand_ratio
    padding=True
    angular_restriction=True

    input_tf=tf.squeeze(input_tf,axis=-1) #[1,Ny,Nz]

    # set the plane of txdr
    wavelength=medium.sound_speed/Freq


    alpha_Np=db2neper(medium.alpha_coeff,medium.alpha_power)*(2*np.pi*Freq)**medium.alpha_power
    if alpha_Np!=0:
        absorbing=True

    # check whether forward prop or backward prop
    if reverse==True:
        input_tf=tf.math.conj(input_tf)

    # padding
    if padding:
        pad_size=int(shape[1])
        txdr_complex_tf=tf.pad(input_tf,[[0,0],[pad_size,pad_size],[pad_size,pad_size]])
        N = int(shape[1])*3

    else:
        txdr_complex_tf=input_tf
        N = int(shape[1])*1
    # compute wave number at driving frequency
    if N%2 == 0:
        k_vec=np.linspace(-N/2,N/2-1,N)*2*np.pi/(N*dx) # create wave number vector
    else:
        k_vec = np.linspace(-(N-1) / 2, (N-1) / 2, N) * 2 * np.pi / (N * dx)  # create wave number vector
    k_vec[math.ceil(N/2)]=0
    k_vec = np.fft.ifftshift(k_vec)
    k=2*np.pi/wavelength # wave number at driving frequency
    kx,ky=np.meshgrid(k_vec,k_vec) #create wavenumber grids
    kx_ky_power=np.power(kx,2)+np.power(ky,2)
    kz=kx_ky_power.astype(np.complex64)
    kz=np.sqrt(np.power(k,2)-kz)
    sqrt_kx2_ky2=np.sqrt(kx_ky_power)

    # compute forward FFT of input plane
    input_fft_tf=tf.signal.fft2d(txdr_complex_tf) # np and tf show same result

    ############ Z-Loop for propagate to multi planes ###################################
    output_list=[]
    for i in range(0,len(Zs)):
        Z=Zs[i]

        # compute H propagator
        H = np.conjugate(np.exp(1j * Z * kz))

        if absorbing:
            H = np.multiply(H, np.exp(np.divide(-alpha_Np * Z * k, kz,out=np.zeros_like(kz),where=kz!=0)))

        # angular restriction
        if angular_restriction:
            fft_length=2**nextpow2(N)
            D=(fft_length-1)*dx # size of computational domain [m]
            kc=k*math.sqrt(0.5*D**2/(0.5*D**2+np.power(Z,2)))
            restrict_condition=sqrt_kx2_ky2>kc
            H=np.where(restrict_condition,0,H)


        H=tf.expand_dims(tf.constant(H,dtype=tf.complex64),axis=0)

        #########################check ifftshift after and before the meshgrid###########################
        # k_vec_fftshift=k_vec=np.fft.ifftshift(k_vec)
        # kx_shift,ky_shift=np.meshgrid(k_vec_fftshift,k_vec_fftshift)
        # kx=np.fft.ifftshift(kx)
        ################################################################################################# both are same

        # angular spectrum method
        #compute using tf
        output_tf=tf.signal.ifft2d(tf.math.multiply(input_fft_tf,H))
        if padding:
            output_tf_comp=output_tf[:,pad_size:-pad_size,pad_size:-pad_size]
        else:
            output_tf_comp=output_tf

    ####################################################################################################
        output_list.append(output_tf_comp)
    output_tf=tf.stack(output_list,axis=3)

    if reverse==True:
        output_tf=tf.math.conj(output_tf)
        output_tf=tf.reverse(output_tf,[0])

    # Check whether out is amplitude or complex
    if return_complex==True:
        return output_tf
    else:
        return tf.cast(tf.abs(tf.pow(output_tf,1)),dtype=tf.dtypes.float32)


def scale(im, nR, nC):
    nR0=len(im)
    nC0=len(im[0])
    return [[im[int(nR0*r/nR)][int(nC0*c/nC)]for c in range(nC)]for r in range(nR)]

def scale_pil(im,sizeR,sizeC):

    im_arr=Image.fromarray(im)
    im_arr=im_arr.resize((sizeR,sizeC),Image.BILINEAR)
    return np.array(im_arr)


def nextpow2(shape):
    return int(np.ceil(np.log2(shape+1)).astype(int))

def phase_expand_func(phase,expand_ratio):
    phase = tf.repeat(phase, expand_ratio, axis=1)
    phase = tf.repeat(phase, expand_ratio, axis=2)
    return phase

def target_resize_func(img,expand_ratio,prop_param):
    expand_ratio=prop_param.input_shape[1]*expand_ratio
    resized = tf.image.resize(img, [expand_ratio, expand_ratio], method='bicubic')
    resized = tf.math.greater(resized,0.5)
    resized = tf.cast(resized,tf.float32)
    return resized

def db2neper(alpha,y):
    # Convert decibels to nepers
    # dB/(MHz^y cm) to units of Nepers/((rad/s)^y m)
    alpha=100*alpha*(1e-6/(2*math.pi))**y/(20*math.log10(math.exp(1)))
    return alpha
