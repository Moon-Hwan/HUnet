import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate,concatenate, Lambda, Conv2D, multiply, \
    Activation,  add, UpSampling2D, MaxPooling2D, SpatialDropout2D, Conv2DTranspose
from tensorflow_addons.layers import GroupNormalization, MaxUnpooling2D
from custom_layer import MaxPoolingWithArgmax2D


initializer = tf.keras.initializers.HeNormal()

def Unet(shape):

    def Conv_block(x,filters):
        x=GroupNormalization()(x)
        x=Activation('relu')(x)
        x=Conv2D(filters,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=initializer)(x)
        return x

    def downsampling(x,filters):
        x=MaxPooling2D(pool_size=(2,2),padding='same')(x)
        x = Conv_block(x,filters)
        x = Conv_block(x, filters)
        return x

    def concat_upsample(x_small,x_big,filters):
        x = UpSampling2D((2,2))(x_small)
        x = GroupNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters,kernel_size=(2,2),padding='same',kernel_initializer=initializer)(x)
        x = Concatenate()([x,x_big])
        x = Conv_block(x, filters)
        x = Conv_block(x, filters)
        return x

    inp = Input(shape=shape,name='target')
    filter_n=64
    x=Conv2D(filters=filter_n,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same',kernel_initializer=initializer)(inp)
    x1 = Conv_block(x,filter_n)
    x2 = downsampling(x,filter_n*2)
    x3 = downsampling(x2,filter_n*4)
    x4 = downsampling(x3, filter_n * 8)
    x5 = downsampling(x4, filter_n * 16)

    x6 = concat_upsample(x5, x4, filter_n * 8)
    x7 = concat_upsample(x6, x3, filter_n * 4)
    x8=concat_upsample(x7,x2,filter_n*2)
    x9=concat_upsample(x8,x1,filter_n)

    x10=Conv2D(filters=1,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=initializer)(x9)

    outp = Activation(activation=None, name='phi')(x10)

    return Model(inp,outp)

def HUnet(shape):

    def Conv_block(x,filters):
        x=GroupNormalization()(x)
        x=Activation('relu')(x)
        x=Conv2D(filters,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=initializer)(x)
        return x

    def downsampling(x,filters):
        x=MaxPooling2D(pool_size=(2,2),padding='same')(x)
        x = Conv_block(x,filters)
        x = Conv_block(x, filters)
        return x

    def concat_upsample(x_small,x_big,filters):
        x = GroupNormalization()(x_small)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters, (5, 5), (2, 2), padding='same',kernel_initializer=initializer)(x)
        x = Concatenate()([x,x_big])
        x = Conv_block(x, filters)
        x = Conv_block(x, filters)
        return x

    inp = Input(shape=shape,name='target')
    filter_n=64
    x=Conv2D(filters=filter_n,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same',kernel_initializer=initializer)(inp)
    x1 = Conv_block(x,filter_n)
    x2 = downsampling(x,filter_n*2)
    x3 = downsampling(x2,filter_n*4)
    x4 = downsampling(x3, filter_n * 8)
    x5 = downsampling(x4, filter_n * 16)

    x6 = concat_upsample(x5, x4, filter_n * 8)
    x7 = concat_upsample(x6, x3, filter_n * 4)
    x8=concat_upsample(x7,x2,filter_n*2)
    x9=concat_upsample(x8,x1,filter_n)

    x10=Conv2D(filters=1,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=initializer)(x9)

    outp = Activation(activation=None, name='phi')(x10)

    return Model(inp,outp)

def InceptionHUnet(shape):
    def Conv_block(x_in,kernel_size=(3,3),filters=64):
        x=Conv2D(filters,kernel_size=kernel_size,strides=(1,1),padding='same',kernel_initializer=initializer)(x_in)
        x=GroupNormalization()(x)
        x=Activation('relu')(x)
        return x

    def downsampling(x,filters):
        x=MaxPooling2D(pool_size=(2,2),padding='same')(x)
        x = Conv_block(x,filters=filters)
        return x

    def concat_upsample(x_small,x_big,filters):
        x = Conv2DTranspose(filters, (5, 5), (2, 2), padding='same',kernel_initializer=initializer)(x_small)
        x = GroupNormalization()(x)
        x = Activation('relu')(x)
        x = Concatenate()([x,x_big])
        x = Conv_block(x, filters=filters)
        x = Conv_block(x, filters=filters)
        return x

    def InceptionA(x):
        #branch 1
        b1=Conv_block(x,kernel_size=(1,1),filters=96)

        #branch 2
        b2=Conv_block(x,kernel_size=(1,1),filters=64)
        b2=Conv_block(b2,kernel_size=(3,3),filters=96)
        b2=Conv_block(b2,kernel_size=(3,3),filters=96)

        #branch 3
        b3=Conv_block(x,kernel_size=(1,1),filters=64)
        b3=Conv_block(b3,kernel_size=(3,3),filters=96)

        #branch 4
        b4=MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
        b4=Conv_block(b4,kernel_size=(1,1),filters=96)

        return concatenate([b1,b2,b3,b4])

    def InceptionB(x):
        #branch 1
        b1=Conv_block(x,kernel_size=(1,1),filters=384)

        #branch 2
        b2=Conv_block(x,kernel_size=(1,1),filters=192)
        b2=Conv_block(b2,kernel_size=(1,7),filters=224)
        b2=Conv_block(b2,kernel_size=(7,1),filters=256)

        #branch 3
        b3=Conv_block(x,kernel_size=(1,1),filters=192)
        b3=Conv_block(b3,kernel_size=(1,7),filters=192)
        b3=Conv_block(b3,kernel_size=(7,1),filters=224)
        b3=Conv_block(b3,kernel_size=(1,7),filters=224)
        b3 = Conv_block(b3, kernel_size=(7, 1), filters=256)

        #branch 4
        b4=MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
        b4=Conv_block(b4,kernel_size=(1,1),filters=128)

        return concatenate([b1,b2,b3,b4])

    def InceptionC(x):
        #branch 1
        b1=Conv_block(x,kernel_size=(1,1),filters=256)

        #branch 2
        b2=Conv_block(x,kernel_size=(1,1),filters=384)
        b2_1=Conv_block(b2,kernel_size=(1,3),filters=256)
        b2_2=Conv_block(b2,kernel_size=(3,1),filters=256)

        #branch 3
        b3=Conv_block(x,kernel_size=(1,1),filters=384)
        b3=Conv_block(b3,kernel_size=(3,1),filters=448)
        b3 = Conv_block(b3, kernel_size=(1, 3), filters=512)
        b3_1 = Conv_block(b3, kernel_size=(3,1),filters=256)
        b3_2 = Conv_block(b3, kernel_size=(1,3),filters=256)

        #branch 4
        b4=MaxPooling2D(pool_size=(3,3),strides=(1,1),padding='same')(x)
        b4=Conv_block(b4,kernel_size=(1,1),filters=256)

        return concatenate([b1,b2_1,b2_2,b3_1,b3_2,b4])

    filter_n = 64

    inp = Input(shape=shape, name='target')
    x1 = Conv_block(inp,filters=filter_n)
    x1 = Conv_block(x1, filters=filter_n) #64
    x2 = downsampling(x1,filters=filter_n*2)
    x2 = InceptionA(x2) #32
    x3 = downsampling(x2,filters=filter_n*4)
    x3 = InceptionB(x3) #16
    x4 = downsampling(x3,filters=filter_n*8)
    x4 = InceptionC(x4) #8
    x5= downsampling(x4,filters=filter_n*16) #4

    x6=concat_upsample(x5,x4,filters=filter_n*8)
    # x6=InceptionC(x6)
    x7=concat_upsample(x6,x3,filters=filter_n*4)
    # x7=InceptionB(x7)
    x8=concat_upsample(x7,x2,filters=filter_n*2)
    # x8=InceptionA(x8)
    x9=concat_upsample(x8,x1,filters=filter_n)

    x10 = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', kernel_initializer=initializer)(x9)


    outp=Activation(activation=None,name='phi')(x10)

    model=Model(inp,outp)
    return model

def AttentionHUnet(shape):
    # shape: [batch,N,N,channel]
    def attention_block_2d(x,g,inter_channel):

        theta_x=Conv2D(filters=inter_channel,kernel_size=(1,1),kernel_initializer=initializer)(x)
        theta_x=GroupNormalization()(theta_x)
        phi_g=Conv2D(filters=inter_channel,kernel_size=(1,1),kernel_initializer=initializer)(g)
        phi_g=GroupNormalization()(phi_g)

        f=Activation(activation='relu')(add([theta_x,phi_g]))

        psi_f=Conv2D(filters=1,kernel_size=(1,1),strides=(1,1),kernel_initializer=initializer)(f)
        psi_f=GroupNormalization(groups=1)(psi_f)
        rate=Activation('sigmoid')(psi_f)
        att_x=multiply([x,rate])
        return att_x

    def attention_up_and_concate(down_layer,layer,K):
        #K: output channels
        in_channel=down_layer.get_shape().as_list()[3]

        x = GroupNormalization()(down_layer)
        x = Activation('relu')(x)
        up = Conv2DTranspose(K, (5, 5), (2, 2), padding='same',kernel_initializer=initializer)(x)
        layer=attention_block_2d(x=layer,g=up, inter_channel=in_channel)

        concat=concatenate([up,layer],axis=3)

        return concat

    inp=Input(shape=shape,name='target')

    depth=4
    filter_N=64
    skips=[]
    x=inp
    #Encode
    for i in range(depth):
        x=Conv2D(filters=filter_N,kernel_size=(3,3),padding='same',kernel_initializer=initializer)(x)
        x=GroupNormalization()(x)
        x=Activation(activation='relu')(x)
        # x=Dropout(0.2)(x)
        x=Conv2D(filters=filter_N,kernel_size=(3,3),activation='relu',padding='same',kernel_initializer=initializer)(x)
        x=GroupNormalization()(x)
        x=Activation(activation='relu')(x)
        skips.append(x)
        x=MaxPooling2D((2,2))(x)
        filter_N=filter_N*2

    #bottleneck
    x=Conv2D(filters=filter_N,kernel_size=(3,3),padding='same',kernel_initializer=initializer)(x)
    x=GroupNormalization()(x)
    x=Activation(activation='relu')(x)
    x=Conv2D(filters=filter_N,kernel_size=(3,3),padding='same',kernel_initializer=initializer)(x)
    x=GroupNormalization()(x)
    x=Activation(activation='relu')(x)

    #Decode
    for i in reversed(range(depth)):
        filter_N=filter_N//2
        x=attention_up_and_concate(x,skips[i],K=filter_N)
        x=Conv2D(filters=filter_N,kernel_size=(3,3),padding='same',kernel_initializer=initializer)(x)
        x=GroupNormalization()(x)
        x=Activation('relu')(x)
        x=Conv2D(filters=filter_N,kernel_size=(3,3),padding='same',kernel_initializer=initializer)(x)
        x=GroupNormalization()(x)
        x=Activation('relu')(x)

    x_=Conv2D(filters=1,kernel_size=(1,1),padding='same',kernel_initializer=initializer)(x)

    outp=Activation(None,name='phi')(x_)

    return Model(inp,outp)

def DenseHUnet(shape):
    #hyperparemters
    global compression, growthrate
    compression=1
    growthrate=0.25
    filter_n=64

    def dense_blk(x,n_filter,n_layer=4,dropout_rate=0.2):
        global growthrate
        k=int(n_filter*growthrate)
        temp = x
        for _ in range(n_layer):
            x=bottleneck_layer(temp,k*4)
            x=GroupNormalization(groups=8)(x)
            x=Activation('relu')(x)
            x=Conv2D(k,(3,3),padding='same',kernel_initializer=initializer)(x)
            if dropout_rate>0:
                    x=SpatialDropout2D(dropout_rate)(x)
            concat=Concatenate(axis=-1)([temp,x])
            temp=concat
        return temp

    def bottleneck_layer(x,filters):
        x=GroupNormalization(groups=8)(x)
        x=Activation('relu')(x)
        x=Conv2D(filters,(1,1),padding='same',kernel_initializer=initializer)(x)
        return x

    def transition_layer(x): #transition (downsampling)

        x=GroupNormalization(groups=8)(x)
        x=Conv2D(int(x.shape[-1]*compression),(1,1),use_bias=False,padding='same',kernel_initializer=initializer)(x)
        x=MaxPooling2D(pool_size=(2,2))(x)
        return x

    def concat_upsample(x_small,x_big,filters):
        x=GroupNormalization(groups=8)(x_small)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters,kernel_size=(5,5),strides=(2,2),padding='same',kernel_initializer=initializer)(x)
        x=  Concatenate()([x,x_big])
        x=GroupNormalization(groups=8)(x)
        x = Activation('relu')(x)
        x = Conv2D(filters/2, (3,3),padding='same',kernel_initializer=initializer)(x)
        return x


    inp = Input(shape=shape,name='target')
    x=Conv2D(filters=filter_n,kernel_size=(3,3),strides=(1,1),activation='relu',padding='same',kernel_initializer=initializer)(inp)
    x = dense_blk(x,n_filter=filter_n) #32 -> 64
    down1 = transition_layer(x) #down sampling 1
    down1 = dense_blk(down1,n_filter=filter_n*2) #64 -> 128
    down2 = transition_layer(down1) # down sampling 2
    down2 = dense_blk(down2,n_filter=filter_n*4) #128->256
    down3 = transition_layer(down2) # down sampling 3
    down3 = dense_blk(down3, n_filter=filter_n*8) #256->512

    bottleneck = transition_layer(down3)
    bottleneck = dense_blk(bottleneck, n_filter=filter_n*16) #512->1024

    up3= concat_upsample(bottleneck,down3,filter_n*16) #1024->512 512+512->1024, 1024 -> 256
    up3 = dense_blk(up3,n_filter=filter_n*8) # 256 -> 512
    up2 = concat_upsample(up3,down2,filter_n*8) #512 -> 256 256+256->512, 512->128
    up2 = dense_blk(up2,filter_n*4)
    up1 = concat_upsample(up2,down1,filter_n*4)
    up1 = dense_blk(up1, filter_n*2)
    out = concat_upsample(up1,x,filter_n*2)
    out = dense_blk(out,filter_n)
    out = Conv2D(filters=1,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=initializer)(out)
    outp = Activation(activation=None, name='phi')(out)
    return Model(inp, outp)

def FusionHUnet(shape):

    def conv_block(x,kernel_n):
        x=Conv2D(kernel_n,(3,3),padding='same',kernel_initializer=initializer)(x)
        x=GroupNormalization()(x)
        x=Activation('relu')(x)
        return x

    def conv_block_R(x,kernel_n):
        x=GroupNormalization()(x)
        x=Activation('relu')(x)
        x=Conv2D(kernel_n,(3,3),padding='same',kernel_initializer=initializer)(x)
        return x

    def res_block(x, kernel_n):
        skip_connection=x
        res_1=conv_block_R(x,kernel_n)
        res_2=conv_block_R(res_1,kernel_n)
        res_3 = conv_block_R(res_2, kernel_n)
        out=add([res_3,skip_connection])
        return out

    def conv_res_block(x,kernel_n):
        x=conv_block(x,kernel_n)
        x=res_block(x,kernel_n)
        x=conv_block(x,kernel_n)
        return x

    def deconv_block(x,kernel_n):
        x=Conv2DTranspose(kernel_n,(5,5),strides=(2,2),padding='same',kernel_initializer=initializer)(x)
        x=GroupNormalization()(x)
        x=Activation('relu')(x)
        return x


    kernels=64
    skip=[]
    input=Input(shape=shape,name='target')

    #Encoder
    down1=conv_res_block(input,kernels)
    skip.append(down1)
    down1=MaxPooling2D((2,2),padding='same')(down1)
    kernels=kernels*2 #128


    down2=conv_res_block(down1,kernels)
    skip.append(down2)
    down2=MaxPooling2D((2,2),padding='same')(down2)
    kernels = kernels * 2 # 256


    down3=conv_res_block(down2,kernels)
    skip.append(down3)
    down3=MaxPooling2D((2,2),padding='same')(down3)
    kernels = kernels * 2 #512


    down4=conv_res_block(down3,kernels)
    skip.append(down4)
    down4=MaxPooling2D((2,2),padding='same')(down4)
    kernels=kernels*2 #1024


    # Bridge
    bridge=conv_res_block(down4,kernels)
    kernels=kernels/2   #512

    # Decoder
    up4=deconv_block(bridge,kernels)
    up4=add([up4,skip[3]])
    up4=conv_res_block(up4,kernels)
    kernels=kernels/2   #256

    up3=deconv_block(up4,kernels)
    up3=add([up3,skip[2]])
    up3=conv_res_block(up3,kernels)
    kernels=kernels/2   #128

    up2=deconv_block(up3,kernels)
    up2=add([up2,skip[1]])
    up2=conv_res_block(up2,kernels)
    kernels=kernels/2   #64

    up1=deconv_block(up2,kernels)
    up1=add([up1,skip[0]])
    up1=conv_res_block(up1,kernels)

    out=Conv2D(1,(1,1),padding='same',kernel_initializer=initializer)(up1)
    out=Activation(activation=None,name='phi')(out)

    return Model(input,out)

def SRResNet(shape):

    def res_block(x_in,filters,scaling):
        x = Conv2D(filters,3,padding='same',kernel_initializer=initializer)(x_in)
        x = GroupNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters,3,padding='same',kernel_initializer=initializer)(x)
        x = GroupNormalization()(x)
        x = add([x_in,x])
        if scaling:
            x=Lambda(lambda t:t*scaling)(x)
        return x

    filter_n=256
    num_res_blocks=8
    res_block_scaling=None
    inp = Input(shape=shape, name='target')
    x = b = Conv2D(filter_n,3,padding='same',kernel_initializer=initializer)(inp)
    b = GroupNormalization()(b)
    for _ in range(num_res_blocks):
        b=res_block(b,filter_n,res_block_scaling)

    b=Conv2D(filter_n,3,padding="same")(b)
    b = GroupNormalization()(b)
    x = Conv2D(1, 1, padding="same",kernel_initializer=initializer)(b)

    outp=Activation(activation=None,name='phi')(x)

    model=Model(inp,outp)
    return model

def MDHGN(shape):

    def downsampling(x,K):
        x=Conv2D(filters=K,kernel_size=(4,4),strides=(2,2),padding='same')(x)
        x=GroupNormalization()(x)
        x=Activation(activation='relu')(x)
        return x
    def upsampling(x,K):
        x=UpSampling2D(size=(2,2))(x)
        x=Conv2D(filters=K,kernel_size=(3,3),strides=(1,1),padding='same')(x)
        x=GroupNormalization()(x)
        x=Activation(activation='relu')(x)
        return x
    def residual(x,K):
        shortcut=x
        x=Conv2D(filters=K,kernel_size=(3,3),strides=(1,1),padding='same')(x)
        x=GroupNormalization()(x)
        x=Activation(activation='relu')(x)
        x = Conv2D(filters=K, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = GroupNormalization()(x)
        x = add([shortcut,x])
        return x


    K=128 #initial filter numbers

    inp=Input(shape=shape,name='target')

    x_1=Conv2D(filters=K,kernel_size=(3,3),strides=(1,1),padding='same')(inp)

    x_2=downsampling(x_1,K*2)
    x_3=downsampling(x_2,K*4)


    x_r=residual(x_3,K*4) #1
    x_r = residual(x_r, K * 4) #2
    x_r = residual(x_r, K * 4) #3
    x_r = residual(x_r, K * 4) #4
    x_r = residual(x_r, K * 4) #5
    x_r = residual(x_r, K * 4) #6
    x_r = residual(x_r, K * 4) #7
    x_r = residual(x_r, K * 4) #8
    x_r = residual(x_r, K * 4) #9
    x_r = residual(x_r, K * 4) #10
    x_r = residual(x_r, K * 4) #11
    x_r = residual(x_r, K * 4) #12
    x_r = residual(x_r, K * 4) #13
    x_r = residual(x_r, K * 4) #14
    x_r = residual(x_r, K * 4)  # 15

    x_4=upsampling(x_r,K*2)
    x_5=upsampling(x_4,K)

    x_6=Conv2D(filters=1,kernel_size=(7,7),strides=(1,1),padding='same')(x_5)
    outp=Activation(activation=None,name='phi')(x_6)

    model=Model(inp,outp)
    return model

def SegNet(shape):

    def Conv_block(x,filters):
        x=Conv2D(filters,(3,3),padding='same',kernel_initializer=initializer)(x)
        x=GroupNormalization()(x)
        x=Activation('relu')(x)
        return x

    inp = Input(shape=shape, name='target')
    conv1=Conv_block(inp,64)
    conv2=Conv_block(conv1,64)

    pool_1,mask_1=MaxPoolingWithArgmax2D((2,2))(conv2)

    conv3=Conv_block(pool_1,128)
    conv4=Conv_block(conv3,128)

    pool_2,mask_2=MaxPoolingWithArgmax2D((2,2))(conv4)

    conv5=Conv_block(pool_2,256)
    conv6=Conv_block(conv5,256)
    conv7=Conv_block(conv6,256)

    pool_3, mask_3 = MaxPoolingWithArgmax2D((2,2))(conv7)

    conv8 = Conv_block(pool_3,512)
    conv9 = Conv_block(conv8,512)
    conv10 = Conv_block(conv9,512)

    pool_4, mask_4 = MaxPoolingWithArgmax2D((2,2))(conv10)

    conv11 = Conv_block(pool_4,512)
    conv12 = Conv_block(conv11,512)
    conv13 = Conv_block(conv12,512)

    pool_5, mask_5 = MaxPoolingWithArgmax2D((2,2))(conv13)

    unpool_1 = MaxUnpooling2D((2,2))(pool_5, mask_5)

    conv14 = Conv_block(unpool_1,512)
    conv15 = Conv_block(conv14,512)
    conv16 = Conv_block(conv15,512)

    unpool_2 = MaxUnpooling2D((2,2))(conv16,mask_4)

    conv17 = Conv_block(unpool_2,512)
    conv18 = Conv_block(conv17,512)
    conv19 = Conv_block(conv18, 256)

    unpool_3 = MaxUnpooling2D((2, 2))(conv19, mask_3)

    conv20 = Conv_block(unpool_3, 256)
    conv21 = Conv_block(conv20, 256)
    conv22 = Conv_block(conv21, 128)

    unpool_4 = MaxUnpooling2D((2, 2))(conv22, mask_2)

    conv23 = Conv_block(unpool_4, 128)
    conv24 = Conv_block(conv23, 64)

    unpool_5 = MaxUnpooling2D((2, 2))(conv24, mask_1)

    conv25 = Conv_block(unpool_5,64)
    conv26 = Conv2D(1,(1,1),padding='same',kernel_initializer=initializer)(conv25)

    outp = Activation(activation=None, name='phi')(conv26)

    return Model(inp,outp)








