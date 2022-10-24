import numpy as np
import math
from us_holo_pkg import env_param
from us_holo_pkg.makeshape import makeDisc

def apply_txdr(pressure, ratio, txdr_output):
    # Apply phisical limit of TXDR in to pressure matrix
    # author: Moon Hwan Lee
    # date: 30th December 2020
    # last update: 30th December 2020
    #
    # INPUT:
    #     txdr_output:
    #         txdr_mask=txdr_output.mask[0,:,:]
    #         elem_pos=txdr_output.ele_pos; [y z]
    #         txdr_points=txdr_output.points; kerf, width, height
    #         ratio: ratio between simulation plane and GS plane
    #         pressure; pressure matrix (2D ndarray)
    # OUTPUT: physical limitation applied matrix

    # call the variable
    txdr_mask=txdr_output.mask[0,:,:]
    elem_pos=txdr_output.ele_pos
    ele_num=elem_pos.shape[0]
    txdr_points=txdr_output.points
    width=txdr_points.width
    p_shape=pressure.shape
    if pressure.ndim==2:
        pressure=pressure.reshape(p_shape[0],p_shape[1],1)
    Z_num=pressure.shape[2]

    #call the size of the simulation plane and GS plane
    # sim_size=txdr_mask.shape
    # GS_size=np.multiply(sim_size,ratio)
    # GS_size=GS_size.tolist()

    matrix=np.zeros(pressure.shape,dtype=complex)
    for k in range(0,Z_num):
        #apply shrinking to match the GS size with the simulation plane size
        pressure_limited=pressure[:,:,k]
        # pressure_limited=pressure_limited[0:GS_size[0]:int(GS_size[0]/sim_size[0]),0:GS_size[1]:int(GS_size[1]/sim_size[1])]
        pressure_limited[np.where(txdr_mask!=1)]=0

        #average phase to assign same phase in each single element
        for i in range(0,ele_num):
            pressure_limited[int(elem_pos[i, 0].item()):int(elem_pos[i, 0].item()) + width,
            int(elem_pos[i, 1].item()):int(elem_pos[i, 1].item()) + width]=np.mean(pressure_limited[int(elem_pos[i,0].item()):int(elem_pos[i,0].item())+width,
                      int(elem_pos[i,1].item()):int(elem_pos[i,1].item())+width])

        #apply expanding (to match the GS plane size)
        # pressure_limited=np.kron(pressure_limited,np.ones((int(GS_size[0]/sim_size[0]),int(GS_size[1]/sim_size[1]))))
        matrix[:, :, k] = pressure_limited

    return matrix

def make_txdr(txdr_param):
    # % make binary transducer for k-wave simulation in python
    # %
    # % author: Moon Hwan Lee
    # % date: 27th December 2020
    # % last update: 27th December 2020
    # %
    # % Input (struct)
    # %     grid_spacing : dx, dy, dz (usually, all spacings are same)
    # %     total number of grid points not including PML : Nx, Ny, Nz
    # %     TXDR type : 'signle_rect', 'single_cir', 'array_1D', 'array_2D'
    # %     element size: (width -> y, height -> z ; 추후 확인 후 수정 필요)
    # %               Single element TXDR; square: width, height / circle:
    # %               diameter (circle은 아직 구현되지 않음)
    # %               Array; 1D array: width, height, kerf, ele_number_width
    # %                       2D array: width, height, kerf (dense array),
    # %                       ele_number_width, ele_number_height
    # % Output: struct
    # % txdr_binary: grid which has the TXDR
    # % ele_pos: position of elements' center [y z] (x=0; default)
    # % center_TXDR: center position of the TXDR
    # % points: the number of points of kerf, ele_width, ele_height (struct)
    grid_spacing=txdr_param.grid_spacing
    grids_number=txdr_param.total_grids
    Nx=(grids_number[0]).item()
    Ny=grids_number[1].item()
    Nz=grids_number[2].item()
    grid = np.zeros((Nx,Ny,Nz))
    txdr_position=np.array([0, math.ceil(Ny/2), math.ceil(Nz/2)]) # middle of the plane
    D_txdr_point=round(txdr_param.D_txdr / grid_spacing)

    txdr_points=env_param.txdr_points
    txdr_output=env_param.txdr_output

    ##create TXDR
    #create element
    txdr_size=txdr_param.size
    txdr_size=txdr_size.tolist()
    points_width=round(txdr_size[0]/grid_spacing)
    points_height=round(txdr_size[1]/grid_spacing)
    element=np.ones((points_width,points_height))

    #merge elements to build 2D array
    #First, define 1D array in width direction, then merge them in height direction to define 2D array
    #1D array in width direction
    points_kerf=round(txdr_size[2]/grid_spacing)
    kerf=np.zeros((points_kerf,points_height))
    temp_txdr=kerf
    for i in range(0,int(txdr_size[3])):
        temp_txdr=np.concatenate((temp_txdr,element),axis=0)
        temp_txdr = np.concatenate((temp_txdr, kerf), axis=0)
    temp_1darray=temp_txdr
    temp_txdr_width=max(temp_txdr.shape)

    #merge 1D arrays (height direction)
    kerf=np.zeros((temp_txdr_width,points_kerf))
    temp_txdr=np.array([])
    temp_txdr=np.concatenate((kerf,temp_1darray),axis=1)
    temp_txdr=np.concatenate((temp_txdr, kerf), axis=1)
    for i in range (0,int(txdr_size[4]-1)):
        temp_txdr = np.concatenate((temp_txdr, temp_1darray), axis=1)
        temp_txdr = np.concatenate((temp_txdr, kerf), axis=1)
    final_size=temp_txdr.shape
    final_txdr_width=final_size[0]
    final_txdr_height=final_size[1]

    #Place the TXDR on the middel of the plane
    first_corner=txdr_position-np.array([0, math.ceil(final_txdr_width/2), math.ceil(final_txdr_height/2)])
    first_corner=first_corner.tolist()
    grid[0,first_corner[1]:first_corner[1]+final_txdr_width,first_corner[2]:first_corner[2]+final_txdr_height]=temp_txdr

    #make matrix to indicate the position of each elements' center
    ele_1st_H=first_corner[2]+points_kerf-1
    ele_1st_H_matrix=np.ones([1,int(txdr_size[3])])*ele_1st_H
    ele_H_spacing_matrix=np.ones([1,int(txdr_size[3])])*(points_kerf+points_height)
    ele_1st_W=first_corner[1]+points_kerf-1
    ele_last_W=ele_1st_W+(txdr_size[3]-1)*(points_kerf+points_width)
    ele_pos_W_matrix=list(range(ele_1st_W, int(ele_last_W) + points_kerf + points_height, points_kerf + points_height))
    ele_pos_W_matrix=np.array(ele_pos_W_matrix)
    ele_pos_W_matrix=np.reshape(ele_pos_W_matrix,(1,ele_pos_W_matrix.shape[0]))
    ele_pos_matrix=np.array([])
    for i in range(1,int(txdr_size[4])+1):
        ele_pos_H_matrix=ele_1st_H_matrix+ele_H_spacing_matrix*(i-1)
        ele_pos_matrix_temp=np.concatenate((ele_pos_W_matrix,ele_pos_H_matrix),axis=0)
        if i==1:
            ele_pos_matrix=ele_pos_matrix_temp
        else:
            ele_pos_matrix=np.concatenate((ele_pos_matrix,ele_pos_matrix_temp),axis=1)
    ele_pos_matrix=ele_pos_matrix.transpose()

    #indicate center of the TXDR
    center_txdr=np.array([0,round(first_corner[1]+final_txdr_width/2),round(first_corner[2]+final_txdr_height/2)])

    if txdr_param.single_txdr:
        Single_mask=makeDisc(Ny,Nz,math.ceil(Ny/2)-1,math.ceil(Nz/2)-1,D_txdr_point/2)
        txdr_binary=np.ones([Nx,Ny,Nz])*Single_mask
    else:
        txdr_binary=grid
    #
    # @dataclass()
    # class txdr_points:
    #     kerf: int
    #     width: int
    #     height: int
    #     txdr_width: int

    txdr_points=txdr_points(kerf=points_kerf,width=points_width,height=points_height,txdr_width=final_txdr_width)

    # @dataclass()
    # class txdr_output:
    #     points: txdr_points
    #     mask: np.ndarray
    #     ele_pos: np.ndarray
    #     center: np.ndarray
    #     phase_levels: int
    #     single_txdr: bool
    #     D_txdr_point: int

    txdr_output=txdr_output(points=txdr_points,mask=txdr_binary, ele_pos=ele_pos_matrix,
                            center=center_txdr,phase_levels=txdr_param.phase_levels,
                            single_txdr=txdr_param.single_txdr,D_txdr_point=D_txdr_point)

    return txdr_output




def phase_quant(input,levels):
    # Apply physical limit of TXDR into pressure matrix
    # ; quantize the phase into desired levels
    #
    # author: Moon Hwan Lee
    # date: 30th December 2020
    # last update: 30th December 2020
    # INPUT:
    #     input: target to be quantized
    #     levels: levels to be quantized
    # OUTPUT:
    #     quantized_phase: limtation applied matrix

    origin=input.shape
    reshaped_input=input.reshape(-1)
    # reshaped_input[np.where(reshaped_input==0)]=np.nan
    drad=2*math.pi/(levels-1)
    bins=np.arange(-math.pi,math.pi+drad,drad)
    inds=np.digitize(reshaped_input,bins)
    b=np.array([bins[i] for i in inds])
    reshaped_input=b.reshape(origin[0],origin[1])
    return reshaped_input


