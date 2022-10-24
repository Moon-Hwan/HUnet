from us_holo_pkg import env_param
from us_holo_pkg.txdr_function import make_txdr
import matplotlib.pyplot as plt
import numpy as np
import os

################### set TXDR & grid ##########################

c0=1500 #[m/s] Speed of Sound
Freq=2e6 #[Hz] Excitation frequency
lamda=c0/Freq #[m] wavelength at Freq

# set the spacing between the grid points
# Hologram and target plane is on YZ axis. Wave propagates toward +X axis.
prop_expand_ratio=3
dilation_factor=int(3/prop_expand_ratio) # Apply dilation to dots and disk if the pixel size decreased
dx=lamda/(3*dilation_factor) #250um (lamda/3) / 83.3um (lamda/9) 187.5um (lamda/4)
dy=dx
dz=dx

# total grid size
D_x=12e-3 #[m]
D_y=16e-3
D_z=16e-3

# set total number of grid points not including the PML
Nx = round(D_x/dx)   # [grid points]
Ny = round(D_y/dy)*dilation_factor # 16e-3 for 64 x 64
Nz = Ny


# Define the medium properties
medium= env_param.medium
medium=medium(sound_speed=c0,density=1000,alpha_coeff=0.0022, alpha_power=1.0001 )

# Define parameter to create transducer (2D array or single element circular transducer)

single_txdr=True
D_txdr=13 #[mm] size of the single element circular transducer
lateral_resol=750e-6 # hologram pixel size: 300um for 3D printed lens & 750um for 2MHz 2D array
axial_resol=40e-6 # Not used. Can be used for define axial resolution of 3D printer or delay resolution of ultrasound system
SoS_3Dm=2495 #[m/s] sound speed of 3D print material (vero white of Stratasys)
txdr_ele_num=round(D_txdr*1e-3/lateral_resol)

# TXDR acoustic property (PZT-5H)
density_t=7500
c_t=5960
Z_t=density_t*c_t

txdr_param= env_param.txdr_param

txdr_param=txdr_param(grid_spacing=dx, total_grids=np.array([Nx,Ny,Nz]),
                      size=np.array([lateral_resol, lateral_resol, 0.0*lamda, txdr_ele_num, txdr_ele_num]),
                      phase_levels=2*np.pi/(2*np.pi*Freq*(1/c0-1/SoS_3Dm)*axial_resol),D_txdr=D_txdr*1e-3 ,single_txdr=single_txdr)

# phase level for 2D array: 1/Freq/delay_resol
#             for ultrasound hologram lens: 2*pi/(2*pi*Freq*(1/c0-1/SoS_m)*axial_resol

txdr_size_points=txdr_param.size/dx
txdr_output=make_txdr(txdr_param)
txdr_position=txdr_output.ele_pos
txdr_points_spec=txdr_output.points

# set total number of grid points not including the PML
Ny_PC = round(Ny/txdr_points_spec.height)*txdr_points_spec.height     # [grid points] 48의 배수 + 20

# Visualize the txdr
elem_position=txdr_output.ele_pos
display_txdr=txdr_output.mask
display_txdr=display_txdr[0,:,:]

fig_txdr=plt.figure()
plt.imshow(display_txdr)
plt.title("Hologram transducer")
plt.show()

# set the projection plane (first plane= 0th idx)
first_plane_pt=1
projection_distance=[5.0] #[mm]
projection_plane_pt=[round(dist*1e-3/dx) for dist in projection_distance]

# calculate a projection distance
proj_dist = [dist * 1e-3 for dist in projection_distance] # in mm scale


# Set propagation parameter group
shape=(Ny,Nz,len(proj_dist))
c_lens=SoS_3Dm
density_lens=1175
Z_lens=c_lens*density_lens
prop_param=env_param.prop_param(input_shape=shape,medium=medium,Freq=Freq,prop_distance=proj_dist,grid_spacing=dx,
                                txdr_output=txdr_output,lens_impedance=Z_lens,txdr_impedance=Z_t,lens_SoS=SoS_3Dm,
                                element_size=lateral_resol,prop_resize_ratio=prop_expand_ratio,PC_padding=Ny-Ny_PC)

FoV = 13 #[mm] #Diameter of Field of view where target objects will be exist.
FoV_pts=round(FoV*1e-3/dx)


# Set IASA parameter
IASA_param= {
            'iteration':200
          }

# Set Diff-PAT parameter
DiffPAT_param={
            'learning_rate': 0.1,
            'iteration':200,
            'loss': 'cosine_similarity',
            'intensity_lamda': '0.1'
            }

#Set DL parameter
DL_param={
        'model':'MDHGN',
        'learning_rate':1e-5,
        'batch_size':8,
        'epochs':200,
        'weights_path':'./model_weights/',
        'MULTIPLEX': False,
        'loss': 'cosine_similarity',
        'intensity_lamda': '0.1'
        }

#Set Dataset generator parameter
dataset_param={
            'path':'Datasets',
            'train_ratio':80/100,
            'shape':shape,
            'object_type':'mixed', # ['Disk', 'Line', 'Dot','mixed','mnist']
            'object_size':[round(250e-6/dx), round(1250e-6/dx)], # pixels (250um~1250um)
            'object_count':[1,50],
            'intensity': 1,
            'normalize': True,
            'centralized':True,
            'FoV': FoV_pts,
            'N':25000,
            'name':'target',
            'compression':'GZIP',
            'dilation_factor': dilation_factor
            }




def check_sim_env():
    print("-----------------------------------------------------")
    print("Following parameters will be used for phase retrieval")
    print("Medium SoS:", c0,"m/s")
    print("Medium grid spacing:", dx*1e6,"um")
    print("Medium total size [X,Y,Z]:", Nx*dx*1e3,",", Ny*dy*1e3,",", Nz*dz*1e3,"mm")
    print("Transducer is located at the center of the YZ planes(x=0)")
    print("Frequency:", Freq / 1e6, "MHz")
    print("Size(diameter):",D_txdr,"mm")
    print("Propagation plane:",projection_distance,"mm")
    print("To change the environment, go to Variables.py")
    print("-------------------------------------------")

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)