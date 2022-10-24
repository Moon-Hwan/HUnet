# Deep learning-based framework for fast and accurate acoustic hologram

## Description
Our framework is for learned neural network which can rapidly generates accurate **acoustic hologram**.
![architecture](https://user-images.githubusercontent.com/70740386/197447338-8e6e0858-f8a6-49f1-a460-f4be6d41442f.png)
The framework contains its architecture, networks, loss functions and datasets for training.
Autoencoder architecture with encoder(neural network) and decoder(simulation method, angular spectrum method).
Trains the encoder to make output of decoder indentical to the input of encoder.

The paper addressed this framework is in revision now.


## Used environment
Python 3.8
Tensorflow 2.5
numpy 1.21.5
matplotlib
PIL
scipy
tqdm
skimage

## Usage
### algorithm
- Set the variables in `Variables.py`.
 ```
 Excitation frequency, spatial step size (pixel size), total grid size, hologram pixel size(lateral_resol), transducer diameter, target plane distance, 
 parameters of the algorithms (ex. learning rate, batch size, loss), parameters of the dataset
 ```

- Select one of the algorithms in `main.py`.
1) Diff-PAT for phase-only holograms
2) Iterative angular spectrum approach (IASA)
3) Ours: HU-Net

- Load target binary image 
 
 ```target_img=algorithm.load_target_img(target_path,plot=True)```
 
- Retrieve phase-only hologram 

```retrieved_phase=algorithm.get_phase(target_img,get_computation_time=True) ```

- Reconstruct the target acoustic field by propagating the retrieved hologram
 
 ``` 
 propagated_pressure=algorithm.propagate(retrieved_phase,expand_ratio)
 -- expand_ratio: Increase spatial sampling for accurate simulation
 ```
 
 - Visualize and assess the result
 
 ```
 algorithm.visualize(target_img, retrieved_phase, propagated_pressure)
 algorithm.assess(target_img, propagated_pressure,expand_ratio)
 ```
 
 - Save the results if needed
 ```
 algorithm.save_result_pressure(propagated_pressure,letter=img_name, save_path)
 algorithm.save_matrix(retireved_phase,letter=img_name, save_path)
 ```

