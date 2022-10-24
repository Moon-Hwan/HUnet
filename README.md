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
- Select one of the algorithms in `main.py`.
1) Diff-PAT for phase-only holograms
2) Iterative angular spectrum approach (IASA)
3) Ours: HU-Net

- Load target binary image 
  ```target_img=algorithm.load_target_img(target_path,plot=True)```
- Retrieve phase-only hologram (algorithm.get_phase(target_img,get_computation_time))
- 

