# cosense
![overview image](https://github.com/HeSunPU/CO-SENSE/blob/master/assets/overview_posci.PNG)
Learning a probabilistic strategy for ***co***mputational imaging ***sen***sor ***se***lection (***cosense***) [ICCP2020](https://arxiv.org/abs/2003.10424)
> Optimized sensing is important for computational imaging in low-resource environments, when images must be recovered from severely limited measurements. In this paper, we propose a physics-motivated, fully  differentiable, autoencoder that learns a probabilistic sensor-sampling strategy for non-linear, correlated measurements, jointly with the image reconstruction procedure, for the purpose of optimized sensor design. The proposed method learns a system's preferred sampling distribution, modeled as an Ising model, that characterizes the correlations between different sensor selections. The learned probabilistic model is achieved by using a Gibbs sampling inspired network architecture, and yields multiple sensor-sampling patterns that perform well.

Further mathematical and implementation details are described in our paper:
```
@inproceedings{sun2020learning,
    author = {He Sun and Adrian V. Dalca and Katherine L. Bouman},
    title = {Learning a Probabilistic Strategy for Computational Imaging Sensor Selection},
    booktitle = {IEEE International Conference on Computational Photography (ICCP)},
    year = {2020},
}
```
If you make use of the code, please cite the paper in any resulting publications.

## Setup
The ***cosense*** package is developed based on Python package "numpy", "tensorflow" and "keras".

It has been tested for a very long baseline interferometry (VLBI) array design problem using Python "eht-imaging" package.

## Run a VLBI array design test
```
python main_vlbi.py array target lr epoch weight1 weight2 resolution fov sefd flux modeltype
```
"array" represents the potential telescopes we are selecting from, 

"target" represents the science target ("sgrA" or "m87"), 

"lr" and "epoch" are the learning rate and number of epochs for training the auto-encoder, 

"weight1" and "weight2" are weights that respectively balance the sparsity and diversity loss, 

"resolution" is the goal reconstruction resolution, 

"fov" is the target field of view, 

"sefd" is the coefficient of the telescopes' thermal flux (0 stands for no thermal noise, 1 stands for site-varying thermal noises, 2 stands for site-equivalent thermal noises), 

"flux" represents whether the flux of training images are constant (0 stands for varying flux, 1 stands for constant flux),

"modeltype" defines the architecture of the reconstruction network ("vis" stands for reconstruction using complex visibilities, and "cpamp" stands for reconstruction using closure phase and visibility amplitude).
