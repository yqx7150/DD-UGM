# DD-UGM    
**Paper**: Universal Generative Modeling in Dual-domain for Dynamic MR Imaging      
**Authors**: Chuanming Yu, Yu Guan, Ziwen Ke, Ke Lei, Dong Liang, Qiegen Liu*
NMR in Biomedicine 36 (12), e5011         
https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/nbm.5011     
     
Date : June-13-2023  
Version : 1.0  
The code and the algorithm are for non-comercial use only.  
Copyright 2022, Department of Mathematics and Computer Sciences, Nanchang University. 

Dynamic magnetic resonance image reconstruction from incomplete k-space data has generated great research interest due to its capa-bility to reduce scan time. Nevertheless, the reconstruction problem remains a thorny issue due to its ill-posed nature. Recently, diffu-sion models, especially the score-based generative models, demonstrated great potential in terms of algorithmic robustness and flexi-bility of utilization. Moreover, the unified framework through the variance exploding stochastic differential equation (VE-SDE) is proposed to enable new sampling methods and further extend the capabilities of score-based generative models. Therefore, by taking advantage of the unified framework, we propose a k-space and image Dual-Domain collaborative Universal Generative Model (DD-UGM) which combines the score-based prior with low-rank regularization penalty to reconstruct highly under-sampled measurements. More precisely, we extract prior components from both image and k-space domains via a universal generative model and adaptively handle these prior components for faster processing while maintaining good generation quality. Experimental comparisons demon-strated the noise reduction and detail preservation abilities of the proposed method. Moreover, DD-UGM can reconstruct data of dif-ferent frames by only training a single frame image, which reflects the flexibility of the proposed model.

## Requirements and Dependencies
    python==3.7.11
    Pytorch==1.7.0
    tensorflow==2.4.0
    torchvision==0.8.0
    tensorboard==2.7.0
    scipy==1.7.3
    numpy==1.19.5
    ninja==1.10.2
    matplotlib==3.5.1
    jax==0.2.26

## Training Demo
``` bash
python main.py --config=configs/ve/SIAT_kdata_ncsnpp.py --workdir=exp --mode=train --eval_folder=result
```

## Test Demo
``` bash
python PCsampling_demo_svd.py
```

## Checkpoints
We provide pretrained checkpoints. You can download pretrained models from [Baidu cloud](https://pan.baidu.com/s/1vo6kpsu8pCgi_Mgw5PMEPw?pwd=gakz)
https://pan.baidu.com/s/1vo6kpsu8pCgi_Mgw5PMEPw?pwd=gakz

## Graphical representation
### The whole pipeline of DD-UGM is illustrated in fig_1
<div align="center"><img src="https://github.com/yqx7150/DD-UGM/blob/main/fig_1.png" >  </div>
The proposed DD-UGM method in dual-domain for DMRI reconstruction. (a) Universal generative model to learn the k-space prior information via de-noising score matching. (b) Universal generative model to learn the image prior information via denoising score matching. (c) Reconstruction to progressively remove aliasing and recover fine details via PC operation and low-rank prior. The red arrow represents the reconstruction process in the image domain while the green arrow indicates the reconstruction process in the k-space domain.

### The key idea of VE-SED in k-space domain is visualized in fig_2.
<div align="center"><img src="https://github.com/yqx7150/DD-UGM/blob/main/fig_2.png" >  </div>
VE-SDE smoothly transforms a data distribution to a known prior distribu-tion by injecting noise in the k-space domain, and a corresponding reverse-time VE-SDE that transforms the prior distribution back into the k-space data distribution by removing the noise.

###  The corresponding bidirectional process of VE-SDE in image domain is described in fig_3
<div align="center"><img src="https://github.com/yqx7150/DD-UGM/blob/main/fig_3.png" >  </div>
The corresponding bidirectional process of VE-SDE in image domain, which performs a slow noise injection process and noise removal process on dynamic MR images.


###  Convergence curve of DD-UGM in terms of PSNR versus iterations
<div align="center"><img src="https://github.com/yqx7150/DD-UGM/blob/main/fig_4.png" >  </div>
 Convergence curve of DD-UGM in terms of PSNR versus iterations.

## Acknowledgement
The implementation is based on this repository: https://github.com/yang-song/score_sde_pytorch.

## Other Related Projects    
  * Homotopic Gradients of Generative Density Priors for MR Image Reconstruction  
[<font size=5>**[Paper]**</font>](https://ieeexplore.ieee.org/abstract/document/9435335)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HGGDP) [<font size=5>**[Slide]**</font>](https://github.com/yqx7150/HGGDP/tree/master/Slide)  

* One-shot Generative Prior in Hankel-k-space for Parallel Imaging Reconstruction  
[<font size=5>**[Paper]**</font>](https://arxiv.org/abs/2208.07181)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/HKGM)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)

* Diffusion Models for Medical Imaging
[<font size=5>**[Paper]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[Code]**</font>](https://github.com/yqx7150/Diffusion-Models-for-Medical-Imaging)   [<font size=5>**[PPT]**</font>](https://github.com/yqx7150/HKGM/tree/main/PPT)  
     
