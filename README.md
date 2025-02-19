# <p align=center>`Polyper: Boundary Sensitive Polyp Segmentation - AAAI 2024`</p>



> If you have any questions about our work, feel free to contact me via e-mail (shaoh@mail.nankai.edu.cn).


The PyTorch vsersion implementation of  [Polyper: Boundary Sensitive Polyp Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/28274) is available at [PyTorch Version]() !!!

## Get Start
> Our experiments are based on ubuntu, and windows is not recommended.
> 
**0. Install**

```
conda create --name medical_seg_jittor python=3.8 -y
conda activate medical_seg_jittor

python -m pip install -r requirements.txt

sudo python setup.py develop

pip install einops opencv-python regex ftfy
```


**1. Dataset**
> The dataset used in the experiment can be obtained in the following methods:
- For polyp segmentation task: [Polypseg](https://github.com/DengPingFan/PraNet): including Kvasir, - CVC-ClinicDB, CVC-ColonDB, EndoScene and ETIS dataset.

**2. Experiments**


### [Polyper: Boundary Sensitive Polyp Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/28274) AAAI 2024

> **Authors:**
> [Hao Shao](https://scholar.google.com/citations?hl=en&user=vB4DPYgAAAAJ), [Yang Zhang](), &[Qibin Hou](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=en&oi=ao).

#### **Abstract**

We present a new boundary sensitive framework for polyp segmentation, called Polyper. Our method is motivated by a clinical approach that seasoned medical practitioners often leverage the inherent features of interior polyp regions to tackle blurred boundaries. Inspired by this, we propose explicitly leveraging polyp regions to bolster the model’s boundary discrimination capability while minimizing computation. Our approach first extracts boundary and polyp regions from the initial segmentation map through morphological operators. Then, we design the boundary sensitive attention that concentrates on augmenting the features near the boundary regions using the interior polyp regions’s characteristics to generate good segmentation results. Our proposed method can be seamlessly integrated with classical encoder networks, like ResNet-50, MiT-B1, and Swin Transformer. To evaluate the effectiveness of Polyper, we conduct experiments on five publicly available challenging datasets, and receive state-of-the-art performance on all of them.

#### Architecture

<p align="center">
    <img src="https://github.com/haoshao-nku/medical_seg/blob/master/fig/pipline_polyper.png"/> <br />
    <em> 
    Figure 1: Overall architecture of Polyper. We use the Swin-T from Swin Transformer as the encoder. The decoder is divided into two main stages. The first potential boundary extraction (PBE) stage aims to capture multi-scale features from the encoder, which are then aggregated to generate the initial segmentation results. Next, we extract the predicted polyps' potential boundary and interior regions using morphology operators. In the second boundary sensitive refinement (BSR) stage, we model the relationships between the potential boundary and interior regions to generate better segmentation results.
    </em>
</p>


<p align="center">
    <img src="https://github.com/haoshao-nku/medical_seg/blob/master/fig/refine_polyper.png"/> <br />
    <em> 
    Figure 2: Detailed structure of boundary sensitive attention (BSA) module. This process is separated into two parallel branches, which systematically capitalize on the distinctive attributes of polyps at various growth stages, both in terms of spatial and channel characteristics. `B' and `M' indicate the number of pixels in the boundary and interior polyp regions within an input of size H*W and C channels.
    </em>
</p>

#### Experiments

### Change dataset path

- 1.Download the [Polypseg](https://github.com/DengPingFan/PraNet) dataset, then decompress the dataset.
- 2.Update the training path and test path of **/medical_seg_jittor/project/_base_/datasets/polyp_512x512.py** in the project, on lines 2.
> We recommend using absolute paths instead of relative paths when updating paths of dataset.

### Training
Please confirm whether you are currently under the mmsegmentation directory. If not, please enter the mmsegmentation directory. Then run the following code in terminal:

- python tools/run_net.py --config-file=/medical_seg_jittor/project/polyper/polyper_polyp_512*512_80k.py --task=train

> During training, verification is performed every 8,000 iterations, and the checkpoint file is saved at the same time. The batch size and validation set evaluation indicators can be changed in **/medical_seg_jittor/project/polyper/polyper_polyp_512*512_80k.py**.

### Testing

The log files and checkpoint files of the training process are saved in /medical_seg/mmsegmentation/work_dirs/polyper_polypseg_224*224_80k/. The command to test the model is as follows:

python tools/run_net.py --config-file=path/to/config --resume=path/to/ckp --save-dir=path/to/save_dir --task=test

>  You can replace ckp to evaluate the performance of different checkpoints.


## Reference

You may want to cite:
```
@inproceedings{shao2024polyper,
  title={Polyper: Boundary Sensitive Polyp Segmentation},
  author={Shao, Hao and Zhang, Yang and Hou, Qibin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={5},
  pages={4731--4739},
  year={2024}
}
```

### License

Code in this repo is for non-commercial use only.