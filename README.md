<h1 align="center">OmniVDiff:<br>Omni Controllable Video Diffusion for Generation and Understandingn</h1>


<p align="center">


  <b>
    <a href="https://xdobetter.github.io/">Dianbing Xi</a><sup>1,2,*</sup>,
    <a href="https://jiepengwang.github.io/">Jiepeng Wang</a><sup>2,*,‡</sup>,
    <a href="https://akira-l.github.io/">Yuanzhi Liang</a><sup>2</sup>,
    Xi Qiu<sup>2</sup>,
    <a href="http://www.cad.zju.edu.cn/home/huo/">Yuchi Huo</a><sup>1</sup>,
    <a href="http://www.cad.zju.edu.cn/home/rwang/">Rui Wang</a><sup>1,†</sup>,
    <a href="https://scholar.google.com/citations?hl=en&user=PXlNTokAAAAJ">Chi Zhang</a><sup>2,†</sup>,
    <a href="https://xuelongli-link.com">Xuelong Li</a><sup>2,†</sup>
  </b>



</p>



<p align="center">
  *Equal contribution. &nbsp; †Corresponding author. &nbsp; ‡Project leader.
</p>

<p align="center">
  <sup>1</sup>State Key Laboratory of CAD&CG, Zhejiang University  
  <br>
  <sup>2</sup>Institute of Artificial Intelligence, China Telecom (TeleAI)  
</p>




<p align="center">
  <a href="https://arxiv.org/pdf/2504.10825" style="font-size:18px;">📄 Paper</a> &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://tele-ai.github.io/OmniVDiff/" style="font-size:18px;">🌐 Project Page</a> &nbsp;&nbsp;·&nbsp;&nbsp;
  <a href="https://modelscope.cn/models/XDoBetter/OmniVDiff" style="font-size:18px;"> 🤗 ModelScope</a>
</p>


 <h3 align="center"><strong>AAAI 2026</strong></h3>


## 📌 Intro

<div align="center">
  <img src="assets/fig1_teaser.png" width="100%">
  <p><em>OmniVDiff enables controllable video generation and understanding in a unified video diffusion framework.</em></p>
</div>

## 📦 Environment Setup

1. Create a conda environment named `omni`:
   ```bash
   conda create -n ovdiff python=3.10.9
   conda activate ovdiff
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Install our modified version of the `diffusers` library.  
   Navigate to the `diffusers` directory and run:
   ```bash
   pip install -e .
   ```


## 🤗 Model Zoo

[OmniVDiff](https://modelscope.cn/models/XDoBetter/OmniVDiff) is available in the ModelScope Hub.

## 🔍 Inference

1. Navigate to the `inference` directory:
   ```bash
   cd inference
   ```

2. Run batch inference:
   ```bash
   python batch_infer.py
   ```

    ```bash
    # -1 no condition, 0:rgb, 1:depth, 2:canny, 3:segment
    python batch_infer.py --idx_cond_modality -1 --output_dir "./output_cond=-1"
    python batch_infer.py --idx_cond_modality 0 --output_dir "./output_cond=0"
    python batch_infer.py --idx_cond_modality 1 --output_dir "./output_cond=1"
    python batch_infer.py --idx_cond_modality 2 --output_dir "./output_cond=2"
    python batch_infer.py --idx_cond_modality 3 --output_dir "./output_cond=3"
    ```

## 🏋️‍♂️ Training




We provide an example configuration for training on **2 GPUs** with `batch_size=1`.  
You can modify the configuration file(.yaml) to adjust the number of GPUs to fit different hardware setups.



1. Navigate to the `finetune` directory:
```bash
cd finetune
```

2. Enable cached latents before training
Before starting the actual training, enable the following option in `train.sh` to use cached latents:

```bash
-check_cache "true"
```


This will generate and store latent representations for faster training.

```bash
bash train.sh
```


3. Disable the option and start training
After the latent cache has been prepared, disable the option (set it to "false" or comment it out) and begin training:

```bash
bash train.sh
```

## 🙏 Acknowledgements

We sincerely thank the developers of the following open-source repositories, whose contributions have been invaluable to our research:

- [CogVideoX](https://github.com/zai-org/CogVideo)
- [NormalCrafter](https://github.com/Binyr/NormalCrafter)
- [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
- [Koala-36M](https://koala36m.github.io/)
- [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)
- [Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM)
- [SAM 2](https://github.com/facebookresearch/sam2)




## 📜 Citation

If you find our work helpful in your research, please consider citing it using the BibTeX entry below.


```bibtex
@article{xdb2025OmniVDiff,
  author    = {Xi, Dianbing and Wang, Jiepeng and Liang, Yuanzhi and Qi, Xi and Huo, Yuchi and Wang, Rui and Zhang, Chi and Li, Xuelong},
  title     = {OmniVDiff: Omni Controllable Video Diffusion for Generation and Understanding},
  journal   = {arXiv preprint arXiv:2504.10825},
  year      = {2025},
}

@misc{xdb2025CtrlVDiff,
      title={CtrlVDiff: Controllable Video Generation via Unified Multimodal Video Diffusion}, 
      author={Dianbing Xi and Jiepeng Wang and Yuanzhi Liang and Xi Qiu and Jialun Liu and Hao Pan and Yuchi Huo and Rui Wang and Haibin Huang and Chi Zhang and Xuelong Li},
      year={2025},
      eprint={2511.21129},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.21129}, 
}

```
