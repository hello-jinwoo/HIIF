# HIIF (CVPR2025) [[Paper](https://arxiv.org/pdf/2412.03748)]
<p align="center">
    <img src="asset/HIIF_archi.png" style="border-radius: 15px">
</p>

```
conda create -n hiif python=3.10.14
conda init
conda activate hiif

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install tqdm
pip install einops
pip install typing_extensions
pip install imageio
pip install timm
pip install pytorch_wavelets
pip install opencv-python
pip install matplotlib
pip install tensorboardX
```

## 📑 Train & Test
EDSR-baseline-HIIF
```
python train.py --config configs/train-div2k/train_edsr-baseline-hiif.yam
bash ./scripts/test-div2k-fast.sh ./save/edsr_baseline_hiif.pth 0
```


---

## 📊 Quantitative Results

We evaluate the performance of **HIIF** on multiple benchmarks against recent state-of-the-art continuous image super-resolution methods under both **in-distribution** and **out-of-distribution** scales.




## 🔍 Qualitative Results
<p align="center">
  <img width="800" src="asset/vs1.png">
</p>

<p align="center">
  <img width="800" src="asset/vs2.png">
</p>

<p align="center">
  <img width="800" src="asset/vs3.png">
</p>

<p align="center">
  <img width="800" src="asset/vs4.png">
</p>

<p align="center">
  <img width="800" src="asset/vs5.png">
</p>


## 🔍 Arbitrary-Scale demo
<p align="center">
  <img width="800" src="asset/vs6.png">
</p>




---
