# HIIF (CVPR2025) [Paper](https://arxiv.org/pdf/2412.03748)]
<p align="center">
    <img src="assets/HIIF_archi_CMR.png" style="border-radius: 15px">
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

## Train & Test
EDSR-baseline-HIIF
```
python train.py --config configs/train-div2k/train_edsr-baseline-hiif.yam
bash ./scripts/test-div2k-fast.sh ./save/edsr_baseline_hiif.pth 0
```


---

## 📊 Quantitative Results

We evaluate the performance of **HIIF** on multiple benchmarks against recent state-of-the-art continuous image super-resolution methods under both **in-distribution** and **out-of-distribution** scales.

### 🔍 Evaluation Datasets


All results are reported in terms of **PSNR (dB)**. For each scale and dataset:
- The **best performance** is highlighted in **red**.
- The **second-best performance** is highlighted in **blue**.
- `-` denotes unavailable results.

---

### 📈 Table 1: Results on DIV2K and Set5

| Dataset | Upscaling Factors | Evaluated Models |
|--------|--------------------|------------------|
| DIV2K | ×2, ×3, ×4 (in-distribution), ×6–×30 (out-of-distribution) | Bicubic, MetaSR, LIIF, LTE, CLIT, CiaoSR, SRNO, **HIIF (Ours)** |
| Set5 | ×2 to ×12 | Same as above |

📌 **HIIF consistently outperforms previous methods across all scales**, especially under extreme out-of-distribution settings.

📷 *See [Table 1](#tbl:results1) in the paper for detailed numbers.*

---

### 📈 Table 2: Results on Set14, BSD100, and Urban100

| Dataset | Upscaling Factors | Evaluated Models |
|--------|--------------------|------------------|
| Set14, BSD100, Urban100 | ×2 to ×12 | Same as Table 1 |

📌 **HIIF achieves state-of-the-art PSNR across all datasets and settings**, demonstrating strong generalization to diverse content and unseen scales.

📷 *See [Table 2](#tbl:results2) in the paper for full breakdown.*

---

📌 **Note**: To reproduce these results, follow the training and evaluation instructions described [here](#training-and-evaluation) and set `--eval_type=benchmark`.

---
