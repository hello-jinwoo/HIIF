# HIIF (CVPR2025)
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
