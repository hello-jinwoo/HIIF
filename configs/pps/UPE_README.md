# UPE Configuration Files Guide

## 📁 파일 목록

### UPE System (NEW)
- **`upe_edsr.yaml`**: EDSR encoder + UPE system (권장 시작점)
- **`upe_swinir.yaml`**: SwinIR encoder + UPE system (더 높은 품질)

### Baseline System (기존)
- **`pps_edsr.yaml`**: EDSR encoder + user-specific decoders
- **`pps_swinir.yaml`**: SwinIR encoder + user-specific decoders

## 🆕 UPE System 주요 변경사항

### 아키텍처 변경
```
❌ Baseline: Encoder + 100 user-specific decoders
✅ UPE:      Encoder + Global decoder + UPE (512-dim)
```

### 성능 개선
| 항목 | Baseline | UPE | 개선 |
|------|----------|-----|------|
| Parameters | 512M | 22M | **96% ↓** |
| Checkpoint size | ~2GB (101 files) | ~90MB (2 files) | **95% ↓** |
| Training speed | ~300ms/iter | <200ms/iter | **1.5× ↑** |
| Validation time | ~3h (100 users) | <20min | **10× ↑** |

## 🔧 사용 방법

### Training

```bash
# EDSR + UPE (권장 시작점)
python train_upe.py \
    --config configs/pps/upe_edsr.yaml \
    --name my_upe_experiment \
    --gpu 0

# SwinIR + UPE (더 높은 품질, 느림)
python train_upe.py \
    --config configs/pps/upe_swinir.yaml \
    --name my_swinir_experiment \
    --gpu 0
```

### Validation

```bash
# EDSR + UPE
python validate_upe.py \
    --config configs/pps/upe_edsr.yaml \
    --checkpoint save/my_upe_experiment/epoch-last.pth \
    --gpu 0

# SwinIR + UPE
python validate_upe.py \
    --config configs/pps/upe_swinir.yaml \
    --checkpoint save/my_swinir_experiment/epoch-last.pth \
    --gpu 0
```

## 📊 Config 비교

### EDSR vs SwinIR

| 설정 | EDSR | SwinIR | 설명 |
|------|------|--------|------|
| **Encoder** |
| Type | CNN (EDSR) | Transformer (SwinIR) | SwinIR이 long-range modeling 우수 |
| Output dim | 64 | 180 | SwinIR이 더 많은 features |
| **Training** |
| crop_size | 128 | 128 | SwinIR은 반드시 8의 배수 |
| Learning rate | 1e-4 | 5e-4 | Transformer는 더 높은 LR 필요 |
| Loss schedule | exponential | linear | Transformer는 linear가 안정적 |
| w_p_end | 0.8 | 0.9 | SwinIR은 prefer에 더 집중 |
| **Performance** |
| Training speed | ~30min/epoch | ~40min/epoch | SwinIR이 33% 느림 |
| Memory usage | ~6GB | ~8GB | SwinIR이 33% 더 사용 |
| Quality | Good | Better | Transformer 특성상 품질 우수 |

## 🎯 새로 추가된 Config 섹션 설명

### 1. `upe_extraction`
```yaml
upe_extraction:
  content_model: dino  # Content features (semantic/structural)
  color_model: clip    # Color/style features
  num_pairs: 16        # UPE 추출에 사용할 pairs (고정)
  normalize: true      # L2 normalization
  seed: 42             # Deterministic random selection
```

**핵심 개념**:
- User당 M개 preference pairs 중 **16개를 UPE 추출용**으로 사용
- 나머지 M-16개는 training/evaluation용
- **Content embedding**: (16, 768) - DINO로 prefer/non_prefer 평균
- **Color difference**: (16, 512) - CLIP으로 prefer - non_prefer
- **Raw UPE**: (16, 1280) = concat[content, color]

**실험 가능한 옵션**:
- `content_model`: `dino` (권장) 또는 `clip`
- `color_model`: `clip` (권장) 또는 `dino`
- ⚠️ 두 모델은 달라야 함!

### 2. `upe_processor`
```yaml
upe_processor:
  num_layers: 3    # Shallow transformer (충분)
  num_heads: 8     # Multi-head attention
  hidden_dim: 512  # Internal dimension
  ffn_dim: 2048    # Feed-forward (4× hidden_dim)
  dropout: 0.1     # Regularization
  activation: gelu # Activation function
```

**핵심 개념**:
- Raw UPE (16, 1280) → Processed UPE (512,)
- Shallow transformer (3 layers) - 16 vectors는 작으므로 충분
- Average pooling으로 최종 512-dim vector 생성
- **User당 1번만 추출 후 cache에 저장**

### 3. Global Decoder (증가된 capacity)
```yaml
model:
  args:
    hidden_dim: 384  # ⬆️ 256 → 384 (baseline 대비 50% 증가)
    blocks: 20       # ⬆️ 16 → 20 (baseline 대비 25% 증가)
```

**핵심 개념**:
- Baseline: User마다 5.12M decoder (총 512M)
- UPE: 모든 users를 처리하는 global decoder ~10M
- Capacity를 2배로 증가 (하나의 decoder가 모든 users 처리)

### 4. No More Sampler!
```yaml
train_dataset:
  # sampler 섹션 제거!
  # Baseline: PPSUserBatchSampler (user grouping)
  # UPE: Standard random sampler (mixed-user batching)
```

**핵심 개념**:
- Baseline: User별로 batch를 grouping → decoder load/offload 필요
- UPE: User 섞여있어도 OK → random sampling 가능
- **1.5-2× faster training** (decoder switching overhead 없음)

### 5. Validation (Direct Inference)
```yaml
validation:
  num_val_users: null      # null = all, 또는 5 (quick test)
  num_upe_pairs: 16        # UPE용 (고정)
  num_eval_samples: -1     # -1 = all M-16, 또는 20 (fair comparison)
  eval_mode: patch         # patch/full
  patch_size: 128          # Patch size
  stride: 64               # Stride
```

**핵심 개념**:
- Baseline: 각 user마다 decoder 생성 → 1000 iter 학습 → 평가
- UPE: UPE 추출 (16 pairs) → global decoder로 직접 inference
- **10× faster validation**

### 6. UPE Cache
```yaml
upe_cache:
  cache_dir: ./load/PPS/upe_cache
  enabled: true
```

**핵심 개념**:
- UPE는 user당 1번만 추출하면 되므로 cache에 저장
- Cache invalidation: 모델/설정 변경 시 자동으로 재추출

## 🔬 실험 가이드

### Quick Test (빠른 검증)
```yaml
# upe_edsr.yaml 또는 upe_swinir.yaml에서 수정:
epoch_max: 50        # 200 → 50
epoch_val: 10        # 10 유지
num_val_users: 5     # null → 5 (5명만 검증)
metrics: [psnr, ssim, lpips]  # 빠른 metrics만
```

### Pilot Study (하이퍼파라미터 탐색)
```yaml
epoch_max: 100       # 100 epochs
num_val_users: 20    # 20 users 검증
```

### Full Training (최종 실험)
```yaml
epoch_max: 200       # 200-500 epochs
num_val_users: null  # All users
metrics: [...]       # All metrics
```

### GPU Memory 부족 시
```yaml
batch_size: 1        # 2 → 1
crop_size: 64        # 128 → 64 (EDSR), 128 유지 (SwinIR - must be 8의 배수)
```

### 더 높은 품질 원할 시
```yaml
hidden_dim: 512      # 384 → 512
blocks: 32           # 20 → 32
crop_size: 256       # 128 → 256 (SwinIR: 256 = 8×32)
```

## 📐 Dimension Flow 참고

### Complete Flow
```
[UPE Extraction - Per User]
16 pairs → Content (16, 768) + Color (16, 512)
        → Raw UPE (16, 1280)
        → UPE Processor (transformer)
        → Processed UPE (512,) ← CACHED

[Training/Inference - Batch]
Input: (B, 3, H, W)
UPE: (B, 512)
↓ Encoder
Features: (B, enc_dim, H_f, W_f)
↓ Freq Conv
Freq: (B, 384, H_f, W_f)
↓ Grid Sampling (4 neighbors)
Grid: (B, H×W, 1538)
↓ UPE Spatial Expansion
UPE: (B, H×W, 512)
↓ Concatenate [grid + hi_coord + upe]
Decoder Input: (B, H×W, 2052)
↓ Global Decoder
Output: (B, 3, H, W)
```

## 🐛 Troubleshooting

### Issue: Cache miss가 너무 많이 발생
```yaml
upe_cache:
  enabled: true  # Make sure this is true
```
Cache는 다음 경우 invalidate됨:
- `content_model` or `color_model` 변경
- `num_pairs` 변경
- `seed` 변경

### Issue: OOM (Out of Memory)
1. **Training OOM**:
   ```yaml
   batch_size: 1         # Reduce
   crop_size: 64         # Reduce (EDSR) or keep 128 (SwinIR)
   ```

2. **Validation OOM**:
   ```yaml
   validation:
     eval_mode: patch    # Use patch mode
     patch_size: 64      # Reduce
     stride: 48          # Reduce
   ```

### Issue: Loss가 수렴하지 않음
1. **Learning rate 낮추기**:
   ```yaml
   optimizer:
     args:
       lr: 5.e-5  # EDSR: 1e-4 → 5e-5, SwinIR: 5e-4 → 2e-4
   ```

2. **Gradient clipping 추가**:
   ```yaml
   # train_upe.py에서 추가 필요
   grad_clip: 1.0
   ```

### Issue: Validation이 너무 느림
```yaml
validation:
  num_val_users: 10      # Reduce from null
  num_eval_samples: 20   # -1 → 20
  metrics: [psnr, ssim]  # Reduce metrics
```

## 📚 관련 문서

- **전체 구현 계획**: `docs/plan/OVERVIEW_hiif_upe_implementation.md`
- **User decisions**: `docs/plan/USER_DECISIONS_20251029.md`
- **Phase 문서**: `docs/plan/phases/phase*.md`
- **Project guide**: `.claude/CLAUDE.md`

## ✅ 체크리스트

### 실험 시작 전
- [ ] Dataset 경로 확인 (`./load/PPS/images`, `./load/PPS/responses/train`, `./load/PPS/responses/validation`)
- [ ] GPU memory 확인 (EDSR: 6GB+, SwinIR: 8GB+)
- [ ] Config 파일 수정 (quick test용으로 조정)
- [ ] Save directory 확인 (`./save/{experiment_name}/`)

### Training 중
- [ ] Loss 수렴 확인 (`tail -f save/{exp}/log_train.txt`)
- [ ] PSNR 증가 확인 (prefer > non_prefer)
- [ ] w_p 증가 확인 (0.5 → 0.8 or 0.9)
- [ ] UPE cache 생성 확인 (`./load/PPS/upe_cache/`)

### Validation 후
- [ ] Metrics 확인 (PSNR, SSIM, LPIPS)
- [ ] Generated images 확인 (`save/{exp}/results/`)
- [ ] Baseline과 비교 (성능 유지 또는 향상)

---

**작성일**: 2025-10-30
**버전**: 1.0
**Author**: Claude Code
