# Training Custom Basketball Detector with YOLO11

## Problem

The pre-trained YOLO11n model with COCO dataset **does not detect basketballs** in game scenarios. The "sports ball" class (32) is too generic and fails to recognize basketballs in real game footage.

**Evidence:**
```
[Frame 50 DEBUG] YOLO Strategy 1 (sports ball): 0 detections
[Frame 100 DEBUG] YOLO Strategy 2 (all classes): 0 total detections
```

## Solution

Fine-tune YOLO11-Large with basketball-specific datasets from Roboflow Universe.

---

## Step 1: Find Datasets on Roboflow Universe

### Recommended Datasets

Go to https://universe.roboflow.com/search?q=basketball and look for:

1. **Basketball Ball Detection**
   - Workspace: `basketball-detection`
   - Project: `basketball-ball-detection`
   - Contains: ~1000+ images with basketball annotations
   - Download format: YOLOv8

2. **Basketball Detection**
   - Search for projects with "basketball detection" or "ball tracking"
   - Look for datasets with:
     - ✓ Multiple angles (overhead, side view, player POV)
     - ✓ Different lighting conditions
     - ✓ Various court types (indoor, outdoor)
     - ✓ Occlusions (ball behind players)

3. **Sports Ball Detection**
   - Some general sports ball datasets include basketballs
   - Check preview images to confirm basketball presence

### Quality Criteria

When selecting datasets:
- **Minimum 500 images** per dataset
- **Diverse angles and lighting**
- **Occlusions included** (ball partially hidden)
- **In-game footage** (not studio shots)
- **Proper annotations** (bounding boxes tight around ball)

---

## Step 2: Get Roboflow API Key

1. Go to https://universe.roboflow.com/
2. Sign in or create free account
3. Click your profile → Settings
4. Copy your Private API Key
5. Set environment variable:
   ```bash
   export ROBOFLOW_API_KEY='your_key_here'
   ```

---

## Step 3: Update Training Script with Datasets

Edit `scripts/train_basketball_detector.py`:

```python
ROBOFLOW_DATASETS = [
    {
        'workspace': 'basketball-detection',
        'project': 'basketball-ball-detection',
        'version': 1,
    },
    {
        'workspace': 'another-workspace',
        'project': 'basketball-tracker',
        'version': 2,
    },
    # Add more datasets for better results
]
```

**Tip:** Using 3-5 diverse datasets gives best results!

---

## Step 4: Train the Model

### Install Requirements

```bash
pip install ultralytics roboflow
```

### Run Training

```bash
python scripts/train_basketball_detector.py
```

This will:
1. Download datasets from Roboflow
2. Combine them into one dataset
3. Train YOLO11-L for 50 epochs (~30-60 minutes on GPU)
4. Save best model to `models/basketball_detector_yolo11l.pt`

### Training Parameters

Default settings (in script):
- Model: YOLO11-L (better accuracy than nano)
- Epochs: 50 (with early stopping)
- Batch size: 16
- Image size: 640x640
- Augmentation: Enabled (rotation, scale, flip, HSV)

### Hardware Requirements

- **GPU recommended** (NVIDIA with CUDA)
- **Minimum 8GB RAM**
- **10GB disk space** (for datasets + models)

Without GPU, training will be **very slow** (10-20x slower).

---

## Step 5: Test the Model

### Quick Test

```bash
python scripts/test_basketball_model.py --model models/basketball_detector_yolo11l.pt --video input_video.mp4
```

Expected output:
```
Frame    Detections   Confidence   Size (px)
--------------------------------------------------
50       1            0.892        24.3
100      1            0.845        22.1
150      1            0.798        26.5
...
Detection rate: 9/10 frames (90%)
Average confidence: 0.845

✅ Model looks good! Detection rate >= 70%
```

### Visual Test

```bash
python scripts/test_basketball_model.py --visualize
```

This saves 10 annotated frames to `outputs/model_test/` for visual inspection.

---

## Step 6: Use in Pipeline

The model is **automatically used** once trained!

The detection code checks for models in this order:
1. `models/basketball_detector_yolo11l.pt` ← Your trained model
2. `models/basketball_detector.pt`
3. `yolo11l.pt` (pre-trained)
4. `yolo11n.pt` (fallback)

Just run the pipeline normally:
```bash
python scripts/pipeline.py
```

You should now see:
```
✓ Custom basketball detector loaded: models/basketball_detector_yolo11l.pt
```

---

## Expected Results

### Before (YOLO11n with COCO):
```
Detection methods:
  auto-refined-hough: 143 (44.3%)
  auto-refined-fallback: 121 (37.5%)
  auto-refined-yolo: 0 (0%)  ← NOTHING DETECTED
```

### After (Fine-tuned YOLO11l):
```
Detection methods:
  auto-refined-yolo: 280 (86.7%)  ← YOLO WORKS!
  manual: 24 (7.4%)
  smooth-interpolated: 19 (5.9%)
```

---

## Troubleshooting

### Problem: "No datasets downloaded"

**Cause:** API key not set or incorrect

**Solution:**
```bash
export ROBOFLOW_API_KEY='your_actual_key'
echo $ROBOFLOW_API_KEY  # Verify it's set
python scripts/train_basketball_detector.py
```

### Problem: "CUDA out of memory"

**Cause:** GPU doesn't have enough memory

**Solution:** Reduce batch size in `train_basketball_detector.py`:
```python
batch_size=8,  # or even 4
```

### Problem: "Model detected nothing"

**Cause:** Poor training data or not enough epochs

**Solutions:**
1. Add more diverse datasets
2. Train for more epochs (100 instead of 50)
3. Check training logs for errors
4. Verify dataset annotations are correct

### Problem: Low detection rate (< 50%)

**Cause:** Training not converged or data mismatch

**Solutions:**
1. Train for more epochs
2. Use datasets with similar camera angles to your video
3. Add data augmentation
4. Try YOLO11x (extra large model)

---

## Alternative: Manual Dataset Collection

If you can't use Roboflow datasets (licensing issues), you can create your own:

### Option A: Label Your Own Frames

1. Extract frames from your video:
   ```bash
   python scripts/extract_frames_for_labeling.py --video input_video.mp4 --count 500
   ```

2. Use labeling tool (LabelImg, CVAT, or Roboflow):
   - Install LabelImg: `pip install labelImg`
   - Run: `labelImg`
   - Draw boxes around basketballs
   - Save in YOLO format

3. Organize dataset:
   ```
   data/custom_basketball/
   ├── train/
   │   ├── images/
   │   └── labels/
   └── valid/
       ├── images/
       └── labels/
   ```

4. Create `data.yaml`:
   ```yaml
   path: /path/to/data/custom_basketball
   train: train/images
   val: valid/images
   names:
     0: basketball
   nc: 1
   ```

5. Train directly:
   ```python
   from ultralytics import YOLO
   model = YOLO('yolo11l.pt')
   model.train(data='data/custom_basketball/data.yaml', epochs=50)
   ```

### Option B: Use Public Datasets

Some basketball datasets are on:
- **Kaggle**: Search "basketball detection"
- **OpenImages**: Filter for "basketball"
- **YouTube-8M**: Extract basketball segments

---

## Performance Tips

### Faster Training
- Use smaller image size: `imgsz=416` instead of 640
- Use fewer augmentations
- Use YOLO11m (medium) instead of large

### Better Accuracy
- Use YOLO11x (extra large) instead of large
- Train for 100+ epochs
- Use 5+ diverse datasets
- Add more data augmentation

### Balanced Approach (Recommended)
- YOLO11l (large model)
- 50 epochs with early stopping
- 3-5 combined datasets
- Default augmentation

---

## Next Steps After Training

1. **Test thoroughly**: Run on your full video
2. **Analyze failures**: Which frames fail? Add similar data.
3. **Fine-tune**: If detection rate < 80%, retrain with more data
4. **Deploy**: Use in production pipeline

---

## Resources

- **Roboflow Universe**: https://universe.roboflow.com/
- **YOLO11 Docs**: https://docs.ultralytics.com/
- **Fine-tuning Guide**: https://docs.ultralytics.com/modes/train/
- **Basketball Detection Papers**: Google Scholar "basketball detection CNN"

---

## Support

Having issues? Check:
1. Training logs: `runs/basketball/basketball_detector_yolo11l/`
2. Validation metrics: Look for mAP > 0.7
3. Test script output: Should show 70%+ detection rate
4. Visual inspection: Check `outputs/model_test/` frames
