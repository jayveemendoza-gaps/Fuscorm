# 🍌 Banana Corm Browning Analyzer - v6.1 (Streamlit)

**Latest Stable Release** | Tested & Production-Ready

## 📋 Overview

This is the latest working version of the Banana Corm Browning Analyzer built with **Streamlit 1.40.0**. The application analyzes banana corm images to detect and quantify fusarium wilt browning using advanced color-space analysis and machine learning-based background removal.

### ✨ Key Features
- **Interactive Drawing Canvas**: Polygon, freeform, and rectangle selection tools
- **Zoom & Pan Controls**: Precise area selection on large images with minimap navigation
- **Scale Calibration**: Real-world measurements (mm²) from ruler-based scale detection
- **Multiple Lesion Detection**: Distinguishes between 5 types of browning lesions
- **Color-Space Analysis**: LAB color space analysis for accurate lesion detection
- **Batch Processing**: Analyze multiple images and export combined results
- **Cloud Optimized**: Tested and optimized for Streamlit Cloud deployment

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- pip or conda package manager

### Installation (Windows)

1. **Create Virtual Environment** (recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run Application**:
```bash
streamlit run appv6.1.py
```

The app will open at `http://localhost:8501` in your browser.

### Installation (macOS/Linux)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run appv6.1.py
```

---

## 📦 Package Versions

This build uses the following critical versions:

| Package | Version | Notes |
|---------|---------|-------|
| **streamlit** | 1.40.0 | Latest with `@st.fragment` support |
| **streamlit-drawable-canvas** | 0.9.3 | Drawing tools and canvas |
| **numpy** | 1.26.4 | Numerical operations |
| **opencv-python-headless** | 4.12.0.88 | Image processing (headless for servers) |
| **Pillow** | 10.4.0 | Image handling |
| **rembg** | 2.0.69 | AI background removal |
| **scikit-image** | 0.25.2 | Image processing filters |
| **scikit-learn** | 1.7.1 | ML utilities |
| **colorspacious** | 1.1.2 | Color space conversions (RGB↔LAB) |

### Version Compatibility Notes

- **Streamlit 1.40.0** includes the `@st.fragment` decorator (added in 1.33.0) which provides significant performance improvements
- **altair 4.2.2** is required by Streamlit 1.40.0 (NOT the latest 5.x)
- **opencv-python-headless** is used instead of **opencv-python** for server deployments (avoids GUI dependencies)

---

## 🎯 Usage Workflow

### Step 1: Upload Image
- Click "Choose corm image" in the sidebar
- Supported formats: PNG, JPG, JPEG
- Max resolution: 1200×1200 (auto-resized for cloud performance)

### Step 2: Calibrate Scale (Optional)
- Draw a line on any known measurement in the image (e.g., ruler)
- Enter the real-world length in millimeters
- The scale (mm/pixel) will be calculated automatically
- Skip this step for pixel-only measurements

### Step 3: Prepare Image
- Click "📋 Prepare Image" button
- Image will be preprocessed and displayed

### Step 4: Select Analysis Area
- Choose selection shape: **Polygon** or **Freeform**
- For polygons: Click points, then click near first point to auto-close
- For freeform: Draw any enclosed shape
- Use zoom/pan controls for precise placement on large images

### Step 5: Analyze
- Click "🔬 Analyze Area" or let it auto-analyze when polygon is complete
- Results show:
  - Total browning percentage
  - Lesion breakdown (5 types)
  - Real-world measurements (if scale calibrated)

---

## 📊 Analysis Output

### Lesion Types Detected

1. **🔴 Reddish Brown (Critical)** - Most severe, associated with active fusarium
2. **⚫ Dark Necrotic (Severe)** - Black lesions indicating necrotic tissue death
3. **🟤 Dark Brown (Severe)** - Deep brown coloration, significant tissue damage
4. **🟠 Normal Brown (Moderate)** - Standard browning, progressive lesion development
5. **🟡 Yellowish Brown (Early)** - Earliest stage of lesion development

### Measurements

**Without Scale:**
- All measurements in pixels
- Browning percentage
- Pixel counts per lesion type

**With Scale:**
- Real-world areas in mm² (or cm² for large areas)
- Exact scale ratio (mm/pixel)
- Comparative analysis with known standards

---

## 🔧 Troubleshooting

### "App not starting" / "Import errors"

**Solution:** Ensure all packages are correctly installed:
```bash
pip install --upgrade -r requirements.txt
```

### "streamlit has no attribute 'fragment'"

**Solution:** This app requires Streamlit 1.33.0 or higher. Update:
```bash
pip install --upgrade streamlit==1.40.0
```

### "Canvas errors" / "Drawing not working"

**Solution:** Clear browser cache and restart:
```bash
streamlit run appv6.1.py --logger.level=debug
```

### "Background removal failing"

This is expected behavior - the app gracefully falls back to color-based detection:
- Ensure good lighting in the image
- Provide clear boundaries between corm and background
- Works better with high-contrast images

### "Polygon not detecting"

**Tips:**
- Click at least 3 points
- Click **within 25 pixels** of your first point to auto-close
- Avoid double-clicking (may remove the last point)
- Try switching between "Point mode" and "Freehand" radio buttons to refresh canvas data

### Memory/Performance Issues

The app includes cloud optimizations:
- Maximum image size: 1200×1200 pixels
- Maximum batch size: 20 images (cloud deployment limit)
- Automatic memory cleanup every 5 reruns
- Large image arrays are cleared after display

For local deployments with more resources, edit `MAX_IMAGE_DIM` and `MAX_BATCH_SIZE` in the code.

---

## 📁 Project Structure

```
BananaCormAnalyzer_v6.1_Streamlit/
├── appv6.1.py                 # Main application (2500+ lines)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── batch_data/
    └── batch_session.json      # Auto-saved batch results (created on first use)
```

---

## 🚀 Deployment

### Streamlit Cloud

1. Push this folder to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New app" → select your repo and `appv6.1.py`
4. Set Python version to 3.11 if available

**Environment Variables:**
- No special setup required
- App uses standard Streamlit Cloud resources

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY appv6.1.py .
EXPOSE 8501
CMD ["streamlit", "run", "appv6.1.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 📝 Performance Notes

**Optimization Features:**
- Fragment-based zoom/pan (arrow buttons only rerun fragment, not full app)
- Lazy image processing (only runs when needed)
- Memory cleanup every 5 application reruns
- Cloud-friendly batch size limits
- Headless OpenCV (no GUI overhead)

**Benchmarks (Streamlit Cloud):**
- Image upload: ~1 second
- Image preparation: ~2 seconds
- Single analysis: ~3-5 seconds
- Batch analysis (20 images): ~1-2 minutes

---

## 🔍 Technical Details

### Color Space Analysis

The application uses CIELAB (CIE L*a*b*) color space for accurate lesion detection:
- **L* channel**: Brightness (0-100)
- **a* channel**: Red-Green spectrum (-128 to +127)
- **b* channel**: Yellow-Blue spectrum (-128 to +127)

Lesion detection thresholds are optimized for banana corm tissue.

### Background Removal

Two-stage process:
1. **AI-based**: Uses rembg (U²-Net model) for intelligent removal
2. **Fallback**: Color-based HSV masking if AI fails

### Polygon Detection

Supports multiple polygon input methods:
- **Point mode**: Click vertices, auto-closes when clicking near start point
- **Freehand mode**: Draw continuous closed shape
- **Point-in-polygon algorithm**: Fills enclosed area for analysis

---

## 📞 Support & Issues

### Common Issues

1. **"No altair.vegalite.v4"**
   ```bash
   pip install altair==4.2.2
   ```

2. **"image_to_url" error**
   - Only occurs with Streamlit >1.30
   - Fixed in this release with compatible versions

3. **Polygon not closing**
   - Try clicking the radio button to refresh canvas state
   - Ensure you click within 25 pixels of the starting point

---

## 📄 License

Developed by the Plant Pathology Laboratory, Institute of Plant Breeding, UPLB.
Co-funded by the Gates Foundation.

---

## 🎯 Version History

**v6.1** (Current) - Streamlit Release
- ✅ Compatible with Streamlit 1.40.0
- ✅ Fragment-based performance optimization
- ✅ Zoom/pan navigation for large images
- ✅ Scale calibration with real-world measurements
- ✅ Cloud-optimized deployment

**v6.0** - Legacy version
- Basic functionality without fragments
- Streamlit 1.17.0

---

## 🌐 Running Locally vs. Cloud

### Local (Recommended for Development)
```bash
streamlit run appv6.1.py --logger.level=info
```
- Full control over resources
- Faster development iteration
- Can use opencv-python (with display)

### Streamlit Cloud
- Auto-deploys from GitHub
- Free tier: up to 3 active apps
- Limitations: 1 CPU, 2GB RAM per app
- Batch processing limited to 20 images
- Auto-sleeps after 7 days of inactivity

---

**Last Updated:** September 28, 2026
**Tested:** Streamlit 1.40.0, Python 3.11, Windows/macOS/Linux
