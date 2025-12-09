import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pydicom
import io
import time  # Ditambahkan untuk timestamp log

# Handle optional dependencies gracefully
try:
    from skimage.metrics import structural_similarity as ssim
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CSS (TEMA FUN & INTERACTIVE)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MRI Brain Fusion",
    layout="wide",
    page_icon="🧠",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan Fun, Ceria, dan Interaktif
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@400;600;700&display=swap');

    /* Global Background - Fun Gradient */
    .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%); /* Sunny Morning Gradient */
        background: linear-gradient(120deg, #e0c3fc 0%, #8ec5fc 100%); /* Soft Purple-Blue Gradient */
        background: #fdfbf7; /* Creamy White Base */
        background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
        background-size: 20px 20px;
        font-family: 'Quicksand', sans-serif;
        color: #2d3748;
    }

    /* Sidebar - Glassmorphism Light */
    section[data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-right: 1px solid #e2e8f0;
        box-shadow: 4px 0 15px rgba(0,0,0,0.05);
    }
    
    /* Headers - Colorful & Bold */
    h1, h2, h3 {
        font-family: 'Quicksand', sans-serif;
        color: #4a5568;
        font-weight: 700;
    }
    
    /* Main Title Gradient */
    h1 {
        background: linear-gradient(to right, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
    }

    /* Custom Buttons - Rounded Pills with Pop */
    div.stButton > button {
        background: linear-gradient(45deg, #FF8008, #FFC837); /* Orange Gradient */
        color: white;
        border: none;
        border-radius: 50px; /* Fully rounded */
        padding: 0.6rem 1.5rem;
        font-weight: 700;
        box-shadow: 0 4px 15px rgba(255, 128, 8, 0.3);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 8px 20px rgba(255, 128, 8, 0.5);
        color: white !important;
    }

    /* File Uploader - Friendly Box */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff;
        border: 2px dashed #a0aec0;
        border-radius: 20px;
        padding: 15px; /* Reduced padding slightly */
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #667eea;
        background-color: #ebf4ff;
    }

    /* Tabs Styling - Floating Cards */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px; /* Reduced gap to prevent overflow */
        background-color: transparent;
        flex-wrap: wrap; /* Allow wrapping if needed */
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 15px;
        color: #718096;
        padding: 8px 16px; /* Compact padding */
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: none;
        font-weight: 600;
        font-size: 0.85rem; /* Slightly smaller text */
        flex-grow: 1; /* Distribute space evenly */
        text-align: center;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea !important;
        color: white !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Fix for tab content spacing */
    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 10px;
    }

    /* Sliders - Chunky & Fun */
    .stSlider > div > div > div > div {
        background-color: #667eea;
        height: 10px;
    }
    .stSlider > div > div > div > div[role="slider"] {
        height: 20px;
        width: 20px;
        background-color: #fff;
        border: 4px solid #667eea;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }

    /* Metrics Box - Colorful Cards */
    .metric-box {
        background: #ffffff;
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        border-left: 8px solid #667eea;
        transition: transform 0.3s ease;
    }
    .metric-box:hover {
        transform: translateY(-5px) rotate(1deg);
    }
    .metric-value {
        font-family: 'Quicksand', sans-serif;
        font-size: 2rem;
        color: #2d3748;
        font-weight: 800;
    }
    .metric-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #718096;
        font-weight: 700;
        margin-bottom: 5px;
    }

    /* Alerts - Soft Bubbles */
    .stAlert {
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #fff;
        border-radius: 10px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. FUNGSI UTILITAS (Backend Processing)
# -----------------------------------------------------------------------------

def load_image(uploaded_file):
    """
    Memuat gambar dari file DICOM, PNG, atau JPEG dan menormalkannya ke array numpy uint8.
    Mengembalikan tuple (image_array, metadata_dict).
    """
    metadata = {}
    
    if uploaded_file is None:
        return None, None

    filename = uploaded_file.name.lower()

    try:
        if filename.endswith('.dcm'):
            ds = pydicom.dcmread(uploaded_file)
            img = ds.pixel_array.astype(float)
            
            # --- FIX: Handle Multi-dimensional DICOM (RGB or Volume) ---
            if img.ndim == 3:
                # Jika formatnya (Rows, Cols, 3) -> RGB -> Convert rata-rata ke Grayscale
                if img.shape[2] == 3:
                    img = np.mean(img, axis=2)
                # Jika formatnya (Slices, Rows, Cols) -> Volume -> Ambil slice tengah
                else:
                    mid_slice = img.shape[0] // 2
                    img = img[mid_slice]
            elif img.ndim == 4:
                # Jika 4D (Time/Volume), ambil frame pertama
                img = img[0, 0]
            # -----------------------------------------------------------
            
            # Normalisasi DICOM ke 0-255 (Hindari pembagian nol)
            max_val = np.max(img)
            if max_val > 0:
                img = (np.maximum(img, 0) / max_val) * 255.0
            else:
                img = np.zeros_like(img)
                
            img = np.uint8(img)
            
            # Ambil metadata DICOM
            metadata = {
                "Patient ID": getattr(ds, 'PatientID', 'Anon'),
                "Modality": getattr(ds, 'Modality', 'MRI'),
                "Study Date": getattr(ds, 'StudyDate', 'N/A'),
                "Resolution": f"{ds.Rows}x{ds.Columns}",
                "Spacing": str(getattr(ds, 'PixelSpacing', 'N/A'))
            }
            
        else:
            image = Image.open(uploaded_file).convert('L') # Convert to Grayscale
            img = np.array(image)
            metadata = {
                "Filename": uploaded_file.name,
                "Format": image.format,
                "Size": f"{image.width}x{image.height}",
                "Mode": image.mode
            }

        return img, metadata

    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None, None

def calculate_metrics(reference, target):
    """Menghitung MSE, PSNR dan SSIM antara citra REFERENSI (Gold Std) dan TARGET (Fusion)."""
    # Pastikan ukuran sama untuk perhitungan (Resize referensi agar cocok dengan target fusion)
    if reference.shape != target.shape:
        reference = cv2.resize(reference, (target.shape[1], target.shape[0]))
    
    # Calculate MSE (Mean Squared Error)
    mse_val = np.mean((reference - target) ** 2)

    # Calculate PSNR
    psnr_val = cv2.PSNR(reference, target)
    
    # SSIM requires scikit-image
    if HAS_SKIMAGE:
        ssim_val = ssim(reference, target, data_range=target.max() - target.min())
    else:
        ssim_val = 0.0
        
    return mse_val, psnr_val, ssim_val

def adjust_window_level(image, contrast, brightness):
    """
    Simulasi Window/Leveling radiologi.
    Contrast ~ Window Width (alpha), Brightness ~ Window Center/Level (beta).
    """
    # Formula standar brightness/contrast OpenCV: new_img = alpha * old_img + beta
    # Contrast range slider (0.5 - 3.0), Brightness range (-100 to 100)
    adj = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    return adj

# -----------------------------------------------------------------------------
# 3. LAYOUT UTAMA APLIKASI
# -----------------------------------------------------------------------------

def main():
    
    # --- SIDEBAR (LAB CONTROL PANEL) ---
    with st.sidebar:
        st.title("🔬 LAB CONTROL")
        st.caption("FUSION WORKBENCH v3.9")
        st.markdown("---")
        
        st.subheader("Experimental Protocol")
        st.markdown("""
        1. **Input Data**: Load T1 & T2 (Source).
        2. **Gold Standard**: Load Reference Image.
        3. **Processing**: Adjust gain & mix ratio.
        4. **Validation**: Compare Fusion vs Gold Std.
        """)
        
        st.markdown("---")
        
        # Dependency Status
        st.caption("SYSTEM STATUS:")
        if not HAS_SKIMAGE:
            st.caption("• Scikit-image: Missing (SSIM Disabled)")
        else:
            st.caption("• Scikit-image: Active")
            
        if not HAS_PLOTLY:
            st.caption("• Plotly: Missing (Static View Mode)")
        else:
            st.caption("• Plotly: Active")
            
        st.markdown("---")
        # Change Department Name
        st.caption("Poltekkes Kemenkes Semarang")
        st.caption("Research Use Only (RUO)")

    # --- HEADER ---
    col_icon, col_header = st.columns([1, 10])
    with col_icon:
        # Changed icon to BRAIN for MRI Context
        st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🧠</h1>", unsafe_allow_html=True)
            
    with col_header:
        # TITLE REMAINS EXACTLY AS REQUESTED
        st.title("FUSI CITRA MRI OTAK SEKUEN 3D SPACE DAN 3D TOF MRA")
        st.markdown("##### Poltekkes Kemenkes Semarang - Digital Imaging Laboratory")
    
    st.markdown("---")

    # --- MAIN WORKSPACE (3 COLUMN LAYOUT) ---
    # Left: Parameters | Center: Scope View | Right: Data Readout
    # UPDATED COLUMN RATIO: Memberi lebih banyak ruang ke kiri agar Tab tidak tertutup
    col_left, col_center, col_right = st.columns([1.3, 2.2, 1])

    # Inisialisasi Session State
    if 'img1_raw' not in st.session_state: st.session_state.img1_raw = None
    if 'img2_raw' not in st.session_state: st.session_state.img2_raw = None
    if 'img_ref_raw' not in st.session_state: st.session_state.img_ref_raw = None
    if 'meta1' not in st.session_state: st.session_state.meta1 = {}
    if 'meta2' not in st.session_state: st.session_state.meta2 = {}
    if 'meta_ref' not in st.session_state: st.session_state.meta_ref = {}

    # --- PANEL KIRI: PARAMETERS ---
    with col_left:
        st.subheader("PARAMETERS")
        
        # Upload Tab (3 Tabs as requested) - SHORTENED LABELS
        tab1, tab2, tab3 = st.tabs(["T1 (SPACE)", "T2 (TOF)", "Gold Std"])
        
        with tab1:
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            file1 = st.file_uploader("Upload Citra T1 (Source)", type=['dcm', 'png', 'jpg', 'jpeg'], key="u1")
            if file1:
                img, meta = load_image(file1)
                st.session_state.img1_raw = img
                st.session_state.meta1 = meta
                st.info("Citra T1 Loaded")

        with tab2:
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            file2 = st.file_uploader("Upload Citra T2 (Overlay)", type=['dcm', 'png', 'jpg', 'jpeg'], key="u2")
            if file2:
                img, meta = load_image(file2)
                st.session_state.img2_raw = img
                st.session_state.img2_raw = img
                st.session_state.meta2 = meta
                st.info("Citra T2 Loaded")

        with tab3: # Gold Standard Uploader
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            file3 = st.file_uploader("Upload Reference (Gold Std)", type=['dcm', 'png', 'jpg', 'jpeg'], key="u3")
            if file3:
                img, meta = load_image(file3)
                st.session_state.img_ref_raw = img
                st.session_state.meta_ref = meta
                st.success("Gold Std Loaded")
            else:
                st.caption("Upload hasil fusi dari workstation (Siemens/GE) sebagai referensi.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("SIGNAL CONTROL")
        
        # Kontrol Windowing (Brightness/Contrast)
        contrast = st.slider("Contrast Gain (α)", 0.5, 3.0, 1.0, 0.1)
        brightness = st.slider("Brightness Offset (β)", -100, 100, 0, 5)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("MIXING")
        # Slider Bobot Fusion
        alpha = st.slider("Fusion Ratio (α)", 0.0, 1.0, 0.5, 0.05, help="0.0 = T1 (SPACE), 1.0 = T2 (TOF)")
        
        # Opsi Colormap
        apply_colormap = st.checkbox("Overlay Heatmap (JET)", value=False)


    # --- PANEL TENGAH: SCOPE VIEW ---
    with col_center:
        st.subheader("SCOPE VIEWPORT")
        
        # Cek apakah gambar sudah ada (T1 & T2 wajib untuk fusi)
        if st.session_state.img1_raw is not None and st.session_state.img2_raw is not None:
            
            # 1. Resize Image B agar sama dengan Image A
            h, w = st.session_state.img1_raw.shape
            img2_resized = cv2.resize(st.session_state.img2_raw, (w, h))
            
            # Ensure Channel Consistency (Grayscale Enforcement)
            if img2_resized.ndim == 3 and st.session_state.img1_raw.ndim == 2:
                img2_resized = cv2.cvtColor(img2_resized, cv2.COLOR_RGB2GRAY)
            
            # 2. Apply Window/Level adjustments
            img1_disp = adjust_window_level(st.session_state.img1_raw, contrast, brightness)
            img2_disp = adjust_window_level(img2_resized, contrast, brightness)

            # 3. Fusion Logic (Hanya T1 dan T2)
            fused_img = cv2.addWeighted(img1_disp, 1 - alpha, img2_disp, alpha, 0)
            
            if apply_colormap:
                fused_display = cv2.applyColorMap(fused_img, cv2.COLORMAP_JET)
                fused_display = cv2.cvtColor(fused_display, cv2.COLOR_BGR2RGB)
            else:
                fused_display = fused_img
            
            # --- DISPLAY LOGIC ---
            if HAS_PLOTLY:
                fig = go.Figure()
                
                if apply_colormap:
                    fig.add_trace(go.Image(z=fused_display))
                else:
                    # Grayscale Heatmap for better scientific viz
                    fig.add_trace(go.Heatmap(z=np.flipud(fused_display), colorscale='Gray', showscale=False))

                # Update Layout Plotly agar Clean Scientific
                fig.update_layout(
                    margin=dict(l=20, r=20, t=20, b=20),
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',  
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False, scaleanchor="x"),
                    dragmode="pan",
                    height=550,
                    title_text="Real-time Fusion Output (T1 + T2)",
                    title_font_size=12,
                    title_font_family="Quicksand, sans-serif",
                    font=dict(color="#2d3748") # Dark text for light theme
                )
                
                # Force axis color
                fig.update_xaxes(color="#2d3748")
                fig.update_yaxes(color="#2d3748")
                
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'scrollZoom': True})
            else:
                # STATIC FALLBACK
                st.image(fused_display, caption="Fused Output [Static Mode - Install Plotly for Zoom]", use_container_width=True, clamp=True, channels="RGB" if apply_colormap else "GRAY")

            # Preview Kecil (T1, T2, dan Gold Standard)
            with st.expander("SOURCE & REFERENCE INSPECTION", expanded=True):
                cols = st.columns(3)
                cols[0].image(img1_disp, caption="Input T1 (SPACE)", use_container_width=True, clamp=True, channels='GRAY')
                cols[1].image(img2_disp, caption="Input T2 (TOF)", use_container_width=True, clamp=True, channels='GRAY')
                
                if st.session_state.img_ref_raw is not None:
                    cols[2].image(st.session_state.img_ref_raw, caption="GOLD STANDARD", use_container_width=True, clamp=True, channels='GRAY')
                else:
                    cols[2].info("No Gold Std")

        else:
            # Placeholder Clean Lab
            st.info("SILAKAN UPLOAD CITRA T1 DAN T2 UNTUK MEMULAI.")
            st.markdown(
                """
                <div style="display: flex; justify-content: center; align-items: center; height: 400px; 
                background: #ffffff; border: 2px dashed #cbd5e0; border-radius: 20px; color: #a0aec0; font-family: 'Quicksand'; font-weight: bold;">
                    AWAITING INPUT SIGNALS (T1 & T2)...
                </div>
                """, 
                unsafe_allow_html=True
            )


    # --- PANEL KANAN: DATA READOUT ---
    with col_right:
        st.subheader("DATA READOUT")
        
        # Metadata
        st.markdown("**METADATA CITRA**")
        if st.session_state.meta1:
            st.text("T1 (3D SPACE):")
            st.json(st.session_state.meta1)
        else:
            st.caption("T1: N/A")
            
        st.markdown("---")
        
        st.markdown("**METRICS VALIDATION**")
        
        if st.session_state.img1_raw is not None and st.session_state.img2_raw is not None:
            # Determine Reference Image for Metrics
            if st.session_state.img_ref_raw is not None:
                reference_img = st.session_state.img_ref_raw
                ref_label = "GOLD STANDARD (Workstation)"
                st.success(f"Comparing: Fusion vs {ref_label}")
            else:
                reference_img = img1_disp
                ref_label = "SELF REFERENCE (T1)"
                st.warning("⚠️ Gold Std Missing. Calculating deviation from T1.")

            # Hitung Metrics (Membandingkan Hasil Fusion dengan Reference)
            mse_val, psnr_val, ssim_val = calculate_metrics(reference_img, fused_img)
            
            ssim_display = f"{ssim_val:.4f}" if HAS_SKIMAGE else "ERR"
            
            # Custom Metric Box UI - Fun Cards
            st.markdown(f"""
            <div style="font-size: 0.8rem; color: #718096; margin-bottom: 5px;">Ref: {ref_label}</div>
            
            <div class="metric-box" style="border-left-color: #ff6b6b;">
                <div class="metric-label">MSE (Mean Squared Error)</div>
                <div class="metric-value">{mse_val:.2f}</div>
                <div style="font-size: 0.7rem; color: #a0aec0;">(Lower is better match)</div>
            </div>

            <div class="metric-box" style="border-left-color: #4ecdc4;">
                <div class="metric-label">PSNR RATIO</div>
                <div class="metric-value">{psnr_val:.2f} <span style="font-size: 0.8rem">dB</span></div>
                <div style="font-size: 0.7rem; color: #a0aec0;">(Higher is better quality)</div>
            </div>
            
            <div class="metric-box" style="border-left-color: #ffe66d;">
                <div class="metric-label">STRUCTURAL INDEX (SSIM)</div>
                <div class="metric-value">{ssim_display}</div>
                <div style="font-size: 0.7rem; color: #a0aec0;">(1.0 = Identical)</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Data untuk Export
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            log_text = f"""
=================================================
MRI FUSION ANALYSIS LOG - POLTEKKES SEMARANG
=================================================
Date/Time: {timestamp}

[CONFIGURATION]
Reference Image : {ref_label}
Alpha (Fusion)  : {alpha}
Contrast (Gain) : {contrast}
Brightness      : {brightness}

[METRICS RESULTS]
MSE (Mean Sq Err): {mse_val:.6f}
PSNR (Peak SNR)  : {psnr_val:.6f} dB
SSIM (Struct Idx): {ssim_display}

[METADATA - T1 SPACE]
{st.session_state.meta1}

[METADATA - T2 TOF]
{st.session_state.meta2}

[METADATA - GOLD STANDARD]
{st.session_state.meta_ref}
=================================================
Digital Imaging Laboratory - RUO
            """

            # Tombol Download Log
            st.download_button(
                label="📥 UNDUH LOG (.TXT)",
                data=log_text,
                file_name=f"fusion_log_{int(time.time())}.txt",
                mime="text/plain",
                help="Simpan hasil analisis ke komputer lokal."
            )
        else:
             st.caption("AWAITING FUSION DATA")

if __name__ == "__main__":
    main()