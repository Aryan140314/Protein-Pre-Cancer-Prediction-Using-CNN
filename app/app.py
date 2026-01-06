"""
Protein Pre-Cancer Prediction Dashboard
Advanced CNN-based protein structure analysis for cancer detection
"""

import os
import sys
import base64
import importlib.util
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import timm
import py3Dmol
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import io

# Optional dependencies
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

@dataclass
class Config:
    """Centralized configuration"""
    PROJECT_ROOT: str = r"D:\Protein_Cancer\Protein_Project"
    MODEL_DIR: str = None
    PREPROCESS_PATH: str = None
    
    # Model paths
    PATH_DENSE: str = None
    PATH_EFF: str = None
    PATH_SER: str = None
    
    # Theme colors
    PINK: str = "#e0006c"
    CANCER_RED: str = "#ff4d4d"
    NONC_GREEN: str = "#00b894"
    
    # Device
    DEVICE: torch.device = None
    
    # Constants
    DISTANCE_THRESHOLD: float = 8.0
    HIGH_BFACTOR_THRESHOLD: float = 50.0
    IMG_NORM_MEAN: List[float] = None
    IMG_NORM_STD: List[float] = None
    
    def __post_init__(self):
        self.MODEL_DIR = os.path.join(self.PROJECT_ROOT, "models_3")
        self.PREPROCESS_PATH = os.path.join(self.PROJECT_ROOT, "Advance_pre-processing.py")
        self.PATH_DENSE = os.path.join(self.MODEL_DIR, "densenet201_best.pth")
        self.PATH_EFF = os.path.join(self.MODEL_DIR, "efficientnet_b4_best.pth")
        self.PATH_SER = os.path.join(self.MODEL_DIR, "seresnet50_best.pth")
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.IMG_NORM_MEAN = [0.485, 0.456, 0.406]
        self.IMG_NORM_STD = [0.229, 0.224, 0.225]

config = Config()

# Hydrophobic amino acids set
HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "CYS", "PRO"}

# ============================================================================
# STREAMLIT PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="BioMedical AI Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

def inject_custom_css():
    """Inject custom CSS for professional styling"""
    st.markdown(f"""
        <style>
        :root {{
            --primary: #2563eb;
            --text: #1e293b;
            --card-bg: #ffffff;
            --pink: {config.PINK};
            --shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
        }}
        
        .stApp {{
            background-color: #f0f4f8;
            font-family: 'Segoe UI', 'Roboto', sans-serif;
        }}
        
        /* Card styling */
        .custom-card {{
            background: var(--card-bg);
            padding: 1.25rem;
            border-radius: 15px;
            box-shadow: var(--shadow);
            margin-bottom: 1.25rem;
            height: 100%;
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        
        .custom-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 12px -2px rgba(0,0,0,0.12);
        }}
        
        /* Statistics styling */
        .stat-label {{
            color: #64748b;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }}
        
        .stat-value {{
            color: var(--text);
            font-size: 1.1rem;
            font-weight: 700;
        }}
        
        /* Heatmap title */
        .heatmap-title {{
            font-size: 1rem;
            font-weight: 600;
            text-align: center;
            margin-bottom: 0.75rem;
            color: #333;
        }}

        /* Scan animation */
        @keyframes scan {{
            0% {{ top: 0%; opacity: 0; }}
            10% {{ opacity: 1; }}
            90% {{ opacity: 1; }}
            100% {{ top: 100%; opacity: 0; }}
        }}
        
        .scan-container {{
            position: relative;
            overflow: hidden;
            border-radius: 10px;
        }}
        
        .scan-line {{
            position: absolute;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, transparent, #ef4444, transparent);
            box-shadow: 0 0 10px #ef4444;
            top: 0;
            left: 0;
            z-index: 5;
            animation: scan 2.5s ease-in-out infinite;
            pointer-events: none;
        }}

        /* Model probability table */
        .model-table {{
            border-collapse: collapse;
            width: 100%;
            font-size: 0.9rem;
            margin-top: 0.5rem;
            box-shadow: var(--shadow);
        }}
        
        .model-table th, .model-table td {{
            border: 1px solid #e2e8f0;
            padding: 0.75rem;
            text-align: center;
        }}
        
        .model-table th {{
            background: var(--pink);
            color: #ffffff;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.8rem;
            letter-spacing: 0.5px;
        }}
        
        .model-table td:first-child {{
            text-align: left;
            font-weight: 600;
        }}
        
        .model-table tbody tr:hover {{
            background-color: #f8fafc;
        }}
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {{
            background-color: #ffffff;
        }}
        
        .sidebar-content {{
            width: 100%;
            text-align: center;
        }}
        
        .sidebar-divider {{
            height: 1px; 
            background: linear-gradient(90deg, transparent, #d1d5db, transparent);
            margin: 1.5rem 0;
        }}
        
        /* Button styling */
        .stButton > button {{
            width: 100%;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }}
        
        /* Loading spinner */
        .stSpinner > div {{
            border-color: var(--pink) !important;
        }}
        </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def img_to_base64(img_pil: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buff = BytesIO()
    img_pil.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode()

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    return numerator / denominator if denominator != 0 else default

# ============================================================================
# PDB ANALYSIS FUNCTIONS
# ============================================================================

class ProteinAnalyzer:
    """Handles protein structure analysis from PDB files"""
    
    @staticmethod
    def compute_secondary_structure(pdb_text: str) -> Tuple[int, int]:
        """
        Calculate secondary structure using DSSP algorithm (via Biotite).
        This works even if HELIX/SHEET records are missing in the PDB file.
        
        Returns:
            Tuple of (alpha_helix_length, beta_sheet_length)
        """
        try:
            # Method 1: Mathematical Calculation using Biotite (DSSP)
            # Create a file-like object from the string
            pdb_file = pdb.PDBFile.read(io.StringIO(pdb_text))
            
            # Get the structure (model 1)
            array = pdb_file.get_structure(model=1)
            
            # Filter for protein backbone to ensure correct annotation
            # (Annotate_sse requires a proper atom array)
            sse = struc.annotate_sse(array)
            
            # Count residues ('a' = alpha helix, 'b' = beta sheet)
            alpha_len = int(np.sum(sse == 'a'))
            beta_len = int(np.sum(sse == 'b'))
            
            return alpha_len, beta_len

        except Exception as e:
            # Method 2: Fallback to old header parsing
            # (Used if Biotite fails or file format is unusual)
            # st.warning(f"Advanced structure calculation failed, using headers: {e}") # Optional debug
            alpha_len = 0
            beta_len = 0
            
            for line in pdb_text.splitlines():
                record_type = line[0:6].strip()
                try:
                    if record_type == "HELIX":
                        start = int(line[21:25].strip())
                        end = int(line[33:37].strip())
                        alpha_len += max(0, end - start + 1)
                    elif record_type == "SHEET":
                        start = int(line[22:26].strip())
                        end = int(line[33:37].strip())
                        beta_len += max(0, end - start + 1)
                except (ValueError, IndexError):
                    continue
            
            return alpha_len, beta_len
    
    @staticmethod
    def compute_residue_properties(pdb_text: str) -> Tuple[int, float]:
        """
        Calculate residue composition and properties
        
        Returns:
            Tuple of (total_residues, hydrophobic_percentage)
        """
        residues = {}
        
        for line in pdb_text.splitlines():
            if not line.startswith("ATOM"):
                continue
            
            try:
                res_name = line[17:20].strip()
                chain_id = line[21].strip()
                res_seq = int(line[22:26].strip())
                residues[(chain_id, res_seq)] = res_name
            except (ValueError, IndexError):
                continue
        
        total_residues = len(residues)
        if total_residues == 0:
            return 0, 0.0
        
        hydrophobic_count = sum(1 for r in residues.values() if r in HYDROPHOBIC_RESIDUES)
        hydrophobic_pct = (hydrophobic_count / total_residues) * 100
        
        return total_residues, hydrophobic_pct
    
    @staticmethod
    def classify_structure(alpha_pct: float, beta_pct: float, coil_pct: float) -> str:
        """Classify protein structure based on secondary structure composition"""
        if coil_pct > 50:
            return "Disordered / Coil-rich"
        elif alpha_pct > 40 and beta_pct < 20:
            return "Alpha-rich"
        elif beta_pct > 40 and alpha_pct < 20:
            return "Beta-rich"
        elif alpha_pct >= 20 and beta_pct >= 20:
            return "Alpha/Beta Mixed"
        else:
            return "Mixed / Other"
    
    @staticmethod
    def calculate_instability_score(avg_conf: float, density: float) -> float:
        """
        Calculate protein instability score based on confidence and density
        
        Returns:
            Score between 0-100 (higher = more unstable)
        """
        conf_component = (1.0 - min(avg_conf, 100.0) / 100.0) * 40.0
        density_component = (1.0 - density) * 60.0
        instability = conf_component + density_component
        return max(0.0, min(100.0, instability))

# ============================================================================
# MODEL MANAGEMENT
# ============================================================================

class ModelManager:
    """Manages loading and inference of multiple CNN models"""
    
    def __init__(self, config: Config):
        self.config = config
        self.models: Dict[str, nn.Module] = {}
    
    def load_model(self, name: str, model_path: str, 
                   model_fn, classifier_config: dict) -> bool:
        """
        Load a single model with error handling
        
        Args:
            name: Model display name
            model_path: Path to model weights
            model_fn: Function to create model architecture
            classifier_config: Dict with classifier modifications
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                st.warning(f"⚠️ Model weights not found: {name}")
                return False
            
            model = model_fn()
            
            # Modify classifier if needed
            if classifier_config:
                for attr, value in classifier_config.items():
                    setattr(model, attr, value)
            
            model.load_state_dict(torch.load(model_path, map_location=self.config.DEVICE))
            model.to(self.config.DEVICE).eval()
            self.models[name] = model
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading {name}: {str(e)}")
            return False
    
    def load_all_models(self):
        """Load all available models"""
        # DenseNet201
        self.load_model(
            "DenseNet201",
            self.config.PATH_DENSE,
            lambda: models.densenet201(weights=None),
            {"classifier": nn.Linear(1920, 1)}
        )
        
        # EfficientNet-B4
        self.load_model(
            "EfficientNetB4",
            self.config.PATH_EFF,
            lambda: models.efficientnet_b4(weights=None),
            {"classifier": nn.Sequential(
                models.efficientnet_b4(weights=None).classifier[0],
                nn.Linear(1792, 1)
            )}
        )
        
        # SE-ResNet50
        self.load_model(
            "SE-ResNet50",
            self.config.PATH_SER,
            lambda: timm.create_model("seresnet50", pretrained=False, num_classes=1),
            {}
        )
        
        if not self.models:
            st.error("❌ No models loaded successfully. Please check model paths.")
            st.stop()
        
        st.success(f"✅ Loaded {len(self.models)} model(s): {', '.join(self.models.keys())}")
    
    def predict(self, input_tensor: torch.Tensor) -> Dict[str, float]:
        """
        Run inference on all loaded models
        
        Returns:
            Dictionary mapping model names to cancer probabilities
        """
        predictions = {}
        
        with torch.no_grad():
            for name, model in self.models.items():
                output = torch.sigmoid(model(input_tensor)).item()
                cancer_prob = 1.0 - output  # Convert to cancer probability
                predictions[name] = cancer_prob
        
        return predictions
    
    def get_target_layer(self, model_name: str) -> Optional[List]:
        """Get the appropriate target layer for Grad-CAM"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        if "DenseNet" in model_name:
            return [model.features[-1]]
        elif "EfficientNet" in model_name:
            return [model.features[-1]]
        elif "SE-ResNet" in model_name or "ResNet" in model_name:
            return [model.layer4[-1]]
        
        return None

# ============================================================================
# GRAD-CAM VISUALIZATION
# ============================================================================

class GradCAMVisualizer:
    """Generate Grad-CAM heatmaps for model interpretability"""
    
    @staticmethod
    def generate(model: nn.Module, target_layer: List, 
                 input_tensor: torch.Tensor, img_pil: Image.Image) -> Optional[Image.Image]:
        try:
            class CancerTarget:
                def __call__(self, model_output):
                    return -1 * model_output  # focus on cancer class

            # FIXED: removed use_cuda
            cam = GradCAM(model=model, target_layers=target_layer)

            grayscale_cam = cam(input_tensor=input_tensor, targets=[CancerTarget()])[0, :]

            # Resize original image
            img_resized = img_pil.resize((grayscale_cam.shape[1], grayscale_cam.shape[0]))
            img_float = np.float32(img_resized) / 255.0

            visualization = show_cam_on_image(img_float, grayscale_cam, use_rgb=True)
            return Image.fromarray(visualization)

        except Exception as e:
            st.warning(f"⚠️ Grad-CAM generation failed: {str(e)}")
            return None

# ============================================================================
# PDF REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """Generate comprehensive PDF diagnostic reports"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate(self, protein_data: dict, model_predictions: dict, 
                 final_diagnosis: str, confidence: float) -> bytes:
        """
        Generate PDF report with all analysis results
        
        Args:
            protein_data: Dictionary with protein analysis results
            model_predictions: Dictionary with model predictions
            final_diagnosis: Final diagnosis string
            confidence: Ensemble confidence score
            
        Returns:
            PDF file as bytes
        """
        pdf_filename = "temp_diagnostic_report.pdf"
        
        with PdfPages(pdf_filename) as pdf:
            # Page 1: Main Report
            self._generate_main_page(pdf, protein_data, model_predictions, 
                                     final_diagnosis, confidence)
            
            # Page 2: Advanced Analytics
            self._generate_analytics_page(pdf, protein_data, model_predictions)
        
        # Read and return PDF bytes
        with open(pdf_filename, "rb") as f:
            pdf_bytes = f.read()
        
        # Cleanup
        if os.path.exists(pdf_filename):
            os.remove(pdf_filename)
        
        return pdf_bytes
    
    def _generate_main_page(self, pdf, protein_data, model_predictions, 
                            final_diagnosis, confidence):
        """Generate the main diagnostic report page"""
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        # Title
        fig.text(0.5, 0.93, "Protein Cancer Diagnosis Report",
                ha="center", va="top", fontsize=16, weight="bold")
        
        # Diagnosis banner
        from matplotlib import patches
        banner = patches.FancyBboxPatch(
            (0.05, 0.86), 0.90, 0.08,
            boxstyle="round,pad=0.02",
            linewidth=2,
            edgecolor=self.config.PINK,
            facecolor=self.config.PINK,
            alpha=0.15
        )
        fig.add_artist(banner)
        
        fig.text(0.5, 0.90, f"Diagnosis: {final_diagnosis}",
                ha="center", va="center", fontsize=12, weight="bold")
        fig.text(0.5, 0.875, f"Confidence: {confidence*100:.2f}%",
                ha="center", va="center", fontsize=10)
        
        # Biological indicators table
        fig.text(0.05, 0.82, "1. Biological Indicators",
                fontsize=11, weight="bold")
        
        bi_headers = [
            "Length", "Alpha", "Beta", "Coil",
            "α/β Ratio", "Hydrophobic", "Instability", "Mut. Vuln."
        ]
        bi_data = [[
            f"{protein_data['protein_len']} aa",
            f"{protein_data['alpha_pct']:.1f}%",
            f"{protein_data['beta_pct']:.1f}%",
            f"{protein_data['coil_pct']:.1f}%",
            protein_data['alpha_beta_text'],
            f"{protein_data['hydrophobic_pct']:.1f}%",
            f"{protein_data['instab_score']:.1f}",
            f"{protein_data['mutation_vulnerability']:.1f}%"
        ]]
        
        ax_table = fig.add_axes([0.05, 0.70, 0.90, 0.10])
        ax_table.axis("off")
        table = ax_table.table(
            cellText=bi_data,
            colLabels=bi_headers,
            loc="center",
            cellLoc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.5)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor(self.config.PINK)
                cell.set_text_props(color="white", weight="bold")
            else:
                cell.set_facecolor("white")
        
        # Model breakdown
        fig.text(0.05, 0.64, "2. AI Model Breakdown",
                fontsize=11, weight="bold")
        
        y_start = 0.58
        y_step = 0.16
        for idx, (name, cancer_prob) in enumerate(model_predictions.items()):
            y = y_start - idx * y_step
            if y < 0.10:
                break
            
            fig.text(0.07, y + 0.03, name, fontsize=10, weight="bold")
            fig.text(0.25, y + 0.03, f"Cancer Risk: {cancer_prob*100:.2f}%", fontsize=10)
            
            # Pie chart
            ax_pie = fig.add_axes([0.55, y - 0.01, 0.18, 0.18])
            ax_pie.pie(
                [cancer_prob, 1 - cancer_prob],
                labels=["Cancer", "Non-Cancer"],
                autopct="%1.0f%%",
                startangle=90,
                colors=[self.config.CANCER_RED, self.config.NONC_GREEN],
                textprops={"fontsize": 7}
            )
            ax_pie.axis("equal")
        
        # Footer
        fig.text(0.5, 0.03,
                "Generated by Protein Pre-Cancer Prediction System | Powered by Deep Learning",
                ha="center", fontsize=7, color="#555555")
        
        pdf.savefig(fig, dpi=300)
        plt.close(fig)
    
    def _generate_analytics_page(self, pdf, protein_data, model_predictions):
        """Generate advanced analytics page"""
        fig, axes = plt.subplots(2, 1, figsize=(8.27, 11.69))
        fig.subplots_adjust(hspace=0.3, left=0.1, right=0.9, top=0.90, bottom=0.08)
        
        # Title
        fig.suptitle("Advanced Model Analytics", fontsize=14, weight="bold")
        
        # Model comparison bar chart
        ax1 = axes[0]
        model_names = list(model_predictions.keys())
        cancer_probs = [model_predictions[m] * 100 for m in model_names]
        
        bars = ax1.bar(model_names, cancer_probs, color=self.config.PINK, alpha=0.8, edgecolor='black')
        ax1.set_ylabel("Cancer Probability (%)", fontsize=10, weight="bold")
        ax1.set_ylim(0, 100)
        ax1.set_title("Model-wise Cancer Risk Assessment", fontsize=11, weight="bold", pad=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels on bars
        for bar, prob in zip(bars, cancer_probs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{prob:.1f}%', ha='center', va='bottom', fontsize=8, weight='bold')
        
        # Structural composition text
        ax2 = axes[1]
        ax2.axis('off')
        
        summary_text = f"""
Protein Structural Summary:

• Structural Classification: {protein_data['struct_class']}
• Secondary Structure Distribution:
  - Alpha Helix: {protein_data['alpha_pct']:.1f}%
  - Beta Sheet: {protein_data['beta_pct']:.1f}%
  - Coil/Random: {protein_data['coil_pct']:.1f}%

• Biochemical Properties:
  - Hydrophobic Residues: {protein_data['hydrophobic_pct']:.1f}%
  - Avg. B-factor (Confidence): {protein_data['avg_conf']:.2f}
  - Structural Density: {protein_data['density']:.3f}

• Stability Indicators:
  - Instability Score: {protein_data['instab_score']:.1f}/100
  - Disorder Score: {protein_data['disorder_score']:.1f}%
  - Mutation Vulnerability: {protein_data['mutation_vulnerability']:.1f}%

• Interpretation:
  Higher instability and mutation vulnerability scores may indicate
  structural regions prone to cancer-associated alterations.
        """
        
        ax2.text(0.1, 0.9, summary_text, fontsize=9, verticalalignment='top',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        # Add QR code if available
        if QR_AVAILABLE:
            try:
                qr = qrcode.QRCode(box_size=3, border=1)
                qr.add_data("https://example.com/protein_viewer")
                qr.make(fit=True)
                img_qr = qr.make_image(fill_color="black", back_color="white")
                
                ax_qr = fig.add_axes([0.78, 0.10, 0.15, 0.15])
                ax_qr.imshow(img_qr, cmap='gray')
                ax_qr.axis("off")
                fig.text(0.855, 0.08, "Scan for 3D Viewer", ha='center', fontsize=7)
            except Exception:
                pass
        
        pdf.savefig(fig, dpi=300)
        plt.close(fig)

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render main dashboard header"""
    st.markdown("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px; margin-bottom: 20px; box-shadow: 0 8px 16px rgba(0,0,0,0.1);">
        <img src="https://www.shutterstock.com/image-vector/breast-cancer-information-logo-design-600nw-2310171931.jpg"
             style="width:80px; display:block; margin:0 auto 10px auto;" />
        <h1 style="margin:0; color:white;">🧬 Protein Pre-Cancer Prediction Using CNN</h1>
        <p style="color:#e0e7ff; margin:5px; font-size:1.1rem;">Advanced Protein Structure Cancer Analysis with Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with controls"""
    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Logo
        st.image("https://www.shutterstock.com/image-vector/breast-cancer-information-logo-design-600nw-2310171931.jpg",
                width=90)
        
        st.markdown("## ⚙️ Settings")
        
        # Visualization mode
        st.markdown("### Visualization Mode")
        view_mode = st.radio(
            "Select Mode",
            ("All Models (Comparison)", "Best Model (DenseNet)", 
             "EfficientNet Only", "SE-ResNet Only"),
            key="view_radio",
            label_visibility="collapsed"
        )
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        # File upload
        st.markdown("### 📁 Upload PDB File")
        uploaded_file = st.file_uploader(
            "Choose a PDB file",
            type=["pdb"],
            help="Upload a protein structure file in PDB format",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name}")
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        # Info
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Protein Cancer Prediction System**
            
            This tool uses ensemble deep learning to analyze protein structures
            and predict cancer associations.
            
            **Models:**
            - DenseNet201
            - EfficientNet-B4
            - SE-ResNet50
            
            **Features:**
            - 3D visualization
            - Grad-CAM interpretability
            - Comprehensive biological metrics
            - PDF report generation
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return view_mode, uploaded_file

def render_3d_viewer(pdb_content: str):
    """Render interactive 3D protein structure viewer"""
    st.markdown('<div class="custom-card"><h3>🧊 3D Structure Viewer</h3>', unsafe_allow_html=True)
    
    view = py3Dmol.view(width=500, height=400)
    view.addModel(pdb_content, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.zoomTo()
    view.spin("y")
    view.zoom(1.2)
    
    components.html(view._make_html(), height=400)
    st.markdown('</div>', unsafe_allow_html=True)

def render_biological_indicators(protein_data: dict):
    """Render biological indicators card"""
    st.markdown(f"""
    <div class="custom-card">
        <h3>📊 Biological Indicators</h3>
        <div style="display:grid; grid-template-columns: 1fr 1fr; gap:15px; margin-top:15px;">
            <div>
                <div class="stat-label">Protein Length</div>
                <div class="stat-value">{protein_data['protein_len']} aa</div>
            </div>
            <div>
                <div class="stat-label">Total Atoms</div>
                <div class="stat-value">{protein_data['total_atoms']}</div>
            </div>
            <div>
                <div class="stat-label">Alpha Helix</div>
                <div class="stat-value">{protein_data['alpha_len']} aa ({protein_data['alpha_pct']:.1f}%)</div>
            </div>
            <div>
                <div class="stat-label">Beta Sheet</div>
                <div class="stat-value">{protein_data['beta_len']} aa ({protein_data['beta_pct']:.1f}%)</div>
            </div>
            <div>
                <div class="stat-label">Coil / Unstructured</div>
                <div class="stat-value">{protein_data['coil_len']} aa ({protein_data['coil_pct']:.1f}%)</div>
            </div>
            <div>
                <div class="stat-label">Alpha / Beta Ratio</div>
                <div class="stat-value">{protein_data['alpha_beta_text']}</div>
            </div>
            <div>
                <div class="stat-label">Hydrophobic Residues</div>
                <div class="stat-value">{protein_data['hydrophobic_pct']:.1f}%</div>
            </div>
            <div>
                <div class="stat-label">Structural Class</div>
                <div class="stat-value">{protein_data['struct_class']}</div>
            </div>
            <div>
                <div class="stat-label">Avg B-factor</div>
                <div class="stat-value">{protein_data['avg_conf']:.2f}</div>
            </div>
            <div>
                <div class="stat-label">Structural Density</div>
                <div class="stat-value">{protein_data['density']:.3f}</div>
            </div>
            <div>
                <div class="stat-label">Instability Score</div>
                <div class="stat-value">{protein_data['instab_score']:.1f}/100</div>
            </div>
            <div>
                <div class="stat-label">Mutation Vulnerability</div>
                <div class="stat-value">{protein_data['mutation_vulnerability']:.1f}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_diagnosis_card(diagnosis: str, confidence: float):
    """Render AI diagnosis card"""
    color = config.CANCER_RED if "CANCER" in diagnosis else config.NONC_GREEN
    
    st.markdown(f"""
    <div class="custom-card" style="border-left: 5px solid {color};">
        <h3 style="margin-top:0;">🤖 AI Diagnosis</h3>
        <h1 style="color:{color}; margin:10px 0; font-size:1.8rem;">{diagnosis}</h1>
        <p style="font-size:1.1rem;"><strong>Ensemble Confidence:</strong> {confidence:.2%}</p>
        <p style="font-size:0.9rem; color:#64748b; margin-top:15px;">
            This prediction is based on ensemble analysis of multiple deep learning models
            trained on protein structural features.
        </p>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application logic"""
    
    # Render header
    render_header()
    
    # Render sidebar and get controls
    view_mode, uploaded_file = render_sidebar()
    
    # Load preprocessing module
    if not os.path.exists(config.PREPROCESS_PATH):
        st.error(f"❌ Preprocessing script not found: {config.PREPROCESS_PATH}")
        st.stop()
    
    try:
        spec = importlib.util.spec_from_file_location("prep", config.PREPROCESS_PATH)
        prep = importlib.util.module_from_spec(spec)
        sys.modules["prep"] = prep
        spec.loader.exec_module(prep)
    except Exception as e:
        st.error(f"❌ Failed to load preprocessing module: {str(e)}")
        st.stop()
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        model_manager = ModelManager(config)
        model_manager.load_all_models()
    
    # Main logic
    if not uploaded_file:
        st.info("👆 Please upload a PDB file to begin analysis")
        
        # Show example
        with st.expander("📖 How to use this tool"):
            st.markdown("""
            1. **Upload a PDB file** using the sidebar
            2. **Select visualization mode** to view different model analyses
            3. **Review the results** including:
               - 3D protein structure visualization
               - Biological and structural indicators
               - AI model predictions with interpretability
            4. **Generate PDF report** for comprehensive documentation
            
            **Supported formats:** PDB (Protein Data Bank)
            """)
        return
    
    # Process uploaded file
    pdb_path = "temp_input.pdb"
    with open(pdb_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        # Read PDB content
        with open(pdb_path, "r") as f:
            pdb_content = f.read()
        
        # Preprocessing
        with st.spinner("🔬 Analyzing protein structure..."):
            data = prep.extract_features(pdb_path)
            if data is None:
                st.error("❌ Invalid PDB file or preprocessing failed")
                return
            
            coords, b_factors = data
            img_bgr, dist_matrix = prep.create_rgb_image(coords, b_factors)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
        
        # Analyze protein structure
        analyzer = ProteinAnalyzer()
        
        protein_len = len(coords)
        
        # NOTE: Using the updated Biotite function if available
        alpha_len, beta_len = analyzer.compute_secondary_structure(pdb_content)
        total_residues, hydrophobic_pct = analyzer.compute_residue_properties(pdb_content)
        
        # Calculate percentages
        alpha_pct = safe_divide(alpha_len, protein_len) * 100
        beta_pct = safe_divide(beta_len, protein_len) * 100
        coil_len = max(0, protein_len - alpha_len - beta_len)
        coil_pct = safe_divide(coil_len, protein_len) * 100
        
        alpha_beta_ratio = safe_divide(alpha_len, beta_len)
        alpha_beta_text = f"{alpha_beta_ratio:.2f}" if beta_len > 0 else "N/A"
        
        # Calculate metrics
        avg_conf = float(np.mean(b_factors)) if len(b_factors) > 0 else 0.0
        density = float(np.mean(dist_matrix < config.DISTANCE_THRESHOLD)) if dist_matrix is not None else 0.0
        
        bf_arr = np.array(b_factors) if len(b_factors) > 0 else np.array([0.0])
        disorder_score = float((bf_arr > config.HIGH_BFACTOR_THRESHOLD).mean() * 100.0)
        
        instab_score = analyzer.calculate_instability_score(avg_conf, density)
        
        # Mutation vulnerability
        try:
            neighbour_counts = (dist_matrix < config.DISTANCE_THRESHOLD).sum(axis=1) - 1
            mutation_vulnerability = float((neighbour_counts < 10).mean() * 100.0)
        except Exception:
            mutation_vulnerability = 0.0
        
        struct_class = analyzer.classify_structure(alpha_pct, beta_pct, coil_pct)
        
        # Prepare protein data dict
        protein_data = {
            'protein_len': protein_len,
            'total_atoms': len(b_factors),
            'alpha_len': alpha_len,
            'alpha_pct': alpha_pct,
            'beta_len': beta_len,
            'beta_pct': beta_pct,
            'coil_len': coil_len,
            'coil_pct': coil_pct,
            'alpha_beta_text': alpha_beta_text,
            'hydrophobic_pct': hydrophobic_pct,
            'struct_class': struct_class,
            'avg_conf': avg_conf,
            'density': density,
            'instab_score': instab_score,
            'disorder_score': disorder_score,
            'mutation_vulnerability': mutation_vulnerability
        }
        
        # Prepare input tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config.IMG_NORM_MEAN, config.IMG_NORM_STD)
        ])
        input_tensor = transform(img_pil).unsqueeze(0).to(config.DEVICE)
        
        # Model predictions
        with st.spinner("🤖 Running AI inference..."):
            model_predictions = model_manager.predict(input_tensor)
        
        # Calculate ensemble prediction
        final_prob = np.mean(list(model_predictions.values()))
        diagnosis = "CANCER ASSOCIATED" if final_prob > 0.5 else "BENIGN / NORMAL"
        
        # ====================================================================
        # RENDER RESULTS
        # ====================================================================
        
        # Row 1: 3D Viewer | Biological Indicators | Diagnosis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            render_3d_viewer(pdb_content)
        
        with col2:
            render_biological_indicators(protein_data)
        
        with col3:
            render_diagnosis_card(diagnosis, final_prob)
            
        # Row 2: Model-wise predictions
        st.markdown("---")
        st.markdown("### 📈 Model-wise Cancer Association")
        
        # Styled table
        rows_html = ""
        for name, c_prob in model_predictions.items():
            rows_html += (
                f"<tr>"
                f"<td>{name}</td>"
                f"<td style='color:{config.CANCER_RED}'><strong>{c_prob*100:.2f}%</strong></td>"
                f"<td style='color:{config.NONC_GREEN}'><strong>{(1-c_prob)*100:.2f}%</strong></td>"
                f"</tr>"
            )
        
        table_html = f"""
        <table class="model-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Cancer Probability</th>
                    <th>Non-Cancer Probability</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
        
        # Pie charts
        st.markdown("#### Probability Distributions")
        all_models = list(model_predictions.items())
        pie_cols = st.columns(len(all_models) + 1)
        
        # Overall ensemble
        with pie_cols[0]:
            st.markdown("**Overall Ensemble**")
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(
                [final_prob, 1 - final_prob],
                labels=["Cancer", "Non-Cancer"],
                autopct="%1.1f%%",
                startangle=90,
                colors=[config.CANCER_RED, config.NONC_GREEN],
                wedgeprops={'edgecolor': 'white', 'linewidth': 2}
            )
            ax.axis("equal")
            st.pyplot(fig)
            plt.close(fig)
        
        # Per-model pies
        for i, (name, c_prob) in enumerate(all_models):
            with pie_cols[i + 1]:
                st.markdown(f"**{name}**")
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.pie(
                    [c_prob, 1 - c_prob],
                    labels=["Cancer", "Non-Cancer"],
                    autopct="%1.1f%%",
                    startangle=90,
                    colors=[config.CANCER_RED, config.NONC_GREEN],
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2}
                )
                ax.axis("equal")
                st.pyplot(fig)
                plt.close(fig)
        
        # Row 3: Grad-CAM visualizations
        st.markdown("---")
        st.markdown("### 🧠 AI Vision (Grad-CAM Analysis)")
        st.markdown("*Highlighted regions indicate areas the AI models focus on for cancer prediction*")
        
        # Determine which models to show
        if view_mode == "All Models (Comparison)":
            target_models = ["DenseNet201", "EfficientNetB4", "SE-ResNet50"]
        elif "DenseNet" in view_mode:
            target_models = ["DenseNet201"]
        elif "EfficientNet" in view_mode:
            target_models = ["EfficientNetB4"]
        elif "SE-ResNet" in view_mode:
            target_models = ["SE-ResNet50"]
        else:
            target_models = []
        
        # Generate heatmaps
        heatmaps = []
        with st.spinner("🔥 Generating neural heatmaps..."):
            for model_name in target_models:
                if model_name in model_manager.models:
                    target_layer = model_manager.get_target_layer(model_name)
                    if target_layer:
                        heatmap = GradCAMVisualizer.generate(
                            model_manager.models[model_name],
                            target_layer,
                            input_tensor,
                            img_pil
                        )
                        if heatmap:
                            heatmaps.append((model_name, img_to_base64(heatmap)))
        
        # Display heatmaps
        cols = st.columns(len(heatmaps) + 1)
        
        # Original input
        img_b64 = img_to_base64(img_pil)
        with cols[0]:
            st.markdown(f"""
            <div class="custom-card">
                <div class="heatmap-title">Biophysical Input</div>
                <div class="scan-container">
                    <div class="scan-line"></div>
                    <img src="data:image/png;base64,{img_b64}"
                         style="width:100%; max-height:300px; object-fit:contain;
                                border-radius:10px; display:block;">
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Heatmaps
        for idx, (name, b64) in enumerate(heatmaps):
            with cols[idx + 1]:
                st.markdown(f"""
                <div class="custom-card">
                    <div class="heatmap-title">{name} Attention</div>
                    <div class="scan-container">
                        <img src="data:image/png;base64,{b64}"
                             style="width:100%; max-height:300px; object-fit:contain;
                                    border-radius:10px; display:block;">
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ====================================================================
        # COLLAPSIBLE MODEL DETAILS SECTION
        # ====================================================================

        st.markdown("---")
        
        with st.expander("📚 View Model Performance & Architecture Details"):
            
            st.markdown("### 📊 Model Performance (Training Results)")

            # Place your model metrics here
            model_metrics = {
                "DenseNet201": {"accuracy": 0.9890, "precision": 0.9868, "recall": 0.8621, "f1": 0.},
                "EfficientNetB4": {"accuracy": 0.9858, "precision": 0.9577, "recall": 0.8448, "f1": 0.8977},
                "SE-ResNet50": {"accuracy": 0.9890, "precision": 0.9933, "recall": 0.8563, "f1": 0.9198},
            }

            # Display metrics
            cols = st.columns(3)
            for i, (name, m) in enumerate(model_metrics.items()):
                with cols[i]:
                    st.markdown(f"""
                    <div class="custom-card" style="text-align:center; background-color: #f8f9fa;">
                        <h4 style="margin-top:0;">{name}</h4>
                        <p style="font-size:1.5rem; margin-bottom:5px;"><b>Accuracy:</b> {m['accuracy']*100:.2f}%</p>
                        <p style="font-size:1.5rem; margin-bottom:5px;"><b>Precision:</b> {m['precision']*100:.2f}%</p>
                        <p style="font-size:1.5rem; margin-bottom:5px;"><b>Recall:</b> {m['recall']*100:.2f}%</p>
                        <p style="font-size:1.5rem; margin-bottom:5px;"><b>F1-Score:</b> {m['f1']*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 🔍 Model Weight & Parameter Summary")

            param_data = {
                "DenseNet201": {
                    "path": config.PATH_DENSE,
                    "params": 18094849
                },
                "EfficientNetB4": {
                    "path": config.PATH_EFF,
                    "params": 17550409
                },
                "SE-ResNet50": {
                    "path": config.PATH_SER,
                    "params": 26041073
                }
            }

            for model_name, info in param_data.items():
                params = info["params"]
                params_million = params / 1_000_000

                st.markdown(f"""
                <div style="background-color:white; padding:15px; border-radius:10px; border:1px solid #eee; margin-bottom:10px;">
                    <h4 style="margin:0;">🔍 {model_name}</h4>
                    <p style="margin:5px 0 0 0;">✅ <b>Weights:</b> {info["path"]}</p>
                    <p style="margin:0;">📊 <b>Parameters:</b> {params:,} ({params_million:.2f} M)</p>
                </div>
                """, unsafe_allow_html=True)

        # ====================================================================
        # PDF GENERATION LOGIC
        # ====================================================================
        st.markdown("---")
        st.markdown("### 📄 Comprehensive Report")
        
        col_a, col_b = st.columns([1, 2])
        with col_a:
            if st.button("📥 Generate PDF Report", use_container_width=True):
                with st.spinner("📝 Generating comprehensive PDF report..."):
                    # Use the ReportGenerator class
                    report_gen = ReportGenerator(config)
                    pdf_bytes = report_gen.generate(
                        protein_data, model_predictions, diagnosis, final_prob
                    )
                    
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"protein_diagnosis_{uploaded_file.name.split('.')[0]}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ Report generated successfully!")
        
        with col_b:
            st.info("💡 The PDF report includes all analysis results, biological metrics, and AI predictions in a professional format suitable for documentation.")

    except Exception as e:
        st.error(f"❌ An error occurred during analysis: {str(e)}")
        st.exception(e)
    
    finally:
        if os.path.exists(pdb_path):
            os.remove(pdb_path)

# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()