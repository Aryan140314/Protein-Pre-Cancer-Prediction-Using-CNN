import os
import csv
import json
import math
import numpy as np
import cv2
from Bio.PDB import PDBParser, MMCIFParser
from tqdm import tqdm
import warnings

# ---------------- CONFIG ----------------
# Update these paths if needed
RAW_PDB_DIR    = r"D:\Protein_Cancer\Protein_Project\raw_pdb"
OUTPUT_IMG_DIR = r"D:\Protein_Cancer\Protein_Project\dataset_images"
BIO_FEAT_PATH  = r"D:\Protein_Cancer\Protein_Project\bio_features.npy"
META_JSON_PATH = r"D:\Protein_Cancer\Protein_Project\metadata.json"
CANCER_ID_CSV  = r"D:\Protein_Cancer\Protein_Project\final_mapping.csv" # List of Cancer IDs

IMG_SIZE        = 299        
MAX_DIST        = 40.0       # for distance normalization
CONTACT_THRESH  = 8.0        # for contact map / density
MIN_RESIDUES    = 10         # skip tiny fragments
# ----------------------------------------

os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)
warnings.simplefilter('ignore')

# ---------- 1. ROBUST ID LOADING ----------
def load_cancer_ids(csv_path):
    """Loads the set of Cancer IDs from your existing file."""
    cancer_ids = set()
    try:
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            # Handles 'ID' or 'Entry' or just the second column
            field = "ID" if "ID" in reader.fieldnames else reader.fieldnames[1]
            for row in reader:
                if row[field]:
                    cancer_ids.add(row[field].strip())
        print(f"✅ Loaded {len(cancer_ids)} Cancer IDs.")
    except Exception as e:
        print(f"⚠️ Error loading CSV: {e}. Using filename matching only.")
    return cancer_ids

# ---------- 2. STRUCTURE EXTRACTION ----------
def extract_features(pdb_path):
    """Safely extracts CA coordinates, B-factors, and residues."""
    # Smart parser selection
    parser = PDBParser(QUIET=True) if pdb_path.endswith(".pdb") else MMCIFParser(QUIET=True)
    
    try:
        structure = parser.get_structure("prot", pdb_path)
        model = structure[0]
        coords = []
        b_factors = []
        
        # Iterate atoms directly (Safest for AlphaFold)
        for atom in model.get_atoms():
            if atom.get_name() == "CA":
                coords.append(atom.get_coord())
                b_factors.append(atom.get_bfactor())

        if len(coords) < MIN_RESIDUES: return None
        
        return np.array(coords, dtype=np.float32), np.array(b_factors, dtype=np.float32)
    except:
        return None

# ---------- 3. IMAGE GENERATION (Hybrid-Plus) ----------
def smart_resize(img):
    """Resizes with padding (No Distortion) - Your best feature!"""
    h, w = img.shape
    size = max(h, w)
    pad_img = np.zeros((size, size), dtype=np.uint8)
    
    # Center
    oh = (size - h) // 2
    ow = (size - w) // 2
    pad_img[oh:oh+h, ow:ow+w] = img
    
    return cv2.resize(pad_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

def create_rgb_image(coords, b_factors):
    # --- Channel 1 (RED): Distance Map ---
    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    dist_norm = np.clip(dist, 0, MAX_DIST) / MAX_DIST
    ch_red = ((1.0 - dist_norm) * 255).astype(np.uint8)

    # --- Channel 2 (GREEN): B-Factor (Confidence) ---
    # High Confidence = Bright Green
    b_norm = np.clip(b_factors / 100.0, 0, 1)
    ch_green = (np.outer(b_norm, b_norm) * 255).astype(np.uint8)

    # --- Channel 3 (BLUE): Z-Depth (3D Orientation) ---
    # Captures depth information
    dz = diff[:, :, 2]
    ch_blue = cv2.normalize(dz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Smart Resize & Merge
    img = cv2.merge([
        smart_resize(ch_blue),
        smart_resize(ch_green),
        smart_resize(ch_red)
    ])
    return img, dist

# ---------- 4. BIO FEATURES (Optional but good) ----------
def compute_bio_vector(coords, b_factors, dist_matrix):
    """Calculates 1D vector of structural properties"""
    L = len(coords)
    density = np.mean(dist_matrix < CONTACT_THRESH)
    instability = np.std(dist_matrix)
    mean_plddt = np.mean(b_factors)
    
    # Simple vector: [Length, Density, Instability, Confidence]
    return np.array([L, density, instability, mean_plddt], dtype=np.float32)
def convert_single_pdb_to_image(pdb_path):
    data = extract_features(pdb_path)
    if data is None:
        return None
    
    coords, b_factors = data
    img, _ = create_rgb_image(coords, b_factors)

    from PIL import Image
    return Image.fromarray(img)
# ---------- MAIN PIPELINE ----------
def main():
    if not os.path.exists(RAW_PDB_DIR):
        print(f"❌ Error: Input folder '{RAW_PDB_DIR}' not found.")
        return

    cancer_ids = load_cancer_ids(CANCER_ID_CSV)
    
    metadata = {}
    bio_features = []
    
    files = [f for f in os.listdir(RAW_PDB_DIR) if f.endswith((".pdb", ".cif"))]
    print(f"🚀 Processing {len(files)} files...")
    
    processed_count = 0
    
    for fname in tqdm(files):
        pdb_path = os.path.join(RAW_PDB_DIR, fname)
        out_name = fname.replace(".pdb", ".png").replace(".cif", ".png")
        out_path = os.path.join(OUTPUT_IMG_DIR, out_name)

        # 1. Determine Label
        # Check if ID is in our cancer list
        is_cancer = any(cid in fname for cid in cancer_ids)
        label = 1 if is_cancer else 0 # 1=Cancer, 0=Normal
        
        # 2. Process
        if os.path.exists(out_path): continue
        
        data = extract_features(pdb_path)
        if data is None: continue
        coords, b_factors = data
        
        # Create Image
        img, dist_matrix = create_rgb_image(coords, b_factors)
        cv2.imwrite(out_path, img)
        
        # Create Features
        bio_vec = compute_bio_vector(coords, b_factors, dist_matrix)
        bio_features.append(bio_vec)
        
        metadata[out_name] = {
            "label": label,
            "source": fname,
            "bio_idx": len(bio_features) - 1
        }
        processed_count += 1

    # Save Metadata
    if bio_features:
        np.save(BIO_FEAT_PATH, np.vstack(bio_features))
    
    with open(META_JSON_PATH, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n✨ DONE!")
    print(f"🖼️ Images saved: {processed_count}")
    print(f"📂 Folder: {OUTPUT_IMG_DIR}")

if __name__ == "__main__":
    main()