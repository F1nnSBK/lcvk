import os
import sys
import struct
import numpy as np

try:
    from PIL import Image
    import torch
    import torch.nn as nn
    from peft import LoraConfig, PeftModel
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from benchmark import PithosEngine

# Configuration
NUM_RECORDS = 1_000_000
DIMENSION = 384
DB_FILE = "temp/benchmark_data/lunar_real_data"
TIERS = np.array([64, 128, 256, 384], dtype=np.int32)

# Fixed seed for reproducibility
np.random.seed(42)

# Define DinoExtractor wrapper only if torch is available
if HAS_TORCH:
    class DinoExtractor(nn.Module):
        def __init__(self, weights_path="models/meta/dinov3_vits16_pretrain_lvd.pth", model_size="vits16",
                     lora_repo="F1nnSBK/lunar-dinov3-lora", use_adapter=False, device=None):
            super().__init__()
            self.model_size = model_size
            self.use_adapter = use_adapter
            if device is None:
                device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            print(f"Loading DINOv3 backbone: dinov3_{model_size} (local weights: {weights_path}) on {self.device}...")
            
            try:
                self.backbone = torch.hub.load("facebookresearch/dinov3", f"dinov3_{model_size}", pretrained=False)
                state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                self.backbone.load_state_dict(state_dict, strict=True)
                self.backbone.to(self.device)
                self.backbone.eval()
            except Exception as e:
                print(f"Error while loading base model: {e}")
                raise
            
            if use_adapter:
                print(f"[Mode B] Loading Lunar LoRA adapter from {lora_repo}...")
                lora_config = LoraConfig(
                    r=32,
                    lora_alpha=32,
                    target_modules=["qkv", "proj", "fc1", "fc2"],
                    lora_dropout=0.1,
                    bias="none"
                )
                try:
                    self.model = PeftModel.from_pretrained(
                        self.backbone,
                        lora_repo,
                        config=lora_config
                    )
                    self.model.to(self.device)
                    self.model.eval()
                except Exception as e:
                    print(f"Error while loading LoRA adapter: {e}")
                    raise
            else:
                print(f"[Mode A] Running naked DINOv3 backbone (no LoRA adapter). Optimal for MNIST system verification on {self.device}.")
                self.model = self.backbone
            
        def forward(self, x):
            return self.model(x.to(self.device))

    def get_feature_extractor(use_adapter=False):
        from torchvision import transforms
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        extractor = DinoExtractor(use_adapter=use_adapter, device=device)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        def extract(images):
            batch = torch.stack([transform(img.convert("RGB")) for img in images])
            with torch.no_grad():
                embeddings = extractor(batch)
            return embeddings.cpu().numpy()
            
        weights = np.eye(DIMENSION, dtype=np.float32)
        if use_adapter:
            try:
                w_qkv = extractor.model.base_model.model.blocks[0].attn.qkv.weight.detach().cpu().numpy()
                weights = w_qkv[:DIMENSION, :DIMENSION].astype(np.float32)
            except Exception:
                q, r = np.linalg.qr(np.random.normal(size=(DIMENSION, DIMENSION)))
                weights = q.astype(np.float32)
                
        return extract, weights

def main():
    use_torch = HAS_TORCH
    if use_torch:
        print("Loading real lunar dataset for Mode B verification...")
        try:
            import glob
            train_pits_path = "/Users/finnhertsch/projects/luna_hole/data/processed/dataset/train/pits/*.png"
            train_negs_path = "/Users/finnhertsch/projects/luna_hole/data/processed/dataset/train/negatives/*.png"
            
            pit_files = sorted(glob.glob(train_pits_path))
            neg_files = sorted(glob.glob(train_negs_path))
            
            print(f"Found {len(pit_files)} train pits and {len(neg_files)} train negatives.")
            
            images = []
            labels = []
            
            # Load pits (label = 1)
            for f in pit_files:
                images.append(Image.open(f))
                labels.append(1)
                
            # Load negatives (label = 0)
            for f in neg_files:
                images.append(Image.open(f))
                labels.append(0)
                
            labels = np.array(labels, dtype=np.int32)
        except Exception as e:
            print(f"Failed to load lunar dataset: {e}. Falling back to synthetic generators...")
            use_torch = False

    if not use_torch:
        print("Running in high-fidelity synthetic fallback mode (simulating Matryoshka-structured visual embeddings)...")
        # Generate synthetic 10,000 base vectors
        embeddings = np.random.normal(0.0, 1.0, size=(10000, DIMENSION)).astype(np.float32)
        labels = np.random.randint(0, 2, size=10000, dtype=np.int32)
        
        # Structure target class 1 to have high similarity (simulating cave morphologies)
        pits_mask = (labels == 1)
        embeddings[pits_mask, :64] += 0.85
        
        # Generate target orthogonal weights matrix W
        q, r = np.linalg.qr(np.random.normal(size=(DIMENSION, DIMENSION)))
        weights = q.astype(np.float32)
    else:
        # Extract embeddings via DINOv3 model with Lunar LoRA adapter (Mode B)
        extract_fn, weights = get_feature_extractor(use_adapter=True)
        print(f"Extracting features for {len(images)} unique images...")
        batch_size = 250
        embeddings_list = []
        for idx in range(0, len(images), batch_size):
            batch_imgs = images[idx : idx + batch_size]
            batch_embs = extract_fn(batch_imgs)
            embeddings_list.append(batch_embs)
        embeddings = np.vstack(embeddings_list)
        print(f"Features extracted. Shape: {embeddings.shape}")

    # L2-Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    
    # Replicate to 1,000,000 records
    print(f"Replicating to {NUM_RECORDS:,} records...")
    tile_indices = np.arange(NUM_RECORDS)
    source_indices = tile_indices % len(embeddings)
    
    db_vectors = embeddings[source_indices].astype(np.float32)
    db_labels = labels[source_indices]
    
    # Resolve native library path
    import platform
    if platform.system() == "Darwin":
        so_paths = [
            "./target/libpithos.dylib",
            "./build-output/libpithos.dylib",
            "./libpithos.dylib",
            "./target/libpithos.so",
            "./build-output/libpithos.so",
        ]
    else:
        so_paths = [
            "./build-output/libpithos.so",
            "./libpithos.so",
            "./target/libpithos.so",
        ]
    lib_path = None
    for p in so_paths:
        if os.path.exists(p):
            lib_path = p
            break
    if not lib_path:
        print("[Error] Pithos native library not found.", file=sys.stderr)
        sys.exit(1)
        
    engine = PithosEngine(lib_path)
    
    # Compile raw float vectors into Pithos database files
    print(f"Compiling Pithos multi-tier database files for {NUM_RECORDS:,} records...")
    ids = np.arange(NUM_RECORDS, dtype=np.int64)
    status = engine.compile_index_file(DB_FILE, 1, 1737400, DIMENSION, TIERS, ids, db_vectors)
    engine.close()
    
    if status != 0:
        print(f"[Error] Compilation failed with code: {status}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Database successfully generated under base name: {DB_FILE}")
    
    # Save target metadata and weights for verification phase
    np.save("temp/benchmark_data/db_labels.npy", db_labels)
    np.save("temp/benchmark_data/weights.npy", weights)
    # np.save("temp/benchmark_data/db_vectors.npy", db_vectors)
    
    if use_torch:
        # Save the raw embeddings of test pits for query generation
        test_pits_path = "/Users/finnhertsch/projects/luna_hole/data/processed/dataset/test/pits/*.png"
        test_pit_files = sorted(glob.glob(test_pits_path))
        test_pit_images = [Image.open(f) for f in test_pit_files]
        print(f"Extracting features for {len(test_pit_images)} test pits for queries...")
        test_pit_embeddings = extract_fn(test_pit_images)
        q_norms = np.linalg.norm(test_pit_embeddings, axis=1, keepdims=True)
        q_norms[q_norms == 0] = 1.0
        test_pit_embeddings = test_pit_embeddings / q_norms
        np.save("temp/benchmark_data/raw_pits.npy", test_pit_embeddings)
    else:
        # Fallback raw pits
        np.save("temp/benchmark_data/raw_pits.npy", embeddings[labels == 1])
    
    # Save first 10,000 raw float vectors for distance verification
    np.save("temp/benchmark_data/db_vectors_subset.npy", db_vectors[:10000])

if __name__ == "__main__":
    main()
