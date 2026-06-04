import os
import sys
import struct
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel

# Configuration
NUM_RECORDS = 1_000_000
DIMENSION = 384
BYTES_PER_RECORD = 64
DB_FILE = "lunar_real_data.bin"

# Fixed seed for reproducibility
np.random.seed(42)

# Generate sign matrix D and H384
D = np.random.choice([-1, 1], size=DIMENSION).astype(np.float32)

def get_hadamard_384():
    def silvester_hadamard(n):
        if n == 1:
            return np.array([[1]])
        H_prev = silvester_hadamard(n // 2)
        return np.block([[H_prev, H_prev], [H_prev, -H_prev]])
        
    H32 = silvester_hadamard(32)
    H12 = np.array([
        [1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
        [1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1],
        [1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1,  1],
        [1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1, -1],
        [1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1, -1],
        [1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1, -1],
        [1, -1, -1, -1,  1, -1, -1,  1, -1,  1,  1,  1],
        [1,  1, -1, -1, -1,  1, -1, -1,  1, -1,  1,  1],
        [1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1,  1],
        [1,  1,  1,  1, -1, -1, -1,  1, -1, -1,  1, -1],
        [1, -1,  1,  1,  1, -1, -1, -1,  1, -1, -1,  1],
        [1,  1, -1,  1,  1,  1, -1, -1, -1,  1, -1, -1]
    ])
    return np.kron(H12, H32).astype(np.float32)

H384 = get_hadamard_384()

class DinoExtractor(nn.Module):
    """
    Dual-mode DINOv3 feature extractor.

    use_adapter=False  -> Naked DINOv3 backbone (System Verification / MNIST baseline).
                          Provides maximum class separation for well-labeled test data.
    use_adapter=True   -> DINOv3 + Lunar LoRA adapter (Production / Planetary Discovery).
                          Optimized to detect lunar pit/cave morphology in real NAC tile data.
    """
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
    import torch
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
        
    return extract

def precondition_and_quantize(embeddings):
    # L2-normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings_norm = embeddings / norms
    
    # Apply diagonal sign preconditioning
    embeddings_sign = embeddings_norm * D
    
    # Apply Hadamard transformation H384 / sqrt(384)
    embeddings_precond = np.dot(embeddings_sign, H384.T) / np.sqrt(DIMENSION)
    
    # 1-bit quantization (val >= 0 -> 1, val < 0 -> 0)
    bits = (embeddings_precond >= 0).astype(np.uint8)
    return bits

def pack_bits(bits_array):
    # bits_array shape: (N, 384)
    # Output byte array: (N, 48)
    n = bits_array.shape[0]
    packed = np.zeros((n, 48), dtype=np.uint8)
    for i in range(48):
        byte_slice = bits_array[:, i*8 : (i+1)*8]
        val = np.zeros(n, dtype=np.uint8)
        for b in range(8):
            val = (val << 1) | byte_slice[:, b]
        packed[:, i] = val
    return packed

def main():
    # 1. Load dataset (MNIST via torchvision)
    print("Loading MNIST dataset for real-data verification...")
    try:
        from torchvision import datasets
        train_dataset = datasets.MNIST(root='./data', train=True, download=True)
        images = [train_dataset[i][0] for i in range(10000)]
        labels = np.array([train_dataset[i][1] for i in range(10000)], dtype=np.int32)
    except Exception as e:
        print(f"Failed to load MNIST via torchvision: {e}. Generating synthetic digits...")
        # Create synthetic images if offline/failed
        images = []
        labels = []
        for i in range(10000):
            # Create a 28x28 mock digit image (circle/lines based on index)
            img = Image.new("L", (28, 28), 0)
            digit = i % 10
            # Draw something simple depending on digit
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            if digit == 7:
                draw.line([(5, 5), (23, 5)], fill=255, width=3)
                draw.line([(23, 5), (10, 23)], fill=255, width=3)
            elif digit == 1:
                draw.line([(14, 5), (14, 23)], fill=255, width=3)
            else:
                draw.ellipse([(5, 5), (23, 23)], outline=255, width=2)
            images.append(img)
            labels.append(digit)
        labels = np.array(labels, dtype=np.int32)

    # 2. Extract embeddings
    # Mode A: use_adapter=False -> naked DINOv3 backbone for maximum MNIST class separation.
    # Mode B: use_adapter=True  -> DINOv3 + Lunar LoRA for real NAC lunar tile ingestion.
    extract_fn = get_feature_extractor(use_adapter=False)
    print("Extracting features for 10,000 unique images (Mode A: DINOv3 backbone, no LoRA)...")
    
    # Process in batches to save memory
    batch_size = 1000
    embeddings_list = []
    for idx in range(0, len(images), batch_size):
        batch_imgs = images[idx : idx + batch_size]
        batch_embs = extract_fn(batch_imgs)
        embeddings_list.append(batch_embs)
    embeddings = np.vstack(embeddings_list)
    print(f"Features extracted. Shape: {embeddings.shape}")
    
    # 3. Apply PolarQuant-Hadamard preconditioning & quantization
    print("Applying PolarQuant-Hadamard preconditioning & 1-bit quantization...")
    bits = precondition_and_quantize(embeddings)
    packed_vectors = pack_bits(bits)
    
    # 4. Replicate to 1,000,000 records
    print(f"Replicating to {NUM_RECORDS:,} records...")
    tile_indices = np.arange(NUM_RECORDS)
    source_indices = tile_indices % 10000
    
    # Sequential Z-order Z-IDs and packed vectors
    db_packed_vectors = packed_vectors[source_indices]
    db_labels = labels[source_indices]
    
    # 5. Write the PLAN binary file
    print(f"Writing binary index to {DB_FILE}...")
    magic = b"PLAN"
    planet_id = 1  # Moon
    header_data = struct.pack("<BQQ", planet_id, NUM_RECORDS, 1737400)
    padding = b"\x00" * 43
    header = magic + header_data + padding
    assert len(header) == 64
    
    # Build 64-byte records
    records = np.zeros((NUM_RECORDS, BYTES_PER_RECORD), dtype=np.uint8)
    
    # Bytes 0-7: Sequential tileId (long)
    ids = np.arange(NUM_RECORDS, dtype=np.uint64)
    records[:, 0:8] = ids.view(np.uint8).reshape(-1, 8)
    
    # Bytes 8-55: Packed vector (48 bytes)
    records[:, 8:56] = db_packed_vectors
    
    # Bytes 56-63: Metadata padding (8 bytes) containing the label (for easy verification)
    metadata = db_labels.astype(np.uint64)
    records[:, 56:64] = metadata.view(np.uint8).reshape(-1, 8)
    
    with open(DB_FILE, "wb") as f:
        f.write(header)
        f.write(records.tobytes())
        
    print(f"Database successfully generated: {DB_FILE}")
    
    # Save target metadata for the verification phase
    np.save("db_labels.npy", db_labels)
    # Save the raw embeddings subset of '7's for query generation
    sevens_indices = np.where(labels == 7)[0]
    np.save("raw_sevens.npy", embeddings[sevens_indices])

if __name__ == "__main__":
    main()
