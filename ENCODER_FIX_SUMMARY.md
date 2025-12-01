# QTMRL Encoder Fix - Window Size Issue

## Problem
The training validation was failing with the error:
```
RuntimeError: Calculated padded input size per channel: (0). Kernel size: (1). 
Kernel size can't be greater than actual input size
```

This occurred because the `TimeCNNEncoder` was using a fixed `kernel_size=3` for 1D convolutions, but during quick validation with minimal data, the window size (W) could be very small (e.g., 1 or 2 timesteps), making the kernel size larger than the input sequence length.

## Root Cause
1. The validation script creates minimal training data for fast testing
2. After data preprocessing and windowing, the actual window dimension becomes very small
3. The Conv1d layer with `kernel_size=3` fails when input width < 3
4. Even with dynamic kernel size adjustment to 1, padding calculations could cause issues

## Solution
Modified `qtmrl/models/encoders.py` - `TimeCNNEncoder` class:

### Key Insight
The problem wasn't just the kernel size - it was that **multiple stacked Conv1d layers** can progressively shrink the sequence length when using integer padding values. With `kernel_size=1` and `padding=0`, each layer maintains length, but with `kernel_size=2` and `padding=1`, the sequence can shrink through multiple layers, eventually reaching 0.

### Changes Made:

1. **Added `kernel_size` parameter** to `__init__`:
   - Default value: 3
   - Stored as `self.base_kernel_size` for dynamic adjustment
   - Removed static conv layer creation in `__init__`

2. **Special handling for very small windows (W < 3)**:
   - Uses simple linear projection instead of convolution
   - Averages features over the time dimension
   - Avoids all convolution-related edge cases completely

3. **Dynamic layer creation with 'same' padding**:
   - Layers are created on-the-fly based on actual input window size
   - Kernel size is adjusted: `kernel_size = min(self.base_kernel_size, W)`
   - **Critical fix**: Use `padding='same'` mode for `kernel_size > 1`
     - This ensures output length = input length through ALL conv layers
     - Prevents sequence shrinkage in stacked convolutions
   - For `kernel_size=1`: use `padding=0` (equivalent to 'same')

4. **Device handling and error messages**:
   - Ensures dynamically created layers are on the same device as input
   - Added try-catch with detailed error messages for debugging

### Code Structure:
```python
def forward(self, features, positions, cash):
    B, W, N, n_feat = features.shape
    
    # Special case: very small windows (W < 3)
    if W < 3:
        # Use linear projection with temporal averaging
        if not hasattr(self, 'linear_proj'):
            self.linear_proj = nn.Linear(self.n_features, self.d_model).to(features.device)
        
        asset_feat_avg = features[:, :, i, :].mean(dim=1)  # Average over time
        encoded = F.relu(self.linear_proj(asset_feat_avg))
        # ... return encodings
    
    # Dynamic kernel size adjustment
    kernel_size = min(self.base_kernel_size, W)
    
    # Use 'same' padding to prevent sequence shrinkage
    padding = 0 if kernel_size == 1 else 'same'
    
    # Build conv layers if needed (cached for efficiency)
    if self.conv_layers is None or self._last_kernel_size != kernel_size:
        layers = []
        # First layer
        layers.append(nn.Conv1d(n_features, d_model, kernel_size, padding=padding))
        layers.append(nn.ReLU())
        # Additional layers (all use same padding)
        for _ in range(n_layers - 1):
            layers.append(nn.Conv1d(d_model, d_model, kernel_size, padding=padding))
            layers.append(nn.ReLU())
        
        self.conv_layers = nn.Sequential(*layers).to(features.device)
        self._last_kernel_size = kernel_size
    
    # ... process with conv layers
    return encodings
```

## Benefits
1. **Robust to variable window sizes**: Works with W=1, 2, 3, ..., any size
2. **Efficient**: Layers are cached and only rebuilt when window size changes
3. **Maintains model quality**: Uses full convolutions when possible, falls back gracefully
4. **Device-aware**: Properly handles GPU/CPU placement
5. **Backward compatible**: Existing code with normal window sizes (10-50) works unchanged

## Testing
The fix should allow the quick validation script to pass the training pipeline test:
```bash
python scripts/quick_validation.py
```

Expected behavior:
- ✓ Step 1: Import Validation - PASS
- ✓ Step 2: Data Preprocessing - PASS  
- ✓ Step 3: Training Pipeline - PASS (previously failing)
- ✓ Step 4: Evaluation Pipeline - PASS

## Next Steps
After validation passes:
1. Run full preprocessing: `python scripts/preprocess.py --config configs/quick_test.yaml`
2. Run training: `python scripts/train.py --config configs/quick_test.yaml`
3. Run evaluation: `python scripts/evaluate.py --config configs/quick_test.yaml`
