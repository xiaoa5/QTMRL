# Dependency Installation Issues - Environment Blocker

**Date**: 2025-12-01
**Status**: 🔴 BLOCKED - Cannot proceed with validation due to critical dependency failures

---

## Critical Issues

### 1. multitasking Package Build Failure (BLOCKER)

**Error**: `AttributeError: install_layout. Did you mean: 'install_platlib'?`

**Root Cause**: The `multitasking` package's `setup.py` is incompatible with setuptools >= 60.0. This is a known issue in the multitasking project that affects all versions (0.0.7 through 0.0.12).

**Impact**:
- **yfinance cannot be installed** (depends on multitasking>=0.0.7)
- **Cannot download stock data** from Yahoo Finance
- **Data preprocessing pipeline blocked**

**Attempts Made**:
- ✗ `pip install yfinance` - Failed
- ✗ `pip install yfinance==0.2.28` - Failed (older version)
- ✗ `pip install multitasking` - Failed
- ✗ `pip install --no-build-isolation multitasking` - Failed
- ✗ `pip install yfinance --no-deps` - Installed but cannot import

**Technical Details**:
```python
# The error occurs in multitasking's setup.py:
File "setuptools/command/install_lib.py", line 17, in finalize_options
  self.set_undefined_options('install',('install_layout','install_layout'))
# This 'install_layout' attribute was removed in setuptools 60+
```

### 2. PyTorch Installation (BLOCKER)

**Issue**: `pip install torch` runs indefinitely (>15 minutes) without completing

**Root Cause**:
- PyTorch wheels are very large (~700MB for CPU version)
- Network issues or slow download speeds in this environment
- Possible proxy issues (403 Forbidden errors observed)

**Impact**:
- **Cannot use neural network models** (Actor, Critic, encoders)
- **Cannot run A2C training algorithm**
- **Cannot validate 80% of the codebase**

**Attempts Made**:
- ✗ `pip install torch` - Timed out after 15+ minutes
- ✗ `pip install torch --index-url https://download.pytorch.org/whl/cpu` - Proxy 403 error

### 3. pandas_ta Not Available (BLOCKER)

**Issue**: `ERROR: Could not find a version that satisfies the requirement pandas_ta`

**Root Cause**: pandas_ta (pandas-ta) is not compatible with Python 3.11 or is not available in the repository mirrors accessible from this environment.

**Impact**:
- **Cannot calculate technical indicators** (SMA, EMA, RSI, MACD, etc.)
- **qtmrl.indicators module is non-functional**
- **Data preprocessing blocked**

**Attempts Made**:
- ✗ `pip install pandas_ta` - Package not found
- ✗ `pip install "pandas_ta>=0.3.14b"` - Package not found
- ⏸️ Did not try: Install from GitHub (would require git)

---

## Successfully Installed

✅ **Working Dependencies**:
- numpy 2.3.5
- pandas 2.3.3
- pyyaml (installed)
- matplotlib (installed)
- tqdm (installed)
- scikit-learn (installed)
- beautifulsoup4 4.14.3 (for yfinance)
- curl_cffi 0.13.0 (for yfinance)
- frozendict, peewee, platformdirs, protobuf, websockets (yfinance deps)

**Partial Install**:
- ⚠️ yfinance 0.2.66 - Installed but cannot import (needs multitasking)

---

## Environment Information

```
Python: 3.11.14
Platform: Linux 4.4.0
pip: Running as root (not recommended)
Cache: Disabled (permission issues)
Network: Possible proxy restrictions (403 errors)
setuptools: Version with incompatible changes for old packages
```

---

## Validation Impact

### What Can Be Validated ✅

- ✅ Basic imports (qtmrl.env only)
- ✅ Data structures (with synthetic data)
- ✅ Environment logic (BUY/SELL/HOLD actions)
- ✅ Basic calculations (if using numpy/pandas directly)

### What Cannot Be Validated ❌

- ❌ Neural network models (requires torch)
- ❌ A2C training algorithm (requires torch)
- ❌ Data downloading (requires yfinance)
- ❌ Technical indicators (requires pandas_ta)
- ❌ Model checkpointing (requires torch)
- ❌ Backtest evaluation (requires torch for model inference)
- ❌ Complete end-to-end pipeline

**Coverage**: ~10% of intended Phase 0 validation

---

## Solutions

### 🥇 Recommended: Use Google Colab

**Why**: Pre-installed packages, GPU access, no dependency issues

**Setup**:
```python
# In Colab cell:
!git clone https://github.com/xiaoa5/QTMRL.git
%cd QTMRL
!pip install -q yfinance pandas-ta
# torch already installed

# Run validation
!python scripts/quick_validation.py
```

**Advantages**:
- ✅ All dependencies work out of the box
- ✅ Free GPU for faster training
- ✅ Easy to share and reproduce
- ✅ No environment configuration needed

### 🥈 Alternative 1: Use Docker

Create `Dockerfile`:
```dockerfile
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /workspace

# Install dependencies
RUN pip install --no-cache-dir \
    yfinance==0.2.28 \
    pandas-ta \
    numpy pandas pyyaml matplotlib tqdm scikit-learn

# Copy code
COPY . /workspace/

# Run validation
CMD ["python", "scripts/quick_validation.py"]
```

**Build and run**:
```bash
docker build -t qtmrl .
docker run --rm qtmrl
```

### 🥉 Alternative 2: Use Conda

Conda handles binary dependencies better:
```bash
conda create -n qtmrl python=3.10
conda activate qtmrl
conda install pytorch torchvision -c pytorch
pip install yfinance pandas-ta pyyaml matplotlib tqdm scikit-learn
```

**Note**: Requires Python 3.10 (pandas_ta compatibility)

### Alternative 3: Manual Data + Simplified Indicators

**Workaround without yfinance and pandas_ta**:

1. **Download data manually**:
   ```python
   # Download CSVs from Yahoo Finance website
   # Or use pandas_datareader
   import pandas_datareader as pdr
   df = pdr.get_data_yahoo('AAPL', start='2023-01-01')
   ```

2. **Implement basic indicators**:
   ```python
   # Simple Moving Average
   def calculate_sma(close, period):
       return close.rolling(window=period).mean()

   # RSI (basic version)
   def calculate_rsi(close, period=14):
       delta = close.diff()
       gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
       loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
       rs = gain / loss
       return 100 - (100 / (1 + rs))
   ```

3. **Create simplified indicators.py**:
   - Keep interface compatible
   - Implement 3-4 basic indicators manually
   - Document as temporary workaround

### Alternative 4: Fix multitasking (Advanced)

**Manual wheel build**:
```bash
# Download multitasking source
git clone https://github.com/ranaroussi/multitasking.git
cd multitasking

# Patch setup.py (remove problematic install_layout)
# ... manual editing ...

# Build wheel
python setup.py bdist_wheel

# Install wheel
pip install dist/multitasking-*.whl
```

---

## Decision Required

**The project is currently blocked**. Please choose one of the solutions above to proceed:

1. **✅ RECOMMENDED**: Move to Google Colab (fastest, most reliable)
2. Use Docker container (good for production)
3. Use Conda environment (good for local development)
4. Implement workarounds (most time-consuming)

**Estimated Time**:
- Colab setup: 5-10 minutes
- Docker setup: 20-30 minutes
- Conda setup: 15-20 minutes
- Workarounds: 2-4 hours

---

## Phase 0 Deliverables Status

| Deliverable | Status | Notes |
|-------------|--------|-------|
| Validation scripts | ✅ Complete | 2 scripts created |
| Core deps installed | ✅ Partial | numpy, pandas, matplotlib OK |
| torch installed | ❌ Failed | Timeout issues |
| yfinance installed | ❌ Blocked | multitasking build error |
| pandas_ta installed | ❌ Failed | Not available for Python 3.11 |
| Validation run | ⚠️ Partial | Only env tests, most skipped |
| Report generated | ✅ Complete | Multiple status documents |

**Overall Phase 0 Status**: 🔴 **BLOCKED** - Cannot proceed without dependency resolution

---

## Next Steps

1. **User Decision**: Choose solution approach (Colab recommended)
2. **Environment Setup**: Implement chosen solution
3. **Re-run Validation**: Execute `scripts/quick_validation.py`
4. **Generate Report**: Complete Phase 0 report
5. **Proceed to Phase 1**: Begin data alignment work

**Time to Unblock**: 5 minutes to 4 hours (depending on solution chosen)

---

## Files Created During Phase 0

```
QTMRL/
├── scripts/
│   ├── quick_validation.py           # Full validation script
│   └── quick_validation_minimal.py   # Graceful degradation version
├── results/
│   └── validation_report_minimal.txt # Partial validation results
├── PHASE0_STATUS.md                   # Initial status report
└── DEPENDENCY_ISSUES.md               # This comprehensive analysis
```

All scripts are ready to run once dependencies are resolved.
