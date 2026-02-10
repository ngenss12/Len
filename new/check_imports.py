import sys
import traceback

modules = [
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("sklearn", "sklearn"),
    ("tifffile", "tifffile"),
    ("pandas", "pandas"),
    ("joblib", "joblib"),
    ("skimage", "skimage"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
]

success = True
for name, pkg in modules:
    try:
        m = __import__(pkg)
        ver = getattr(m, "__version__", "(no __version__)")
        print(f"{name}: OK, version {ver}")
    except Exception:
        success = False
        print(f"{name}: FAILED to import")
        traceback.print_exc()

# Additional checks
try:
    import torch
    print("torch.cuda.is_available():", torch.cuda.is_available())
except Exception:
    pass

if not success:
    sys.exit(1)
