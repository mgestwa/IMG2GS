import os
import subprocess
import torch
import sys

def run_cmd(cmd):
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
        return out.decode('utf-8', errors='ignore').strip()
    except subprocess.CalledProcessError as e:
        return f"ERROR: {e.output.decode('utf-8', errors='ignore').strip()}"
    except Exception as e:
        return str(e)

print("-" * 50)
print("ENVIRONMENT DIAGNOSTIC")
print("-" * 50)

print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA: {torch.version.cuda}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

print("-" * 20)
print("CUDA_HOME:", os.environ.get("CUDA_HOME", "Not Set"))
print("CUDA_PATH:", os.environ.get("CUDA_PATH", "Not Set"))

print("-" * 20)
print("NVCC Version:")
print(run_cmd("nvcc --version"))

print("-" * 20)
print("CL (MSVC) Version:")
print(run_cmd("cl"))

print("-" * 20)
print("Ninja Version:")
print(run_cmd("ninja --version"))

print("-" * 50)
