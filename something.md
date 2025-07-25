

base) cevheri@ubuntu:~$ sudo docker commit 005146d10083 my_saved_image:latest
sha256:09453d4403e6b72280c8a8af548535b295833d8e2dcfc631ab0bce66755e4cd0




torch.cuda.is_available(): True
torch.cuda.device_count(): 1
os.environ['CUDA_VISIBLE_DEVICES']: cuda

2025-07-25 07:26:52,556 - INFO - Cleaning up...
2025-07-25 07:26:52,566 - INFO - System cleanup complete
root@005146d10083:/workspace# python3 cevheri.py
Using device: cuda
YOLOv8n summary (fused): 168 layers, 3151904 parameters, 0 gradients, 8.7 GFLOPs
WARNING ⚠️ gun_detecktor.pt appears to require 'ultralytics.nn.modules.conv', which is not in ultralytics requirements.
AutoInstall will run now for 'ultralytics.nn.modules.conv' but this feature will be removed in the future.
Recommend fixes are to train a new model using the latest 'ultralytics' package or to run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'
requirements: YOLOv8 requirement "ultralytics.nn.modules.conv" not found, attempting AutoUpdate...
ERROR: Could not find a version that satisfies the requirement ultralytics.nn.modules.conv (from versions: none)
ERROR: No matching distribution found for ultralytics.nn.modules.conv
requirements: ❌ Command 'pip install "ultralytics.nn.modules.conv"  ' returned non-zero exit status 1.
Failed to load models: No module named 'ultralytics.nn.modules.conv'; 'ultralytics.nn.modules' is not a package
System failed: No module named 'ultralytics.nn.modules.conv'; 'ultralytics.nn.modules' is not a package
root@005146d10083:/workspace# pip uninstall ultralytics
Found existing installation: ultralytics 8.0.50
Uninstalling ultralytics-8.0.50:
  Would remove:
    /usr/local/bin/ultralytics
    /usr/local/bin/yolo
    /usr/local/lib/python3.8/dist-packages/ultralytics-8.0.50.dist-info/*
    /usr/local/lib/python3.8/dist-packages/ultralytics/*
Proceed (y/n)? y
Your response ('pip install ultralytics==8.0.196y') was not one of the expected responses: y, n
Proceed (y/n)? y
  Successfully uninstalled ultralytics-8.0.50
root@005146d10083:/workspace# pip install ultralytics==8.0.196
Collecting ultralytics==8.0.196
  Downloading ultralytics-8.0.196-py3-none-any.whl (631 kB)
     |████████████████████████████████| 631 kB 1.6 MB/s 
Requirement already satisfied: matplotlib>=3.3.0 in /usr/local/lib/python3.8/dist-packages (from ultralytics==8.0.196) (3.7.5)
Requirement already satisfied: numpy>=1.22.2 in /usr/local/lib/python3.8/dist-packages (from ultralytics==8.0.196) (1.24.4)
Collecting opencv-python>=4.6.0
  Using cached opencv_python-4.12.0.88-cp37-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl (45.9 MB)
Requirement already satisfied: pillow>=7.1.2 in /usr/local/lib/python3.8/dist-packages/Pillow-9.4.0-py3.8-linux-aarch64.egg (from ultralytics==8.0.196) (9.4.0)
Requirement already satisfied: pyyaml>=5.3.1 in /usr/lib/python3/dist-packages (from ultralytics==8.0.196) (5.3.1)
Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.8/dist-packages (from ultralytics==8.0.196) (2.28.2)
Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.8/dist-packages (from ultralytics==8.0.196) (1.10.0)
Requirement already satisfied: torch>=1.8.0 in /usr/local/lib/python3.8/dist-packages (from ultralytics==8.0.196) (2.0.0a0+ec3941ad.nv23.2)
Requirement already satisfied: torchvision>=0.9.0 in /usr/local/lib/python3.8/dist-packages/torchvision-0.14.1a0+5e8e2f1-py3.8-linux-aarch64.egg (from ultralytics==8.0.196) (0.14.1a0+5e8e2f1)
Requirement already satisfied: tqdm>=4.64.0 in /usr/local/lib/python3.8/dist-packages (from ultralytics==8.0.196) (4.67.1)
Requirement already satisfied: pandas>=1.1.4 in /usr/local/lib/python3.8/dist-packages (from ultralytics==8.0.196) (1.5.3)
Requirement already satisfied: seaborn>=0.11.0 in /usr/local/lib/python3.8/dist-packages (from ultralytics==8.0.196) (0.13.2)
Requirement already satisfied: psutil in /usr/local/lib/python3.8/dist-packages (from ultralytics==8.0.196) (5.9.4)
Requirement already satisfied: py-cpuinfo in /usr/local/lib/python3.8/dist-packages (from ultralytics==8.0.196) (9.0.0)
Requirement already satisfied: thop>=0.1.1 in /usr/local/lib/python3.8/dist-packages (from ultralytics==8.0.196) (0.1.1.post2209072238)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.8/dist-packages (from matplotlib>=3.3.0->ultralytics==8.0.196) (1.1.1)
Requirement already satisfied: cycler>=0.10 in /usr/lib/python3/dist-packages (from matplotlib>=3.3.0->ultralytics==8.0.196) (0.10.0)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.8/dist-packages (from matplotlib>=3.3.0->ultralytics==8.0.196) (4.57.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/lib/python3/dist-packages (from matplotlib>=3.3.0->ultralytics==8.0.196) (1.0.1)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.8/dist-packages (from matplotlib>=3.3.0->ultralytics==8.0.196) (23.0)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/lib/python3/dist-packages (from matplotlib>=3.3.0->ultralytics==8.0.196) (2.4.6)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.8/dist-packages (from matplotlib>=3.3.0->ultralytics==8.0.196) (2.8.2)
Requirement already satisfied: importlib-resources>=3.2.0; python_version < "3.10" in /usr/local/lib/python3.8/dist-packages (from matplotlib>=3.3.0->ultralytics==8.0.196) (5.10.2)
Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.8/dist-packages (from requests>=2.23.0->ultralytics==8.0.196) (3.0.1)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.8/dist-packages (from requests>=2.23.0->ultralytics==8.0.196) (3.4)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.8/dist-packages (from requests>=2.23.0->ultralytics==8.0.196) (1.26.14)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.8/dist-packages (from requests>=2.23.0->ultralytics==8.0.196) (2022.12.7)
Requirement already satisfied: networkx in /usr/local/lib/python3.8/dist-packages (from torch>=1.8.0->ultralytics==8.0.196) (3.0)
Requirement already satisfied: sympy in /usr/local/lib/python3.8/dist-packages (from torch>=1.8.0->ultralytics==8.0.196) (1.11.1)
Requirement already satisfied: typing-extensions in /usr/local/lib/python3.8/dist-packages (from torch>=1.8.0->ultralytics==8.0.196) (4.4.0)
Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.8/dist-packages (from pandas>=1.1.4->ultralytics==8.0.196) (2022.7.1)
Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.8/dist-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics==8.0.196) (1.16.0)
Requirement already satisfied: zipp>=3.1.0; python_version < "3.10" in /usr/local/lib/python3.8/dist-packages (from importlib-resources>=3.2.0; python_version < "3.10"->matplotlib>=3.3.0->ultralytics==8.0.196) (3.12.0)
Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.8/dist-packages (from sympy->torch>=1.8.0->ultralytics==8.0.196) (1.2.1)
Installing collected packages: opencv-python, ultralytics
Successfully installed opencv-python-4.12.0.88 ultralytics-8.0.196
root@005146d10083:/workspace# python3 cevheri.py
WARNING ⚠️ Ultralytics settings reset to default values. This may be due to a possible problem with your settings or a recent ultralytics package update. 
View settings with 'yolo settings' or at '/root/.config/Ultralytics/settings.yaml'
Update settings with 'yolo settings key=value', i.e. 'yolo settings runs_dir=path/to/dir'.
Using device: cuda
YOLOv8n summary (fused): 168 layers, 3151904 parameters, 0 gradients, 8.7 GFLOPs
Model summary (fused): 218 layers, 25840339 parameters, 0 gradients, 78.7 GFLOPs
YOLOv8n summary (fused): 168 layers, 3005843 parameters, 0 gradients, 8.1 GFLOPs
✅ All three models loaded successfully
🚀 TRIPLE-MODEL DETECTION SYSTEM STARTED
🚗 Vehicles: YOLOv8 | 🔫 Guns: gun_detecktor.pt | 🚙 Plates: license_plate_detector.pt
FPS: 0.0 | V:0 | P:0
