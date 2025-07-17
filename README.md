# for algorithm
 - File to modify: /home/cevheri/cevheri_algo/env/lib/python3.10/site-packages/ultralytics/nn/tasks.py
- Line to change: Around line 511 in the torch_safe_load function
Change this line:
```python
return torch.load(file, map_location='cpu'), file  # load
```
```python
return torch.load(file, map_location='cpu', weights_only=False), file  # load
```

## for jetson
Jetson cihazlarda normal PC'lerde olduğu gibi pip ile PyTorch kuramamanızın birkaç temel nedeni var:

## Mimari Farklılıklar

**ARM64 vs x86_64**: Jetson cihazları ARM64 işlemci kullanırken, çoğu PC x86_64 kullanır. PyPI'daki PyTorch'un önceden derlenmiş wheel'leri x86_64 mimarisi için yapılmıştır ve ARM64'te çalışmaz.

**CUDA Sürüm Uyumluluğu**: Jetson cihazları belirli CUDA sürümleri ile JetPack çalıştırır ve bu sürümler standart PyTorch wheel'lerinin derlendiği CUDA sürümleri ile eşleşmeyebilir.

## Çözümler

### 1. NVIDIA'nın Önceden Hazırlanmış Wheel'lerini Kullanın
NVIDIA, Jetson için özel PyTorch wheel'leri sağlar:

```bash
# JetPack 4.6+ için
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Veya NVIDIA'nın doğrudan wheel'lerini kullanın
wget https://nvidia.box.com/shared/static/[belirli-wheel-linki].whl
pip3 install [indirilen-wheel].whl
```

### 2. Jetson Container'larını Kullanın
NVIDIA PyTorch ile önceden hazırlanmış Docker container'ları sağlar:

```bash
sudo docker run -it --rm --runtime nvidia --network host nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3
```

### 3. Kaynak Koddan Derleyin
Bu birkaç saat sürer ama size en fazla kontrolü verir:

```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export USE_CUDA=1
export USE_CUDNN=1
python3 setup.py install
```

### 4. JetPack SDK Manager'ı Kullanın
En kolay yaklaşım - JetPack genellikle AI/ML paketlerinde PyTorch'u içerir.

