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
