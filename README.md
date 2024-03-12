[![PyPI license](https://img.shields.io/pypi/l/ansicolortags.svg)](https://pypi.python.org/pypi/ansicolortags/)
 <img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252" /> 
 
Here I used a pretrained YOLOv5 model to perform object detection on an image.
First of all, I installed the following requirements on my Google Colab environment.

`!pip install torch torchvision torchaudio
!pip install cython
!pip install -U 'git+https://github.com/facebookresearch/fvcore.git'
!pip install 'git+https://github.com/facebookresearch/detectron2.git'`

```python
!pip install torch torchvision torchaudio
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -r requirements.txt
```

    Requirement already satisfied: torch in /usr/local/lib/python3.10/dist-packages (2.1.0+cu121)
    Requirement already satisfied: torchvision in /usr/local/lib/python3.10/dist-packages (0.16.0+cu121)
    Requirement already satisfied: torchaudio in /usr/local/lib/python3.10/dist-packages (2.1.0+cu121)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch) (3.13.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch) (4.10.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch) (3.1.3)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch) (2023.6.0)
    Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch) (2.1.0)
    Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from torchvision) (1.25.2)
    Requirement already satisfied: requests in /usr/local/lib/python3.10/dist-packages (from torchvision) (2.31.0)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.10/dist-packages (from torchvision) (9.4.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (2.1.5)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests->torchvision) (2024.2.2)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch) (1.3.0)
    Cloning into 'yolov5'...
    remote: Enumerating objects: 16517, done.[K
    remote: Counting objects: 100% (115/115), done.[K
    remote: Compressing objects: 100% (98/98), done.[K
    remote: Total 16517 (delta 47), reused 55 (delta 17), pack-reused 16402[K
    Receiving objects: 100% (16517/16517), 15.12 MiB | 19.26 MiB/s, done.
    Resolving deltas: 100% (11309/11309), done.
    /content/yolov5
    Collecting gitpython>=3.1.30 (from -r requirements.txt (line 5))
      Downloading GitPython-3.1.42-py3-none-any.whl (195 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m195.4/195.4 kB[0m [31m3.2 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: matplotlib>=3.3 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 6)) (3.7.1)
    Requirement already satisfied: numpy>=1.23.5 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 7)) (1.25.2)
    Requirement already satisfied: opencv-python>=4.1.1 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 8)) (4.8.0.76)
    Requirement already satisfied: Pillow>=9.4.0 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 9)) (9.4.0)
    Requirement already satisfied: psutil in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 10)) (5.9.5)
    Requirement already satisfied: PyYAML>=5.3.1 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 11)) (6.0.1)
    Requirement already satisfied: requests>=2.23.0 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 12)) (2.31.0)
    Requirement already satisfied: scipy>=1.4.1 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 13)) (1.11.4)
    Collecting thop>=0.1.1 (from -r requirements.txt (line 14))
      Downloading thop-0.1.1.post2209072238-py3-none-any.whl (15 kB)
    Requirement already satisfied: torch>=1.8.0 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 15)) (2.1.0+cu121)
    Requirement already satisfied: torchvision>=0.9.0 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 16)) (0.16.0+cu121)
    Requirement already satisfied: tqdm>=4.64.0 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 17)) (4.66.2)
    Collecting ultralytics>=8.0.232 (from -r requirements.txt (line 18))
      Downloading ultralytics-8.1.27-py3-none-any.whl (721 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m721.2/721.2 kB[0m [31m11.3 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: pandas>=1.1.4 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 27)) (1.5.3)
    Requirement already satisfied: seaborn>=0.11.0 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 28)) (0.13.1)
    Requirement already satisfied: setuptools>=65.5.1 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 42)) (67.7.2)
    Requirement already satisfied: wheel>=0.38.0 in /usr/local/lib/python3.10/dist-packages (from -r requirements.txt (line 50)) (0.42.0)
    Collecting gitdb<5,>=4.0.1 (from gitpython>=3.1.30->-r requirements.txt (line 5))
      Downloading gitdb-4.0.11-py3-none-any.whl (62 kB)
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.7/62.7 kB[0m [31m7.1 MB/s[0m eta [36m0:00:00[0m
    [?25hRequirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3->-r requirements.txt (line 6)) (1.2.0)
    Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3->-r requirements.txt (line 6)) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3->-r requirements.txt (line 6)) (4.49.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3->-r requirements.txt (line 6)) (1.4.5)
    Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3->-r requirements.txt (line 6)) (23.2)
    Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3->-r requirements.txt (line 6)) (3.1.2)
    Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib>=3.3->-r requirements.txt (line 6)) (2.8.2)
    Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->-r requirements.txt (line 12)) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->-r requirements.txt (line 12)) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->-r requirements.txt (line 12)) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.23.0->-r requirements.txt (line 12)) (2024.2.2)
    Requirement already satisfied: filelock in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->-r requirements.txt (line 15)) (3.13.1)
    Requirement already satisfied: typing-extensions in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->-r requirements.txt (line 15)) (4.10.0)
    Requirement already satisfied: sympy in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->-r requirements.txt (line 15)) (1.12)
    Requirement already satisfied: networkx in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->-r requirements.txt (line 15)) (3.2.1)
    Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->-r requirements.txt (line 15)) (3.1.3)
    Requirement already satisfied: fsspec in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->-r requirements.txt (line 15)) (2023.6.0)
    Requirement already satisfied: triton==2.1.0 in /usr/local/lib/python3.10/dist-packages (from torch>=1.8.0->-r requirements.txt (line 15)) (2.1.0)
    Requirement already satisfied: py-cpuinfo in /usr/local/lib/python3.10/dist-packages (from ultralytics>=8.0.232->-r requirements.txt (line 18)) (9.0.0)
    Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas>=1.1.4->-r requirements.txt (line 27)) (2023.4)
    Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython>=3.1.30->-r requirements.txt (line 5))
      Downloading smmap-5.0.1-py3-none-any.whl (24 kB)
    Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.7->matplotlib>=3.3->-r requirements.txt (line 6)) (1.16.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch>=1.8.0->-r requirements.txt (line 15)) (2.1.5)
    Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch>=1.8.0->-r requirements.txt (line 15)) (1.3.0)
    Installing collected packages: smmap, gitdb, thop, gitpython, ultralytics
    Successfully installed gitdb-4.0.11 gitpython-3.1.42 smmap-5.0.1 thop-0.1.1.post2209072238 ultralytics-8.1.27



```python
import torch
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Set confidence threshold
model.conf = 0.3

# Function to perform object detection on an image
def detect_objects(image_url):
    # Extract image ID from URL
    image_id = image_url.split('/')[-2]

    # Construct direct image URL
    direct_url = f"https://drive.google.com/uc?id={image_id}"

    # Load the image
    response = requests.get(direct_url)
    img = Image.open(BytesIO(response.content))

    # Perform object detection
    results = model(img)

    # Display the results with smaller size
    plt.figure(figsize=(4, 2))
    results.show()

    # Return the results
    return results

# URL of the image
image_url = "https://drive.google.com/file/d/16NybAvrjTcPVbSYxznlzoESWmPSCFlD_/view"

# Perform object detection on the image
detect_objects(image_url)
```

    /usr/local/lib/python3.10/dist-packages/torch/hub.py:294: UserWarning: You are about to download and run code from an untrusted repository. In a future release, this won't be allowed. To add the repository to your trusted list, change the command to {calling_fn}(..., trust_repo=False) and a command prompt will appear asking for an explicit confirmation of trust, or load(..., trust_repo=True), which will assume that the prompt is to be answered with 'yes'. You can also use load(..., trust_repo='check') which will only prompt for confirmation if the repo is not already trusted. This will eventually be the default behaviour
      warnings.warn(
    Downloading: "https://github.com/ultralytics/yolov5/zipball/master" to /root/.cache/torch/hub/master.zip
    YOLOv5 🚀 2024-3-11 Python-3.10.12 torch-2.1.0+cu121 CPU
    
    Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt to yolov5s.pt...
    100%|██████████| 14.1M/14.1M [00:00<00:00, 110MB/s] 
    
    Fusing layers... 
    YOLOv5s summary: 213 layers, 7225885 parameters, 0 gradients, 16.4 GFLOPs
    Adding AutoShape... 



    
![png](README_files/README_1_1.png)
    





    YOLOv5 <class 'models.common.Detections'> instance
    image 1/1: 2272x2822 8 persons, 2 cars, 5 traffic lights
    Speed: 162.9ms pre-process, 358.4ms inference, 35.8ms NMS per image at shape (1, 3, 544, 640)
The image was downloaded from the following [link](https://www.pexels.com/fr-fr/photo/deux-hommes-qui-courent-sur-une-route-en-beton-1564470/).  

`DISCLAIMER`: I don't warrant this code in any way whatsoever. This code is provided "as-is" to be used at your own risk.
