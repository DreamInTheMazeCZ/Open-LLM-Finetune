# Conversation AI - Llama 3 Version


### 배포 환경
- 서버 : Ubuntu 22.04.4 LTS Linux Server 121.78.118.25 (GPU - GeForce RTX4080)
- NVIDIA Driver : 550.78
- CUDA : 12.4
- 파이썬 버전 : Python 3.10.12
- 스크립트 배포 : /home/python/llama3/
- 포트 : 8000


### 가상환경
- 가상환경 실행 : source /home/llm_test/bin/activate
- 가상환경 해제 : deactivate


## 프로젝트
```
/
        | llm_api.py - API 서버
        | llm_utils.py - LLM 로딩, Chat 기능, 프롬프트 핸들러 스크립트
        | nohup.out - 서버 가동 로그
        | requirements.txt - 필요 라이브러리
        | /logs
             | LLama3_log_YYYY-MM-DD.logs - 스크립트 로그
        | /PromptFile
             | llm_prompt.csv - User:Assistant 프롬프트 관리 파일
        | /templates
             | chat 1.html - 임시 웹 LLM 인터페이스
        | /static
```


## 필요 라이브러리
```
accelerate==0.29.3
aiohttp==3.9.5
aiosignal==1.3.1
annotated-types==0.6.0
anyio==4.3.0
async-timeout==4.0.3
attrs==23.2.0
bitsandbytes==0.43.1
certifi==2024.2.2
charset-normalizer==3.3.2
click==8.1.7
dataclasses==0.6
datasets==2.19.1
dill==0.3.8
docstring_parser==0.16
einops==0.8.0
exceptiongroup==1.2.1
fastapi==0.110.3
filelock==3.13.4
flash-attn==2.5.8
frozenlist==1.4.1
fsspec==2024.3.1
h11==0.14.0
huggingface-hub==0.22.2
idna==3.7
Jinja2==3.1.3
markdown-it-py==3.0.0
MarkupSafe==2.1.5
mdurl==0.1.2
mpmath==1.3.0
multidict==6.0.5
multiprocess==0.70.16
networkx==3.3
ninja==1.11.1.1
numpy==1.26.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==8.9.2.26
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.4.127
nvidia-nvtx-cu12==12.1.105
packaging==24.0
pandas==2.2.2
peft==0.10.0
pillow==10.3.0
protobuf==5.26.1
psutil==5.9.8
pyarrow==16.0.0
pyarrow-hotfix==0.6
pydantic==2.7.1
pydantic_core==2.18.2
Pygments==2.18.0
python-dateutil==2.9.0.post0
pytz==2024.1
PyYAML==6.0.1
regex==2024.4.28
requests==2.31.0
rich==13.7.1
safetensors==0.4.3
shtab==1.7.1
six==1.16.0
sniffio==1.3.1
starlette==0.37.2
sympy==1.12
tensorboardX==2.6.2.2
tokenizers==0.19.1
torch==2.3.0
torchaudio==2.3.0
torchvision==0.18.0
tqdm==4.66.2
transformers==4.40.1
triton==2.3.0
trl==0.8.6
typing_extensions==4.11.0
tyro==0.8.3
tzdata==2024.1
urllib3==2.2.1
uvicorn==0.29.0
xxhash==3.4.1
yarl==1.9.4
```

## 기타
- 일반 모델 로드 시 약 14.5GB, 8bit 로드 시 약 9GB, 4bit 로드 시 약 6GB 정도의 GPU 자원 소모
- 4bit는 사용이 어려울 수준의 성능이므로 최소 8bit로 사용 권장
- 프롬프트가 늘어나면 max_new_tokens 수 조절 필요 - 속도가 저하될 수 있으므로 적절한 수 설정 필요