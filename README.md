cd /tmp
curl https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh --output anaconda.sh
bash anaconda.sh
source ~/.bashrc
conda create --name rag_label_env python=3.10
conda activate rag_label_env
pip3 install torch
pip install pyarrow==14.* langchain sentence-transformers chromadb faiss-gpu python-multipart fastapi uvicorn[standard]
pip install -q -U git+https://github.com/huggingface/transformers.git@v4.35-release
pip install -q -U trl accelerate git+https://github.com/huggingface/peft.git
pip install -q datasets bitsandbytes wandb

uvicorn src.service.app:app --reload
