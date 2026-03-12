# 프로젝트 초기 설정 가이드

NLP 대화 요약 프로젝트를 시작하기 위한 단계별 설정 가이드입니다.

---

## 📋 필수 조건

- Python 3.8+
- CUDA 11.8+ (GPU 사용 시)
- Git

---

## 🚀 빠른 시작 (5분)

### 1. 프로젝트 클론 및 이동
```bash
cd /path/to/your/project
```

### 2. 템플릿 파일 복사
```bash
# 환경 변수 템플릿 복사
cp .claude/skills/nlp-dialogue-summarization/references/.env.template .env

# .gitignore 복사
cp .claude/skills/nlp-dialogue-summarization/references/.gitignore .gitignore

# requirements.txt 복사 (또는 직접 사용)
cp .claude/skills/nlp-dialogue-summarization/references/requirements.txt requirements.txt
```

### 3. 환경 변수 설정
```bash
# .env 파일 열기
vim .env  # 또는 nano .env

# 다음 값들을 수정:
# WANDB_API_KEY=your-actual-api-key  # https://wandb.ai/authorize 에서 발급
# WANDB_ENTITY=your-username
# WANDB_PROJECT=dialogue_summarization
```

### 4. 가상 환경 생성 및 패키지 설치
```bash
# Conda 사용 (권장)
conda create -n dialogue_sum python=3.10
conda activate dialogue_sum

# 또는 venv 사용
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

### 5. Hydra Config 디렉토리 생성
```bash
mkdir -p conf/model conf/training conf/inference
```

---

## 📦 상세 설정

### A. PyTorch 설치 (GPU 사용 시)

#### CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### CUDA 12.1
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### CPU only
```bash
pip install torch torchvision torchaudio
```

### B. 한국어 형태소 분석기 설치 (Optional)

#### Ubuntu/Debian
```bash
sudo apt-get install g++ openjdk-8-jdk python3-dev python3-pip curl
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
pip install konlpy
```

#### macOS
```bash
brew install mecab mecab-ko mecab-ko-dic
pip install konlpy
```

### C. Hydra Plugins 설치 (Optional)

#### Optuna Sweeper (자동 하이퍼파라미터 최적화)
```bash
pip install hydra-optuna-sweeper
```

#### Joblib Launcher (병렬 실행)
```bash
pip install hydra-joblib-launcher
```

---

## 📁 프로젝트 구조 생성

```bash
# 프로젝트 루트에서 실행
mkdir -p data src/data src/models src/utils checkpoints prediction

# 확인
tree -L 2
```

예상 구조:
```
NLP/
├── .env                     # 환경 변수 (git 제외)
├── .gitignore               # Git 제외 목록
├── requirements.txt         # 패키지 목록
├── conf/                    # Hydra config
│   ├── config.yaml
│   ├── model/
│   ├── training/
│   └── inference/
├── src/
│   ├── data/
│   ├── models/
│   ├── utils/
│   ├── train.py
│   └── inference.py
├── data/                    # 데이터셋
├── checkpoints/             # 모델 체크포인트
└── prediction/              # 추론 결과
```

---

## ✅ 설치 확인

### 1. Python 패키지 확인
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import hydra; print(f'Hydra: {hydra.__version__}')"
python -c "import wandb; print(f'WandB: {wandb.__version__}')"
```

### 2. GPU 확인 (GPU 사용 시)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### 3. WandB 로그인 테스트
```bash
wandb login
# 또는 .env 파일의 API key 사용
python -c "import wandb; import os; from dotenv import load_dotenv; load_dotenv(); wandb.login(key=os.getenv('WANDB_API_KEY')); print('WandB login successful!')"
```

---

## 🎯 다음 단계

### 1. Config 파일 생성
`conf/config.yaml` 파일을 생성하고 기본 설정을 작성합니다.
자세한 내용은 `SKILL.md`의 "Hydra Config 구조" 섹션 참고.

### 2. 데이터 준비
```bash
# 데이터를 data/ 디렉토리에 배치
cp /path/to/train.csv data/
cp /path/to/dev.csv data/
cp /path/to/test.csv data/
```

### 3. 첫 실험 실행
```bash
# 간단한 테스트 실행
python src/train.py training.num_train_epochs=1 debug=true
```

---

## 🔧 문제 해결

### ImportError: No module named 'xxx'
```bash
pip install xxx
# 또는
pip install -r requirements.txt --force-reinstall
```

### WandB 연결 오류
```bash
# 오프라인 모드로 테스트
WANDB_MODE=offline python src/train.py

# 또는 .env 파일의 WANDB_API_KEY 확인
cat .env | grep WANDB_API_KEY
```

### CUDA out of memory
```bash
# Batch size 줄이기
python src/train.py training.per_device_train_batch_size=16

# 또는 FP16 활성화
python src/train.py training.fp16=true
```

### Hydra config not found
```bash
# Config 경로 확인
ls -la conf/
# config.yaml 파일이 존재하는지 확인

# 또는 상대 경로 수정
# src/train.py의 @hydra.main(config_path="../conf") 확인
```

---

## 📚 추가 자료

- **Hydra 공식 문서**: https://hydra.cc/docs/intro/
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers/
- **WandB 가이드**: https://docs.wandb.ai/
- **PyTorch 공식 문서**: https://pytorch.org/docs/stable/

---

## 💡 팁

### 최소 설치로 빠르게 시작
```bash
# Core만 설치 (Hydra plugins 제외)
pip install torch transformers hydra-core omegaconf wandb rouge pandas python-dotenv
```

### 개발 환경 설정 (Optional)
```bash
# Jupyter Notebook
pip install jupyter ipython

# Code formatting
pip install black isort flake8

# 실행
jupyter notebook
```

### 환경 변수 자동 로드
```python
# src/train.py 상단에 추가
from dotenv import load_dotenv
load_dotenv()  # .env 파일 자동 로드
```

---

## 🎉 설정 완료!

모든 설정이 완료되었습니다. 이제 본격적으로 학습을 시작할 수 있습니다:

```bash
# 기본 학습
python src/train.py

# 모델 변경
python src/train.py model=kot5

# Sweep 실행
python src/train.py -m training.learning_rate=1e-5,3e-5,5e-5
```

궁금한 점이 있다면 `SKILL.md` 문서를 참고하세요!
