코드 구조
코드 파일은 아래와 같이 구성되어 있습니다.

code 폴더에는 대회에서 사용될 baseline code인 baseline.ipynb, 베이스라인 코드 실행에 필요한 기본적인 패키지 정보들을 모아둔 requirement.txt, 그리고 베이스라인 모델의 매개변수 정보들을 모아둔 config.yaml이 들어있습니다.

코드 설명 (baseline.ipynb)
1. 데이터 가공 및 데이터셋 클래스 구축하기
데이터셋을 불러와 BART 모델의 encoder와 decoder의 입력 형태로 가공해줍니다.

가공된 데이터를 torch dataset class 로 구축하여 모델에 입력가능한 형태로 만듭니다.

1.1. class Preprocess
: 데이터 전처리를 위한 클래스로, 데이터셋을 데이터프레임으로 변환하고 인코더와 디코더의 입력을 생성합니다.

1.2. class DatasetForTrain, DatasetForVal, DatasetForInference
: Train, validation, test에 사용되는 Dataset 클래스를 정의합니다.

1.3. Define prepare_train_dataset
: Tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
: encoder의 input 데이터와 decoder의 input, ouput 데이터를 생성합니다.



2. Trainer 및 TrainingArguments 구축하기
Huggingface 의 Trainer 와 TrainingArguments를 활용하여 모델 학습을 일괄적으로 처리해주는 클래스를 정의합니다.

2.1. Define compute_metrics
: 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.

2.2. Define load_trainer_for_train
: 학습을 위한 Trainer 클래스와 매개변수를 정의합니다.

2.3. Define load_tokenizer_and_model_for_train
: 학습을 위한 BART 기반의 tokenizer와 사전 학습된 모델을 불러옵니다. 본 베이스라인은 한국어에 특화된 KoBART (https://huggingface.co/digit82/kobart-summarization)를 사용합니다.
: tokenizer를 정의할 때, 도메인마다 문장에서 자주 등장하는 고유 단어들이 분해되는 것을 방지하기 위하여 special_tokens을 지정해줍니다.

3. 모델 학습하기
Trainer 클래스를 불러온 후 모델 학습을 진행합니다.



4. 모델 추론하기
4.1. Define prepare_test_dataset
: Tokenization 과정까지 진행된 최종적으로 모델에 입력될 데이터를 출력합니다.
: 학습 과정에서와는 다르게 추론 시에는 decoder의 output 데이터를 생성할 필요없이 encoder와 decoder에 입력될 데이터만 생성합니다.

4.2. Define load_tokenizer_and_model_for_test
: 추론을 위한 tokenizer와 학습시킨 모델을 checkpoint 경로를 통해 불러옵니다.

4.3. Define inference
: 학습된 모델이 생성한 요약문의 출력 결과를 보여줍니다.



코드 설명 (solar_api.ipynb)
1. Solar Chat API 요약 성능 확인하기
Solar Chat API을 이용하여 train 및 validation dataset에 포함된 dialogue 샘플을 요약해 봅니다.

1.1 Define compute_metrics
: 모델 성능에 대한 평가 지표를 정의합니다. 본 대회에서는 ROUGE 점수를 통해 모델의 성능을 평가합니다.

1.2 Define build_prompt
: Dialogue를 입력으로 받아, Solar Chat API에 보낼 Prompt를 생성하는 함수를 정의합니다.

1.3 Define summarization
: Solar Chat API를 활용해 Summarization을 수행하는 함수를 정의합니다.

1.4 Define test_on_train_data
: Train data 중 처음 3개의 대화를 요약합니다. num_samples 값을 변경하여 원하는 양의 대화를 요약할 수 있습니다.
: For문을 이용하여 루프를 돌면서, 각각의 sample을 앞서 정의한 summarization 함수로 요약하고, 그 결과를 출력합니다.

1.5 Define validation
: Validation data의 대화를 요약합니다. num_samples 값을 변경하여 원하는 양의 대화를 요약할 수 있습니다.
: For문을 이용하여 루프를 돌면서, 각각의 sample을 앞서 정의한 summarization 함수로 요약하고, compute_metrics 함수로 점수를 측정합니다. 이후, 최종 평균 점수를 출력합니다.


2. Solar Chat API로 요약하기
Solar Chat API을 이용하여 test dataset에 포함된 dialogue를 요약하고 제출용 파일을 생성합니다.

2.1 Define inference
: Test data의 대화를 요약하여, 최종 제출 파일을 생성합니다.
: 요약 진행 시, RPM 제한에 걸리지 않도록, 분당 최대 100개의 요청만 하는 로직이 구현되어 있습니다.
: 요약이 완료되면 최종 결과 파일을 csv 형태로 저장합니다.


3. Prompt Engineering
Prompt engineering을 통해 요약 성능 향상을 시도합니다.

2.1 Re-define build_prompt
: 앞서 정의한 build_prompt 함수를 재정의 하여, 다른 형태로 프롬프트를 생성하도록 합니다. 이후 기존 test_on_train_data 함수, validation 함수 및 inference 함수를 동일하게 이용하여 결과 확인, validation 진행 및 최종 추론 진행을 할 수 있습니다.

학습시간 및 학습장비
Stages GPU 서버를 활용했을 때의 실험 결과는 아래와 같습니다.

Training time

CPU times: user 18min 10s, sys: 2min 20s, total: 20min 11s Wall time: 20min 56s

Inference time

CPU times: user 14.4 s, sys: 740 ms, total: 15.1 s Wall time: 12.6 s

Public data 성능
제공된 baseline 기준 전체 test 데이터 중 250개의 public data에서 측정한 성능입니다.

ROUGE-F1 score: 47.1244

LinkedIn
Instagram
Facebook
YouTube
Discord
