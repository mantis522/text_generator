# text_generator

### 처음엔 GPT-2로 시작해 text_generation으로 data augmentation을 시도하는 프로젝트였지만 어쩌다보니 Bert까지 쓰게됨
### 그냥 트랜스포머 기반 기술들을 사용한 data augmentation의 집합이다... 라고 생각

## LSTM

 - GPT-2로 data augmentation 한 것에 대해 RNN-LSTM 모델로 성능 평가한 것.
 - 키워드 : GPT-2, RNN-LSTM, IMDB dataset
 
## Bert

 - 단순히 Bert가 어떻게 돌아가나 확인하려고 만든 패키지로, 정확도도 89%고 테스트셋 나누는 것도 좀 빡세서 패스해도 됨
 
## Bert_imdb_93

 - ipynb에 설명 나와있음. 93.5% 정확도를 보이고, 트레인 / 테스트셋도 사이킷런으로 나눠서 이게 편할 듯. 이걸 써보자
