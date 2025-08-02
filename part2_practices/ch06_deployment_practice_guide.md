# CHAPTER 6 실습 가이드: 상용 배포 준비

이 문서는 `ch06_deployment_practice.ipynb` 노트북을 VS Code에서 단계별로 실행하는 방법을 요약한 가이드입니다.

---

## 1. 사전 준비

1. **가상환경 활성화**  
   ```bash
   source venv/bin/activate
   ```  
   프롬프트에 `(venv)` 표시 확인.

2. **필요 라이브러리 설치**  
   ```bash
   pip install pandas seaborn scikit-learn joblib mlflow
   ```
   (`requirements.txt` 있으면 `pip install -r requirements.txt`)

3. **VS Code 확장 설치 확인**  
   - Python  
   - Jupyter

---

## 2. 노트북 열기 및 커널 선택

1. VS Code에서 `ch06_deployment_practice.ipynb` 파일 열기  
2. 우측 상단 **Select Kernel** 클릭  
3. `Python 3.x.x ('venv': venv)` 또는 `/path/to/mlops-study/venv/bin/python` 선택

---

## 3. 셀별 실행 순서

### 3.1 마크다운 셀 확인
- 노트북 상단 소개 내용 확인

### 3.2 모델 학습 및 저장
```python
# 실행 셀 코드 예시
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import joblib

df = sns.load_dataset('titanic').dropna(subset=['age','embarked'])
df['sex'] = df['sex'].map({'male':0,'female':1})
df['embarked'] = df['embarked'].astype('category').cat.codes
features = ['pclass','sex','age','sibsp','parch','fare','embarked']
X = df[features]
y = df['survived']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

joblib.dump(model, 'model_rf.pkl')
print('✅ 모델 저장 완료')
```

### 3.3 MLflow 모델 로깅
```python
import mlflow
import mlflow.sklearn

mlflow.set_experiment('deployment_test')
with mlflow.start_run():
    mlflow.log_param('n_estimators', 100)
    mlflow.log_metric('accuracy', model.score(X, y))
    mlflow.sklearn.log_model(model, 'model')
print('✅ MLflow 로깅 완료')
```

### 3.4 Dockerfile 예시 출력
```python
dockerfile = '''
# Dockerfile for model serving
FROM python:3.11-slim
WORKDIR /app

COPY model_rf.pkl ./
COPY requirements.txt ./
RUN pip install -r requirements.txt

CMD ['python', 'serve.py']
'''
print(dockerfile)
```

---

## 4. 후속 작업 (옵션)

- **`requirements.txt` 작성**  
  ```text
  pandas
  seaborn
  scikit-learn
  joblib
  mlflow
  flask
  ```
- **`serve.py` 템플릿**  
  ```python
  from flask import Flask, request, jsonify
  import joblib

  app = Flask(__name__)
  model = joblib.load('model_rf.pkl')

  @app.route('/predict', methods=['POST'])
  def predict():
      data = request.json
      features = [data[k] for k in ['pclass','sex','age','sibsp','parch','fare','embarked']]
      pred = model.predict([features])[0]
      return jsonify({'survived': int(pred)})

  if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8000)
  ```
- **컨테이너 빌드 & 실행**  
  ```bash
  docker build -t mlops-demo .
  docker run -p 8000:8000 mlops-demo
  ```
