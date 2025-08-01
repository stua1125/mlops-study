# CHAPTER 6. 상용 배포

6장은 머신러닝 모델을 실제 운영 환경에 배포하는 과정을 설명합니다. 단순히 모델을 저장하는 것을 넘어서, 자동화된 파이프라인과 안전한 배포 전략, 확장 가능한 인프라 구성 등을 포함한 실무적 관점에서 다룹니다.

---

## 6.1 CI/CD 파이프라인

- 지속적 통합(Continuous Integration)과 지속적 배포(Continuous Deployment) 전략 필요
- 모델 테스트, 검증, 패키징, 릴리스 자동화
- Git, Docker, Jenkins, GitHub Actions, Airflow 등 도구 사용

## 6.2 머신러닝 아티팩트 개발

- 모델뿐만 아니라 관련된 코드, 설정, 전처리기 등도 아티팩트로 관리
- 모델이 예측에 필요한 모든 의존성을 포함해야 함
- reproducibility 확보를 위해 아티팩트 버전 관리 필요

## 6.3 배포 전략

- **배치 vs 실시간 예측**: 사용 시나리오에 따라 배포 방식 선택
- **A/B 테스트**, **점진적 배포**, **블루-그린 배포** 등의 전략 활용
- 롤백 전략도 함께 설계되어야 함

## 6.4 컨테이너화

- 모델 실행 환경을 Docker와 같은 컨테이너로 패키징
- 일관된 실행 보장, 이식성 향상
- Kubernetes, ECS, SageMaker 등과 연계 가능

## 6.5 배포 확장

- 고가용성과 확장성 확보가 필요
- 오토스케일링, 로드 밸런싱 설계 포함
- 마이크로서비스 구조 내 모델 서빙 고려

## 6.6 요구사항과 도전 과제

- 데이터 프라이버시, 응답 시간, 트래픽 예측 등 고려해야 할 실무 이슈
- 모델과 API 간 분리 여부 결정
- 모델 재학습과 배포의 연계성까지 전체 흐름 고려

## 6.7 마치며

- 상용 배포는 기술적 요소와 운영 전략을 통합하는 과정
- DevOps, ML 엔지니어, 데이터 사이언티스트의 협력이 필수
