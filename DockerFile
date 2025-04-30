# 고성능 PyTorch + CUDA 환경 기반 이미지
FROM runpod/pytorch:2.1.2-py3.10-cuda12.1.0

# 작업 디렉토리 설정
WORKDIR /app

# requirements 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 전체 코드 복사
COPY . .

# 앱 실행 (필요시 app.py 수정 가능)
CMD ["python", "app.py"]
