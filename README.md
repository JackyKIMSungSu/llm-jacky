# llm-jacky

LangChain 기반 LLM 앱 개인 프로젝트.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # API 키 입력
```

## Run

CLI:
```bash
python main.py "<주제>"            # 1~5단계 + draft 저장
python main.py "<주제>" --publish  # 1~6단계 (WP 업로드)
```

Web UI (Streamlit):
```bash
streamlit run app.py               # http://localhost:8501
```
- **Pipeline** 페이지: 주제 → 초안/SEO/출처 검토 → WordPress 업로드
- **Evaluation** 페이지: 시드 토픽 평가 → 6 메트릭 표

## Local WordPress (Docker)

발행(6단계) 테스트용 로컬 WordPress.

```bash
docker compose up -d              # localhost:8080 에서 기동
docker compose logs -f wordpress  # 초기화 로그
docker compose down               # 종료 (데이터 유지)
docker compose down -v            # 종료 + 볼륨 삭제 (초기화)
```

최초 1회 셋업:

1. 브라우저로 http://localhost:8080 접속 → 5분 설치 (admin 계정 생성)
2. **Users → Add New** → username `llm-bot`, role `Editor` 로 봇 계정 생성
3. 봇 계정으로 재로그인 → 본인 아바타 → **Edit Profile** 페이지 하단 **Application Passwords**
   에서 이름 `llm-jacky` 로 발급 → 24자 패스워드 즉시 복사
4. `.env` 에 입력:
   ```
   WORDPRESS_URL=http://localhost:8080
   WORDPRESS_USERNAME=llm-bot
   WORDPRESS_APP_PASSWORD=xxxx xxxx xxxx xxxx xxxx xxxx
   ```
5. 퍼머링크 + .htaccess 셋업 (REST API `/wp-json/...` 라우팅용):
   ```bash
   ./docker/init-permalinks.sh
   ```
6. 검증: `curl -u "llm-bot:xxxx ..." http://localhost:8080/wp-json/wp/v2/users/me`

> `docker compose down -v` 로 볼륨을 초기화하면 5번을 다시 실행해야 합니다.
