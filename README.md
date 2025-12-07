# 🚀 RL Hands-on Framework: LunarLander Agent

\<div align="center"\>
\<img width="100%" alt="LunarLander Simulation" src="[https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6](https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6)" /\>
<br>
\<h3\>🤖 AI 조교와 함께하는 강화학습 실습 플랫폼\</h3\>
\</div\>

## 📖 과제 개요 (Overview)

본 프로젝트는 **2026학년도 강화학습(Agent RL)** 수업을 위한 실습 프레임워크입니다.
여러분은 달 착륙선(**LunarLander-v2**)이 표면에 안전하게 착륙할 수 있도록 제어하는 \*\*인공지능 에이전트(Agent)\*\*를 개발해야 합니다.

단순히 코드를 짜는 것을 넘어, 로컬 LLM(Ollama)이 탑재된 **AI 조교**와 대화하며 이론을 학습하고 문제를 해결해 보세요.

### 🎯 미션 목표

1.  **Environment:** `LunarLander-v2` (Gymnasium)
2.  **Input (State):** 8차원 벡터 (좌표, 속도, 각도, 다리 접촉 여부 등)
3.  **Output (Action):** 4가지 행동 (0: 대기, 1: 좌측 엔진, 2: 메인 엔진, 3: 우측 엔진)
4.  **Goal:** 에피소드 당 **Total Reward 200점 이상** 달성

-----

## 🛠️ 기술 스택 (Tech Stack)

이 프로젝트는 최신 AI 및 웹 기술로 구축되었습니다. 코드를 분석하며 웹 서비스 아키텍처도 함께 공부해 보세요.

  * **Frontend:** [React](https://react.dev/), [Vite](https://vitejs.dev/), [Tailwind CSS](https://tailwindcss.com/)
  * **Infrastructure:** [Docker](https://www.docker.com/), [Docker Compose](https://docs.docker.com/compose/)
  * **AI Engine:** [Ollama](https://ollama.com/) (Llama 3.1 Model)
  * **Network:** [Cloudflare Tunnel](https://www.cloudflare.com/) (외부 접속 지원)

-----

## 🚀 실행 방법 (Getting Started)

이 프로젝트는 **Docker** 환경에서 원클릭으로 실행되도록 설계되었습니다. 복잡한 환경 설정 없이 아래 명령어를 따라오세요.

### 1\. 사전 준비 (Prerequisites)

  * **Docker Desktop** 설치 (필수)
  * (선택) NVIDIA GPU 드라이버 (AI 응답 속도 향상용)

### 2\. 설치 및 실행

터미널(PowerShell 또는 CMD)을 열고 프로젝트 폴더에서 다음 명령어를 입력합니다.

```bash
# 1. 저장소 복제 (이미 받았다면 생략)
git clone https://github.com/mathcom/rl-hands-on-framework.git
cd rl-hands-on-framework

# 2. Docker 컨테이너 빌드 및 실행
docker compose up --build -d
```

### 3\. 접속하기

  * **PC 접속:** 브라우저를 열고 `http://localhost:5137` 접속
  * **모바일/외부 접속:** 실행 로그에 표시된 Cloudflare 주소 확인
    ```bash
    docker compose logs tunnel
    # https://xxxx-xxxx.trycloudflare.com 형식의 주소를 복사하세요.
    ```

-----

## 📂 프로젝트 구조 (Structure)

여러분이 주로 작업해야 할 공간은 `src` 폴더입니다.

```text
📦 rl-hands-on-framework
 ┣ 📂 src
 ┃ ┣ 📂 components      # UI 컴포넌트 (채팅창 등)
 ┃ ┣ 📜 App.tsx         # 메인 웹 애플리케이션 로직
 ┃ ┣ 📜 agent_template.py # [중요] 여러분이 완성해야 할 에이전트 코드
 ┃ ┗ 📜 main.py         # 학습 및 시각화 엔진
 ┣ 📜 docker-compose.yml # 컨테이너 오케스트레이션 설정
 ┗ 📜 Dockerfile         # 프론트엔드 빌드 설정
```

-----

## 🎓 학습 가이드 (Study Roadmap)

이 과제를 통해 여러분은 다음 단계로 성장할 수 있습니다.

### Lv.1: 강화학습 기초 (Reinforcement Learning)

  * \*\*상태(State), 행동(Action), 보상(Reward)\*\*의 개념을 코드로 이해합니다.
  * `agent_template.py`의 `get_action` 함수를 수정하며 **Epsilon-Greedy** 전략을 구현해 봅니다.
  * DQN(Deep Q-Network) 논문을 읽고 네트워크 구조를 설계해 봅니다.

### Lv.2: 프롬프트 엔지니어링 (Using AI Assistant)

  * AI 조교에게 정답을 바로 묻지 말고, \*\*"원리"\*\*와 \*\*"힌트"\*\*를 물어보세요.
  * *Good Prompt:* "DQN에서 Experience Replay가 왜 필요한지 비유를 들어 설명해줘."
  * *Bad Prompt:* "코드 그냥 짜줘."

### Lv.3: 풀스택 엔지니어링 (System Architecture)

  * **Docker Compose**가 어떻게 웹 서버(Frontend)와 AI 서버(Ollama)를 연결하는지 분석해 보세요.
  * `vite.config.ts`의 **Proxy** 설정이 왜 필요한지(CORS 문제 해결) 탐구해 보세요.
  * 스마트폰에서 접속했을 때 UI가 변하는 **반응형 웹(Responsive Web)** 코드를 `App.tsx`에서 찾아보세요.

-----

## 📚 추천 자료 (References)

더 깊이 공부하고 싶은 학생들을 위한 추천 자료입니다.

1.  **강화학습 이론:** [Sutton & Barto, "Reinforcement Learning: An Introduction"](http://incompleteideas.net/book/the-book-2nd.html)
2.  **OpenAI Spinning Up:** [Deep RL 입문자를 위한 최고의 가이드](https://spinningup.openai.com/)
3.  **PyTorch Tutorials:** [Reinforcement Learning (DQN) Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

-----

## ⚠️ 문제 해결 (Troubleshooting)

  * **Q. `localhost` 연결 거부됨**
      * Docker가 실행 중인지 확인하세요 (`docker compose ps`).
  * **Q. AI가 답변을 안 해요.**
      * Ollama 모델 다운로드에 시간이 걸릴 수 있습니다. `docker compose logs ollama`로 `Success` 메시지가 떴는지 확인하세요.
  * **Q. 모바일에서 키보드가 화면을 가려요.**
      * 안드로이드 구형 버전에서는 UI가 겹칠 수 있습니다. 가로 모드보다는 세로 모드를 권장합니다.

-----

\<div align="center"\>
MIT License | Created by Agent RL Class (2025)
\</div\>