# 🚀 RL Hands-on Framework: LunarLander Agent

<p align="center">
  <img width="100%" alt="LunarLander Simulation" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Built%20with-Google%20AI%20Studio-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Google AI Studio" />
  <img src="https://img.shields.io/badge/Powered%20by-Gemini%20Pro-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini Pro" />
</p>

<p align="center">
  <b>🤖 AI 조교(RAG)와 함께하는 강화학습 실습 플랫폼</b>
</p>

## 📖 과제 개요 (Overview)

본 프로젝트는 **2026학년도 강화학습(Agent RL)** 수업을 위한 실습 프레임워크입니다.
여러분은 달 착륙선(**LunarLander-v2**)이 표면에 안전하게 착륙할 수 있도록 제어하는 **인공지능 에이전트(Agent)**를 개발해야 합니다.

단순히 코드를 짜는 것을 넘어, 프로젝트 코드를 모두 이해하고 있는 **AI 조교(RAG System)**와 대화하며 이론을 학습하고 문제를 해결해 보세요.

### 🎯 미션 목표

1.  **Environment:** `LunarLander-v2` (Gymnasium)
2.  **Input (State):** 8차원 벡터 (좌표, 속도, 각도, 다리 접촉 여부 등)
3.  **Output (Action):** 4가지 행동 (0: 대기, 1: 좌측 엔진, 2: 메인 엔진, 3: 우측 엔진)
4.  **Goal:** 에피소드 당 **Total Reward 200점 이상** 달성

---

## 🛠️ 기술 스택 (Tech Stack)

이 프로젝트는 최신 AI 및 웹 기술로 구축되었습니다. MSA(Microservice Architecture) 구조를 학습해 보세요.

* **Frontend:** [React](https://react.dev/), [Vite](https://vitejs.dev/), [Tailwind CSS](https://tailwindcss.com/)
* **Backend (RAG):** [FastAPI](https://fastapi.tiangolo.com/), [LangChain](https://www.langchain.com/), [ChromaDB](https://www.trychroma.com/)
* **Infrastructure:** [Docker](https://www.docker.com/), [Docker Compose](https://docs.docker.com/compose/)
* **AI Engine:** [Ollama](https://ollama.com/) (Llama 3.1 + Nomic-Embed-Text)
* **Network:** [Cloudflare Tunnel](https://www.cloudflare.com/) (외부 접속 지원)

---

## 🚀 실행 방법 (Getting Started)

이 프로젝트는 **Docker** 환경에서 원클릭으로 실행되도록 설계되었습니다.

### 1. 사전 준비 (Prerequisites)

* **Docker Desktop** 설치 (필수)
* (선택) NVIDIA GPU 드라이버 (AI 응답 속도 향상용)

### 2. 설치 및 실행

터미널(PowerShell 또는 CMD)을 열고 프로젝트 폴더에서 다음 명령어를 입력합니다.

```bash
# 1. 저장소 복제 (이미 받았다면 생략)
git clone [https://github.com/mathcom/rl-hands-on-framework.git](https://github.com/mathcom/rl-hands-on-framework.git)
cd rl-hands-on-framework

# 2. Docker 컨테이너 빌드 및 실행
# (Backend가 코드를 학습하는 동안 약 1분 정도 소요될 수 있습니다)
docker compose up --build -d
````

### 3\. 접속하기

  * **PC 접속:** 브라우저를 열고 `http://localhost:3000` 접속
  * **모바일/외부 접속:** 실행 로그에 표시된 Cloudflare 주소 확인
    ```bash
    docker compose logs tunnel
    # [https://xxxx-xxxx.trycloudflare.com](https://xxxx-xxxx.trycloudflare.com) 형식의 주소를 복사하세요.
    ```

-----

## 📂 프로젝트 구조 (Structure)

여러분이 주로 작업해야 할 공간은 `src` 폴더입니다. `backend`는 AI 조교를 위한 서버입니다.

```text
📦 rl-hands-on-framework
 ┣ 📂 backend           # (New) RAG AI 서버 (FastAPI + ChromaDB)
 ┣ 📂 src
 ┃ ┣ 📂 components      # UI 컴포넌트 (채팅창, 코드블록 등)
 ┃ ┣ 📜 App.tsx         # 메인 웹 애플리케이션 로직
 ┃ ┣ 📜 constants.ts    # [중요] 레벨별 에이전트 코드 및 가이드 데이터
 ┃ ┗ 📜 main.py         # 학습 및 시각화 엔진 (Python)
 ┣ 📜 docker-compose.yml # 컨테이너 오케스트레이션 설정
 ┗ 📜 Dockerfile         # 프론트엔드 빌드 설정
```

-----

## 🎓 학습 가이드 (Study Roadmap)

이 과제를 통해 여러분은 다음 단계로 성장할 수 있습니다.

### Lv.1: Tabular Methods (Q-Learning)

  * \*\*상태(State), 행동(Action), 보상(Reward)\*\*의 개념을 코드로 이해합니다.
  * 연속적인 상태를 \*\*이산화(Discretization)\*\*하여 Q-Table을 직접 구현해 봅니다.

### Lv.2: Value-based Deep RL (DQN)

  * 테이블 대신 \*\*신경망(Neural Network)\*\*을 사용하여 Q-Function을 근사합니다.
  * PyTorch를 이용해 **DQN**을 구현하고, Experience Replay의 중요성을 학습합니다.

### Lv.3: Policy-based Deep RL (PPO)

  * 최신 알고리즘인 **Actor-Critic (PPO)** 방식을 이해합니다.
  * 가치뿐만 아니라 \*\*정책(Policy)\*\*을 직접 최적화하여 부드러운 제어를 구현합니다.

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
      * Ollama 모델 다운로드 및 백엔드 초기화에 시간이 걸릴 수 있습니다. `docker compose logs backend`를 확인하여 "RAG 시스템 준비 완료" 메시지가 떴는지 보세요.
  * **Q. 모바일에서 키보드가 화면을 가려요.**
      * 안드로이드 구형 버전에서는 UI가 겹칠 수 있습니다. 가로 모드보다는 세로 모드를 권장합니다.

-----

## ⚡ Credits

This framework was developed using **Google AI Studio** and **Gemini**.
Special thanks to the **Gemini 3 Pro** model for assistance in code generation, architecture design, and documentation.

-----

<p align="center"\>
MIT License | Created by Jonghwan Choi (2026)
</p\>
