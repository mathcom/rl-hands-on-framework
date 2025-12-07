export const APP_TITLE = "RL Hands-on Framework";

export const REQUIREMENTS_TXT = `gymnasium[box2d]
torch
numpy
matplotlib
swig`;

// =====================================================================
// [Level 1] 동적 프로그래밍 & Tabular Method (Q-Table)
// =====================================================================
export const LEVEL1_CODE = `import numpy as np
import random
from typing import Tuple

class Agent:
    def __init__(self, state_dim: int, action_dim: int):
        """
        [Level 1] Tabular Agent (Q-Learning)
        목표: 연속적인 LunarLander의 상태를 이산화(Discrete)하여 Q-Table을 만드세요.
        """
        self.action_dim = action_dim
        self.lr = 0.1  # 학습률
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995

        # TODO 1: Q-Table 초기화
        # 힌트: 상태를 몇 개의 구간으로 나눌지 결정해야 합니다. (예: 6개의 구간)
        # self.q_table = {} 

    def discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        """
        상태 벡터(8개)를 이산적인 키(Tuple)로 변환하는 함수
        예: [0.12, -0.4, ...] -> (1, 3, 0, ...)
        """
        # TODO 2: np.digitize 또는 np.linspace를 사용하여 구현하세요.
        return tuple()

    def get_action(self, state: np.ndarray) -> int:
        # Epsilon-Greedy 구현
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        # TODO 3: Q-Table에서 가장 높은 가치를 가진 행동 선택
        state_key = self.discretize(state)
        return 0 # argmax 구현 필요

    def update(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]):
        state, action, reward, next_state, done = transition
        
        # TODO 4: Bellman Equation 구현 (Q-Learning Update)
        # Q(s,a) = Q(s,a) + lr * (reward + gamma * max(Q(s', a')) - Q(s,a))
        pass
`;

// =====================================================================
// [Level 2] Value-based RL (DQN)
// =====================================================================
export const LEVEL2_CODE = `import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # TODO 1: 신경망 레이어 정의 (Input -> Hidden -> Output)
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

class Agent:
    def __init__(self, state_dim: int, action_dim: int):
        """
        [Level 2] DQN Agent
        목표: 신경망을 사용하여 Q-Function을 근사(Approximation)하세요.
        """
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.0005)
        self.memory = [] # Replay Buffer (deque 사용 권장)

    def get_action(self, state: np.ndarray) -> int:
        # TODO 2: 신경망을 통과시켜 행동 결정 (epsilon-greedy)
        return 0

    def update(self, transition):
        # TODO 3: Experience Replay 구현
        # 1. 메모리에 저장
        # 2. 배치 샘플링
        # 3. Loss 계산 (MSELoss)
        # 4. Backprop
        pass
`;

// =====================================================================
// [Level 3] Policy-based RL (PPO / Actor-Critic)
// =====================================================================
export const LEVEL3_CODE = `import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # TODO 1: Actor(정책)와 Critic(가치) 네트워크 정의 (공유 레이어 사용 가능)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        return x

class Agent:
    def __init__(self, state_dim: int, action_dim: int):
        """
        [Level 3] Policy Gradient Agent (PPO/A2C)
        목표: 확률적 정책(Stochastic Policy)을 직접 학습하세요.
        """
        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0003)

    def get_action(self, state):
        # TODO 2: Softmax 확률 분포에서 행동 샘플링 (Categorical 활용)
        # return action, log_prob
        return 0, 0

    def update(self, transitions):
        # TODO 3: Policy Gradient Loss 계산
        # Loss = - log_prob * advantage
        pass
`;

export const MAIN_PY = `import argparse
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

# 의존성 확인 및 예외 처리
try:
    import gymnasium as gym
except ImportError:
    print("오류: gymnasium이 설치되지 않았습니다. (pip install gymnasium)")
    sys.exit(1)

# 학생들에게는 "자신의 레벨에 맞는 코드를 'agent.py'로 저장해서 실행하세요"라고 안내하는 것이 가장 깔끔합니다.
from agent_template import Agent

def parse_args():
    parser = argparse.ArgumentParser(description="RL Hands-on Framework: LunarLander")
    parser.add_argument('--train', action='store_true', help='학습 모드 (Training Mode)')
    parser.add_argument('--test', action='store_true', help='테스트 모드 (Test Mode)')
    parser.add_argument('--render', action='store_true', help='화면 출력 켜기 (Enable Rendering)')
    return parser.parse_args()

def plot_durations(episode_rewards, ax):
    """
    실시간으로 에피소드 보상을 그래프로 그립니다.
    """
    ax.clear()
    ax.set_title('Training Progress (LunarLander-v2)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.plot(episode_rewards, label='Reward')
    
    # 최근 50개 에피소드 이동 평균
    if len(episode_rewards) >= 50:
        means = [np.mean(episode_rewards[i-50:i]) for i in range(50, len(episode_rewards)+1)]
        ax.plot(range(50, len(episode_rewards)+1), means, label='Avg (50 eps)', color='orange')
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.pause(0.001)

def main():
    args = parse_args()
    
    # 렌더링 모드 설정
    render_mode = 'human' if args.render or args.test else None
    
    print("환경을 초기화하는 중... (Initializing Environment...)")
    try:
        env = gym.make("LunarLander-v2", render_mode=render_mode)
    except gym.error.DependencyNotInstalled:
        print("\\n" + "="*60)
        print("🚨 오류: Box2D 의존성이 설치되지 않았습니다.")
        print("해결 방법: pip install \"gymnasium[box2d]\" 명령어를 실행하세요.")
        print("Window 사용자의 경우 'swig' 설치가 필요할 수 있습니다.")
        print("="*60 + "\\n")
        sys.exit(1)
    
    state_dim = env.observation_space.shape[0] # 8
    action_dim = env.action_space.n        # 4
    
    print(f"State Dim: {state_dim} (좌표, 속도, 각도 등)")
    print(f"Action Dim: {action_dim} (0:No-op, 1:Left, 2:Main, 3:Right)")
    
    # 에이전트 생성
    agent = Agent(state_dim=state_dim, action_dim=action_dim)
    
    # 모델 불러오기 (테스트 모드)
    if args.test:
        try:
            agent.model.load_state_dict(torch.load('lunar_lander_model.pth'))
            agent.epsilon = 0.0
            print("💾 모델을 성공적으로 불러왔습니다.")
        except Exception as e:
            print(f"⚠️ 모델 불러오기 실패: {e}")
            print("랜덤 에이전트로 실행합니다.")
    
    rewards_history = []
    
    # 그래프 초기화
    if args.train:
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 5))
    
    num_episodes = 500 if args.train else 5
    
    for i_episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            if args.train:
                transition = (state, action, reward, next_state, done)
                agent.update(transition)
            
            state = next_state
            episode_reward += reward
            
            if args.render:
                env.render()
        
        rewards_history.append(episode_reward)
        print(f"Episode {i_episode+1}: Total Reward {episode_reward:.2f}")
        
        # LunarLander는 200점 이상이면 해결된 것으로 간주
        if episode_reward > 200:
            print(f"🚀 Good Job! Episode {i_episode+1} solved!")

        if args.train:
            plot_durations(rewards_history, ax)
            
            # 주기적 저장
            if (i_episode + 1) % 50 == 0:
                if agent.model:
                    torch.save(agent.model.state_dict(), 'lunar_lander_model.pth')
                    print(f"Saved model at episode {i_episode+1}")

    env.close()
    
    if args.train:
        if agent.model:
            torch.save(agent.model.state_dict(), 'lunar_lander_model.pth')
            print("최종 모델 저장 완료.")
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
`;

export const SYSTEM_INSTRUCTION_KOREAN = `
당신은 강화학습(Reinforcement Learning) 실습 수업의 AI 조교입니다.
현재 학생들은 **LunarLander-v2** 환경에서 착륙선이 안전하게 착륙하도록 학습시키는 과제를 수행 중입니다.
질문에 대해 한국어로 명확하고 친절하게 답변하세요.
정답 코드를 직접 주기보다는 개념 설명과 힌트를 제공하여 학습을 유도하세요.

주요 개념:
- State (8차원): [x좌표, y좌표, x속도, y속도, 각도, 각속도, 다리접촉1, 다리접촉2]
- Action (4개): [0: 아무것도 안함, 1: 왼쪽 엔진 점화, 2: 메인 엔진 점화, 3: 오른쪽 엔진 점화]
- Reward: 안전 착륙 시 +200점, 추락 시 감점 등.
`;

export const RUN_GUIDE_MD = `
### 🌕 LunarLander 강화학습 챌린지 가이드

본 프레임워크는 한 학기 동안 여러분의 강화학습 에이전트를 **Tabular Method**부터 **Deep RL**까지 점진적으로 발전시킬 수 있도록 설계되었습니다.

---

### 📚 단계별 학습 목표 (Curriculum)

왼쪽 상단의 **[Select Level]** 메뉴에서 단계를 선택하여 템플릿 코드를 확인하세요.

#### Level 1: Tabular Methods (Q-Learning)
- **목표:** 연속적인 상태 공간(Continuous Space)을 어떻게 테이블에 저장할지 고민해봅니다.
- **핵심:** \`discretize\` 함수를 구현하여 상태를 이산화(Bucket)하고, Q-Table을 갱신합니다.
- **난관:** 상태를 너무 잘게 쪼개면 테이블이 폭발하고, 너무 크게 쪼개면 학습이 안 되는 딜레마를 경험해 보세요.

#### Level 2: Value-based Deep RL (DQN)
- **목표:** 테이블 대신 **신경망(Neural Network)**으로 Q-Function을 근사(Approximation)합니다.
- **핵심:** PyTorch를 사용하여 Q-Network를 구현하고, **Experience Replay**와 **Target Network**가 왜 필요한지 코드로 체감합니다.

#### Level 3: Policy-based Deep RL (PPO)
- **목표:** 최신 알고리즘인 PPO(Proximal Policy Optimization)를 구현합니다.
- **핵심:** 가치(Value)뿐만 아니라 **정책(Policy)**을 직접 최적화하여, 더 부드럽고 안정적인 비행을 구현합니다.

---

### 🛠️ 실행 방법 (How to Run)

이 프레임워크는 로컬 Python 환경에서 실행됩니다. 다음 순서대로 진행해 주세요.

#### 1. 환경 설정 (Environment Setup)
Box2D 물리 엔진 구동을 위해 필수 라이브러리를 설치해야 합니다.

\`\`\`bash
# 1. 의존성 설치 (Windows/Mac/Linux 공통)
pip install swig
pip install -r requirements.txt
\`\`\`
> **⚠️ 주의:** Windows 사용자는 \`swig\` 설치 시 에러가 발생할 수 있습니다. 에러 발생 시 [링크](http://www.swig.org/download.html)에서 바이너리를 다운받거나 Anaconda를 사용하세요.

#### 2. 에이전트 코드 준비
1. 왼쪽 탭에서 도전할 **Level**을 선택합니다.
2. 코드를 복사하여 로컬 폴더에 **\`agent.py\`** 라는 이름으로 저장합니다.
3. 주석에 달린 **\`TODO\`** 부분을 채워 알고리즘을 완성합니다.

#### 3. 학습 (Training)
에이전트가 달 착륙을 스스로 학습하는 과정을 지켜봅니다.
\`\`\`bash
python main.py --train
\`\`\`
* 실시간으로 보상(Reward) 그래프가 그려집니다.
* **목표 점수:** 200점 이상 (안전하게 착륙하여 멈춤)

#### 4. 테스트 (Testing)
학습된 모델(\`lunar_lander_model.pth\` 또는 \`q_table.pkl\`)을 불러와 실제 렌더링 화면을 확인합니다.
\`\`\`bash
python main.py --test
\`\`\`
`;