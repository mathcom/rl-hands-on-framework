export const APP_TITLE = "RL Hands-on Framework (LunarLander)";

export const REQUIREMENTS_TXT = `gymnasium[box2d]
torch
numpy
matplotlib
swig`;

export const AGENT_TEMPLATE_PY = `import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Tuple, List, Dict

class Agent:
    def __init__(self, state_dim: int, action_dim: int):
        """
        에이전트 초기화 (Agent Initialization)
        
        LunarLander-v2 환경 정보:
        - state_dim: 8 (착륙선의 좌표 x,y, 속도 vx,vy, 각도, 각속도, 다리 접촉 여부 등)
        - action_dim: 4 (0: 아무것도 안함, 1: 왼쪽 엔진, 2: 메인 엔진, 3: 오른쪽 엔진)

        Args:
            state_dim (int): 상태 공간의 차원 (Dimension of state space)
            action_dim (int): 행동 공간의 차원 (Dimension of action space)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # TODO: 신경망 모델을 정의하세요 (예: nn.Sequential 사용)
        # 힌트: 입력(8) -> 은닉층 -> 출력(4) 구조가 필요합니다.
        self.model = None 
        
        # TODO: 최적화(Optimizer)와 손실 함수(Loss function)를 정의하세요
        self.optimizer = None
        
        # 하이퍼파라미터
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

    def get_action(self, state: np.ndarray) -> int:
        """
        현재 상태(state)를 받아 행동(action)을 결정합니다.
        
        Args:
            state (np.ndarray): 현재 상태 (shape: (8,))
            
        Returns:
            int: 선택된 행동 (0~3)
        """
        # TODO: 엡실론-그리디(Epsilon-Greedy) 전략을 구현하세요
        # 1. random.random() < self.epsilon 이면 무작위 행동 (0~3) 선택
        # 2. 그렇지 않으면 모델을 통해 최적의 행동 선택 (torch.argmax 활용)
        
        # 더미 로직 (구현 후 삭제): 랜덤 행동
        return random.randint(0, self.action_dim - 1)

    def update(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> float:
        """
        학습 데이터를 받아 모델을 업데이트합니다.
        
        Args:
            transition (Tuple): (state, action, reward, next_state, done)
            
        Returns:
            float: 계산된 손실(Loss) 값 (로깅용)
        """
        state, action, reward, next_state, done = transition
        
        # 데이터 변환 (numpy -> tensor)
        state_t = torch.FloatTensor(state)
        next_state_t = torch.FloatTensor(next_state)
        action_t = torch.LongTensor([action])
        reward_t = torch.FloatTensor([reward])
        done_t = torch.FloatTensor([0.0 if done else 1.0])
        
        # TODO: DQN 학습 로직 구현
        # 1. 현재 Q값: q_values = self.model(state_t)[action_t]
        # 2. 타겟 Q값: target = reward + gamma * max(self.model(next_state_t)) * (1 - done)
        # 3. Loss 계산 및 역전파 (Backpropagation)
        
        # Epsilon 감소
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return 0.0 # Loss 반환
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
### 실행 방법 (How to Run)

이 프레임워크는 로컬 Python 환경에서 **LunarLander-v2**를 실행하도록 설정되었습니다.

1. **파일 준비**
   - 상단 탭의 \`requirements.txt\`, \`agent_template.py\`, \`main.py\` 코드를 복사하여 로컬에 저장합니다.

2. **환경 설정 (중요)**
   Box2D 물리 엔진이 필요하므로 다음 명령어로 라이브러리를 설치하세요.
   \`\`\`bash
   # Windows/Mac/Linux 공통
   pip install swig
   pip install -r requirements.txt
   \`\`\`
   *참고: Windows 사용자는 swig 설치 에러 시 [링크](http://www.swig.org/download.html)에서 바이너리를 다운받거나 conda를 사용하세요.*

3. **에이전트 구현**
   \`agent_template.py\`를 열고 \`TODO\`를 따라 구현합니다.
   - **입력:** 8차원 벡터 (상태)
   - **출력:** 4차원 벡터 (각 행동에 대한 Q값)

4. **학습 (Training)**
   \`\`\`bash
   python main.py --train
   \`\`\`
   실시간으로 보상(Reward) 그래프가 그려집니다. 목표 점수는 200점 이상입니다.

5. **테스트 (Testing)**
   \`\`\`bash
   python main.py --test
   \`\`\`
   학습된 모델(\`lunar_lander_model.pth\`)을 불러와 실제 착륙 장면을 렌더링합니다.
`;