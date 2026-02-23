"""
=todo=

cramped_room 프롬프트 수정
2. Instruction, 연습공간 - 싱글 플레이? 협동 플레이?
3. 벤치마크 시스템 구축 - Manhatten vs 절대 좌표 비교 - 만들어봤긴한데 서버 고치고 다시 해봐야 할 듯.
생각하는 동안 / 기다리는 동안은 상하좌우 랜덤행동으로

=실험 설계=
    =정해야 할 질문들=
        맵3개시킬지 4개 시킬지
    1 experiment안에 여러개의 block (맵 1개가 1개의 block), 각 block마다 쉬는시간.
    맵 순서 -> 쉬운거부터 어려운 순으로 - cramped_room -> asymmetric_advantages, coordination_ring, counter_circuit
    의도 표시는 Balanced latin square 사용
    law Log 따두기 (필요한 것: 어떤 의도 표현 방식인지, 각 timestep, object 상태, player 상태, 행동, reward)
    사후 설문 - Cognitive load - nasa xls(매번) + 선호도(block당), 얼마나 도움이 되었는가(매번) + agency: agent를 내 마음대로 움직일 수 있었다.
        Agency 찾아보기
        1block마다 사후 설문 받기

=완성사항=
-2월 마지막 주-
#7. 자연어, 이모지 좀 더 고민해보기
#말풍선에 투명도 추가하기
#정보에 service location 위치도 추가하기
#무한 interaction 방지: 1.2초동안 멈춰있음(근데 wait이 아님) -> 다시 프롬프트 보내기
#5. 1번질문 reference 다시 찾기: timestep이 player가 움직이지 않아도 가는 것에 대한 reference - 참고논문에 있음.
#6. coordination_ring, counter_circuit, #cramped_room 프롬프트 제작하기


#마지막 명령 후 3초뒤에 호출이 없으면, 새로하기
#본인의 계획만 출력하는 경우 있어야함.
#문제: 어떤게 가까운 것인지 알 지 못하는 것 같음. : 프롬프트에서 각 object와 얼마나 거리가 있는지에 대한 정보 주기.
-> 각 object의 위치를 주지 말고, 현재 위치에서 각 object가 얼마나 떨어져 있는지를 주기.
#1. 말풍선으로 수정하기
#1-1 render_game 별도 공간으로 두기.
=연구=
    불확실성?
=문제점=
    LLM이 반대쪽에 ex, 초록색 구역이 아닌곳에 초록색으로 의도를 파악하는 경우가 있음.
=궁금증=
    양쪽 에이전트다 구현해야하나?
    proagent는 counter_circuit에서 협력하는가?
    - 예시를 구체적으로 주어야 하는가?
=벤치마크=
    내 환경의 ProAgent와 비교
=논문=
    Limitation: 의도추론이 완벽하지 않음 / 협업완벽하지않음
    ProAgent가 문단의 첫 문장에 나올 것이 아니라, detail에 조금만 들어가야 한다.
=진행한 것들=
    파랑머리가 이야기하는 것처럼 (✅ 렌더링 함수에 구현됨)
    파랑머리: 의도, 계획
    이동 경로를 모두 저장했다가 보여주기 -> 이전 3개 정도만 보여주기
    프롬프트 예시 추가하기

참고 논문 및 아이디어
    LLM-Powered Hierarchical Language Agent for Real-time Human-AI Coordination
        2.5Hz 기본, 3.5Hz 게임플레이 박진감. -> 유저의 평균속도에 agent가 맞춰서 행동하는 거였음.
        100초 진행. -> 250step -> 근데 이거는 제한시간에 task를 완료하는거라서 좀 애매함.
    가장 가까운 곳이 아니라, LLM이 어떤 pot에 넣을지까지 정하기 (✅ 방금 action(index) 파싱 및 하이라이트로 구현 완료)
참고 논문 2
    Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration
        In the real-time settings, each timestep corresponds to 0.25 seconds in the real world.
        500timestep / 250ms -> 125초

수정사항들
    cook time argument -> 요리가 완료됨을 추가하는 argument + layout가서 따로 수정해주기
    IntentionResponsiveAgent
"""
import warnings
import os
# 1. FutureWarnings (numpy 관련) 무시
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

# 2. TensorFlow 내부 C++ 로그(INFO, WARNING) 숨기기
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 2는 에러/워닝, 3은 에러만 표시

import time
import datetime
import os
import json
import numpy as np
import pygame
import visualization_utils as vu
from argparse import ArgumentParser
from distutils.util import strtobool

# Rich Progress Bar
from rich import print as rprint
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn
)

# Environment & Agents
import importlib_metadata
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import AgentGroup, Agent
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent

# Tensorflow Warning Suppression
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

# --- Constants & Configuration ---
try:
    VERSION = importlib_metadata.version("overcooked_ai")
except:
    VERSION = "0.0.1"

LLM_AGENT_TYPES = ['ProAgent', 'EIRA', 'EIRAAsync']

def boolean_argument(value):
    return bool(strtobool(value))

# --- Helper Classes & Functions ---

class HumanAgent(Agent):
    """키보드 입력을 받아 행동하는 에이전트"""
    def __init__(self):
        super().__init__()
        self.next_action = None 

    def set_next_action(self, action):
        self.next_action = action

    def action(self, state):
        if self.next_action is not None:
            a = self.next_action
            self.next_action = None
            return a
        return Action.STAY

def get_combined_thought(agents_list):
    for i, agent in enumerate(agents_list):
        if hasattr(agent, 'current_thought') and agent.current_thought:
            return i, agent.current_thought
    return -1, None

def main(variant):
    # 1. 설정 변수 로드
    layout_name = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']
    mode = variant['mode']
    render = variant['render'] 
    cook_time = variant['cook_time']
    visual_level = variant['visual_level']
    show_intention = variant['show_intention']
    
    # [핵심] 글로벌 게임 스텝 주기
    game_timestep = variant['timestep'] 

    # 2. MDP & 환경 초기화
    mdp_layout = NEW_LAYOUTS.get(layout_name, layout_name) if VERSION == '1.1.0' else OLD_LAYOUTS.get(layout_name, layout_name)
    mdp = OvercookedGridworld.from_layout_name(mdp_layout)
    layout_dict = vu.generate_layout_dict(mdp)
    env = OvercookedEnv(mdp, horizon=horizon)
    
    # 3. 시각화 초기화
    visualizer = None
    window_surface = None
    if render:
        pygame.init()
        visualizer = StateVisualizer(cook_time=cook_time)
        window_surface = pygame.display.set_mode((900, 600)) 
        pygame.display.set_caption(f"Overcooked AI - {layout_name}")
    
    # 4. 에이전트 생성
    p0_algo, p1_algo = variant['p0'], variant['p1']
    rprint(f"\n[bold green]=== Global Timestep: {game_timestep}ms (AI & Human Sync) ===[/bold green]\n")
    start_time = time.time()
    results = []
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"), 
        BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), 
        TimeRemainingColumn(), TextColumn(" | Mean Reward: [bold cyan]{task.fields[mean_score]}"), 
    ) as progress:
        
        task_id = progress.add_task(f"Episodes...", total=episode, mean_score="0")
        if 'Human' in [p0_algo, p1_algo]: render = True
            
        for ep in range(episode):  
            agents_list = []
            for alg in [p0_algo, p1_algo]:
                if alg in LLM_AGENT_TYPES:
                    agent = make_agent(alg, mdp, layout_name, model=variant['gpt_model'], 
                                       prompt_level=variant['prompt_level'], 
                                       belief_revision=variant['belief_revision'], 
                                       retrival_method=variant['retrival_method'], K=variant['K'])
                elif alg == "Human": agent = HumanAgent()
                else: agent = make_agent(alg, mdp, layout_name)
                agents_list.append(agent)
            
            team = AgentGroup(*agents_list)
            team.reset(); env.reset()
            r_total = 0

            # 5-1. [WarmUp] (생략 가능하나 첫 프레임 딜레이 방지용으로 유지)
            if render and mode == 'exp':
                for idx, agent in enumerate(agents_list):
                    if hasattr(agent, 'generate_ml_action'):
                        vu.draw_centered_text(window_surface, f"AI Initializing...", "Thinking...", color=(0, 0, 255))
                        _ = agent.generate_ml_action(env.state)
                for i in range(3, 0, -1):
                    window_surface.fill((255, 255, 255))
                    vu.draw_centered_text(window_surface, "Game starts in...", str(i), color=(255, 0, 0))
                    pygame.display.flip()
                    start_ticks = pygame.time.get_ticks()
                    while pygame.time.get_ticks() - start_ticks < 1000: # 1초(1000ms) 동안 반복
                        pygame.event.pump() # OS 이벤트 큐를 비워주어 창이 얼어붙지 않게 숨통을 트여줌
                        pygame.time.delay(10) # CPU 과부하 방지용 짧은 휴식

            # 5-2. Step 루프
            clock = pygame.time.Clock()
            if mode == 'exp' and render:
                for t in range(1, horizon + 1):
                    step_start_time = pygame.time.get_ticks()
                    
                    chosen_action = (0, 0)
                    action_chosen = False
                    
                    # --- 설정된 timestep 동안 입력 감지 ---
                    while pygame.time.get_ticks() - step_start_time < game_timestep:
                        clock.tick(60)
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT: pygame.quit(); return
                            if event.type == pygame.KEYDOWN and not action_chosen:
                                key = event.key
                                action = None
                                if key == pygame.K_UP: action = (0, -1)
                                elif key == pygame.K_DOWN: action = (0, 1)
                                elif key == pygame.K_LEFT: action = (-1, 0)
                                elif key == pygame.K_RIGHT: action = (1, 0)
                                elif key == pygame.K_SPACE: action = Action.INTERACT
                                if action: chosen_action = action; action_chosen = True 
                    
                    # --- 행동 실행 (모든 에이전트가 매 timestep마다 action 호출) ---
                    s_t = env.state
                    actions_t = []
                    for agent in agents_list:
                        if isinstance(agent, HumanAgent):
                            agent.set_next_action(chosen_action)
                        actions_t.append(agent.action(s_t)) # AI도 여기서 즉시 action() 수행

                    obs, reward, done, env_info = env.step(tuple(actions_t))
                    r_total += reward
                    
                    # 렌더링
                    thought_idx, thought_msg = get_combined_thought(agents_list)
                    vu.render_game(window_surface, visualizer, env, t, horizon, r_total, 
                                   thought_idx, visual_level, layout_dict, thought_msg, show_intention)
                    
                    if done: break
            
            results.append(r_total)
            progress.update(task_id, advance=1, mean_score=str(int(np.mean(results))))

    end_time = time.time()
    print(f"\n\nTotal Cost time : {end_time - start_time:.3f}s-----\n")

    # 결과 저장 로직 (동일)
    result_dict = {
        "input": variant,
        "raw_results": results,
        "mean_result": int(np.mean(results)),
        "std_result": float(np.std(results))
    }
    
    # ... (저장 로직 기존과 동일) ...
    if variant['save']:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if variant['log_dir'] is None:
            log_dir = f"experiments/{timestamp}_{layout_name}_{horizon}_{p0_algo}_{p1_algo}_{episode}numep"
        else:
            log_dir = variant['log_dir']

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        print(f"Result saved to {log_dir}")
        safe_model_name = str(variant.get('gpt_model', 'model')).replace('/', '-')
        
        filename = f"results_{episode}_{horizon}"
        if p0_algo in LLM_AGENT_TYPES or p1_algo in LLM_AGENT_TYPES:
            filename += f"_{safe_model_name}_{variant['prompt_level']}"
        filename += ".json"
            
        with open(os.path.join(log_dir, filename), "w") as f:
            json.dump(result_dict, f, indent=4)


if __name__ == '__main__':
    parser = ArgumentParser(description='OvercookedAI Experiment')

    # Basic Parsers
    parser.add_argument('--layout', '-l', type=str, default='cramped_room', 
                        choices=['cramped_room', 'asymmetric_advantages', 'coordination_ring', 'forced_coordination', 'counter_circuit'])
    parser.add_argument('--p0',  type=str, default='ProAgent', help='Algorithm for P0')
    parser.add_argument('--p1', type=str, default='Human', help='Algorithm for P1')
    parser.add_argument('--horizon', type=int, default=400, help='Horizon steps')
    parser.add_argument('--cook_time', type=int, default=20)
    parser.add_argument('--episode', type=int, default=1, help='Number of episodes')
    parser.add_argument('--render', type=boolean_argument, default=True, help='Visualization on/off')
    parser.add_argument('--visual_level', type=int, default=2, help='0: baseline, 1: emoji, 2: NL, 3: highlight')
    parser.add_argument('--show_intention', type=boolean_argument, default=True, help='Show Intention bubble')

    # [NEW] Global Timestep (Default 400ms = Same speed as human)
    parser.add_argument('--timestep', type=int, default=400, help='Global timestep in ms')

    # LLM Agent Parsers
    parser.add_argument('--gpt_model', type=str, default='Qwen/Qwen3-VL-8B-Instruct', help='Model name')
    parser.add_argument('--prompt_level', '-pl', type=str, default='l3-aip', choices=['l1-p', 'l2-ap', 'l3-aip'])
    parser.add_argument('--belief_revision', '-br', type=boolean_argument, default=False)
    parser.add_argument('--retrival_method', type=str, default="recent_k", choices=['recent_k', 'bert_topk'])
    parser.add_argument('--K', type=int, default=1)

    # Misc Parsers
    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'demo'])                                
    parser.add_argument('--save', type=boolean_argument, default=True)
    parser.add_argument('--log_dir', type=str, default=None)

    args = parser.parse_args()
    variant = vars(args)

    main(variant)