"""
=todo=
프롬프팅 수정
#study 실시간 / 비실시간 분리하기
#U누르고 로딩.
#not responding 해결

2. Instruction, 연습공간 - 싱글 플레이? 협동 플레이?
3. 벤치마크 시스템 구축 - Manhatten vs 절대 좌표 비교 - 만들어봤긴한데 서버 고치고 다시 해봐야 할 듯.
로그따기
counter_circuit의 경우에서, 왼쪽 / 오른쪽이 아님. 다시 프롬프팅하기.

highlight에 물음표 넣기
실시간성 있는 거 / 없는 거 구현 - 비실시간이 사람만?

맵순서


=실험 설계=
    =정해야 할 질문들=
        맵3개시킬지 4개 시킬지
    instruction에서 추론근거 설명하기
    1 experiment안에 여러개의 block (맵 1개가 1개의 block), 각 block마다 쉬는시간.
    맵 순서 -> 랜덤
    의도 표시는 Balanced latin square 사용
    law Log 따두기 (필요한 것: 어떤 의도 표현 방식인지, 각 timestep, object 상태, player 상태, 행동, reward)
    사후 설문 - Cognitive load - nasa xls(매번) + 선호도(block당), 얼마나 도움이 되었는가(매번) + agency: agent를 내 마음대로 움직일 수 있었다.
        Agency 찾아보기
        1block마다 사후 설문 받기

=완성사항=
#cramped_room 프롬프트 수정
#생각하는 동안 / 기다리는 동안은 상하좌우 랜덤행동으로
#충돌구현(EIRA가 움직였을 때 같은 위치로 이동한다면, 한번 wait하기)
#혼자 말하는 것처럼 수정하기.
#색깔 지우기 / left, right onion으로 수정
#object에 bold
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
import time
import datetime
import json
import numpy as np
import pygame
import threading
import sys
import platform
import visualization_utils as vu
from argparse import ArgumentParser
from distutils.util import strtobool
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.agent import AgentGroup, Agent
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TARGET_SCORES = {'simple': 60, 'cramped_room': 60, 'asymmetric_advantages': 100, 'coordination_ring': 60, 'counter_circuit': 60}

def boolean_argument(value):
    return bool(strtobool(value))

def get_mdp(layout_name):
    try: return OvercookedGridworld.from_layout_name(layout_name)
    except: return OvercookedGridworld.from_grid(["XXXPX", "O   D", "X1 2X", "X   X", "XXSXX"])

class OvercookedLogger:
    def __init__(self, log_dir="experiments/logs", variant=None):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        layout = variant.get('layout', 'unknown') if variant else 'unknown'
        cond = variant.get('name', 'unknown') if variant else 'unknown'
        self.filepath = os.path.join(self.log_dir, f"log_{timestamp}_{layout}_{cond}.jsonl")
        if variant: self._write_line({"metadata": variant})
    def log_step(self, step_data): self._write_line(step_data)
    def _write_line(self, data):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

class HumanAgent(Agent):
    def __init__(self): super().__init__(); self.next_action = None 
    def set_next_action(self, action): self.next_action = action
    def action(self, state):
        a = self.next_action if self.next_action is not None else Action.STAY
        self.next_action = None
        return a

def get_combined_thought(agents_list):
    for i, agent in enumerate(agents_list):
        if hasattr(agent, 'current_thought') and agent.current_thought: return i, agent.current_thought
    return -1, None

def get_parser():
    p = ArgumentParser()
    p.add_argument('--layout', type=str, default='simple'); p.add_argument('--p0', default='Human'); p.add_argument('--p1', default='EIRAAsync')
    p.add_argument('--horizon', type=int, default=40); p.add_argument('--episode', type=int, default=1)
    p.add_argument('--render', type=boolean_argument, default=True); p.add_argument('--visual_level', type=int, default=2)
    p.add_argument('--async', dest='async', type=boolean_argument, default=True)
    p.add_argument('--timestep', type=int, default=400); p.add_argument('--log_dir', default=None); p.add_argument('--name', default='unknown')
    p.add_argument('--gpt_model', default='Qwen/Qwen3-VL-8B-Instruct'); p.add_argument('--prompt_level', default='l3-aip'); p.add_argument('--show_intention', type=boolean_argument, default=True)
    p.add_argument('--cook_time', type=int, default=20)
    p.add_argument('--save', type=boolean_argument, default=True)
    return p

def main(variant, surface=None):
    layout_name = variant.get('layout', 'cramped_room')
    horizon = variant.get('horizon', 400); render = variant.get('render', True)
    visual_level = variant.get('visual_level', 2); async_mode = variant.get('async', True)
    game_timestep = variant.get('timestep', 400); target_score = TARGET_SCORES.get(layout_name, 60)
    
    mdp = get_mdp(layout_name); layout_dict = vu.generate_layout_dict(mdp)
    env = OvercookedEnv(mdp, horizon=horizon); window_surface = surface
    if render and window_surface is None: window_surface = pygame.display.set_mode((900, 600))
    visualizer = StateVisualizer(cook_time=variant.get('cook_time', 20))

    agents_list = []
    for i, alg in enumerate([variant.get('p0', 'Human'), variant.get('p1', 'EIRAAsync')]):
        if alg == "Human": agent = HumanAgent()
        else: agent = make_agent(alg, mdp, layout_name, model=variant.get('gpt_model', 'Qwen/Qwen3-VL-8B-Instruct'), prompt_level=variant.get('prompt_level', 'l3-aip'))
        
        # 💡 [에러 해결] 필요한 속성 강제 초기화
        agent.set_agent_index(i)
        agent.async_mode = async_mode
        if not hasattr(agent, 'teammate_intentions_dict'): agent.teammate_intentions_dict = {}
        if not hasattr(agent, 'current_timestep'): agent.current_timestep = 0
        
        agents_list.append(agent)
    
    # 💡 [수정] duration 값을 저장할 배열 추가
    results_score, results_step, results_col, results_dur = [], [], [], []
    for ep in range(variant.get('episode', 1)):
        logger = OvercookedLogger(log_dir=variant.get('log_dir', 'experiments/logs'), variant=variant)
        env.reset(); r_total, inter_col_count = 0, 0; final_t = horizon
        
        actual_start_time = time.time() # 만약을 대비한 기본값 초기화
        
        if render:
            vu.draw_centered_text(window_surface, "잠시만 기다려주세요...", "AI가 첫 번째 행동을 분석하고 있습니다.", color=(255, 200, 0))
            pygame.display.flip()
            
            init_threads = []
            def init_agent(agent):
                if hasattr(agent, 'generate_ml_action'): 
                    agent.generate_ml_action(env.state)
            
            for a in agents_list:
                th = threading.Thread(target=init_agent, args=(a,))
                init_threads.append(th)
                th.start()
                
            # 초기 모델 추론 대기 중 응답 없음 방지
            while any(th.is_alive() for th in init_threads):
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT: 
                        pygame.quit()
                        sys.exit()
                pygame.event.pump()
                pygame.time.delay(10)
                
            for th in init_threads:
                th.join()

            # 💡 [핵심] AI 초기화 직후 첫 화면을 한 번 그려주고, 입력 대기 안내 표시
            idx, msg = get_combined_thought(agents_list)
            vu.render_game(window_surface, visualizer, env, 0, target_score, r_total, idx, visual_level, layout_dict, msg, variant.get('show_intention', True))
            
            # OS별 한글 폰트 감지 후 안내 텍스트 렌더링
            os_name = platform.system()
            if os_name == 'Windows': korean_font_name = 'malgungothic'
            elif os_name == 'Darwin': korean_font_name = 'applegothic'
            else: korean_font_name = 'nanumgothic'
            
            guide_font = pygame.font.SysFont(korean_font_name, 24, bold=True)
            guide_surf = guide_font.render("방향키나 스페이스바를 누르면 시작됩니다.", True, (255, 50, 50))
            guide_rect = guide_surf.get_rect(center=(450, 60))
            window_surface.blit(guide_surf, guide_rect)
            pygame.display.flip()

        for t in range(1, horizon + 1):
            # 💡 매 스텝 에이전트의 타임스텝 업데이트
            for a in agents_list: a.current_timestep = t
            
            start_t = pygame.time.get_ticks(); human_act = Action.STAY; chosen = False
            if async_mode:
                # 💡 [핵심] 첫 번째 스텝(t==1)일 때는 시간 제한 없이 무한 대기
                if t == 1:
                    while not chosen:
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT: pygame.quit(); return 0,0,0,0
                            if ev.type == pygame.KEYDOWN:
                                if ev.key == pygame.K_UP: human_act = Direction.NORTH; chosen = True
                                elif ev.key == pygame.K_DOWN: human_act = Direction.SOUTH; chosen = True
                                elif ev.key == pygame.K_LEFT: human_act = Direction.WEST; chosen = True
                                elif ev.key == pygame.K_RIGHT: human_act = Direction.EAST; chosen = True
                                elif ev.key == pygame.K_SPACE: human_act = Action.INTERACT; chosen = True
                                
                                # 💡 유효한 키가 눌려 chosen이 True가 된 순간 측정 시작
                                if chosen: actual_start_time = time.time()
                        pygame.time.delay(10)
                else:
                    while pygame.time.get_ticks() - start_t < game_timestep:
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT: pygame.quit(); return 0,0,0,0
                            if ev.type == pygame.KEYDOWN and not chosen:
                                if ev.key == pygame.K_UP: human_act = Direction.NORTH; chosen = True
                                elif ev.key == pygame.K_DOWN: human_act = Direction.SOUTH; chosen = True
                                elif ev.key == pygame.K_LEFT: human_act = Direction.WEST; chosen = True
                                elif ev.key == pygame.K_RIGHT: human_act = Direction.EAST; chosen = True
                                elif ev.key == pygame.K_SPACE: human_act = Action.INTERACT; chosen = True
                        pygame.time.delay(1)
            else:
                idx, msg = get_combined_thought(agents_list)
                vu.render_game(window_surface, visualizer, env, t, target_score, r_total, idx, visual_level, layout_dict, msg, variant.get('show_intention', True))
                while not chosen:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT: pygame.quit(); return 0,0,0,0
                        if ev.type == pygame.KEYDOWN:
                            if ev.key == pygame.K_UP: human_act = Direction.NORTH; chosen = True
                            elif ev.key == pygame.K_DOWN: human_act = Direction.SOUTH; chosen = True
                            elif ev.key == pygame.K_LEFT: human_act = Direction.WEST; chosen = True
                            elif ev.key == pygame.K_RIGHT: human_act = Direction.EAST; chosen = True
                            elif ev.key == pygame.K_SPACE: human_act = Action.INTERACT; chosen = True
                            
                            # 💡 비실시간 환경에서도 첫 스텝의 액션 시점에 측정 시작
                            if chosen and t == 1: actual_start_time = time.time()
                    pygame.time.delay(10)

            actions = [None] * len(agents_list)
            action_threads = []
            
            def fetch_action(index, agent):
                if isinstance(agent, HumanAgent): 
                    agent.set_next_action(human_act)
                    actions[index] = agent.action(env.state)
                else: 
                    actions[index] = agent.action(env.state, partner_action=human_act)

            for idx, a in enumerate(agents_list):
                th = threading.Thread(target=fetch_action, args=(idx, a))
                action_threads.append(th)
                th.start()
                
            # 에이전트 행동 추론(API 대기 등) 중 응답 없음 방지
            while any(th.is_alive() for th in action_threads):
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT: 
                        pygame.quit()
                        sys.exit()
                pygame.event.pump()
                pygame.time.delay(10)
                
            for th in action_threads:
                th.join()
            
            old_p0, old_p1 = env.state.players[0].position, env.state.players[1].position
            int_p0 = (old_p0[0] + actions[0][0], old_p0[1] + actions[0][1]) if actions[0] in Direction.ALL_DIRECTIONS else old_p0
            int_p1 = (old_p1[0] + actions[1][0], old_p1[1] + actions[1][1]) if actions[1] in Direction.ALL_DIRECTIONS else old_p1
            is_col = (int_p0 == int_p1 and (actions[0] != Action.STAY or actions[1] != Action.STAY)) or \
                     (int_p0 == old_p1 and int_p1 == old_p0 and actions[0] != Action.STAY and actions[1] != Action.STAY) or \
                     (actions[0] in Direction.ALL_DIRECTIONS and int_p0 == old_p1 and actions[1] not in Direction.ALL_DIRECTIONS) or \
                     (actions[1] in Direction.ALL_DIRECTIONS and int_p1 == old_p0 and actions[0] not in Direction.ALL_DIRECTIONS)
            if is_col: inter_col_count += 1

            _, reward, is_tout, _ = env.step(tuple(actions)); r_total += reward
            done = is_tout or (r_total >= target_score); idx, msg = get_combined_thought(agents_list)
            logger.log_step({"timestep": t, "inter_collision": is_col, "reward": reward, "cumulative_reward": r_total})
            if render and async_mode: vu.render_game(window_surface, visualizer, env, t, target_score, r_total, idx, visual_level, layout_dict, msg, variant.get('show_intention', True))
            if done: final_t = t; break
            
        # 💡 에피소드 종료 시점 측정
        ep_duration = time.time() - actual_start_time
        results_score.append(r_total); results_step.append(final_t); results_col.append(inter_col_count); results_dur.append(ep_duration)
        
    # 💡 [수정] duration(시간)도 함께 반환
    return int(np.mean(results_score)), int(np.mean(results_step)), int(np.mean(results_col)), np.mean(results_dur)

if __name__ == '__main__':
    parser = get_parser(); args = parser.parse_args(); main(vars(args))