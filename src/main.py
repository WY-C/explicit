"""
todo
=연구=
1. 어떤 실험들을 할 것인가 + 이 실험을 하기 위해서 어떤 연구, 구현이 필요한 가를 정리하기.
 - 사람이 누르지 않을 때도 시간이 가게 할 것인가?
 - 자연어로 어떻게 표현할 것인가
 - 이모지로 어떻게 표현할 것인가
 - 어떤 맵을 사용할 것인가

 - 실험자는 몇 명이 필요한가?

 - 비교대상: 자연어 / 이모지 / 없음

=논문에서의 참조사항=
LLM + 사람 = 400ms + 100초 (250step) 
Agent(FCP) + 사람 = 200ms + 60초 (300step) 요리시간: 20step
-> 약 10step으로 줄인다?


 - 400 time steps 기본적으로 사용함 - 주로 agent끼리 게임했을때의 결과임.
 - 100초 동안 진행된다 + 약 2.5Hz

2. “벤치마크와 proagent 점수 (내가 수정한 proagent의 점수)”
-> 기존 proagent의 코드 + greedy agent, self-play, fcp, human proxy와 비교하기.
- 고민해볼 사항: 현재 병렬적으로 움직이게 되어있는데, 기존의 코드에서도 병렬적으로 움직이게 한 다음에 하기? - 생각해보니, 모두가 동기되면서 움직이는 게 (기다리면서) 맞는 듯.


3. Proagent와 사람이 실험할 수 있는 세팅 (관련 연구를 더 찾아보기) - 실시간 overcookedAI UI 좀 찾아보기.

4. 프롬프팅 줄이기 (일반적인 형태로 변환하기)
=구현=
“사람이 움직이는 것과 agent가 움직이는 sync 맞추기 (agent만 너무 빠르게 움직이는 것 방지하기)” <- 구현 완료

유저가 누를때마다 step 전환 + 250ms의 delay : 구현 완료. 

방법: 맵 상에서 보여주기, 표현

컨텐츠: 내가 이렇게 하겠다. 너가 이렇게 할 것이다.

(논문을 좀 찾아보기)

액션의 순서 - 플레이어의 쿼리를 받고나서, 행동 할 것을 보고 LLM 쿼리에 넣기.
즉, 현재 LLM의 프롬프트에서
    1. 물품들의 위치
    2. 플레이어 전 스텝의 위치
    3. 플레이어 현 스텝의 위치
    4. 플레이어의 행동
을 추가해서 넘겨주어야 함.
완료
행동 생각중, 로딩 중이라는 글자 표시하기.


Proagent의 exmaple생성방식 찾아보기





이모티콘
자연어

수정사항들
cook time argument -> 요리가 완료됨을 추가하는 argument + layout가서 따로 수정해주기

"""
import time
import datetime
import os
import json
from argparse import ArgumentParser
import numpy as np

# Rich 라이브러리 임포트
from rich import print as rprint
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeRemainingColumn
)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*") 

from distutils.util import strtobool
def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

import importlib_metadata
try:
    VERSION = importlib_metadata.version("overcooked_ai")
except:
    VERSION = "0.0.1"

import pygame
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

print(f'\n----This overcook version is {VERSION}----\n')

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentGroup
from overcooked_ai_py.mdp.actions import Action

from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent

from overcooked_ai_py.agents.agent import Agent

# [수정됨] HumanAgent: set_next_action 구조 사용
class HumanAgent(Agent):
    def __init__(self):
        super().__init__()
        self.next_action = None 

    def set_next_action(self, action):
        """키 입력 이벤트에서 호출되어 행동을 저장함"""
        self.next_action = action

    def action(self, state):
        """환경이 step을 진행할 때 호출됨. 저장된 행동을 반환하고 초기화."""
        if self.next_action is not None:
            a = self.next_action
            self.next_action = None
            return a
        return (0,0)
    
def render_game(window, visualizer, env, step, horizon, reward, thought_msg=None, is_thinking=False):
    if not window or not visualizer:
        return

    # 1. 배경 및 게임 화면 (하단 배치)
    window.fill((255, 255, 255)) 
    screen_width, screen_height = window.get_size()

    state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
    surf_width, surf_height = state_surface.get_size()
    
    start_x = (screen_width - surf_width) // 2
    start_y = screen_height - surf_height - 30 
    if start_y < 300: start_y = 300 

    window.blit(state_surface, (start_x, start_y))

    # 2. 상단 상태 정보
    font = pygame.font.SysFont("malgungothic", 30) 
    text = font.render(f"Step: {step}/{horizon} | Reward: {reward}", True, (0, 0, 0))
    window.blit(text, (10, 10))

    # 3. [핵심] 텍스트 내용을 아이콘으로 '완전 대체'하여 그리기
    if thought_msg:
        # 메시지 정리
        if "Intention for Player" in thought_msg:
            thought_msg = thought_msg[thought_msg.find("Intention for Player"):]
        if "Plan for Player 0" in thought_msg:
            thought_msg = thought_msg.replace("Plan for Player 0", "\nPlan for Agent")
        
        lines = thought_msg.split('\n')
        
        # 폰트 및 설정
        bubble_font = pygame.font.SysFont("malgungothic", 28) 
        text_color = (0, 0, 0)
        
        icon_size = 45    
        line_height = 55  
        current_y = 60    
        left_margin = 10  

        # 검색할 키워드 순서 중요 (긴 단어 우선 혹은 중요도 순)
        # 키워드가 발견되면 해당 이미지를 매핑
        icon_map = {
            "onion": "onion", "양파": "onion",
            #"tomato": "tomato", "토마토": "tomato",
            "dish": "dish", "접시": "dish",
            #"pot": "pot", "냄비": "pot",
            #"soup": "soup_done", "수프": "soup_done",
            #"serve": "serve" # serve 관련 텍스트가 있을 경우
        }

        for line in lines:
            line_lower = line.lower()
            sprite_name = None
            
            # 1. 문장 안에 재료/도구 키워드가 있는지 확인
            for key, s_name in icon_map.items():
                if key in line_lower:
                    sprite_name = s_name
                    break # 하나 찾으면 중단 (예: onion 찾으면 끝)
            
            # 2. 라벨(Header) 추출 (콜론 앞부분만 따오기)
            # 예: 'Intention for Player 1: "put_onion()"' -> 'Intention for Player 1'
            if ":" in line:
                label_text = line.split(":")[0].strip() + " : "
            else:
                label_text = line # 콜론이 없으면 그냥 통째로 라벨 취급

            # 3. 그리기
            # 텍스트 렌더링 (라벨만)
            text_surf = bubble_font.render(label_text, True, text_color)
            
            # 텍스트 수직 중앙 위치
            text_y_pos = current_y + (line_height - text_surf.get_height()) // 2
            window.blit(text_surf, (left_margin, text_y_pos))
            
            # 4. 아이콘이 있으면 텍스트 바로 뒤에 그림
            if sprite_name:
                icon_x = left_margin + text_surf.get_width() + 10 # 10px 간격
                
                temp_surf = pygame.Surface((visualizer.UNSCALED_TILE_SIZE, visualizer.UNSCALED_TILE_SIZE), pygame.SRCALPHA)
                visualizer.OBJECTS_IMG.blit_on_surface(temp_surf, (0, 0), sprite_name)
                scaled_icon = pygame.transform.scale(temp_surf, (icon_size, icon_size))
                
                icon_y_pos = current_y + (line_height - icon_size) // 2
                window.blit(scaled_icon, (icon_x, icon_y_pos))
            
            # 아이콘이 없는 문장이면(키워드 못 찾음), 뒤에 원래 텍스트 내용을 그냥 흐리게라도 보여줄지, 
            # 아니면 아예 안 보여줄지 결정해야 함. 
            # 여기서는 요청하신 대로 '아이콘이 없으면 그냥 빈칸' 혹은 '원본 텍스트 내용'을 띄웁니다.
            # (현재 코드는 키워드 없으면 라벨만 나옴. 내용을 보고 싶으면 아래 else 주석 해제)
            else:
                 # 키워드를 못 찾았을 때 뒤에 내용을 텍스트로라도 보여주고 싶다면:
                 full_text_surf = bubble_font.render(line, True, text_color)
                 window.blit(full_text_surf, (left_margin, text_y_pos))

            current_y += line_height

    pygame.display.flip()

# def render_game(window, visualizer, env, step, horizon, reward, thought_msg=None, is_thinking=False):
#     """
#     게임 화면, 상태 텍스트, 에이전트 생각, 로딩 표시를 그리는 통합 함수
#     """
#     if not window or not visualizer:
#         return

#     # 1. 기본 배경 및 게임 상태 그리기
#     window.fill((255, 255, 255)) # 흰색 배경
    
#     state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
#     screen_width, screen_height = window.get_size()
#     surf_width, surf_height = state_surface.get_size()
    
#     start_x = (screen_width - surf_width) // 2
#     start_y = (screen_height - surf_height) // 2
#     window.blit(state_surface, (start_x, start_y))

#     # 2. 상단 상태 정보 (Step, Reward)
#     font = pygame.font.SysFont(None, 36)
#     text = font.render(f"Step: {step}/{horizon} | Reward: {reward}", True, (0, 0, 0))
#     window.blit(text, (10, 10))

#     # 3. 에이전트 생각(Thought) 텍스트 출력
#     if thought_msg:
#         # 메시지 파싱 (기존 로직 유지)
#         if "Intention for Player" in thought_msg:
#             thought_msg = thought_msg[thought_msg.find("Intention for Player"):]
#             thought_msg = thought_msg.replace("Intention for Player 1", "Intention for Player")
#         if "Plan for Player 0" in thought_msg:
#             thought_msg = thought_msg.replace("Plan for Player 0", "\nPlan for Agent")
        
#         lines = thought_msg.split('\n')
        
#         # 텍스트 그리기
#         bubble_font = pygame.font.SysFont("malgungothic", 20)
#         current_y = 50
#         for line in lines:
#             text_surf = font.render(line, True, (0, 0, 0), (255, 255, 255))
#             window.blit(text_surf, (10, current_y))
#             current_y += 15 # 줄간격 약간 조정

#     # 4. [핵심] 생각 중 표시 (is_thinking=True 일 때만)
#     if is_thinking:
#         loading_font = pygame.font.SysFont("malgungothic", 40, bold=True)
#         loading_text = font.render("Agent is thinking... ", True, (0, 0, 0)) # 빨간색
        
#         # 화면 중앙 상단에 배치
#         text_rect = loading_text.get_rect(center=(screen_width // 2, 80))
        
#         # 배경에 반투명 박스를 깔아주면 글자가 더 잘 보임 (선택사항)
#         bg_rect = text_rect.inflate(20, 10)
#         pygame.draw.rect(window, (255, 255, 255), bg_rect)
#         pygame.draw.rect(window, (0, 0, 0), bg_rect, 2) # 테두리
        
#         window.blit(loading_text, text_rect)

#     # 5. 화면 업데이트
#     pygame.display.flip()


def main(variant):
    #step_duration = 500
    layout = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']
    mode = variant['mode']
    render = variant['render'] # 렌더링 여부 확인
    cook_time = variant['cook_time']
    
    if VERSION == '1.1.0':
        mdp = OvercookedGridworld.from_layout_name(NEW_LAYOUTS[layout])
    elif VERSION == '0.0.1':
        mdp = OvercookedGridworld.from_layout_name(OLD_LAYOUTS[layout])

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()
    
    # 렌더링 옵션이 켜져 있을 때만 Pygame 초기화
    visualizer = None
    window_surface = None
    if render:
        pygame.init()
        visualizer = StateVisualizer(cook_time = cook_time)
        window_surface = pygame.display.set_mode((800, 600)) 
        pygame.display.set_caption("ProAgent vs Human")
    
    p0_algo = variant['p0']
    p1_algo = variant['p1']
    print(f"\n===P0 agent: {p0_algo} | P1 agent: {p1_algo}===\n")

    start_time = time.time()
    results = []

    # [Rich Progress Bar 설정]
    with Progress(
        SpinnerColumn(),        # 로딩 스피너
        TextColumn("[progress.description]{task.description}"), # 설명 텍스트
        BarColumn(),            # 진행 바
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"), # 퍼센트
        TimeRemainingColumn(),  # 남은 시간
        TextColumn(" | Mean Reward: [bold cyan]{task.fields[mean_score]}"), # 커스텀 필드 (평균 점수)
        transient=False         # 완료 후에도 바를 사라지게 하지 않음
    ) as progress:
        
        task_id = progress.add_task(f"[green]Simulating {episode} Episodes...", total=episode, mean_score="0")
        if variant['p0'] == 'Human' or variant['p1']== 'Human':
            render = True
            
        for i in range(episode):  
            agents_list = []
            for alg in [p0_algo, p1_algo]:
                if alg == "ProAgent" or alg == "MyAgent" or alg == "MyAgentAsync":
                    gpt_model = variant['gpt_model']
                    retrival_method = variant['retrival_method']
                    K = variant['K']
                    prompt_level = variant['prompt_level']
                    belief_revision = variant['belief_revision']
                    agent = make_agent(alg, mdp, layout, model=gpt_model, 
                                       prompt_level=prompt_level, 
                                       belief_revision=belief_revision, 
                                       retrival_method=retrival_method, K=K)
                elif alg == "Human":
                    agent = HumanAgent()
                elif alg == "BC":
                    agent = make_agent(alg, mdp, layout, seed_id=i) # 매 에피소드마다 시드 변경
                else:
                    agent = make_agent(alg, mdp, layout)
                agents_list.append(agent)
            print("agents_list: ", *agents_list)
            team = AgentGroup(*agents_list)
            team.reset()
            env.reset()
            r_total = 0

            clock = pygame.time.Clock()
            if mode == 'exp':
                if (variant['p0'] == 'Human') or (variant['p1'] == 'Human'):
                    has_proagent = 1
                    if variant['p0'] == 'Human':
                        num_human = 0
                    else:
                        num_human = 1
                    if variant['p0'] == 'ProAgent' or variant['p0'] == 'MyAgent' or variant['p0'] == 'MyAgentAsync':
                        num_pro = 0
                    elif variant['p1'] == 'ProAgent'or variant['p1'] == 'MyAgent' or variant['p1'] == 'MyAgentAsync':
                        num_pro = 1
                    else:
                        has_proagent = 0
                    
                    for t in range(horizon):
                        # 1. 스텝 시작 시간 및 변수 초기화
                        step_start_time = pygame.time.get_ticks()
                        step_duration = 400  # 400ms 고정
                        
                        # [중요] event.clear() 제거! (입력 씹힘 방지)
                        # 대신, 이번 턴에 행동을 정했는지 확인하는 플래그 사용
                        action_chosen = False 
                        chosen_action = (0, 0) # 기본값: 정지(Stay)

                        # 2. 렌더링 (첫 프레임)
                        first = True
                        if render and first:
                            first = False
                            if has_proagent == 1:
                                current_thought = agents_list[num_pro].current_thought 
                            else:
                                current_thought = "No ProAgent"
                            render_game(window_surface, visualizer, env, t, horizon, r_total, current_thought, is_thinking=False)
                        
                        # 3. [핵심] 400ms가 지날 때까지 무조건 루프를 돕니다.
                        # 입력을 빨리 했어도 여기서 시간을 떼웁니다 (Non-blocking Wait).
                        while True:
                            # 60FPS 유지
                            clock.tick(60)
                            
                            current_time = pygame.time.get_ticks()
                            elapsed = current_time - step_start_time

                            # 400ms가 지나면 루프 탈출 -> 다음 스텝 진행
                            if elapsed >= step_duration:
                                break

                            # 이벤트 처리
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    return
                                
                                # 키 입력 처리 (이미 이번 턴에 행동을 결정했다면 무시)
                                if event.type == pygame.KEYDOWN and not action_chosen:
                                    key = event.key
                                    action = None
                                    if key == pygame.K_UP: action = (0, -1)
                                    elif key == pygame.K_DOWN: action = (0, 1)
                                    elif key == pygame.K_LEFT: action = (-1, 0)
                                    elif key == pygame.K_RIGHT: action = (1, 0)
                                    elif key == pygame.K_SPACE: action = "interact"
                                    
                                    if action is not None:
                                        chosen_action = action
                                        action_chosen = True 
                                        # [중요] break 하지 않음! 
                                        # 행동은 입력받았지만, 시간(400ms)은 다 채우고 나갑니다.
                        
                        # 4. 루프가 끝나면(400ms 경과) 결정된 행동을 에이전트에 주입
                        # 입력이 없었으면 기본값 (0,0)이 들어감
                        agents_list[num_human].set_next_action(chosen_action)

                        # 5. 환경 업데이트 (Step)
                        s_t = env.state
                        a_t = []
                        a_t.append(team.agents[0].action(s_t))
                        
                        if render:
                            if has_proagent == 1:
                                current_thought = agents_list[num_pro].current_thought
                            render_game(window_surface, visualizer, env, t, horizon, r_total, current_thought, is_thinking=True)
                            pygame.event.pump() 

                        a_t.append(team.agents[1].action(s_t))
                        a_t = tuple(a_t)
                        
                        obs, reward, done, env_info = env.step(a_t)
                        r_total += reward
                        render_game(window_surface, visualizer, env, t, horizon, r_total, current_thought, is_thinking=False)
                        
                        # [삭제] 마지막의 pygame.time.wait()는 이제 필요 없습니다.
                        # 위쪽의 while 루프가 정확히 시간을 맞춰줍니다.
                        
                        if done: break
                else:
                    
                    for t in range(horizon):
                        clock.tick(3)
                        # [렌더링 로직] render=True일 때만 실행
                        if render:
                            state_surface = visualizer.render_state(env.state, grid=env.mdp.terrain_mtx)
                            
                            screen_width, screen_height = window_surface.get_size()
                            surf_width, surf_height = state_surface.get_size()
                            start_x = (screen_width - surf_width) // 2
                            start_y = (screen_height - surf_height) // 2
                            
                            window_surface.fill((255, 255, 255)) 
                            window_surface.blit(state_surface, (start_x, start_y))
                            
                            font = pygame.font.SysFont(None, 36)
                            text = font.render(f"Step: {t}/{horizon} | Reward: {r_total}", True, (0, 0, 0))
                            window_surface.blit(text, (10, 10))        
                    
                        s_t = env.state
                        
                        # 에이전트 행동 결정
                        a_t = team.joint_action(s_t)

                        #print("a_t:", a_t)
                        obs, reward, done, env_info = env.step(a_t)
                        pygame.display.flip()
                        pygame.event.pump() 
                        r_total += reward

            elif mode == 'demo':
                pass
            
            # 결과 저장 및 진행 바 업데이트
            results.append(r_total)
            current_mean = int(np.mean(results))
            
            # 진행 바 한 칸 전진 및 평균 점수 텍스트 갱신
            progress.update(task_id, advance=1, mean_score=str(current_mean))

    # 전체 루프 종료 후
    end_time = time.time()
    print(f"\n\nTotal Cost time : {end_time - start_time:.3f}s-----\n")

    result_dict = {
        "input": variant,
        "raw_results": results,
        "mean_result": int(np.mean(results)),
        "std_result": float(np.std(results))
    }
    
    # 결과 요약 출력
    for (k,v) in result_dict.items():
        if k != "raw_results":
            print(f'{k}: {v}')

    # 파일 저장
    if variant['save']:
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        if variant['log_dir'] == None:
            log_dir = f"experiments/{timestamp}_{layout}_{horizon}_{p0_algo}_{p1_algo}_{episode}numep"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        else:
            log_dir = variant['log_dir']

        print(f"Result saved to {log_dir}")
        safe_model_name = str(variant.get('gpt_model', 'model')).replace('/', '-')
        
        if p0_algo == "ProAgent"  or p1_algo == "ProAgent":
            json_file = f"{log_dir}/results_{episode}_{horizon}_{safe_model_name}_{variant['prompt_level']}.json"
        else:
            json_file = f"{log_dir}/results_{episode}_{horizon}.json"
            
        with open(json_file, "w") as f:
            json.dump(result_dict, f, indent=4)

    
if __name__ == '__main__':
    parser = ArgumentParser(description='OvercookedAI Experiment')

    # Basic Parsers
    parser.add_argument('--layout', '-l', type=str, default='cramped_room', choices=['cramped_room', 'asymmetric_advantages', 'coordination_ring', 'forced_coordination', 'counter_circuit'])
    parser.add_argument('--p0',  type=str, default='ProAgent', help='Algorithm for P0 agent 0')
    parser.add_argument('--p1', type=str, default='Greedy', help='Algorithm for P1 agent 1')
    parser.add_argument('--horizon', type=int, default=400, help='Horizon steps in one game')
    parser.add_argument('--cook_time', type=int, default=20)
    parser.add_argument('--episode', type=int, default=1, help='Number of episodes')
    parser.add_argument('--render', type=boolean_argument, default=True, help='Visualization on/off')

    # ProAgent Parsers
    parser.add_argument('--gpt_model', type=str, default='Qwen/Qwen3-VL-8B-Instruct', help='Model name')
    parser.add_argument('--api_base', type=str, default='http://localhost:8000/v1', help='Local LLM API URL')
    parser.add_argument('--api_key', type=str, default='EMPTY', help='API Key')
    parser.add_argument('--prompt_level', '-pl', type=str, default='l3-aip', choices=['l1-p', 'l2-ap', 'l3-aip'])
    parser.add_argument('--belief_revision', '-br', type=boolean_argument, default=False)
    parser.add_argument('--retrival_method', type=str, default="recent_k", choices=['recent_k', 'bert_topk'])
    parser.add_argument('--K', type=int, default=1)

    # Misc Parsers
    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'demo'])                                
    parser.add_argument('--save', type=boolean_argument, default=True)
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--debug', type=boolean_argument, default=True)

    

    args = parser.parse_args()
    variant = vars(args)

    main(variant)