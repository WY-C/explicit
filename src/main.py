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

# 에이전트 재사용을 위한 전역 캐시
GLOBAL_AGENT_CACHE = {}

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

# ==========================================
# 💡 에이전트 정신 차리게 만드는 초기화 함수
# ==========================================
def reset_agent_internals(agent):
    """에이전트의 내부 상태를 완전히 새 게임 상태로 초기화"""
    if agent is None or isinstance(agent, HumanAgent):
        return
    
    # 1. 렌더링용 텍스트 초기화
    if hasattr(agent, 'current_thought'): 
        agent.current_thought = None
    
    # 2. 타임스텝 및 대화 기록 초기화
    agent.current_timestep = 0
    if hasattr(agent, 'history'): 
        agent.history = []  # LLM 대화 내역 삭제
    if hasattr(agent, 'messages'): 
        agent.messages = [] # API 호출용 메시지 삭제
        
    # 3. 협동 관련 데이터 초기화
    if hasattr(agent, 'teammate_intentions_dict'):
        agent.teammate_intentions_dict = {}
    if hasattr(agent, 'last_action'):
        agent.last_action = Action.STAY
        
    # 4. 에이전트 클래스 자체에 reset 메소드가 있다면 실행
    if hasattr(agent, 'reset'):
        try:
            agent.reset()
        except:
            pass
def render_loading_screen(surface, message, dot_count):
    surface.fill((40, 44, 52)) # 깔끔한 다크 테마 배경
    font = pygame.font.SysFont("arial", 40, bold=True)
    dots = "." * dot_count
    text_surf = font.render(f"{message}{dots}", True, (255, 255, 255))
    text_rect = text_surf.get_rect(center=(surface.get_width()//2, surface.get_height()//2 - 20))
    surface.blit(text_surf, text_rect)

    sub_font = pygame.font.SysFont("arial", 24)
    sub_surf = sub_font.render("Loading LLM weights and Environment... Please wait.", True, (171, 178, 191))
    sub_rect = sub_surf.get_rect(center=(surface.get_width()//2, surface.get_height()//2 + 30))
    surface.blit(sub_surf, sub_rect)
    pygame.display.flip()
def get_parser():
    p = ArgumentParser()
    p.add_argument('--layout', type=str, default='simple'); p.add_argument('--p0', default='Human'); p.add_argument('--p1', default='EIRAAsync')
    p.add_argument('--horizon', type=int, default=400); p.add_argument('--episode', type=int, default=1)
    p.add_argument('--render', type=boolean_argument, default=True); p.add_argument('--visual_level', type=int, default=2)
    p.add_argument('--async', dest='async', type=boolean_argument, default=True)
    p.add_argument('--timestep', type=int, default=400); p.add_argument('--log_dir', default=None); p.add_argument('--name', default='unknown')
    p.add_argument('--gpt_model', default='Qwen/Qwen3-VL-8B-Instruct'); p.add_argument('--prompt_level', default='l3-aip'); p.add_argument('--show_intention', type=boolean_argument, default=True)
    p.add_argument('--cook_time', type=int, default=20)
    p.add_argument('--save', type=boolean_argument, default=True)
    return p

def main(variant, surface=None):
    pygame.init()
    layout_name = variant.get('layout', 'cramped_room')
    horizon = variant.get('horizon', 400)
    render = variant.get('render', True)
    visual_level = variant.get('visual_level', 2)
    async_mode = variant.get('async', True)
    game_timestep = variant.get('timestep', 400)
    target_score = TARGET_SCORES.get(layout_name, 60)

    mdp = get_mdp(layout_name); layout_dict = vu.generate_layout_dict(mdp)
    env = OvercookedEnv(mdp, horizon=horizon); window_surface = surface
    
    if render and window_surface is None: window_surface = pygame.display.set_mode((1280, 800))
    visualizer = StateVisualizer(cook_time=variant.get('cook_time', 20))

    agents_list = []
    loading_complete = False
    
    def load_agents_task():
        nonlocal agents_list, loading_complete
        for i, alg in enumerate([variant.get('p0', 'Human'), variant.get('p1', 'EIRAAsync')]):
            if alg == "Human": 
                agent = HumanAgent()
            else:
                cache_key = (layout_name, alg, variant.get('gpt_model'))
                if cache_key in GLOBAL_AGENT_CACHE:
                    print(f"[*] Reusing and Resetting cached agent: {layout_name}-{alg}")
                    agent = GLOBAL_AGENT_CACHE[cache_key]
                    reset_agent_internals(agent)
                else:
                    print(f"[*] Creating new agent: {layout_name}-{alg}")
                    agent = make_agent(alg, mdp, layout_name, 
                                       model=variant.get('gpt_model', 'Qwen/Qwen3-VL-8B-Instruct'), 
                                       prompt_level=variant.get('prompt_level', 'l3-aip'))
                    GLOBAL_AGENT_CACHE[cache_key] = agent
            
            agent.set_agent_index(i)
            agent.async_mode = async_mode
            agents_list.append(agent)
        loading_complete = True

    if render:
        loader_thread = threading.Thread(target=load_agents_task)
        loader_thread.start()
        
        dot_timer = 0
        dots = 0
        # 로딩이 끝날 때까지 이벤트 펌핑하며 애니메이션 재생
        while not loading_complete:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: 
                    pygame.quit()
                    sys.exit()
            
            if time.time() - dot_timer > 0.5:
                dots = (dots + 1) % 4
                dot_timer = time.time()
            
            render_loading_screen(window_surface, "Initializing AI Models", dots)
            pygame.time.delay(50)
            
        loader_thread.join()
        pygame.display.set_caption("EIRA Agent - Overcooked AI") # 로딩 끝나면 제목 변경
    else:
        # 렌더링을 안 할 때는 그냥 일반 동기 로딩
        load_agents_task()

    # ==========================================
    
    ai_idx = 1
    for i, a in enumerate(agents_list):
        if not isinstance(a, HumanAgent):
            ai_idx = i
            break

    results_score, results_step, results_col, results_dur = [], [], [], []
    for ep in range(variant.get('episode', 1)):
        logger = OvercookedLogger(log_dir=variant.get('log_dir', 'experiments/logs'), variant=variant)
        env.reset(); r_total, inter_col_count = 0, 0; final_t = horizon
        
        # 매 에피소드 시작 시 인덱스와 타임스텝 재확인
        for i, a in enumerate(agents_list):
            reset_agent_internals(a)
            a.set_agent_index(i)
        
        actual_start_time = time.time() 
        
        if render:
            vu.draw_centered_text(window_surface, "Please wait...", "AI is analyzing the first action.", color=(255, 200, 0))
            pygame.display.flip()
            
            init_threads = []
            def init_agent(agent):
                if hasattr(agent, 'generate_ml_action'): 
                    agent.generate_ml_action(env.state)
            
            for a in agents_list:
                th = threading.Thread(target=init_agent, args=(a,))
                init_threads.append(th)
                th.start()
                
            while any(th.is_alive() for th in init_threads):
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT: 
                        pygame.quit()
                        sys.exit()
                pygame.event.pump()
                pygame.time.delay(10)
                
            for th in init_threads:
                th.join()

            _, msg = get_combined_thought(agents_list)
            vu.render_game(window_surface, visualizer, env, 0, target_score, r_total, ai_idx, visual_level, layout_dict, msg, variant.get('show_intention', True), pid=variant.get('pid'), trial=variant.get('trial'), condition=variant.get('condition'))
            
            guide_font = pygame.font.SysFont("arial", 24, bold=True)
            guide_surf = guide_font.render("Press arrow keys or SPACE to start.", True, (255, 50, 50))
            guide_rect = guide_surf.get_rect(center=(640, 60))
            window_surface.blit(guide_surf, guide_rect)
            pygame.display.flip()

        for t in range(1, horizon + 1):
            for a in agents_list: a.current_timestep = t
            
            start_t = pygame.time.get_ticks(); human_act = Action.STAY; chosen = False
            if async_mode:
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
                _, msg = get_combined_thought(agents_list)
                vu.render_game(window_surface, visualizer, env, t, target_score, r_total, ai_idx, visual_level, layout_dict, msg, variant.get('show_intention', True), pid=variant.get('pid'), trial=variant.get('trial'), condition=variant.get('condition'))
                while not chosen:
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT: pygame.quit(); return 0,0,0,0
                        if ev.type == pygame.KEYDOWN:
                            if ev.key == pygame.K_UP: human_act = Direction.NORTH; chosen = True
                            elif ev.key == pygame.K_DOWN: human_act = Direction.SOUTH; chosen = True
                            elif ev.key == pygame.K_LEFT: human_act = Direction.WEST; chosen = True
                            elif ev.key == pygame.K_RIGHT: human_act = Direction.EAST; chosen = True
                            elif ev.key == pygame.K_SPACE: human_act = Action.INTERACT; chosen = True
                            if chosen and t == 1: actual_start_time = time.time()
                    pygame.time.delay(10)

            actions = [None] * len(agents_list)
            action_threads = []
            def fetch_action(index, agent):
                if isinstance(agent, HumanAgent): 
                    agent.set_next_action(human_act)
                    actions[index] = agent.action(env.state)
                else: actions[index] = agent.action(env.state, partner_action=human_act)

            for idx, a in enumerate(agents_list):
                th = threading.Thread(target=fetch_action, args=(idx, a))
                action_threads.append(th); th.start()
            while any(th.is_alive() for th in action_threads):
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT: pygame.quit(); sys.exit()
                pygame.event.pump(); pygame.time.delay(10)
            for th in action_threads: th.join()
            
            old_p0, old_p1 = env.state.players[0].position, env.state.players[1].position
            int_p0 = (old_p0[0] + actions[0][0], old_p0[1] + actions[0][1]) if actions[0] in Direction.ALL_DIRECTIONS else old_p0
            int_p1 = (old_p1[0] + actions[1][0], old_p1[1] + actions[1][1]) if actions[1] in Direction.ALL_DIRECTIONS else old_p1
            is_col = (int_p0 == int_p1 and (actions[0] != Action.STAY or actions[1] != Action.STAY)) or \
                     (int_p0 == old_p1 and int_p1 == old_p0 and actions[0] != Action.STAY and actions[1] != Action.STAY) or \
                     (actions[0] in Direction.ALL_DIRECTIONS and int_p0 == old_p1 and actions[1] not in Direction.ALL_DIRECTIONS) or \
                     (actions[1] in Direction.ALL_DIRECTIONS and int_p1 == old_p0 and actions[0] not in Direction.ALL_DIRECTIONS)
            if is_col: inter_col_count += 1

            try:
                _, reward, is_tout, _= env.step(actions)
            except ValueError as e:
                print(f"[Tutorial Env Error] Step failed: {e}. Attempting recovery with STAY.")
                _, reward, is_tout, _ = env.step((actions[0], Action.STAY))
                agents_list[1].generate_ml_action(env.state)
            
            r_total += reward
            done = is_tout or (r_total >= target_score); _, msg = get_combined_thought(agents_list)
            
            logger.log_step({
                "timestep": t, "p0_action": str(actions[0]), "p1_action": str(actions[1]), 
                "inter_collision": is_col, "reward": reward, "cumulative_reward": r_total
            })
            
            if render and async_mode: vu.render_game(window_surface, visualizer, env, t, target_score, r_total, ai_idx, visual_level, layout_dict, msg, variant.get('show_intention', True), pid=variant.get('pid'), trial=variant.get('trial'), condition=variant.get('condition'))
            if done: final_t = t; break
            
        ep_duration = time.time() - actual_start_time
        results_score.append(r_total); results_step.append(final_t); results_col.append(inter_col_count); results_dur.append(ep_duration)
        
    return int(np.mean(results_score)), int(np.mean(results_step)), int(np.mean(results_col)), np.mean(results_dur)

if __name__ == '__main__':
    parser = get_parser(); args = parser.parse_args(); main(vars(args))