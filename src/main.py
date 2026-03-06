import warnings
import os
import time
import datetime
import json
import numpy as np
import pygame
import visualization_utils as vu
from argparse import ArgumentParser
from distutils.util import strtobool
from rich import print as rprint
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

import importlib_metadata
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.agents.agent import AgentGroup, Agent
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent

# 환경 설정
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    VERSION = importlib_metadata.version("overcooked_ai")
except:
    VERSION = "0.0.1"

LLM_AGENT_TYPES = ['ProAgent', 'EIRA', 'EIRAAsync']

def boolean_argument(value):
    return bool(strtobool(value))

class OvercookedLogger:
    def __init__(self, log_dir="experiments/logs", variant=None):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        layout = variant.get('layout', 'unknown')
        self.filepath = os.path.join(self.log_dir, f"log_{timestamp}_{layout}.jsonl")
        if variant: self._write_line({"metadata": variant})
            
    def log_step(self, step_data): self._write_line(step_data)
    def _write_line(self, data):
        with open(self.filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

class HumanAgent(Agent):
    def __init__(self):
        super().__init__()
        self.next_action = None 
    def set_next_action(self, action): self.next_action = action
    def action(self, state):
        a = self.next_action if self.next_action is not None else Action.STAY
        self.next_action = None
        return a

def get_combined_thought(agents_list):
    for i, agent in enumerate(agents_list):
        if hasattr(agent, 'current_thought') and agent.current_thought:
            return i, agent.current_thought
    return -1, None

def get_parser():
    parser = ArgumentParser(description='OvercookedAI Experiment')
    parser.add_argument('--layout', '-l', type=str, default='cramped_room')
    parser.add_argument('--p0', type=str, default='Human')
    parser.add_argument('--p1', type=str, default='EIRAAsync')
    parser.add_argument('--horizon', type=int, default=250)
    parser.add_argument('--cook_time', type=int, default=20)
    parser.add_argument('--episode', type=int, default=1)
    parser.add_argument('--render', type=boolean_argument, default=True)
    parser.add_argument('--visual_level', type=int, default=2)
    parser.add_argument('--show_intention', type=boolean_argument, default=True)
    parser.add_argument('--async', dest='async', type=boolean_argument, default=True)
    parser.add_argument('--timestep', type=int, default=400)
    parser.add_argument('--gpt_model', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--prompt_level', type=str, default='l3-aip')
    parser.add_argument('--belief_revision', type=boolean_argument, default=False)
    parser.add_argument('--retrival_method', type=str, default="recent_k")
    parser.add_argument('--K', type=int, default=1)
    parser.add_argument('--mode', type=str, default='exp')
    parser.add_argument('--save', type=boolean_argument, default=True)
    parser.add_argument('--log_dir', type=str, default=None)
    return parser

def main(variant, surface=None):
    layout_name = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']
    render = variant['render'] 
    visual_level = variant['visual_level']
    async_mode = variant.get('async', True)
    game_timestep = variant['timestep'] 

    mdp_layout = NEW_LAYOUTS.get(layout_name, layout_name) if VERSION == '1.1.0' else OLD_LAYOUTS.get(layout_name, layout_name)
    mdp = OvercookedGridworld.from_layout_name(mdp_layout)
    layout_dict = vu.generate_layout_dict(mdp)
    env = OvercookedEnv(mdp, horizon=horizon)
    
    # Pygame 화면 설정
    window_surface = surface
    if not pygame.get_init(): pygame.init()
    if render and window_surface is None:
        window_surface = pygame.display.set_mode((900, 600))
    #pygame.display.set_caption(f"Overcooked AI - {layout_name}")
    
    visualizer = StateVisualizer(cook_time=variant['cook_time'])
    
    # 에이전트 생성 및 필수 속성 강제 주입
    agents_list = []
    for alg in [variant['p0'], variant['p1']]:
        if alg in LLM_AGENT_TYPES:
            agent = make_agent(alg, mdp, layout_name, model=variant['gpt_model'], prompt_level=variant['prompt_level'])
        elif alg == "Human": 
            agent = HumanAgent()
        else: 
            agent = make_agent(alg, mdp, layout_name)
        
        # 💡 [속성 에러 방지] 필수 변수들 강제 초기화
        agent.async_mode = async_mode
        agent.current_timestep = 0
        if not hasattr(agent, 'teammate_intentions_dict'): agent.teammate_intentions_dict = {}
        if not hasattr(agent, 'current_thought'): agent.current_thought = None
        agents_list.append(agent)
    
    team = AgentGroup(*agents_list)
    results = []

    for ep in range(episode):
        logger = OvercookedLogger(log_dir=variant.get('log_dir', 'experiments/logs'), variant=variant)
        env.reset()
        r_total = 0
        
        # 웜업 로직
        if render:
            vu.draw_centered_text(window_surface, "AI Initializing...", "Thinking...", color=(0, 0, 255))
            pygame.display.flip()
            for agent in agents_list:
                if hasattr(agent, 'generate_ml_action'): agent.generate_ml_action(env.state)

        clock = pygame.time.Clock()
        for t in range(1, horizon + 1):
            # 💡 에이전트들에게 현재 스텝 전달
            for agent in agents_list: agent.current_timestep = t
                
            step_start_time = pygame.time.get_ticks()
            human_action = Action.STAY
            action_chosen = False

            if async_mode:
                while pygame.time.get_ticks() - step_start_time < game_timestep:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: pygame.quit(); return 0
                        if event.type == pygame.KEYDOWN and not action_chosen:
                            if event.key == pygame.K_UP: human_action = (0, -1); action_chosen = True
                            elif event.key == pygame.K_DOWN: human_action = (0, 1); action_chosen = True
                            elif event.key == pygame.K_LEFT: human_action = (-1, 0); action_chosen = True
                            elif event.key == pygame.K_RIGHT: human_action = (1, 0); action_chosen = True
                            elif event.key == pygame.K_SPACE: human_action = Action.INTERACT; action_chosen = True
                    pygame.time.delay(1)
            else:
                thought_idx, thought_msg = get_combined_thought(agents_list)
                vu.render_game(window_surface, visualizer, env, t, horizon, r_total, thought_idx, visual_level, layout_dict, thought_msg, variant['show_intention'])
                while not action_chosen:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: pygame.quit(); return 0
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_UP: human_action = (0, -1); action_chosen = True
                            elif event.key == pygame.K_DOWN: human_action = (0, 1); action_chosen = True
                            elif event.key == pygame.K_LEFT: human_action = (-1, 0); action_chosen = True
                            elif event.key == pygame.K_RIGHT: human_action = (1, 0); action_chosen = True
                            elif event.key == pygame.K_SPACE: human_action = Action.INTERACT; action_chosen = True
                    pygame.time.delay(10)

            actions = []
            for agent in agents_list:
                if isinstance(agent, HumanAgent):
                    agent.set_next_action(human_action); actions.append(agent.action(env.state))
                else: actions.append(agent.action(env.state, partner_action=human_action))

            _, reward, done, _ = env.step(tuple(actions))
            r_total += reward
            
            if async_mode:
                thought_idx, thought_msg = get_combined_thought(agents_list)
                vu.render_game(window_surface, visualizer, env, t, horizon, r_total, thought_idx, visual_level, layout_dict, thought_msg, variant['show_intention'])
            
            if done: break
        
        results.append(r_total)
    
    return int(np.mean(results))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(vars(args))