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
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.agents.agent import AgentGroup, Agent
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from utils import NEW_LAYOUTS, OLD_LAYOUTS, make_agent

# 환경 설정
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

TARGET_SCORES = {
    'cramped_room': 60,
    'asymmetric_advantages': 100,
    'coordination_ring': 60,
    'counter_circuit': 60
}

def boolean_argument(value):
    return bool(strtobool(value))

class OvercookedLogger:
    def __init__(self, log_dir="experiments/logs", variant=None):
        self.log_dir = log_dir
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        layout = variant.get('layout', 'unknown')
        cond_name = variant.get('name', 'unknown')
        self.filepath = os.path.join(self.log_dir, f"log_{timestamp}_{layout}_{cond_name}.jsonl")
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
    parser.add_argument('--horizon', type=int, default=400)
    parser.add_argument('--cook_time', type=int, default=20)
    parser.add_argument('--episode', type=int, default=1)
    parser.add_argument('--render', type=boolean_argument, default=True)
    parser.add_argument('--visual_level', type=int, default=2)
    parser.add_argument('--show_intention', type=boolean_argument, default=True)
    parser.add_argument('--async', dest='async', type=boolean_argument, default=True)
    parser.add_argument('--timestep', type=int, default=400)
    parser.add_argument('--gpt_model', type=str, default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--prompt_level', type=str, default='l3-aip')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--name', type=str, default='unknown')
    return parser

def main(variant, surface=None):
    layout_name = variant.get('layout', 'cramped_room')
    horizon = variant.get('horizon', 400)
    episode = variant.get('episode', 1)
    render = variant.get('render', True)
    visual_level = variant.get('visual_level', 2)
    async_mode = variant.get('async', True)
    game_timestep = variant.get('timestep', 400)
    target_score = TARGET_SCORES.get(layout_name, 60)

    mdp = OvercookedGridworld.from_layout_name(layout_name)
    layout_dict = vu.generate_layout_dict(mdp)
    env = OvercookedEnv(mdp, horizon=horizon)
    
    window_surface = surface
    if not pygame.get_init(): pygame.init()
    if render and window_surface is None:
        window_surface = pygame.display.set_mode((900, 600))
    
    visualizer = StateVisualizer(cook_time=variant.get('cook_time', 20))
    
    agents_list = []
    for alg in [variant.get('p0', 'Human'), variant.get('p1', 'EIRAAsync')]:
        if alg in LLM_AGENT_TYPES:
            agent = make_agent(alg, mdp, layout_name, model=variant.get('gpt_model', 'Qwen/Qwen3-VL-8B-Instruct'), prompt_level=variant.get('prompt_level', 'l3-aip'))
        elif alg == "Human": agent = HumanAgent()
        else: agent = make_agent(alg, mdp, layout_name)
        
        agent.async_mode = async_mode
        agent.current_timestep = 0
        agents_list.append(agent)
    
    results_score, results_step, results_col = [], [], []

    for ep in range(episode):
        logger = OvercookedLogger(log_dir=variant.get('log_dir', 'experiments/logs'), variant=variant)
        env.reset()
        r_total, inter_col_count = 0, 0
        final_t = horizon
        
        if render:
            vu.draw_centered_text(window_surface, "AI Initializing...", "Thinking...", color=(0, 0, 255))
            pygame.display.flip()
            for agent in agents_list:
                if hasattr(agent, 'generate_ml_action'): agent.generate_ml_action(env.state)

        for t in range(1, horizon + 1):
            for agent in agents_list: agent.current_timestep = t
            step_start_time = pygame.time.get_ticks()
            human_action = Action.STAY
            action_chosen = False

            if async_mode:
                while pygame.time.get_ticks() - step_start_time < game_timestep:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: pygame.quit(); return 0, 0, 0
                        if event.type == pygame.KEYDOWN and not action_chosen:
                            if event.key == pygame.K_UP: human_action = Direction.NORTH; action_chosen = True
                            elif event.key == pygame.K_DOWN: human_action = Direction.SOUTH; action_chosen = True
                            elif event.key == pygame.K_LEFT: human_action = Direction.WEST; action_chosen = True
                            elif event.key == pygame.K_RIGHT: human_action = Direction.EAST; action_chosen = True
                            elif event.key == pygame.K_SPACE: human_action = Action.INTERACT; action_chosen = True
                    pygame.time.delay(1)
            else:
                thought_idx, thought_msg = get_combined_thought(agents_list)
                vu.render_game(window_surface, visualizer, env, t, horizon, r_total, thought_idx, visual_level, layout_dict, thought_msg, variant.get('show_intention', True))
                while not action_chosen:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT: pygame.quit(); return 0, 0, 0
                        if event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_UP: human_action = Direction.NORTH; action_chosen = True
                            elif event.key == pygame.K_DOWN: human_action = Direction.SOUTH; action_chosen = True
                            elif event.key == pygame.K_LEFT: human_action = Direction.WEST; action_chosen = True
                            elif event.key == pygame.K_RIGHT: human_action = Direction.EAST; action_chosen = True
                            elif event.key == pygame.K_SPACE: human_action = Action.INTERACT; action_chosen = True
                    pygame.time.delay(10)

            actions = []
            for agent in agents_list:
                if isinstance(agent, HumanAgent):
                    agent.set_next_action(human_action); actions.append(agent.action(env.state))
                else: actions.append(agent.action(env.state, partner_action=human_action))

            # 💡 상호 충돌 감지 로직
            old_p0, old_p1 = env.state.players[0].position, env.state.players[1].position
            int_p0 = (old_p0[0] + actions[0][0], old_p0[1] + actions[0][1]) if actions[0] in Direction.ALL_DIRECTIONS else old_p0
            int_p1 = (old_p1[0] + actions[1][0], old_p1[1] + actions[1][1]) if actions[1] in Direction.ALL_DIRECTIONS else old_p1
            
            is_col = False
            if int_p0 == int_p1 and (actions[0] != Action.STAY or actions[1] != Action.STAY): is_col = True
            elif int_p0 == old_p1 and int_p1 == old_p0 and actions[0] != Action.STAY and actions[1] != Action.STAY: is_col = True
            elif actions[0] in Direction.ALL_DIRECTIONS and int_p0 == old_p1 and actions[1] not in Direction.ALL_DIRECTIONS: is_col = True
            elif actions[1] in Direction.ALL_DIRECTIONS and int_p1 == old_p0 and actions[0] not in Direction.ALL_DIRECTIONS: is_col = True
            if is_col: inter_col_count += 1

            _, reward, is_timeout, _ = env.step(tuple(actions))
            r_total += reward
            is_success = r_total >= target_score
            done = is_timeout or is_success
            
            thought_idx, thought_msg = get_combined_thought(agents_list)
            logger.log_step({"timestep": t, "inter_collision": is_col, "reward": reward, "cumulative_reward": r_total, "done": done})

            if render and async_mode:
                vu.render_game(window_surface, visualizer, env, t, horizon, r_total, thought_idx, visual_level, layout_dict, thought_msg, variant.get('show_intention', True))
            
            if done: 
                final_t = t
                break
        
        results_score.append(r_total)
        results_step.append(final_t)
        results_col.append(inter_col_count)
    
    return int(np.mean(results_score)), int(np.mean(results_step)), int(np.mean(results_col))

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(vars(args))