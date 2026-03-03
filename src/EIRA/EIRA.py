import itertools, os, json, re
from collections import defaultdict
import numpy as np
import pkg_resources
import sys 
import copy 
from .modules import Module
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.planning.search import find_path 
from overcooked_ai_py.planning.search import get_intersect_counter 
from overcooked_ai_py.planning.search import query_counter_states 
import time
import random 
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState

cwd = os.getcwd()
openai_key_file = os.path.join(cwd, "openai_key.txt")
PROMPT_DIR = os.path.join(cwd, "prompts")

NAME_TO_ACTION = {
    "NORTH": Direction.NORTH,
    "SOUTH": Direction.SOUTH,
    "EAST": Direction.EAST,
    "WEST": Direction.WEST,
    "INTERACT": Action.INTERACT,
    "STAY": Action.STAY
}

class ProAgent(object):
    """
    This agent uses GPT-3.5/Qwen to generate actions.
    """
    def __init__(self, model="Qwen/Qwen2-VL-7B-Instruct-AWQ"):
        self.agent_index = None
        self.model = model

        self.openai_api_keys = []
        self.load_openai_keys()
        self.key_rotation = True
        
        # 끼임(Stuck) 감지용 변수
        self.stuck_steps = 0
        self.last_pos_for_stuck = None

    def load_openai_keys(self):
        with open(openai_key_file, "r") as f:
            context = f.read()
        self.openai_api_keys = context.split('\n')

    def openai_api_key(self):
        if self.key_rotation:
            self.update_openai_key()
        return self.openai_api_keys[0]

    def update_openai_key(self):
        self.openai_api_keys.append(self.openai_api_keys.pop(0))

    def set_agent_index(self, agent_index):
        raise NotImplementedError

    def action(self, state):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class ProMediumLevelAgent(ProAgent):
    """
    This agent default to use GPT to generate medium level actions.
    Synchronous Execution (Blocking).
    """
    def __init__(
            self,
            mlam,
            layout,
            model = "Qwen/Qwen3-VL-8B-Instruct",
            prompt_level='l2-ap', 
            belief_revision=False,
            retrival_method="recent_k",
            K=3,
            auto_unstuck=True,
            controller_mode='new', 
            debug_mode='N', 
            agent_index=None,
            outdir = None,
    ):
        super().__init__(model=model)
        self.trace = True 
        self.debug_mode = 'Y' 
        self.controller_mode = controller_mode 
        self.mlam = mlam
        self.layout = layout
        self.mdp = self.mlam.mdp
        
        self.out_dir = outdir 
        self.agent_index = agent_index

        self.prompt_level = prompt_level
        self.belief_revision = belief_revision

        self.retrival_method = retrival_method
        self.K = K
        
        self.prev_state = None
        self.auto_unstuck = auto_unstuck

        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.possible_motion_goals = None
        self.pot_id_to_pos = []
        self.global_id_mapping = {
            "onion_dispenser": [],
            "dish_dispenser": [],
            "tomato_dispenser": [],
            "serving": [],
            "pot": []
        }
        self.cached_terrain_matrix = {
            'matrix': copy.deepcopy(self.mlam.mdp.terrain_mtx), 
            'height': len(self.mlam.mdp.terrain_mtx),
            'width': len(self.mlam.mdp.terrain_mtx[0])
        }
        self.action_regex = re.compile(r'\((\s*\d+\s*)\)')
        self.overcooked_version = pkg_resources.get_distribution("overcooked_ai").version

        # [초기화]
        self.layout_prompt = ""
        self.partner_move_history = []
        self.prev_partner_move = None
        self.current_thought = ""

    def set_mdp(self, mdp):
        self.mdp = mdp

    def create_gptmodule(self, module_name, file_type='txt', retrival_method='recent_k', K=10):
        print(f"\n--->Initializing GPT {module_name}<---\n")    

        if "gpt" in self.model or "text-davinci" in self.model or "Qwen" in self.model:
            model_name = "gpt"
        elif "claude" in self.model:
            model_name = "claude"
    
        if module_name == "planner":
            prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, self.prompt_level.strip(), f'{self.layout}_{self.agent_index}.{file_type}')
        elif module_name == "explainer":
            prompt_file = os.path.join(PROMPT_DIR, model_name, module_name, f'player{self.agent_index}.{file_type}')
        else:
            raise Exception(f"Module {module_name} not supported.")
        
        with open(prompt_file, "r") as f:
            if file_type == 'json':
                messages = json.load(f)
            elif file_type == 'txt':
                messages = [{"role": "system", "content": f.read()}]
            else:
                print("Unsupported file format.")
        
        return Module(messages, self.model, retrival_method, K)

    def reset(self):
        self.planner.reset()
        self.explainer.reset()
        self.prev_state = None
        self.current_ml_action = None
        self.current_ml_action_steps = 0
        self.time_to_wait = 0
        self.possible_motion_goals = None
        self.current_timestep = 0
        self.teammate_ml_actions_dict = {}
        self.teammate_intentions_dict = {}
        self.stuck_steps = 0
        self.last_pos_for_stuck = None

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index
        self.planner = self.create_gptmodule("planner", retrival_method=self.retrival_method, K=self.K)
        self.explainer = self.create_gptmodule("explainer", retrival_method='recent_k', K=self.K)

        print(self.planner.instruction_head_list[0]['content'])
      
    def generate_layout_prompt(self, my_pos, other_pos):
        self.global_id_mapping = {
            "onion_dispenser": [], "dish_dispenser": [], "tomato_dispenser": [], "serving": [], "pot": []
        }
        self.pot_id_to_pos = [] 

        name_map = {
            "onion_dispenser": "OnionD",    
            "dish_dispenser": "DishD",      
            "serving": "Serve",             
            "pot": "Pot"
        }

        layout_prompt = "Layout: "
        
        for key, readable_name in name_map.items():
            locations = getattr(self.mdp, f"get_{key}_locations")()
            self.global_id_mapping[key] = locations
            
            if not locations:
                continue
                
            items_str_list = []
            for i, pos in enumerate(locations):
                if self.agent_index == 0:
                    dist_p0 = abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1])
                    dist_p1 = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
                else: 
                    dist_p1 = abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1])
                    dist_p0 = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])

                items_str_list.append(f"<{readable_name} {i}> [P0:{dist_p0}, P1:{dist_p1}]")
                
                if key == "pot":
                    self.pot_id_to_pos.append(pos)
            
            layout_prompt += f"{', '.join(items_str_list)}; "

        return layout_prompt.strip() + "\n"
        
    def generate_state_prompt(self, state):
        ego = state.players[self.agent_index]
        teammate = state.players[1 - self.agent_index]
        self.layout_prompt = self.generate_layout_prompt(ego.position, teammate.position)
            
        history_prompt = ""

        curr_partner_pos = teammate.position
        partner_idx = 1 - self.agent_index
        
        history_prompt += f"\n<Player {partner_idx}> History: "

        if not self.partner_move_history:
            move_str = f"Start -> {curr_partner_pos}"
        else:
            past_moves_str = " -> ".join([str(pos) for pos in self.partner_move_history])
            move_str = f"{past_moves_str} -> {curr_partner_pos}"
        
        history_prompt += f"Moved: {move_str}"
        
        self.partner_move_history.append(curr_partner_pos)
        if len(self.partner_move_history) >= 5:
            self.partner_move_history.pop(0)

        time_prompt = f"Scene {state.timestep}: "
        
        ego_object = ego.held_object.name if ego.held_object else "nothing"
        ego_state_prompt = f"<Player {self.agent_index}> holds "
        if ego_object == 'soup':
            ego_state_prompt += f"a dish with {ego_object} and needs to deliver soup. "
        elif ego_object == 'nothing':
            ego_state_prompt += f"{ego_object}. "
        else:
            ego_state_prompt += f"one {ego_object}. "
        ego_state_prompt += f" at {ego.position}. "
        
        teammate_object = teammate.held_object.name if teammate.held_object else "nothing"
        teammate_state_prompt = f"<Player {1-self.agent_index}> holds "
        if teammate_object == 'soup':
            teammate_state_prompt += f"a dish with {teammate_object}. "
        elif teammate_object == "nothing":
            teammate_state_prompt += f"{teammate_object}. "
        else:
            teammate_state_prompt += f"one {teammate_object}. "
        teammate_state_prompt += f" at {teammate.position}. "

        kitchen_state_prompt = "Kitchen states: "
        prompt_dict = {
            "empty": "<Pot {id}> is empty; ",
            "cooking": "<Pot {id}> starts cooking, the soup will be ready after {t} timesteps; ",
            "ready": "<Pot {id}> has already cooked the soup; ",
            "1_items": "<Pot {id}> has 1 onion; ",
            "2_items": "<Pot {id}> has 2 onions; ",
            "3_items": "<Pot {id}> has 3 onions and is full; "
        }

        pot_states_dict = self.mdp.get_pot_states(state)   
        
        if self.overcooked_version== '1.1.0':
            for key in pot_states_dict.keys():
                if key == "cooking":
                    for pos in pot_states_dict[key]:
                        pot_id = self.pot_id_to_pos.index(pos)
                        soup_object = state.get_object(pos)
                        kitchen_state_prompt += prompt_dict[key].format(id=pot_id, t=soup_object.cook_time_remaining)
                else:
                    for pos in pot_states_dict[key]:
                        pot_id = self.pot_id_to_pos.index(pos)
                        kitchen_state_prompt += prompt_dict[key].format(id=pot_id) 
        
        elif self.overcooked_version== '0.0.1':
            for key in pot_states_dict.keys():
                if key == "empty":
                    for pos in pot_states_dict[key]: 
                        pot_id = self.pot_id_to_pos.index(pos)
                        kitchen_state_prompt += prompt_dict[key].format(id=pot_id)     
                else: 
                    for soup_key in pot_states_dict[key].keys():
                        for pos in pot_states_dict[key][soup_key]:
                            pot_id = self.pot_id_to_pos.index(pos)
                            soup_object = state.get_object(pos)
                            soup_type, num_items, cook_time = soup_object.state
                            if soup_key == "cooking":
                                kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id, t=self.mdp.soup_cooking_time-cook_time)
                            elif soup_key == "partially_full":
                                pass
                            else:
                                kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id)

        if self.layout == 'forced_coordination': 
            from utils import get_intersect_counter, query_counter_states
            intersect_counters = get_intersect_counter(
                                state.players_pos_and_or[self.agent_index], 
                                state.players_pos_and_or[1 - self.agent_index], 
                                self.mdp, 
                                self.mlam
                            )
            counter_states = query_counter_states(self.mdp, state)  
            
            kitchen_state_prompt += '{} counters can be visited by <Player {}>. Their states are as follows: '.format(len(intersect_counters), self.agent_index)
            count_states = {}  
            for i in intersect_counters:  
                obj_i = 'nothing' 
                if counter_states[i] != ' ': 
                    obj_i = counter_states[i]                
                if obj_i in count_states:  
                    count_states[obj_i] += 1
                else: 
                    count_states[obj_i]  = 1 
            total_obj = ['onion', 'dish']
            for i in count_states:   
                if i == 'nothing': 
                    continue 
                kitchen_state_prompt += f'{count_states[i]} counters have {i}. '   
            for i in total_obj: 
                if i not in count_states:        
                    kitchen_state_prompt += f'No counters have {i}. ' 

            teammate_state_prompt = "" 

        scene_block = time_prompt + ego_state_prompt + teammate_state_prompt

        parts = [
            self.layout_prompt.strip(),  
            scene_block.strip(),         
            history_prompt.strip(),      
            kitchen_state_prompt.strip() 
        ]
        
        final_prompt = "\n".join([p for p in parts if p])
        print("PROMPT:", final_prompt)
        return final_prompt
    
    def generate_belief_prompt(self):
        ego_id = self.agent_index
        intention_prompt = f"All <Player {ego_id}> infered intentions about <Player {1-ego_id}>: {self.teammate_intentions_dict}.\n"
        real_behavior_prompt = f"<Player {1-ego_id}> real behaviors: {self.teammate_ml_actions_dict}.\n"
        belief_prompt = intention_prompt + real_behavior_prompt
        return belief_prompt

    def action(self, state):
        """
        동기 방식(Synchronous) 제어 및 스텝 처리.
        """
        current_pos = state.players[self.agent_index].position

        # 1. 현재 수행 중인 High-Level Action 관리
        if self.current_ml_action is not None:
            
            # --- 제자리 멈춤 감지 ---
            if self.last_pos_for_stuck == current_pos:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0  
            
            self.last_pos_for_stuck = current_pos

            # 강제 취소 (3스텝 이상 막혔을 때)
            if self.stuck_steps >= 3 and "wait" not in self.current_ml_action:
                print(f"\n[Stuck] 제자리에 3스텝 이상 막혀서 '{self.current_ml_action}' 강제 취소! 다시 생각합니다.")
                self.trace = False
                self.current_ml_action = None
                self.stuck_steps = 0
                    
            elif self.check_current_ml_action_done(state):
                self.generate_success_feedback(state)
                self.current_ml_action = None 
                self.stuck_steps = 0 
                
            elif not self.validate_current_ml_action(state):
                self.trace = False
                self.current_ml_action = None
                self.stuck_steps = 0 

        # 2. 생각(Thinking) - 동기적으로 처리
        if self.current_ml_action is None:
            plan = self.generate_ml_action(state)
            self.current_ml_action = plan
            self.current_ml_action_steps = 0
            
            if "wait" in self.current_ml_action:
                import re
                nums = re.findall(r'\d+', self.current_ml_action)
                self.time_to_wait = int(nums[0]) if nums else 1

            if hasattr(self, 'last_llm_timing'):
                print(f" >> LLM Timing: {json.dumps(self.last_llm_timing)}")

        # 3. Low-Level Motion Planning (이동)
        self.trace = True 
        chosen_action = Action.STAY
        
        if "wait" in self.current_ml_action:
            self.current_ml_action_steps += 1
            self.time_to_wait -= 1
            if self.time_to_wait <= 0:
                self.current_ml_action = None
            lis_actions = self.mdp.get_valid_actions(state.players[self.agent_index])
            chosen_action = lis_actions[np.random.randint(0,len(lis_actions))]
        else:
            possible_motion_goals = self.find_motion_goals(state)    
            current_motion_goal, chosen_action = self.choose_motion_goal(
                state.players_pos_and_or[self.agent_index], 
                possible_motion_goals, 
                state
            )
            if chosen_action is None:
                 self.current_ml_action = "wait(1)"
                 self.time_to_wait = 1
                 chosen_action = Action.STAY

        self.prev_state = state
        self.current_ml_action_steps += 1

        if self.overcooked_version == '1.1.0':
            return chosen_action, {}
        elif self.overcooked_version == '0.0.1':
            return chosen_action

    def parse_ml_action(self, response, agent_index): 
        if agent_index == 0: 
            pattern = r'layer\s*0: (.+)'
        elif agent_index == 1: 
            pattern = r'layer\s*1: (.+)'
        else:
            raise ValueError("Unsupported agent index.")

        match = re.search(pattern, response)
        action_string = match.group(1).strip() if match else response.strip()

        if "wait" in action_string:
            def parse_wait_string(s):
                if s == "wait": return 1
                nums = re.findall(r'\d+', s)
                if nums: return int(nums[0])
                return 1
            
            wait_time = parse_wait_string(action_string)
            if self.layout == 'forced_coordination': 
                wait_time = max(3, wait_time)
            
            return f"wait({wait_time})"

        if re.search(r'\w+\(\d+\)', action_string):
            if "," in action_string:
                action_string = action_string.split(',')[0].strip()
            return action_string

        ml_action = action_string.split()[0] 

        if "place_obj" in action_string: ml_action = "place_obj_on_counter"
        elif "deliver" in action_string: ml_action = "deliver_soup"
        elif "pick" in action_string:
            if "onion" in action_string: ml_action = "pickup_onion" 
            elif "tomato" in action_string: ml_action = "pickup_tomato"
            elif "dish" in action_string: ml_action = "pickup_dish"
        elif "put" in action_string:
            if "onion" in action_string: ml_action = "put_onion_in_pot"
            elif "tomato" in action_string: ml_action = "put_tomato_in_pot"
        elif "fill" in action_string:   
            ml_action = "fill_dish_with_soup"
        
        return ml_action

    def generate_ml_action(self, state):
        breakdown = {}
        t_start = time.perf_counter()

        if self.prompt_level == "l3-aip" and self.belief_revision:
            belief_prompt = self.generate_belief_prompt()
        else:
            belief_prompt = ''
        
        state_prompt = belief_prompt + self.generate_state_prompt(state)

        state_message = {"role": "user", "content": state_prompt}
        self.planner.current_user_message = state_message
        
        t_prompt = time.perf_counter()
        breakdown['1_Prompt_Prep'] = t_prompt - t_start

        response = self.planner.query(key=self.openai_api_key(), stop='Scene', trace=self.trace)
        
        t_inference = time.perf_counter()
        breakdown['2_LLM_Inference'] = t_inference - t_prompt

        if 'wait' not in response:
            self.planner.add_msg_to_dialog_history(state_message) 
            self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})
        
        print(f"\n\n\n### GPT Planner module\n")   
        print("====== GPT Query ======")
        print(response)  
        self.current_thought = response

        if self.prompt_level == "l3-aip":
            generated_intention = self.parse_ml_action(response, 1-self.agent_index)
            self.teammate_intentions_dict[str(self.current_timestep)] = generated_intention

        ml_action = self.parse_ml_action(response, self.agent_index)

        if "wait" not in ml_action:
            self.planner.add_msg_to_dialog_history({"role": "assistant", "content": ml_action})
        
        self.current_ml_action_steps = 0

        t_parse = time.perf_counter()
        breakdown['3_Parsing'] = t_parse - t_inference
        
        self.last_llm_timing = breakdown
        
        return ml_action


    def check_current_ml_action_done(self, state):
        player = state.players[self.agent_index]
        action = self.current_ml_action
        
        if not action:
            return True

        if "pickup" in action:
            target_obj = None
            if "onion" in action: target_obj = "onion"
            elif "dish" in action: target_obj = "dish"
            elif "tomato" in action: target_obj = "tomato"
            
            return player.has_object() and player.get_object().name == target_obj
        
        elif "fill" in action:
            return player.held_object is not None and player.held_object.name == 'soup'
        
        elif "put" in action or "place" in action or "deliver" in action:
            return not player.has_object()
        
        elif "wait" in action:
            return self.time_to_wait <= 0
            
        return False

    def validate_current_ml_action(self, state):
        if self.current_ml_action is None:
            return False
            
        action = self.current_ml_action
        player = state.players[self.agent_index]
        pot_states_dict = self.mdp.get_pot_states(state)
        
        has_object = player.has_object()
        has_onion = has_object and player.get_object().name == 'onion'
        has_dish = has_object and player.get_object().name == 'dish'
        has_soup = has_object and player.get_object().name == 'soup'
        
        if self.overcooked_version == '1.1.0':
            soup_ready = len(pot_states_dict['ready']) > 0
            soup_cooking = len(pot_states_dict['cooking']) > 0
            pot_available_for_onion = len(pot_states_dict["empty"] + self.mdp.get_partially_full_pots(pot_states_dict)) > 0
        else:
            soup_ready = len(pot_states_dict['onion']['ready']) > 0
            soup_cooking = len(pot_states_dict['onion']['cooking']) > 0
            pot_available_for_onion = len(pot_states_dict["empty"] + pot_states_dict["onion"]['partially_full']) > 0

        if "pickup_onion" in action:   
            if len(self.find_motion_goals(state)) == 0: return False
            return not has_object and len(self.mdp.get_onion_dispenser_locations()) > 0

        elif "pickup_dish" in action:
            if len(self.find_motion_goals(state)) == 0: return False
            return not has_object and len(self.mdp.get_dish_dispenser_locations()) > 0
            
        elif "put_onion" in action:
            return has_onion and pot_available_for_onion
            
        elif "place_obj" in action:
            return has_object and len(self.mdp.get_empty_counter_locations(state)) > 0
            
        elif "fill_dish" in action:
            return has_dish and (soup_ready or soup_cooking)
            
        elif "deliver" in action:
            return has_soup
            
        elif "wait" in action:
            return True
            
        return False
   
    def generate_success_feedback(self, state):
        success_feedback = f"### Controller Validation\nPlayer {self.agent_index} succeeded at {self.current_ml_action}. \n"
        print(success_feedback)  
        if 'wait' not in success_feedback:
            self.planner.add_msg_to_dialog_history({"role": "user", "content": f'Player {self.agent_index} succeeded at {self.current_ml_action}.'})
        
    def generate_failure_feedback(self, state):
        failure_feedback = self.generate_state_prompt(state)
        failure_feedback += f" Player {self.agent_index} failed at {self.current_ml_action}."
        failure_feedback += f" Why did Player {self.agent_index} fail ?"     
        print(f"\n~~~~~~~~ Explainer~~~~~~~~\n{failure_feedback}")  
        failure_message = {"role": "user", "content": failure_feedback}
        self.explainer.current_user_message = failure_message
        failure_explanation = self.explainer.query(self.openai_api_key())
        print(failure_explanation)  
        if "wait" not in failure_explanation or self.layout == 'forced_coodination':
            self.explainer.add_msg_to_dialog_history({"role": "user", "content": failure_feedback})
            self.explainer.add_msg_to_dialog_history({"role": "assistant", "content": failure_explanation})
        self.planner.add_msg_to_dialog_history({"role": "user", "content": failure_explanation}) 
  
        
    def find_shared_counters(self, state, mlam):  
        counter_dicts = query_counter_states(self.mdp, state) 

        counter_list  = get_intersect_counter(state.players_pos_and_or[self.agent_index],
                        state.players_pos_and_or[1 - self.agent_index], 
                        self.mdp, 
                        self.mlam
                    )    

        lis = [] 
        for i in counter_list:  
            if counter_dicts[i] == ' ':  
                lis.append(i)       
        available_plans = mlam._get_ml_actions_for_positions(lis)
        return available_plans          

    def find_motion_goals(self, state):
        am = self.mlam
        motion_goals = []
        player = state.players[self.agent_index]
        
        def get_interact_states_from_pos(target_pos):
            valid_states = []
            x, y = target_pos
            width = len(self.mdp.terrain_mtx[0])
            height = len(self.mdp.terrain_mtx)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                adj_pos = (nx, ny)
                
                if 0 <= nx < width and 0 <= ny < height:
                    if self.mdp.get_terrain_type_at_pos(adj_pos) == ' ':
                        face_dir = (-dx, -dy)
                        valid_states.append((adj_pos, face_dir))
                        
            return valid_states

        raw_action = self.current_ml_action.strip()
        target_index = -1
        
        match = self.action_regex.search(raw_action)
        if match:
            target_index = int(match.group(1).strip())

        if target_index != -1:
            target_list_key = None
            
            if "pickup_onion" in raw_action:
                target_list_key = "onion_dispenser"
            elif "pickup_dish" in raw_action:
                target_list_key = "dish_dispenser"
            elif "pickup_tomato" in raw_action:
                target_list_key = "tomato_dispenser"
            elif "put_onion" in raw_action or "put_tomato" in raw_action or "fill_dish" in raw_action:
                target_list_key = "pot"
            elif "deliver" in raw_action:
                target_list_key = "serving"
            
            if target_list_key:
                targets = self.global_id_mapping.get(target_list_key, [])
                
                if 0 <= target_index < len(targets):
                    target_pos = targets[target_index]
                    motion_goals = get_interact_states_from_pos(target_pos)
                else:
                    print(f"[Warning] Index {target_index} out of bounds for {raw_action}. Avail: {len(targets)}")

        if not motion_goals:
            if "place_obj_on_counter" in raw_action:
                motion_goals = self.find_shared_counters(state, self.mlam)     
                if len(motion_goals) == 0: 
                    motion_goals = am.place_obj_on_counter_actions(state)
            
            elif "wait" in raw_action:
                motion_goals = am.wait_actions(player)

            elif target_index == -1:
                pot_states_dict = self.mdp.get_pot_states(state)
                counter_objects = self.mdp.get_counter_objects_dict(
                    state, list(self.mdp.terrain_pos_dict["X"])
                )
                
                if "pickup_onion" in raw_action:
                    motion_goals = am.pickup_onion_actions_new(state, counter_objects, state.players_pos_and_or, self.agent_index)
                elif "pickup_dish" in raw_action:
                    motion_goals = am.pickup_dish_actions_new(state, counter_objects, state.players_pos_and_or, self.agent_index)
                elif "put_onion" in raw_action:
                    motion_goals = am.put_onion_in_pot_actions(pot_states_dict)
                elif "fill_dish" in raw_action:
                    motion_goals = am.pickup_soup_with_dish_actions(pot_states_dict, only_nearly_ready=True)
                elif "deliver" in raw_action:
                    motion_goals = am.deliver_soup_actions()
                
        motion_goals = [
            mg for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]

        return motion_goals
 
    def choose_motion_goal(self, start_pos_and_or, motion_goals, state = None):
        if self.controller_mode == 'new':
            (
                chosen_goal,
                chosen_goal_action,
            ) = self.get_lowest_cost_action_and_goal_new(
                start_pos_and_or, motion_goals, state
            )
        else: 
            (
                chosen_goal,
                chosen_goal_action,
            ) = self.get_lowest_cost_action_and_goal(
                start_pos_and_or, motion_goals
            )
        return chosen_goal, chosen_goal_action
    
    def get_lowest_cost_action_and_goal(self, start_pos_and_or, motion_goals):
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:
            action_plan, _, plan_cost = self.mlam.motion_planner.get_plan(
                start_pos_and_or, goal
            )
            if plan_cost < min_cost:
                best_action = action_plan[0]
                min_cost = plan_cost
                best_goal = goal
        return best_goal, best_action
 
    def get_lowest_cost_action_and_goal_new(self, start_pos_and_or, motion_goals, state): 
        min_cost = np.Inf
        best_action, best_goal = None, None
        for goal in motion_goals:   
            action_plan, plan_cost = self.real_time_planner(
                start_pos_and_or, goal, state
            )     
            if plan_cost < min_cost:
                best_action = action_plan
                min_cost = plan_cost
                best_goal = goal     
        if best_action is None: 
            return self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
        return best_goal, best_action

    def real_time_planner(self, start_pos_and_or, goal, state):   
        other_pos_and_or = state.players_pos_and_or[1 - self.agent_index]
        action_plan, plan_cost = find_path(start_pos_and_or, other_pos_and_or, goal, self.cached_terrain_matrix) 
        return action_plan, plan_cost
    
class ProPlanningAgent(ProAgent):
    def __init__(self, model="Qwen/Qwen2-VL-7B-Instruct-AWQ"):
        super().__init__(model=model)