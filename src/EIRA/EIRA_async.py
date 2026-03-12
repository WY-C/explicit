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
import threading  # [NEW] 멀티스레딩을 위한 모듈
import random # 고민중에 random 액션
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
    This agent uses GPT-3.5 to generate actions.
    """
    def __init__(self, model=""):
        self.agent_index = None
        self.model = model

        self.openai_api_keys = []
        self.load_openai_keys()
        self.key_rotation = True
        # [NEW] 끼임(Stuck) 감지용 변수
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
    This agent default to use GPT-3.5 to generate medium level actions.
    Now supports Asynchronous Execution (Non-blocking).
    """
    def __init__(
            self,
            mlam,
            layout,
            model = "Qwen/Qwen3-VL-8B-Instruct",
            prompt_level='l3-aip', 
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

        # [NEW] 비동기 처리 및 타임아웃을 위한 변수
        self.is_thinking = False       
        self.think_thread = None       
        self.next_ml_action = None     
        self.lock = threading.Lock()   
        self.prev_partner_move = None
        self.current_thought = ""
        
        # [NEW] 타임아웃/재요청 관리 변수
        self.thinking_start_time = 0.0  # 생각 시작 시간
        self.thinking_request_id = 0    # 요청 고유 ID (오래된 요청 무시용)
        self.TIMEOUT_SECONDS = 4.0      # 3초 타임아웃 설정

        self.is_llm_called_this_step = False
        self.last_llm_response_time = 0.0


    def set_mdp(self, mdp):
        self.mdp = mdp

    def create_gptmodule(self, module_name, file_type='txt', retrival_method='recent_k', K=10):
        print(f"\n--->Initializing GPT {module_name}<---\n")    

        # prompt_file = os.path.join(PROMPT_DIR, self.model, module_name, self.layout+f'_{self.agent_index}.'+file_type)

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
        
        # [NEW] 비동기 상태 초기화
        with self.lock:
            self.is_thinking = False
            self.next_ml_action = None

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index
        self.planner = self.create_gptmodule("planner", retrival_method=self.retrival_method, K=self.K)
        self.explainer = self.create_gptmodule("explainer", retrival_method='recent_k', K=self.K)

        print(self.planner.instruction_head_list[0]['content'])
      
    def generate_layout_prompt(self, my_pos, other_pos):
        """
        [최적화 V3] 서빙 카운터(Serve) 추가 및 포맷 통일
        출력 예시: <Serve 0> [P0:5, P1:2]
        """
        # 1. 매핑 초기화
        self.global_id_mapping = {
            "onion_dispenser": [], "dish_dispenser": [], "tomato_dispenser": [], "serving": [], "pot": []
        }
        self.pot_id_to_pos = [] 

        # 2. 이름 매핑 (Serve 추가됨)
        name_map = {
            "onion_dispenser": "OnionD",    # Dispenser
            "dish_dispenser": "DishD",      # Dispenser
            "serving": "Serve",             # [중요] Serving Location 추가
            "pot": "Pot"
        }

        layout_prompt = "Layout: "
        
        # 3. 객체 순회 및 거리 계산
        for key, readable_name in name_map.items():
            # MDP에서 위치 정보 가져오기 (get_serving_locations 등)
            locations = getattr(self.mdp, f"get_{key}_locations")()
            self.global_id_mapping[key] = locations
            
            if not locations:
                continue
                
            items_str_list = []
            for i, pos in enumerate(locations):
                # --- 거리 계산 로직 (Player ID에 맞춰 할당) ---
                # self.agent_index가 0이면 my_pos가 P0, other_pos가 P1
                if self.agent_index == 0:
                    dist_p0 = abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1])
                    dist_p1 = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
                else: # self.agent_index가 1이면 my_pos가 P1, other_pos가 P0
                    dist_p1 = abs(pos[0] - my_pos[0]) + abs(pos[1] - my_pos[1])
                    dist_p0 = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])

                # [최종 포맷] 좌표 제거, Serve 포함, 단축된 거리 표기
                # 예: <Serve 0> [P0:5, P1:2]
                items_str_list.append(f"<{readable_name} {i}> [P0:{dist_p0}, P1:{dist_p1}]")
                
                if key == "pot":
                    self.pot_id_to_pos.append(pos)
            
            # 항목별로 묶어서 추가 (예: Serve: <Serve 0> [...], <Serve 1> [...]; )
            layout_prompt += f"{', '.join(items_str_list)}; "

        return layout_prompt.strip() + "\n"
    def generate_state_prompt(self, state):
        ego = state.players[self.agent_index]
        teammate = state.players[1 - self.agent_index]
        self.layout_prompt = self.generate_layout_prompt(ego.position, teammate.position)
            

        history_prompt = ""
        #self.partner_move_history = []

        curr_partner_pos = teammate.position
        partner_idx = 1 - self.agent_index
        
        # [수정] 프롬프트 ID를 파트너 ID로 변경
        history_prompt += f"\n<Player {partner_idx}> History: "

        # 경로 문자열 생성 (예: (1,1) -> (1,2) -> (2,2) -> (2,3))
        if not self.partner_move_history:
            # 첫 시작인 경우
            move_str = f"Start -> {curr_partner_pos}"
            is_blocked = False
        else:
            # 과거 기록들을 화살표로 연결
            past_moves_str = " -> ".join([str(pos) for pos in self.partner_move_history])
            move_str = f"{past_moves_str} -> {curr_partner_pos}"
        

        history_prompt += f"Moved: {move_str}"
        
        self.partner_move_history.append(curr_partner_pos)
        if len(self.partner_move_history) >= 5:
            self.partner_move_history.pop(0)
            


        # =========================================================
        # 3. 기존 정보들 (시간, 플레이어 상태)
        # =========================================================
        time_prompt = f"Scene {state.timestep}: "
        
        # Ego(나) 상태 설명
        ego_object = ego.held_object.name if ego.held_object else "nothing"
        ego_state_prompt = f"<Player {self.agent_index}> holds "
        if ego_object == 'soup':
            ego_state_prompt += f"a dish with {ego_object} and needs to deliver soup. "
        elif ego_object == 'nothing':
            ego_state_prompt += f"{ego_object}. "
        else:
            ego_state_prompt += f"one {ego_object}. "
        ego_state_prompt += f" at {ego.position}. "
        
        # Teammate(동료) 상태 설명
        teammate_object = teammate.held_object.name if teammate.held_object else "nothing"
        teammate_state_prompt = f"<Player {1-self.agent_index}> holds "
        if teammate_object == 'soup':
            teammate_state_prompt += f"a dish with {teammate_object}. "
        elif teammate_object == "nothing":
            teammate_state_prompt += f"{teammate_object}. "
        else:
            teammate_state_prompt += f"one {teammate_object}. "
        teammate_state_prompt += f" at {teammate.position}. "

        # =========================================================
        # 4. 주방 냄비 상태 (Kitchen State)
        # =========================================================
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
        # 버전별 냄비 상태 처리 로직
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

        # Forced Coordination 맵 특수 처리
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

            teammate_state_prompt = "" # 이 맵에서는 동료 정보 숨김

        # =========================================================
        # 5. [핵심 수정] 최종 조합 방식 변경
        # =========================================================
        # 각 파트를 리스트로 만들고, .strip()으로 앞뒤 공백/개행을 제거한 뒤
        # '\n'으로 연결하여 중복 줄바꿈을 방지하고 깔끔한 포맷을 만듭니다.
        
        # Scene 설명과 플레이어 정보는 보통 한 줄이나 붙어있는 문단으로 취급하므로 합칩니다.
        scene_block = time_prompt + ego_state_prompt + teammate_state_prompt

        # print("LAYOUT PROMPT:", self.layout_prompt)
        # print("SCENE BLOCK:", scene_block)
        # print("HISTORY PROMPT:", history_prompt)
        # print("KITCHEN STATE PROMPT:", kitchen_state_prompt)

        parts = [
            self.layout_prompt.strip(),  # Layout
            scene_block.strip(),         # Scene + Players
            history_prompt.strip(),      # History
            kitchen_state_prompt.strip() # Kitchen
        ]
        
        # 내용이 있는 파트만 \n으로 연결
        final_prompt = "\n".join([p for p in parts if p])
        
        # 디버깅용 출력
        print("PROMPT:", final_prompt)
        return final_prompt
    
    
    def generate_belief_prompt(self):
        ego_id = self.agent_index
        intention_prompt = f"All <Player {ego_id}> infered intentions about <Player {1-ego_id}>: {self.teammate_intentions_dict}.\n"
        real_behavior_prompt = f"<Player {1-ego_id}> real behaviors: {self.teammate_ml_actions_dict}.\n"
        belief_prompt = intention_prompt + real_behavior_prompt
        return belief_prompt
    
    ##################
    '''
    The followings are the Planner part (THREADED)
    '''
    ##################
    def _thinking_process(self, state_dict, request_id):
        """
        [Modified] request_id를 인자로 받아서, 최신 요청일 때만 결과를 반영합니다.
        """
        try:
            current_state = OvercookedState.from_dict(state_dict)
            
            # LLM 호출 (시간이 걸림)
            plan = self.generate_ml_action(current_state)

            with self.lock:
                # [중요] 현재 처리한 결과가 최신 요청(request_id)과 일치하는지 확인
                # 만약 메인 스레드에서 이미 타임아웃으로 request_id를 올렸다면, 
                # 이 결과는 버려집니다 (늦게 도착한 패킷 무시).
                if self.thinking_request_id == request_id:
                    self.next_ml_action = plan
                    self.is_thinking = False
                # else: 
                #     print(f" [Thread] Discarded old result (Req:{request_id} != Curr:{self.thinking_request_id})")

        except Exception as e:
            # 에러 발생 시에도 상태 초기화를 위해 체크
            import traceback
            traceback.print_exc()
            with self.lock:
                if self.thinking_request_id == request_id:
                    self.is_thinking = False

    def _correction_process(self, state_snapshot):
        """
        [NEW] Thread Function: Correction (Failure Recovery)
        검증 실패 시 Explainer 호출 후 행동 재생성
        """
        try:
            print(f"\n[Agent] Validation Failed! Asking Explainer & Re-planning...")
            
            # # 1. 실패 피드백 생성 (Explainer LLM 호출 - 시간 소요)
            # 실시간성을 위해 스킵하기
            self.generate_failure_feedback(state_snapshot)
            
            # 2. 행동 재생성 (Planner LLM 호출 - 시간 소요)
            new_action = self.generate_ml_action(state_snapshot)
            
            # 결과 저장
            with self.lock:
                self.next_ml_action = new_action
                self.is_thinking = False
                
        except Exception as e:
            print(f"[Error in Correction Thread] {e}")
            with self.lock:
                self.is_thinking = False

    def action(self, state, partner_action=None):
        """
        [Final Logic] 
        1. Stuck 감지: 3스텝 이상 제자리일 경우 현재 계획 강제 취소.
        2. 공평한 시간 비용(Penalty): 비실시간(Sync) 모드에서 '생각 시작 시점' 기준 페널티 부여.
        3. 랜덤 이동: 생각 중이거나 페널티 대기 중일 때 상하좌우 랜덤 이동 수행.
        4. 사전 충돌 방지: 파트너의 다음 수와 내 다음 수가 겹치면 STAY로 양보.
        """
        self.is_llm_called_this_step = False
        self.last_llm_response_time = 0.0
        
        current_step = state.timestep
        current_pos = state.players[self.agent_index].position

        # ==========================================
        # 🏁 [1] High-Level Action 상태 관리 및 Stuck 감지
        # ==========================================
        if self.current_ml_action is not None:
            # 제자리 멈춤(Stuck) 감지 로직
            if self.last_pos_for_stuck == current_pos:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0
            
            self.last_pos_for_stuck = current_pos

            # 3스텝 이상 끼었을 경우 강제 재사고 유도
            if self.stuck_steps >= 3 and "wait" not in self.current_ml_action:
                print(f"\n[Stuck Avoidance] '{self.current_ml_action}' Cancelled -> Re-planning...")
                self.current_ml_action = None
                self.stuck_steps = 0
                with self.lock: self.is_thinking = False
                    
            elif self.check_current_ml_action_done(state):
                self.generate_success_feedback(state)
                self.current_ml_action = None 
                
            elif not self.validate_current_ml_action(state):
                # 환경 변화로 인해 계획이 무효화된 경우
                self.current_ml_action = None
                with self.lock: self.is_thinking = False

        # ==========================================
        # 🧠 [2] 비동기 추론 관리 및 비실시간 페널티 제어
        # ==========================================
        if self.current_ml_action is None:
            with self.lock:
                current_time = time.time()
                
                # (A) LLM 응답 도착 및 페널티 조건 확인
                if self.next_ml_action is not None:
                    # 💡 getattr를 사용하여 async_mode가 없더라도 기본적으로 True(실시간)로 간주해 에러 방지
                    is_async = getattr(self, 'async_mode', True)
                    
                    if not is_async: 
                        # 비실시간(Sync) 환경 전용 페널티 로직
                        required_steps = getattr(self, 'latest_llm_required_steps', 1)
                        target_release_step = self.thinking_start_step + required_steps
                        
                        if current_step >= target_release_step:
                            self.current_ml_action = self.next_ml_action
                            self.next_ml_action = None
                            self.is_thinking = False
                        else:
                            #print(f"\n[Sync Penalty] Current Step: {current_step}, Target Release Step: {target_release_step}. Waiting...")
                            pass
                    else:
                        # 실시간(Async) 환경은 즉시 실행
                        self.current_ml_action = self.next_ml_action
                        self.next_ml_action = None
                        self.is_thinking = False

                # (B) 타임아웃 체크 및 재요청 (4초 이상 무응답 시)
                elif self.is_thinking and (current_time - self.thinking_start_time > self.TIMEOUT_SECONDS):
                    print(f"\n[Timeout] Retrying with NEW state...")
                    self.thinking_request_id += 1
                    self.thinking_start_time = current_time
                    # 새로운 생각 시작 스텝 업데이트 (재시도 시점 기준)
                    self.thinking_start_step = current_step 
                    self.think_thread = threading.Thread(target=self._thinking_process, args=(state.to_dict(), self.thinking_request_id))
                    self.think_thread.start()

                # (C) 최초 생각 시작
                elif not self.is_thinking:
                    self.is_thinking = True
                    self.thinking_start_time = current_time
                    self.thinking_request_id += 1
                    
                    # 💡 [핵심] 생각이 시작된 '스텝 번호'를 기록
                    self.thinking_start_step = current_step
                    
                    self.think_thread = threading.Thread(target=self._thinking_process, args=(state.to_dict(), self.thinking_request_id))
                    self.think_thread.start()

        # ==========================================
        # 🎲 [3] 행동 결정 (생각/페널티 중일 때 랜덤 이동)
        # ==========================================
        # 계획이 아직 확정되지 않은 모든 상태(추론 중, 페널티 대기 중)
        if self.current_ml_action is None:
            # 💡 상하좌우 중 랜덤 선택하여 활발한 움직임 연출
            random_dir = random.choice([Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST])
            if self.overcooked_version == '1.1.0':
                return random_dir, {}
            return random_dir

        # 계획된 행동(Plan) 수행
        chosen_action = Action.STAY
        
        if "wait" in self.current_ml_action:
            self.current_ml_action_steps += 1
            self.time_to_wait -= 1
            if self.time_to_wait <= 0:
                self.current_ml_action = None
            # wait 행동 수행 중에도 제자리 랜덤 이동으로 심심하지 않게 표현
            chosen_action = random.choice([Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST])
        else:
            # Low-level Motion Planning을 통한 최단 경로 이동
            possible_motion_goals = self.find_motion_goals(state)    
            current_motion_goal, chosen_action = self.choose_motion_goal(
                state.players_pos_and_or[self.agent_index], 
                possible_motion_goals, 
                state
            )
            # 경로를 찾을 수 없는 경우에도 랜덤 이동으로 탈출 시도
            if chosen_action is None:
                 chosen_action = random.choice([Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST])

        # ==========================================
        # 🛡️ [4] 사전 충돌 방지 로직 (Proactive Avoidance)
        # ==========================================
        if isinstance(chosen_action, tuple) and chosen_action != Action.STAY:
            my_pos = state.players[self.agent_index].position
            partner_pos = state.players[1 - self.agent_index].position
            my_next_pos = (my_pos[0] + chosen_action[0], my_pos[1] + chosen_action[1])
            
            # 파트너의 행동 정보가 실시간으로 들어오는 경우
            if partner_action is not None and isinstance(partner_action, tuple) and partner_action != Action.STAY:
                partner_next_pos = (partner_pos[0] + partner_action[0], partner_pos[1] + partner_action[1])
                
                # Case A: 동일 타일 진입 / Case B: 서로의 위치를 크로스하여 교환하려는 경우
                if my_next_pos == partner_next_pos: 
                    chosen_action = Action.STAY
                elif my_next_pos == partner_pos and partner_next_pos == my_pos: 
                    chosen_action = Action.STAY
            else:
                # 파트너가 가만히 있거나 상호작용 중일 때 파트너의 위치로 돌진 방지
                if my_next_pos == partner_pos: 
                    chosen_action = Action.STAY

        self.prev_state = state
        self.current_ml_action_steps += 1

        # 최종 행동 반환
        if self.overcooked_version == '1.1.0':
            return chosen_action, {}
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

        # 1. Wait 커맨드 먼저 처리 (시간 조정 로직 등을 위해)
        if "wait" in action_string:
            def parse_wait_string(s):
                if s == "wait": return 1
                # 숫자만 추출
                nums = re.findall(r'\d+', s)
                if nums: return int(nums[0])
                return 1
            
            # forced_coordination 맵일 경우 최소 3초 대기
            wait_time = parse_wait_string(action_string)
            if self.layout == 'forced_coordination': 
                wait_time = max(3, wait_time)
            
            return f"wait({wait_time})"

        # 2. 인덱스가 포함된 함수 호출형 액션 (pickup_onion(0)) 즉시 반환
        # wait는 위에서 처리했으므로 제외됨
        if re.search(r'\w+\(\d+\)', action_string):
            if "," in action_string:
                action_string = action_string.split(',')[0].strip()
            return action_string

        # 3. 레거시(구형) 텍스트 처리 (LLM이 인덱스 없이 말했을 경우 대비)
        ml_action = action_string.split()[0] # 기본 단어 추출

        if "place_obj" in action_string: ml_action = "place_obj_on_counter"
        elif "deliver" in action_string: ml_action = "deliver_soup"
        elif "pick" in action_string:
            if "onion" in action_string: ml_action = "pickup_onion" # (0)이 없으면 자동 할당됨(find_motion_goals에서)
            elif "tomato" in action_string: ml_action = "pickup_tomato"
            elif "dish" in action_string: ml_action = "pickup_dish"
        elif "put" in action_string:
            if "onion" in action_string: ml_action = "put_onion_in_pot"
            elif "tomato" in action_string: ml_action = "put_tomato_in_pot"
        elif "fill" in action_string:   
            ml_action = "fill_dish_with_soup"
        
        return ml_action

    def generate_ml_action(self, state):
            """
            [Modified] 전체 실행 시간을 측정하여 필요한 스텝 수(Total Time / 400ms 올림)를 계산합니다.
            이 데이터는 비실시간(Sync) 모드에서 공평한 시간 비용을 지불하는 데 사용됩니다.
            """
            import math
            
            # 💡 [NEW] 호출 상태 표시
            self.is_llm_called_this_step = True

            # 상세 타이밍 기록용 딕셔너리
            breakdown = {}
            t_start = time.perf_counter() # 전체 프로세스 시작 시간

            # -------------------------------------------------
            # 1. 프롬프트 생성 (Prompt Construction)
            # -------------------------------------------------
            if self.prompt_level == "l3-aip" and self.belief_revision:
                belief_prompt = self.generate_belief_prompt()
            else:
                belief_prompt = ''
            
            state_prompt = belief_prompt + self.generate_state_prompt(state)

            state_message = {"role": "user", "content": state_prompt}
            self.planner.current_user_message = state_message
            
            t_prompt = time.perf_counter()
            breakdown['1_Prompt_Prep'] = t_prompt - t_start

            # -------------------------------------------------
            # 2. LLM 추론 (Inference / API Call)
            # -------------------------------------------------
            response = self.planner.query(key=self.openai_api_key(), stop='Scene', trace=self.trace)
            
            t_inference = time.perf_counter()
            breakdown['2_LLM_Inference'] = t_inference - t_prompt
            
            # 메인 루프 로깅용 순수 추론 시간 저장
            self.last_llm_response_time = breakdown['2_LLM_Inference'] * 1000

            if 'wait' not in response:
                self.planner.add_msg_to_dialog_history(state_message) 
                self.planner.add_msg_to_dialog_history({"role": "assistant", "content": response})
            
            # 디버깅 출력
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

            # -------------------------------------------------
            # 3. 파싱 및 최종 시간 계산
            # -------------------------------------------------
            t_parse = time.perf_counter()
            breakdown['3_Parsing'] = t_parse - t_inference
            
            # 💡 [핵심] 전체 소요 시간 계산 (프롬프트 준비 + 추론 + 파싱)
            total_duration_ms = (t_parse - t_start) * 1000
            
            # 💡 [핵심] 400ms 단위로 올림하여 필요 스텝 수 결정
            # 예: 1200ms -> 3스텝, 1300ms -> 4스텝
            required_steps = math.ceil(total_duration_ms / 400)

            # 클래스 멤버 변수에 기록 (action 메서드에서 페널티 부여 시 사용)
            self.last_llm_timing = breakdown
            self.latest_llm_time_ms = total_duration_ms # 전체 시간 저장
            self.latest_llm_required_steps = required_steps # 계산된 필요 스텝 수
            self.llm_just_finished = True 
            
            return ml_action     


    def check_current_ml_action_done(self, state):
        """
        checks if the current ml action is done
        Supports indexed actions like 'pickup_onion(0)'.
        """
        player = state.players[self.agent_index]
        action = self.current_ml_action
        
        # None 체크 (안전장치)
        if not action:
            return True

        # 1. Pickup 계열: 해당 물건을 손에 쥐었는지 확인
        if "pickup" in action:
            target_obj = None
            if "onion" in action: target_obj = "onion"
            elif "dish" in action: target_obj = "dish"
            elif "tomato" in action: target_obj = "tomato"
            
            # 물건을 들고 있고, 그 물건 이름이 목표와 같으면 완료
            return player.has_object() and player.get_object().name == target_obj
        
        # 2. Fill 계열: 빈 접시가 수프가 담긴 접시로 변했는지 확인
        elif "fill" in action:
            # 손에 든 것이 있고, 그 이름이 'soup'이면 완료
            return player.held_object is not None and player.held_object.name == 'soup'
        
        # 3. 손을 비우는 행동들 (Put, Place, Deliver)
        # 냄비에 넣거나, 카운터에 두거나, 서빙하면 -> 손이 빔
        elif "put" in action or "place" in action or "deliver" in action:
            return not player.has_object()
        
        # 4. Wait 계열: 대기 시간이 끝났는지 확인
        elif "wait" in action:
            return self.time_to_wait <= 0
            
        return False

    def validate_current_ml_action(self, state):
        """
        make sure the current_ml_action exists and is valid
        Supports indexed actions (e.g., pickup_onion(0))
        Tomato logic removed.
        """
        if self.current_ml_action is None:
            return False
            
        action = self.current_ml_action
        player = state.players[self.agent_index]
        pot_states_dict = self.mdp.get_pot_states(state)
        
        # 플레이어 상태 확인
        has_object = player.has_object()
        has_onion = has_object and player.get_object().name == 'onion'
        has_dish = has_object and player.get_object().name == 'dish'
        has_soup = has_object and player.get_object().name == 'soup'
        
        # 냄비 상태 계산
        import pkg_resources
        if self.overcooked_version == '1.1.0':
            soup_ready = len(pot_states_dict['ready']) > 0
            soup_cooking = len(pot_states_dict['cooking']) > 0
            pot_available_for_onion = len(pot_states_dict["empty"] + self.mdp.get_partially_full_pots(pot_states_dict)) > 0
        else:
            # 0.0.1 버전
            soup_ready = len(pot_states_dict['onion']['ready']) > 0
            soup_cooking = len(pot_states_dict['onion']['cooking']) > 0
            pot_available_for_onion = len(pot_states_dict["empty"] + pot_states_dict["onion"]['partially_full']) > 0

        # --- 검증 로직 ---
        
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

    ##################
    '''
    The followings are the Controller part almost inherited from GreedyHumanModel class
    '''
    ##################  
        
    def find_shared_counters(self, state, mlam):  
        counter_dicts = query_counter_states(self.mdp, state) 

        counter_list  = get_intersect_counter(state.players_pos_and_or[self.agent_index],
                        state.players_pos_and_or[1 - self.agent_index], 
                        self.mdp, 
                        self.mlam
                    )    

        #print('counter_list = {}'.format(counter_list))  
        lis = [] 
        for i in counter_list:  
            if counter_dicts[i] == ' ':  
                lis.append(i)       
        available_plans = mlam._get_ml_actions_for_positions(lis)
        return available_plans          

    def find_motion_goals(self, state):
        """
        Generates motion goals based on LLM output strings like 'pickup_onion(0)'.
        Uses self.global_id_mapping to ensure ID consistency.
        """
        am = self.mlam
        motion_goals = []
        player = state.players[self.agent_index]
        
        # -----------------------------------------------------------
        # 1. Helper Function
        # -----------------------------------------------------------
        # -----------------------------------------------------------
        # 1. Helper Function: Calculate interaction spots
        # -----------------------------------------------------------
        def get_interact_states_from_pos(target_pos):
            """Returns valid (pos, orientation) tuples to interact with target_pos."""
            valid_states = []
            x, y = target_pos
            
            # MDP에서 맵 크기 가져오기
            width = len(self.mdp.terrain_mtx[0])
            height = len(self.mdp.terrain_mtx)

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                adj_pos = (nx, ny)
                
                # [수정] 맵 범위를 벗어나는지 먼저 확인해야 함 (IndexError 방지)
                if 0 <= nx < width and 0 <= ny < height:
                    # 범위 안일 때만 지형 타입 확인
                    if self.mdp.get_terrain_type_at_pos(adj_pos) == ' ':
                        face_dir = (-dx, -dy)
                        valid_states.append((adj_pos, face_dir))
                        
            return valid_states

        # -----------------------------------------------------------
        # 2. Parsing Logic
        # -----------------------------------------------------------
        raw_action = self.current_ml_action.strip()
        target_index = -1
        
        # [수정] 문자열 슬라이싱 대신 정규표현식 사용
        # 괄호 '('와 ')' 사이의 숫자를 찾습니다. 앞뒤 공백이나 끝의 점(.)이 있어도 문제없습니다.
        match = self.action_regex.search(raw_action)
        if match:
            # 공백 제거 후 정수 변환
            target_index = int(match.group(1).strip())

        # -----------------------------------------------------------
        # 3. Handling Indexed Actions (Target Specific Object)
        # -----------------------------------------------------------
        # print("self.global_id_mapping :", self.global_id_mapping)
        # print("target_index :", target_index)
        if target_index != -1:
            target_list_key = None
            
            # 액션 이름에 따라 매핑 키 선택
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
            
            # [수정됨] 저장해둔 global_id_mapping에서 좌표 리스트를 가져옴
            
            if target_list_key:
                targets = self.global_id_mapping.get(target_list_key, [])
                
                if 0 <= target_index < len(targets):
                    target_pos = targets[target_index]
                    motion_goals = get_interact_states_from_pos(target_pos)
                else:
                    print(f"[Warning] Index {target_index} out of bounds for {raw_action}. Avail: {len(targets)}")

        # -----------------------------------------------------------
        # 4. Handling Non-Indexed Actions or Fallbacks
        # -----------------------------------------------------------
        if not motion_goals:
            # 인덱스가 없는 경우 기존 로직(가까운 곳 찾기 등) 유지
            if "place_obj_on_counter" in raw_action:
                motion_goals = self.find_shared_counters(state, self.mlam)     
                if len(motion_goals) == 0: 
                    motion_goals = am.place_obj_on_counter_actions(state)
            
            elif "wait" in raw_action:
                motion_goals = am.wait_actions(player)

            # 인덱스 파싱 실패 혹은 LLM이 인덱스를 안 줬을 때의 폴백(Fallback)
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
                # 토마토 등 추가 가능
                
        # -----------------------------------------------------------
        # 5. Validation
        # -----------------------------------------------------------
        motion_goals = [
            mg for mg in motion_goals
            if self.mlam.motion_planner.is_valid_motion_start_goal_pair(
                player.pos_and_or, mg
            )
        ]

        return motion_goals
 
    def choose_motion_goal(self, start_pos_and_or, motion_goals, state = None):
        """
        For each motion goal, consider the optimal motion plan that reaches the desired location.
        Based on the plan's cost, the method chooses a motion goal (either boltzmann rationally
        or rationally), and returns the plan and the corresponding first action on that plan.
        """

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
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """
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
        """
        Chooses motion goal that has the lowest cost action plan.
        Returns the motion goal itself and the first action on the plan.
        """   
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
            # print('\n\n\nBlocking Happend, executing default path\n\n\n')
            # print('current position = {}'.format(start_pos_and_or)) 
            # print('goal position = {}'.format(motion_goals))        
            # if np.random.rand() < 0.5:  
            #     return None, Action.STAY
            # else: 
            return self.get_lowest_cost_action_and_goal(start_pos_and_or, motion_goals)
        return best_goal, best_action

    def real_time_planner(self, start_pos_and_or, goal, state):   
        other_pos_and_or = state.players_pos_and_or[1 - self.agent_index]
        # 미리 만들어둔 self.cached_terrain_matrix 사용
        action_plan, plan_cost = find_path(start_pos_and_or, other_pos_and_or, goal, self.cached_terrain_matrix) 
        return action_plan, plan_cost
    
class ProPlanningAgent(ProAgent):
    def __init__(self, model="Qwen/Qwen2-VL-7B-Instruct-AWQ"):
        super().__init__(model=model)


# def generate_state_prompt(self, state):
#     ego = state.players[self.agent_index]
#     teammate = state.players[1 - self.agent_index]

#     time_prompt = f"Scene {state.timestep}: "
#     ego_object = ego.held_object.name if ego.held_object else "nothing"
#     teammate_object = teammate.held_object.name if teammate.held_object else "nothing"
#     ego_state_prompt = f"<Player {self.agent_index}> holds "
#     if ego_object == 'soup':
#         ego_state_prompt += f"a dish with {ego_object} and needs to deliver soup.  "
#     elif ego_object == 'nothing':
#         ego_state_prompt += f"{ego_object}. "
#     else:
#         ego_state_prompt += f"one {ego_object}. "
    
#     teammate_state_prompt = f"<Player {1-self.agent_index}> holds "
#     if teammate_object == 'soup':
#         teammate_state_prompt += f"a dish with {teammate_object}. "
#     elif teammate_object == "nothing":
#         teammate_state_prompt += f"{teammate_object}. "
#     else:
#         teammate_state_prompt += f"one {teammate_object}. "

    
#     kitchen_state_prompt = "Kitchen states: "
#     prompt_dict = {
#         "empty": "<Pot {id}> is empty; ",
#         "cooking": "<Pot {id}> starts cooking, the soup will be ready after {t} timesteps; ",
#         "ready": "<Pot {id}> has already cooked the soup; ",
#         "1_items": "<Pot {id}> has 1 onion; ",
#         "2_items": "<Pot {id}> has 2 onions; ",
#         "3_items": "<Pot {id}> has 3 onions and is full; "
#     }

#     pot_states_dict = self.mdp.get_pot_states(state)   

#     if pkg_resources.get_distribution("overcooked_ai").version == '1.1.0':
#         for key in pot_states_dict.keys():
#             if key == "cooking":
#                 for pos in pot_states_dict[key]:
#                     pot_id = self.pot_id_to_pos.index(pos)
#                     soup_object = state.get_object(pos)
#                     kitchen_state_prompt += prompt_dict[key].format(id=pot_id, t=soup_object.cook_time_remaining)
#             else:
#                 for pos in pot_states_dict[key]:
#                     pot_id = self.pot_id_to_pos.index(pos)
#                     kitchen_state_prompt += prompt_dict[key].format(id=pot_id) 
    
#     elif pkg_resources.get_distribution("overcooked_ai").version == '0.0.1':
#         for key in pot_states_dict.keys():
#             if key == "empty":
#                 for pos in pot_states_dict[key]: 
#                     pot_id = self.pot_id_to_pos.index(pos)
#                     kitchen_state_prompt += prompt_dict[key].format(id=pot_id)     
#             else: # key = 'onion' or 'tomota'
#                 for soup_key in pot_states_dict[key].keys():
#                     # soup_key: ready, cooking, partially_full
#                     for pos in pot_states_dict[key][soup_key]:
#                         pot_id = self.pot_id_to_pos.index(pos)
#                         soup_object = state.get_object(pos)
#                         soup_type, num_items, cook_time = soup_object.state
#                         if soup_key == "cooking":
#                             kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id, t=self.mdp.soup_cooking_time-cook_time)
#                         elif soup_key == "partially_full":
#                             pass
#                         else:
#                             kitchen_state_prompt += prompt_dict[soup_key].format(id=pot_id)


#     intersect_counters = get_intersect_counter(
#                             state.players_pos_and_or[self.agent_index], 
#                             state.players_pos_and_or[1 - self.agent_index], 
#                             self.mdp, 
#                             self.mlam
#                         )
#     counter_states = query_counter_states(self.mdp, state)  

#     if self.layout == 'forced_coordination': 
#         kitchen_state_prompt += '{} counters can be visited by <Player {}>. Their states are as follows: '.format(len(intersect_counters), self.agent_index)
#         count_states = {}  
#         for i in intersect_counters:  
#             obj_i = 'nothing' 
#             if counter_states[i] != ' ': 
#                 obj_i = counter_states[i]                
#             if obj_i in count_states:  
#                 count_states[obj_i] += 1
#             else: 
#                 count_states[obj_i]  = 1 
#         total_obj = ['onion', 'dish']
#         for i in count_states:   
#             if i == 'nothing': 
#                 continue 
#             kitchen_state_prompt += f'{count_states[i]} counters have {i}. '   
#         for i in total_obj: 
#             if i not in count_states:        
#                 kitchen_state_prompt += f'No counters have {i}. ' 

#     if self.layout == 'forced_coordination': 
#         teammate_state_prompt = ""
#     print("PROMPT:",self.layout_prompt + time_prompt + ego_state_prompt +
#             teammate_state_prompt + kitchen_state_prompt)
#     return (self.layout_prompt + time_prompt + ego_state_prompt +
#             teammate_state_prompt + kitchen_state_prompt)