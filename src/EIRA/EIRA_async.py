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
    def __init__(self, model="Qwen/Qwen2-VL-7B-Instruct-AWQ"):
        self.agent_index = None
        self.model = model

        self.openai_api_keys = []
        self.load_openai_keys()
        self.key_rotation = True

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
            prompt_level='l2-ap', # ['l1-p', 'l2-ap', 'l3-aip']
            belief_revision=False,
            retrival_method="recent_k",
            K=3,
            auto_unstuck=True,
            controller_mode='new', # the default overcooked-ai Greedy controller
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
            # [중요] 원본 mdp가 오염되지 않도록 deepcopy를 필수적으로 사용해야 합니다.
            'matrix': copy.deepcopy(self.mlam.mdp.terrain_mtx), 
            'height': len(self.mlam.mdp.terrain_mtx),
            'width': len(self.mlam.mdp.terrain_mtx[0])
        }
        self.action_regex = re.compile(r'\((\s*\d+\s*)\)')
        self.overcooked_version = pkg_resources.get_distribution("overcooked_ai").version

        self.layout_prompt = self.generate_layout_prompt()

        # [NEW] 비동기 처리를 위한 변수 초기화
        self.is_thinking = False       # 현재 스레드가 돌고 있는지
        self.think_thread = None       # 스레드 객체
        self.next_ml_action = None     # 스레드 결과 저장소
        self.lock = threading.Lock()   # 데이터 경쟁 방지용 락
        self.prev_partner_move = None
        self.current_thought = ""
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
        
        # [NEW] 비동기 상태 초기화
        with self.lock:
            self.is_thinking = False
            self.next_ml_action = None

    def set_agent_index(self, agent_index):
        self.agent_index = agent_index
        self.planner = self.create_gptmodule("planner", retrival_method=self.retrival_method, K=self.K)
        self.explainer = self.create_gptmodule("explainer", retrival_method='recent_k', K=self.K)

        print(self.planner.instruction_head_list[0]['content'])
      
    def generate_layout_prompt(self):
        # 1. 매핑 초기화
        self.global_id_mapping = {
            "onion_dispenser": [],
            "dish_dispenser": [],
            "tomato_dispenser": [],
            "serving": [],
            "pot": []
        }
        self.pot_id_to_pos = [] # 레거시 호환용

        # 2. 출력할 이름 정의
        name_map = {
            "onion_dispenser": "Onion Dispenser",
            "dish_dispenser": "Dish Dispenser",
            "tomato_dispenser": "Tomato Dispenser",
            "serving": "Serving Loc",
            "pot": "Pot"
        }

        layout_prompt = "Layout Information: "
        
        # 3. 각 객체 타입별로 순회하며 "ID at 좌표" 형식 생성
        for key, readable_name in name_map.items():
            # MDP에서 위치 정보 가져오기
            locations = getattr(self.mdp, f"get_{key}_locations")()
            
            # [중요] ID 매핑 저장 (Action 실행 시 사용됨)
            self.global_id_mapping[key] = locations
            
            # 해당 객체가 맵에 없으면 건너뜀
            if not locations:
                continue
                
            # 문자열 생성: "<Pot 0> at (4, 2)" 형태
            items_str_list = []
            for i, pos in enumerate(locations):
                items_str_list.append(f"<{readable_name} {i}> at {pos}")
                
                # 레거시 호환용
                if key == "pot":
                    self.pot_id_to_pos.append(pos)
            
            # 리스트를 콤마로 연결하여 추가
            # 예: "Pots: <Pot 0> at (4, 2), <Pot 1> at (4, 3); "
            layout_prompt += f"{readable_name}s: {', '.join(items_str_list)}; "

        layout_prompt = layout_prompt.strip() + "\n"
        return layout_prompt

    def generate_state_prompt(self, state):
        ego = state.players[self.agent_index]
        teammate = state.players[1 - self.agent_index]
            
        layout_info = ""

        history_prompt = ""
        self.partner_move_history = []

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
        if len(self.partner_move_history) > 5:
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
        self.layout_prompt = self.generate_layout_prompt()
        parts = [
            self.layout_prompt.strip(),  # Layout
            layout_info.strip(),         # (비어있을 수 있음)
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
    def _thinking_process(self, state_dict):
        try:
            # 1. 딕셔너리를 다시 객체로 복구 (OvercookedState)
            # 이 상태의 'current_state'는 맵 정보는 모르고 플레이어 위치만 아는 상태입니다.
            current_state = OvercookedState.from_dict(state_dict)
            
            # 2. LLM 호출 함수 실행
            # self.generate_ml_action 내부에서는 
            # self.mdp (맵 정보)와 current_state (플레이어 정보)를 둘 다 사용할 것입니다.
            # self는 공유되므로 접근 가능합니다.
            
            plan = self.generate_ml_action(current_state)

            with self.lock:
                self.next_ml_action = plan

        except Exception as e:
            # 에러 처리
            import traceback
            traceback.print_exc()
        finally:
            with self.lock:
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
            #self.generate_failure_feedback(state_snapshot)
            
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

    def action(self, state):
        """
        [Modified] Main Game Loop Action
        이 함수는 절대 Blocking되면 안 됩니다.
        """
        # [Timing] 전체 틱 소요 시간 (가벼운 연산만 포함됨)
        total_start_time = time.perf_counter()
        
        # 1. 현재 수행 중인 High-Level Action 관리
        if self.current_ml_action is not None:
            # (A) 완료 체크
            if self.check_current_ml_action_done(state):
                self.generate_success_feedback(state)
                self.current_ml_action = None 
            
            # (B) 유효성 검증 (실패 시 즉시 중단 및 복구 스레드 시작)
            elif not self.validate_current_ml_action(state):
                self.trace = False
                self.current_ml_action = None # 행동 취소
                
                # 복구(Correction) 스레드 시작
                with self.lock:
                    if not self.is_thinking:
                        self.is_thinking = True
                        # 메인 게임 상태가 변하므로 스냅샷(Deepcopy) 전달 필수
                        state_snapshot = copy.deepcopy(state)
                        self.think_thread = threading.Thread(target=self._correction_process, args=(state_snapshot,))
                        self.think_thread.start()

        # 2. 수행할 행동이 없을 때 (생각 결과 확인 또는 생각 시작)
        if self.current_ml_action is None:
            with self.lock:
                # (A) 스레드 결과가 도착했는지 확인
                if self.next_ml_action is not None:
                    self.current_ml_action = self.next_ml_action
                    self.next_ml_action = None
                    self.current_ml_action_steps = 0
                    #print(f"\n[Agent] New Plan Arrived: {self.current_ml_action}")
                    # 상세 LLM 타이밍 로그가 있다면 출력 (generate_ml_action에서 저장됨)
                    if hasattr(self, 'last_llm_timing'):
                        print(f" >> LLM Timing: {json.dumps(self.last_llm_timing)}")
                
                # (B) 결과도 없고, 생각 중도 아니라면 -> 일반 계획(Thinking) 시작
                elif not self.is_thinking:
                    self.is_thinking = True
                    #print(f"\n[Agent] Start Thinking... (Non-blocking)")
                    
                    safe_data = state.to_dict()
                    self.think_thread = threading.Thread(target=self._thinking_process, args=(safe_data,))
                    self.think_thread.start()
                    

        # 3. 실제 Low-Level Action 반환 (Motion Planning)
        
        # 아직 생각 중이거나 행동이 정해지지 않았으면 대기(STAY)
        if self.current_ml_action is None:
            random_action = random.choice(Direction.ALL_DIRECTIONS)
            if self.overcooked_version == '1.1.0':
                return Action.STAY, {}
            return Action.STAY

        # 행동이 정해져 있으면 Motion Planner 실행 (빠른 연산이므로 메인 스레드에서 수행)
        self.trace = True 
        chosen_action = Action.STAY
        
        if "wait" in self.current_ml_action:
            self.current_ml_action_steps += 1
            self.time_to_wait -= 1
            if self.time_to_wait <= 0:
                self.current_ml_action = None
            
            # 랜덤 움직임 (Optional: 대기 중 무작위성)
            lis_actions = self.mdp.get_valid_actions(state.players[self.agent_index])
            chosen_action = lis_actions[np.random.randint(0,len(lis_actions))]
        else:
            # 일반 이동
            possible_motion_goals = self.find_motion_goals(state)    
            current_motion_goal, chosen_action = self.choose_motion_goal(
                state.players_pos_and_or[self.agent_index], 
                possible_motion_goals, 
                state
            )
            
            # 경로가 없거나 실패 시
            if chosen_action is None:
                 self.current_ml_action = "wait(1)"
                 self.time_to_wait = 1
                 chosen_action = Action.STAY

        self.prev_state = state
        self.current_ml_action_steps += 1

        # [Timing Report - Optional] 너무 잦은 출력 방지를 위해 10스텝마다 출력하거나 주석 처리 가능
        # print(f"[Tick {state.timestep}] Cost: {time.perf_counter() - total_start_time:.5f}s")

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
        [Modified] 내부 실행 시간을 측정하여 self.last_llm_timing에 저장합니다.
        이 함수는 이제 백그라운드 스레드에서 호출됩니다.
        """
        # 상세 타이밍 기록용 딕셔너리
        breakdown = {}
        t_start = time.perf_counter()

        # -------------------------------------------------
        # 1. 프롬프트 생성 (Prompt Construction)
        # -------------------------------------------------
        if self.prompt_level == "l3-aip" and self.belief_revision:
            belief_prompt = self.generate_belief_prompt()
            # print("belief_prompt: ", belief_prompt) # 로그가 너무 길면 주석 처리
        else:
            belief_prompt = ''
        
        state_prompt = belief_prompt + self.generate_state_prompt(state)

        # print(f"\n\n### Observation module to GPT\n")   
        # print(f"{state_prompt}")

        state_message = {"role": "user", "content": state_prompt}
        self.planner.current_user_message = state_message
        
        t_prompt = time.perf_counter()
        breakdown['1_Prompt_Prep'] = t_prompt - t_start

        # -------------------------------------------------
        # 2. LLM 추론 (Inference / API Call)
        # -------------------------------------------------
        # 실제 시간이 가장 오래 걸리는 구간
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
            print(f"Intention for Player {1 - self.agent_index}: {generated_intention}")

        ml_action = self.parse_ml_action(response, self.agent_index)

        if "wait" not in ml_action:
            self.planner.add_msg_to_dialog_history({"role": "assistant", "content": ml_action})
        
        #print(f"Player {self.agent_index}: {ml_action}")
        self.current_ml_action_steps = 0

        t_parse = time.perf_counter()
        breakdown['3_Parsing'] = t_parse - t_inference
        
        # [NEW] 클래스 멤버 변수에 상세 기록 저장 (action 메서드에서 읽기 위함)
        self.last_llm_timing = breakdown
        
        return ml_action

    ##################
    '''
    The followings are the Verificator part
    '''
    ##################

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
            if np.random.rand() < 0.5:  
                return None, Action.STAY
            else: 
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