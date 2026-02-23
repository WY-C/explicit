import openai
from rich import print as rprint
import time
from typing import Union
from .utils import convert_messages_to_prompt, retry_with_exponential_backoff
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# 토큰 제한 테이블 (그대로 유지)
TOKEN_LIMIT_TABLE = {
    "text-davinci-003": 4080,
    "gpt-3.5-turbo": 4096,
    "gpt-3.5-turbo-0301": 4096,
    "gpt-3.5-turbo-16k": 16384,
    "gpt-4": 8192,
    "gpt-4-0314": 8192,
    "gpt-4-32k": 32768,
    "gpt-4-32k-0314": 32768,
    "Qwen/Qwen3-VL-8B-Instruct": 8192,
    "Qwen/Qwen2-VL-7B-Instruct-AWQ": 8192
}

class Module(object):
    """
    This module is responsible for communicating with GPTs or Local LLMs.
    """
    def __init__(self, 
                 role_messages, 
                 model="Qwen/Qwen3-VL-8B-Instruct",
                 retrival_method="recent_k",
                 K=3,
                 api_base="http://192.168.0.19:8000/v1", # [추가] 기본값 설정
                 api_key=None):                        # [추가]
        '''
        args:  
        api_base: URL for the API server
        '''

        self.model = model
        self.retrival_method = retrival_method
        self.K = K
        
        # [추가] API 설정 저장
        self.api_base = api_base
        self.api_key = api_key

        # 모델명에 gpt, claude, qwen 등이 있으면 chat 모델로 간주
        self.chat_model = True 
        # (원하시면 구체적인 필터링 로직을 넣어도 됩니다)
        # self.chat_model = True if any(x in self.model for x in ["gpt", "claude", "Qwen", "Llama"]) else False

        self.instruction_head_list = role_messages
        self.dialog_history_list = []
        self.current_user_message = None
        self.cache_list = None

    def add_msgs_to_instruction_head(self, messages: Union[list, dict]):
        if isinstance(messages, list):
            self.instruction_head_list += messages
        elif isinstance(messages, dict):
            self.instruction_head_list += [messages]

    def add_msg_to_dialog_history(self, message: dict):
        self.dialog_history_list.append(message)
    
    def get_cache(self)->list:
        if self.retrival_method == "recent_k":
            if self.K > 0:
                return self.dialog_history_list[-self.K:]
            else: 
                return []
        else:
            return None 
           
    @property
    def query_messages(self)->list:
        return self.instruction_head_list + self.cache_list + [self.current_user_message]
    
    @retry_with_exponential_backoff
    def query(self, key=None, stop=None, temperature=0.0, debug_mode = 'Y', trace = True):
        # [핵심 수정] 요청을 보내기 직전에 주소와 키를 강제로 설정합니다.
        openai.api_base = self.api_base  # <--- 이 부분이 빠지면 OpenAI로 연결됩니다!
        
        # 인자로 받은 key가 있으면 그걸 쓰고, 없으면 self.api_key를 씁니다.
        # (ProAgent에서 key를 넘겨줄 때 "EMPTY"가 넘어오더라도 여기서 api_base가 localhost면 작동합니다)
        if key:
            openai.api_key = key
        elif self.api_key:
            openai.api_key = self.api_key

        rec = self.K  
        if trace == True: 
            self.K = 0 
        self.cache_list = self.get_cache()
        messages = self.query_messages
        if trace == False: 
            messages[len(messages) - 1]['content'] += " Based on the failure explanation and scene description, analyze and plan again." 
        self.K = rec 
        response = "" 
        
        get_response = False
        retry_count = 0
        
        while not get_response:  
            if retry_count > 3:
                rprint("[red][ERROR][/red]: Query GPT failed for over 3 times!")
                return {}
            try:  
                if self.model in ['text-davinci-003']:
                    prompt = convert_messages_to_prompt(messages) 
                    response = openai.Completion.create(
                        model=self.model,
                        prompt=prompt,
                        stop=stop,
                        temperature= 0.1, 
                        max_tokens = 512
                    )
                    #time.sleep(1) # 로컬 속도에 맞춰 조절
                
                # 대부분의 채팅 모델(Qwen 포함)은 여기로 진입
                else:
                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=messages,
                        stop=stop,
                        temperature=0.0, 
                        max_tokens = 350
                    )
                    #time.sleep(1) 
                
                get_response = True

            except Exception as e:
                retry_count += 1
                rprint(f"[red][OPENAI ERROR][/red]: {e}")
                rprint(f"[yellow]Check URL:[/yellow] {openai.api_base}") # 디버깅용 출력 추가
                #time.sleep(5)  
        return self.parse_response(response)

    def parse_response(self, response):
        if self.model == 'claude': 
            return response 
        elif self.model in ['text-davinci-003']:
            return response["choices"][0]["text"]
        else:
            # ChatCompletion 응답 구조
            return response["choices"][0]["message"]["content"]

    def restrict_dialogue(self):
        limit = TOKEN_LIMIT_TABLE.get(self.model, 1024) # 키 에러 방지용 get 사용
        # print(f'Current token: {self.prompt_token_length}')
        # ... (기존 코드 내용이 있다면 유지) ...
        pass
        
    def reset(self):
        self.dialog_history_list = []