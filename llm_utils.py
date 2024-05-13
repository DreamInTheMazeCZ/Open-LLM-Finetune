#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import warnings, ast, datetime, logging, os
warnings.filterwarnings('ignore')

from logging import handlers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed

quantization_config=BitsAndBytesConfig(load_in_8bit=True)

# ========== LLM Configure ==========
model_id = "beomi/Llama-3-Open-Ko-8B-Instruct-preview"

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    quantization_config=quantization_config,
)

# ========== Logging ==========
now_date = datetime.datetime.now().strftime("%Y-%m-%d") # Current Date
now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # Current Time

if not os.path.isdir(f'./logs'):
    os.mkdir(f'./logs')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = handlers.TimedRotatingFileHandler(
    f'logs/GPTAPI_log_{now_date}.log',
    when='d',
    interval=7,
    backupCount=7,
    atTime='midnight',
    encoding='utf-8'
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)

logger = logging.getLogger()
if not logger.handlers:
    logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# Prompt File Directory
file_dir = './PromptFile/'

if not os.path.isdir(file_dir):
    os.mkdir(file_dir)

def today():
    return datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d %H:%M:%S %A')


class Prompt():
    """
    
    """
    __prompt_list = []
    
    def __init__(self):
        '''
        
        '''
        if os.path.isfile(f'{file_dir}llm_prompt.csv'):
            Prompt.__prompt_list = pd.read_csv(f'{file_dir}llm_prompt.csv', encoding='utf-8-sig', names=['Prompt'], header=0)                
        else:
            pd.DataFrame(columns=['Prompt']).to_csv(f'{file_dir}llm_prompt.csv', encoding='utf-8-sig', index=False)
        
    def get_prompt(self) -> pd.DataFrame:
        '''
        
        '''
        return Prompt.__prompt_list
        
    def add_prompt(self, user_prompt: str, assistant_prompt: str) -> pd.DataFrame:
        '''
        
        '''
        dict_process = [{"role":"user","content":user_prompt}, {"role":"assistant","content":assistant_prompt}]
        add_process = pd.DataFrame([str(dict_process[0]) + ', ' + str(dict_process[1])], columns=['Prompt'])
        Prompt.__prompt_list = Prompt.__prompt_list._append(add_process, ignore_index = True)
        Prompt.__prompt_list.to_csv(f'{file_dir}llm_prompt.csv', encoding='utf-8-sig', index=False)
        return Prompt.__prompt_list
    
    def delete_prompt(self, index_num: int) -> pd.DataFrame:
        '''
        
        '''
        Prompt.__prompt_list = Prompt.__prompt_list.drop(index=index_num).reset_index(drop=True)
        Prompt.__prompt_list.to_csv(f'{file_dir}llm_prompt.csv', encoding='utf-8-sig', index=False)
        return Prompt.__prompt_list
    
    def make_prompt(self) -> list:
        '''
        
        '''
        prompts = []
        for pt in Prompt.__prompt_list.Prompt.tolist():
            temp_list = [ast.literal_eval(pt)[0]] + [ast.literal_eval(pt)[1]]
            prompts += temp_list
        return prompts
    
# ========== Setup Datetime ==========

def calc_day(new:bool, day:int):
    if new:
        new_day = datetime.datetime.strftime(datetime.datetime.strptime(today(), '%Y-%m-%d %H:%M:%S %A') + datetime.timedelta(day), '%Y-%m-%d ')
        return new_day
    else:
        old_day = datetime.datetime.strftime(datetime.datetime.strptime(today(), '%Y-%m-%d %H:%M:%S %A') - datetime.timedelta(day), '%Y-%m-%d ')
        return old_day

reserve_list = [
    {'role':'user', 'content':'내일 오전 10시~11시까지 중회의실 예약해줘'},
    {'role':'assistant', 'content':f'{{"inquirytype":"reserveMeeting", "meetingRoom":"중회의실", "startDate": "{calc_day(True, 1)} 10:00:00", "endDate": "{calc_day(True, 1)} 11:00:00"}}'},
    {'role':'user', 'content':'모레 오후 3시부터 5시까지 대회의실 예약해 줄래?'},
    {'role':'assistant', 'content':f'{{"inquirytype":"reserveMeeting", "meetingRoom":"대회의실", "startDate": "{calc_day(True, 2)} 15:00:00", "endDate": "{calc_day(True, 2)} 17:00:00"}}'},
    ]

def llm_answer(prompt_list:list, input_text:str):
    
    prompt = '''당신은 텍스트에서 엔터티를 추출하도록 설계된 도우미다.
    사용자는 텍스트 문자열을 붙여넣고 텍스트에서 추출한 엔터티를 JSON 개체로 응답한다.

    오늘의 날짜 및 시각은 ''' + today() + '''이다.
    출력 형식의 예는 다음과 같다.

    USER : "홍길동의 주소는 뭐야?"
    ASSISTANT : {{"inquirytype":"call", "name":"홍길동", "reqData":"주소"}}

    USER : "내일 오전 10시~11시까지 중회의실 예약해줘"
    ASSISTANT : {{"inquirytype":"reserveMeeting", "meetingRoom":"중회의실", "startDate": "{0} 10:00:00", "endDate": "{0} 11:00:00"}}
    
    - 다른 내용 없이 JSON 내용만 반환한다.
    - 여러 inquery일 경우 inquery 별 JSON으로 최대한 자세하게 반환한다.
    - 예가 없는 경우 "이해할 수 없습니다."라고 반환한다.'''.format(calc_day(True, 1))

    messages = [{"role": "system", "content": prompt}] + prompt_list + reserve_list + [{"role": "user", "content": input_text}]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # ========== Set Seed ==========
    set_seed(8020)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=200,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    ).to("cuda")

    response = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(response, skip_special_tokens=True)