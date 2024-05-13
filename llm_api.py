#!/usr/bin/env python
# coding: utf-8

from fastapi import FastAPI, APIRouter, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware # 필요 시 사용
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse
from typing import List, Union
from pydantic import BaseModel

from llm_utils import Prompt, logger, llm_answer, today

import json, uvicorn, torch

# class InputData(BaseModel):  # API 요청 폼
#     text: str      # 챗봇에 입력하는 텍스트
#     userId: str
#     tenantId: str
#     userType: str

prompt_router = APIRouter(prefix='/prompt')
router = APIRouter(prefix='/api')

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# HTML Templates
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=['GET', 'POST', 'DELETE'],
    allow_headers=["*"],
    allow_credentials=True,
)

@app.get("/", response_class=HTMLResponse)
async def MainPage(request: Request):
    return templates.TemplateResponse(
        "chat 1.html", {"request":request}, 
    )

@router.get('/')
def HelloWorld():
    return {"message":"Hello World"}

# ========== LLM Prompt ========== 

@prompt_router.get('/GetPrompt')
def GetSQLPrompt(prompt: Prompt = Depends(Prompt)) -> dict:
    return JSONResponse(prompt.get_prompt().to_dict()['Prompt'], status_code = 200, headers = None, media_type = None)

@prompt_router.post('/AddPrompt')
def AddSQLPrompt(user_prompt:str, assistant_prompt:str, prompt: Prompt = Depends(Prompt)) -> dict:
    return JSONResponse(prompt.add_prompt(user_prompt, assistant_prompt).to_dict()['Prompt'], status_code = 200, headers = None, media_type = None)

@prompt_router.delete('/DeletePrompt')
def DeleteSQLPrompt(prompt_idx:int, prompt: Prompt = Depends(Prompt)) -> dict:
    try:
        return JSONResponse(prompt.delete_prompt(prompt_idx).to_dict()['Prompt'], status_code = 200, headers = None, media_type = None)
    except:
        raise HTTPException(status_code=404, detail="Prompt index number not found")

# ========== LLM Process ========== 

@router.get('/GetAnswer')
async def GetAnswer(input_text:str, prompt: Prompt = Depends(Prompt)):
    result = llm_answer(prompt.make_prompt(), input_text)
    try:
        return JSONResponse([json.loads(result)], status_code = 200, headers = None, media_type = None)
        # return str(result)
    except torch.cuda.OutOfMemoryError:
        logger.exception(f'{today()} - Out Of Memory')
        return HTTPException(status_code=400, detail='Out of memory')
    except Exception:
        try:
            if result[0] == '{' and result[-1] != '}':
                result += '}'
                return JSONResponse([json.loads(result)], status_code = 200, headers = None, media_type = None)
            return str(result)
        except Exception:
            logger.exception('API error')
            return "입력하신 정보를 찾을 수 없습니다."

# @router.post('/GetGPTAns')
# async def GetGPTAns(
#     input_data: List[InputData],                            # API 요청 데이터
#     schema_prompt: SchemaPrompt = Depends(SchemaPrompt),
#     sql_prompt: SQLPrompt = Depends(SQLPrompt),
#     qa_prompt: QAPrompt = Depends(QAPrompt)
#     ):
    
#     input_data = input_data[0]
    
#     # Input 파라미터 추출
#     text = input_data.text
#     user_id = input_data.userId
#     tenant_id = input_data.tenantId
#     user_type = input_data.userType
    
#     gpt_msg = get_gpt_to_sql(sql_prompt.make_sql_prompt(), schema_prompt.make_schema_prompt(), qa_prompt.make_qa_prompt() , str(text), user_id, tenant_id, user_type)
    
#     logger.info(f"User Input - {text}\nGPT answer - {gpt_msg}\n")
    
#     if gpt_msg:
#         return str(gpt_msg)
#     else:
#         return "입력하신 정보를 찾을 수 없습니다."

# Routing Process

app.include_router(prompt_router)
app.include_router(router)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)