import asyncio
import random
import types

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Request, Depends, HTTPException, Form, Response
from fastapi import Request, HTTPException, Form, Security
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from starlette.background import BackgroundTask

import utils.globals as globals
from app import app, templates, security_scheme
from chatgpt.ChatService import ChatService
from chatgpt.ChatServiceCache import ChatServiceCache
from chatgpt.authorization import refresh_all_tokens
from utils.Client import Client
from utils.Logger import logger
from utils.configs import api_prefix, scheduled_refresh, proxy_url_list
from utils.retry import async_retry

scheduler = AsyncIOScheduler()

# 创建缓存实例，6分钟TTL，1分钟清理间隔
chat_service_cache = ChatServiceCache(ttl=360, cleanup_interval=60)

@app.on_event("startup")
async def app_start():
    asyncio.create_task(chat_service_cache.start_cleanup_loop())
    if scheduled_refresh:
        scheduler.add_job(id='refresh', func=refresh_all_tokens, trigger='cron', hour=3, minute=0, day='*/2',
                          kwargs={'force_refresh': True})
        scheduler.start()
        asyncio.get_event_loop().call_later(0, lambda: asyncio.create_task(refresh_all_tokens(force_refresh=False)))


async def to_send_conversation(request_data, req_token, sentinel_token):
    chat_service = None
    try:
        if sentinel_token and sentinel_token.get("chat_token") and sentinel_token.get("proof_token") and sentinel_token.get("oai_device_id"):
            oai_device_id = sentinel_token["oai_device_id"]
            chat_service = chat_service_cache.get(oai_device_id)

            if chat_service:
                logger.info(f"get chat_requirements oai_device_id: {oai_device_id}")
                chat_service.chat_token = sentinel_token["chat_token"]
                chat_service.proof_token = sentinel_token["proof_token"]
                chat_service.oai_device_id = oai_device_id
                chat_service.base_headers['oai-device-id'] = oai_device_id
                chat_service.persona = 'chatgpt-freeaccount'
            else:
                # 缓存未命中时，创建新的实例
                chat_service = ChatService(req_token)
                await chat_service.set_dynamic_data(request_data)
                await chat_service.get_chat_requirements()
        else:
            # 无 sentinel_token 时直接创建新的 ChatService 实例
            chat_service = ChatService(req_token)
            await chat_service.set_dynamic_data(request_data)
            await chat_service.get_chat_requirements()

            if sentinel_token is None and chat_service.requirement_data:
                oai_device_id = chat_service.oai_device_id
                logger.info(f"add chat_requirements oai_device_id: {oai_device_id}")
                chat_service_cache.set(oai_device_id, chat_service)

        return chat_service
    except HTTPException as e:
        if chat_service:
            await chat_service.close_client()
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        if chat_service:
            await chat_service.close_client()
        logger.error(f"Server error, {str(e)}")
        raise HTTPException(status_code=500, detail="Server error")


async def process(request_data, req_token, sentinel_token):
    chat_service = await to_send_conversation(request_data, req_token, sentinel_token)
    try:
        await chat_service.prepare_send_conversation()
        res = await chat_service.send_conversation()
        return chat_service, res
    except HTTPException as e:
        await chat_service.close_client()
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        await chat_service.close_client()
        logger.error(f"Server error, {str(e)}")
        raise HTTPException(status_code=500, detail="Server error")


@app.post(f"/{api_prefix}/v1/chat/completions" if api_prefix else "/v1/chat/completions")
async def send_conversation(request: Request, credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    req_token = credentials.credentials
    try:
        # 从请求头中读取数据
        chat_token = request.headers.get("openai-sentinel-chat-requirements-token", None)
        proof_token = request.headers.get("openai-sentinel-proof-token", None)
        oai_device_id = request.headers.get("oai-device-id", None)
        sentinel_token = {
            "chat_token": chat_token,
            "proof_token": proof_token,
            "oai_device_id": oai_device_id
        }

        request_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": "Invalid JSON body"})
    chat_service, res = await async_retry(process, request_data, req_token, sentinel_token)
    try:
        if isinstance(res, types.AsyncGeneratorType):
            background = BackgroundTask(chat_service.close_client)
            return StreamingResponse(res, media_type="text/event-stream", background=background)
        else:
            background = BackgroundTask(chat_service.close_client)
            return JSONResponse(res, media_type="application/json", background=background)
    except HTTPException as e:
        await chat_service.close_client()
        if e.status_code == 500:
            logger.error(f"Server error, {str(e)}")
            raise HTTPException(status_code=500, detail="Server error")
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        await chat_service.close_client()
        logger.error(f"Server error, {str(e)}")
        raise HTTPException(status_code=500, detail="Server error")


@app.get(f"/{api_prefix}/tokens" if api_prefix else "/tokens", response_class=HTMLResponse)
async def upload_html(request: Request):
    tokens_count = len(set(globals.token_list) - set(globals.error_token_list))
    return templates.TemplateResponse("tokens.html",
                                      {"request": request, "api_prefix": api_prefix, "tokens_count": tokens_count})


@app.post(f"/{api_prefix}/tokens/upload" if api_prefix else "/tokens/upload")
async def upload_post(text: str = Form(...)):
    lines = text.split("\n")
    for line in lines:
        if line.strip() and not line.startswith("#"):
            globals.token_list.append(line.strip())
            with open(globals.TOKENS_FILE, "a", encoding="utf-8") as f:
                f.write(line.strip() + "\n")
    logger.info(f"Token count: {len(globals.token_list)}, Error token count: {len(globals.error_token_list)}")
    tokens_count = len(set(globals.token_list) - set(globals.error_token_list))
    return {"status": "success", "tokens_count": tokens_count}


@app.post(f"/{api_prefix}/tokens/clear" if api_prefix else "/tokens/clear")
async def upload_post():
    globals.token_list.clear()
    globals.error_token_list.clear()
    with open(globals.TOKENS_FILE, "w", encoding="utf-8") as f:
        pass
    logger.info(f"Token count: {len(globals.token_list)}, Error token count: {len(globals.error_token_list)}")
    tokens_count = len(set(globals.token_list) - set(globals.error_token_list))
    return {"status": "success", "tokens_count": tokens_count}


@app.post(f"/{api_prefix}/tokens/error" if api_prefix else "/tokens/error")
async def error_tokens():
    error_tokens_list = list(set(globals.error_token_list))
    return {"status": "success", "error_tokens": error_tokens_list}


@app.get(f"/{api_prefix}/tokens/add/{{token}}" if api_prefix else "/tokens/add/{token}")
async def add_token(token: str):
    if token.strip() and not token.startswith("#"):
        globals.token_list.append(token.strip())
        with open(globals.TOKENS_FILE, "a", encoding="utf-8") as f:
            f.write(token.strip() + "\n")
    logger.info(f"Token count: {len(globals.token_list)}, Error token count: {len(globals.error_token_list)}")
    tokens_count = len(set(globals.token_list) - set(globals.error_token_list))
    return {"status": "success", "tokens_count": tokens_count}

@app.post("/oauth/token")
async def proxy_oauth_token(request: Request):
    data = await request.json()
    client = Client(proxy=random.choice(proxy_url_list) if proxy_url_list else None)

    try:
        # 将请求转发到OpenAI OAuth端点
        r = await client.post("https://auth0.openai.com/oauth/token", json=data, timeout=5)

        # 直接返回OpenAI的响应
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers={key: value for key, value in r.headers.items() if key.lower() != "content-encoding"}
        )
    except Exception as e:
        logger.error(f"无法代理OAuth令牌请求：{str(e)}")
        raise HTTPException(status_code=500, detail="无法代理OAuth令牌请求。")
    finally:
        await client.close()
        del client


@app.post(f"/{api_prefix}/backend-api/sentinel/chat-requirements" if api_prefix else "/backend-api/sentinel/chat-requirements")
async def chat_requirements(credentials: HTTPAuthorizationCredentials = Security(security_scheme)):
    req_token = credentials.credentials
    request_data = {
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": "Say this is a test!"
            }
        ],
        "req_type": "requirements"
    }
    chat_service = await async_retry(to_send_conversation, request_data, req_token, None)
    try:
        #background = BackgroundTask(chat_service.close_client)
        #, background=background
        return JSONResponse(chat_service.requirement_data, media_type="application/json")
    except HTTPException as e:
        await chat_service.close_client()
        if e.status_code == 500:
            logger.error(f"Server error, {str(e)}")
            raise HTTPException(status_code=500, detail="Server error")
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except Exception as e:
        await chat_service.close_client()
        logger.error(f"Server error, {str(e)}")
        raise HTTPException(status_code=500, detail="Server error")