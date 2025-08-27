import asyncio
import random
import types

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import Request, Depends, HTTPException, Form, Response
from fastapi import Request, HTTPException, Form, Security
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from starlette.background import BackgroundTask
from curl_cffi import requests

import utils.globals as globals
from app import app, templates, security_scheme
from chatgpt.ChatService import ChatService
from chatgpt.ChatServiceCache import ChatServiceCache
from chatgpt.authorization import refresh_all_tokens
from utils.Client import Client
from utils.Logger import logger
from utils.configs import api_prefix, scheduled_refresh, proxy_url_list, remote_pow
from utils.retry import async_retry

scheduler = AsyncIOScheduler()

# 创建缓存实例，10分钟TTL，1分钟清理间隔
chat_service_cache = ChatServiceCache(ttl=600, cleanup_interval=60)


async def check_proxy_ip():
    """检查代理IP地址"""
    if not proxy_url_list:
        logger.info("没有配置代理，跳过代理IP检查")
        return
    
    for i, proxy_url in enumerate(proxy_url_list):
        try:
            logger.info(f"正在检查代理 {i+1}/{len(proxy_url_list)}: {proxy_url}")
            
            # 设置代理
            proxies = {
                'http': proxy_url,
                'https': proxy_url
            }
            
            # 发送请求获取IP
            headers = {
                'User-Agent': 'curl/7.68.0',
                'Accept': 'text/plain'
            }
            response = requests.get(
                'https://ifconfig.co', 
                proxies=proxies, 
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                proxy_ip = response.text.strip()
                logger.info(f"代理 {proxy_url} 的出口IP: {proxy_ip}")
            else:
                logger.warning(f"代理 {proxy_url} 检查失败，状态码: {response.status_code}")
                
        except Exception as e:
            logger.error(f"检查代理 {proxy_url} 时发生错误: {str(e)}")


@app.on_event("startup")
async def app_start():
    # 检查代理IP
    await check_proxy_ip()
    
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
            oai_device_id = sentinel_token.get("oai_device_id")
            chat_service = chat_service_cache.get(oai_device_id)

            if chat_service:
                logger.info(f"Retrieved chat requirements for oai_device_id: {oai_device_id}, current cache size: {chat_service_cache.get_cache_size()}")
                chat_service.chat_token = sentinel_token.get("chat_token")
                chat_service.proof_token = sentinel_token.get("proof_token")
                chat_service.persona = 'chatgpt-freeaccount'

                chat_service.data = request_data
                await chat_service.set_model()
                chat_service.api_messages = request_data.get("messages", chat_service.api_messages)
                chat_service.max_tokens = chat_service.data.get("max_tokens", 2147483647)
                if not isinstance(chat_service.max_tokens, int):
                    chat_service.max_tokens = 2147483647

            else:
                if remote_pow:
                    logger.error(f"Cache miss for oai_device_id: {oai_device_id}. Raising an exception.")
                    raise HTTPException(status_code=500, detail="Cache miss")
                else:
                    logger.info(f"Cache miss for oai_device_id: {oai_device_id}, creating a new instance.")
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
                oai_device_id = chat_service.requirement_data.get("oai_device_id")
                #logger.info(f"add chat_requirements oai_device_id: {oai_device_id}")
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

        # 如果 remote_pow 为 True 并且其中一个参数为空，则抛出异常
        if remote_pow:
            if not chat_token or not proof_token or not oai_device_id:
                logger.error(f"Missing one of the required headers: chat_token, proof_token, or oai_device_id.")
                raise HTTPException(status_code=400, detail="Missing required headers: chat_token, proof_token, or oai_device_id")

        sentinel_token = {
            "chat_token": chat_token,
            "proof_token": proof_token,
            "oai_device_id": oai_device_id
        }

        request_data = await request.json()
        #logger.info(f"request_data: {request_data}")
    except Exception as e:
        error_message = f"Invalid JSON body: {str(e)}"
        raise HTTPException(status_code=400, detail=error_message)
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
async def clear_tokens():
    globals.token_list.clear()
    globals.error_token_list.clear()
    with open(globals.TOKENS_FILE, "w", encoding="utf-8") as f:
        pass
    logger.info(f"Token count: {len(globals.token_list)}, Error token count: {len(globals.error_token_list)}")
    tokens_count = len(set(globals.token_list) - set(globals.error_token_list))
    return {"status": "success", "tokens_count": tokens_count}

@app.post(f"/{api_prefix}/seed_tokens/clear" if api_prefix else "/seed_tokens/clear")
async def clear_seed_tokens():
    globals.seed_map.clear()
    globals.conversation_map.clear()
    with open(globals.SEED_MAP_FILE, "w", encoding="utf-8") as f:
        f.write("{}")
    with open(globals.CONVERSATION_MAP_FILE, "w", encoding="utf-8") as f:
        f.write("{}")
    logger.info(f"Seed token count: {len(globals.seed_map)}")
    return {"status": "success", "seed_tokens_count": len(globals.seed_map)}

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

async def forward_request(request: Request, endpoint: str):
    data = await request.json()
    client = Client(proxy=random.choice(proxy_url_list) if proxy_url_list else None)
    try:
        headers = {
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br, zstd',
            'accept-language': 'en-US,en;q=0.9',
            'content-type': 'application/json',
            'priority': 'u=1, i',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin'
        }
        r = await client.post(endpoint, headers=headers, json=data, timeout=5)
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

@app.post("/oauth/token")
async def proxy_oauth_token(request: Request):
    return await forward_request(request, "https://auth0.openai.com/oauth/token")

@app.post("/api/accounts/oauth/token")
async def proxy_oauth_token(request: Request):
    return await forward_request(request, "https://auth.openai.com/api/accounts/oauth/token")

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

@app.post(f"/{api_prefix}/backend-api/sentinel/delete-requirements" if api_prefix else "/backend-api/sentinel/delete-requirements")
async def delete_requirements(request: Request):
    chat_service = None
    try:
        # 从请求头中读取数据
        oai_device_id = request.headers.get("oai-device-id")
        chat_service = chat_service_cache.get(oai_device_id)

        if chat_service:
            await chat_service.close_client()

        return JSONResponse('success', media_type="application/json")

    except HTTPException as e:
        if chat_service:
            await chat_service.close_client()
        logger.error(f"HTTPException encountered: {str(e)}")
        raise HTTPException(status_code=e.status_code, detail=e.detail)

    except Exception as e:
        if chat_service:
            await chat_service.close_client()
        logger.error(f"Unexpected server error: {str(e)}")
        raise HTTPException(status_code=500, detail="Server error")