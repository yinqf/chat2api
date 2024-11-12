import time
import warnings

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from utils.Logger import logger
from utils.configs import enable_gateway, api_prefix

warnings.filterwarnings("ignore")


log_config = uvicorn.config.LOGGING_CONFIG
default_format = "%(asctime)s | %(levelname)s | %(message)s"
access_format = r'%(asctime)s | %(levelname)s | %(client_addr)s: %(request_line)s %(status_code)s'
log_config["formatters"]["default"]["fmt"] = default_format
log_config["formatters"]["access"]["fmt"] = access_format

app = FastAPI(
    docs_url=f"/{api_prefix}/docs",    # 设置 Swagger UI 文档路径
    redoc_url=f"/{api_prefix}/redoc",  # 设置 Redoc 文档路径
    openapi_url=f"/{api_prefix}/openapi.json"  # 设置 OpenAPI JSON 路径
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 自定义中间件记录接口执行时长，并以特定格式输出日志
@app.middleware("http")
async def log_execution_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time

    # 构建符合格式的日志输出
    client_info = f"{request.client.host}:{request.client.port}"
    log_message = (
        f"{client_info}: {request.method} {request.url.path} "
        f"{response.status_code} OK - {duration:.2f} seconds"
    )
    logger.info(log_message)

    return response

templates = Jinja2Templates(directory="templates")
security_scheme = HTTPBearer()

from app import app

import api.chat2api

if enable_gateway:
    import gateway.share
    import gateway.login
    import gateway.chatgpt
    import gateway.gpts
    import gateway.admin
    import gateway.v1
    import gateway.backend
else:
    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "TRACE"])
    async def reverse_proxy():
        raise HTTPException(status_code=404, detail="Gateway is disabled")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5005, access_log=False)
    # uvicorn.run("app:app", host="0.0.0.0", port=5005, ssl_keyfile="key.pem", ssl_certfile="cert.pem")
