import hashlib
import json
import random
import re
import time
import uuid
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

import pybase64

from utils.Logger import logger
from utils.configs import conversation_only

cores = [8, 16, 24, 32]
timeLayout = "%a %b %d %Y %H:%M:%S"

cached_scripts = []
cached_dpl = ""
cached_time = 0
cached_require_proof = ""

navigator_key = [
    "registerProtocolHandler−function registerProtocolHandler() { [native code] }",
    "storage−[object StorageManager]",
    "locks−[object LockManager]",
    "appCodeName−Mozilla",
    "permissions−[object Permissions]",
    "share−function share() { [native code] }",
    "webdriver−false",
    "managed−[object NavigatorManagedData]",
    "canShare−function canShare() { [native code] }",
    "vendor−Google Inc.",
    "vendor−Google Inc.",
    "mediaDevices−[object MediaDevices]",
    "vibrate−function vibrate() { [native code] }",
    "storageBuckets−[object StorageBucketManager]",
    "mediaCapabilities−[object MediaCapabilities]",
    "getGamepads−function getGamepads() { [native code] }",
    "bluetooth−[object Bluetooth]",
    "share−function share() { [native code] }",
    "cookieEnabled−true",
    "virtualKeyboard−[object VirtualKeyboard]",
    "product−Gecko",
    "mediaDevices−[object MediaDevices]",
    "canShare−function canShare() { [native code] }",
    "getGamepads−function getGamepads() { [native code] }",
    "product−Gecko",
    "xr−[object XRSystem]",
    "clipboard−[object Clipboard]",
    "storageBuckets−[object StorageBucketManager]",
    "unregisterProtocolHandler−function unregisterProtocolHandler() { [native code] }",
    "productSub−20030107",
    "login−[object NavigatorLogin]",
    "vendorSub−",
    "login−[object NavigatorLogin]",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "mediaDevices−[object MediaDevices]",
    "locks−[object LockManager]",
    "webkitGetUserMedia−function webkitGetUserMedia() { [native code] }",
    "vendor−Google Inc.",
    "xr−[object XRSystem]",
    "mediaDevices−[object MediaDevices]",
    "virtualKeyboard−[object VirtualKeyboard]",
    "virtualKeyboard−[object VirtualKeyboard]",
    "appName−Netscape",
    "storageBuckets−[object StorageBucketManager]",
    "presentation−[object Presentation]",
    "onLine−true",
    "mimeTypes−[object MimeTypeArray]",
    "credentials−[object CredentialsContainer]",
    "presentation−[object Presentation]",
    "getGamepads−function getGamepads() { [native code] }",
    "vendorSub−",
    "virtualKeyboard−[object VirtualKeyboard]",
    "serviceWorker−[object ServiceWorkerContainer]",
    "xr−[object XRSystem]",
    "product−Gecko",
    "keyboard−[object Keyboard]",
    "gpu−[object GPU]",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "webkitPersistentStorage−[object DeprecatedStorageQuota]",
    "doNotTrack",
    "clearAppBadge−function clearAppBadge() { [native code] }",
    "presentation−[object Presentation]",
    "serial−[object Serial]",
    "locks−[object LockManager]",
    "requestMIDIAccess−function requestMIDIAccess() { [native code] }",
    "locks−[object LockManager]",
    "requestMediaKeySystemAccess−function requestMediaKeySystemAccess() { [native code] }",
    "vendor−Google Inc.",
    "pdfViewerEnabled−true",
    "language−zh-CN",
    "setAppBadge−function setAppBadge() { [native code] }",
    "geolocation−[object Geolocation]",
    "userAgentData−[object NavigatorUAData]",
    "mediaCapabilities−[object MediaCapabilities]",
    "requestMIDIAccess−function requestMIDIAccess() { [native code] }",
    "getUserMedia−function getUserMedia() { [native code] }",
    "mediaDevices−[object MediaDevices]",
    "webkitPersistentStorage−[object DeprecatedStorageQuota]",
    "sendBeacon−function sendBeacon() { [native code] }",
    "hardwareConcurrency−32",
    "credentials−[object CredentialsContainer]",
    "storage−[object StorageManager]",
    "cookieEnabled−true",
    "pdfViewerEnabled−true",
    "windowControlsOverlay−[object WindowControlsOverlay]",
    "scheduling−[object Scheduling]",
    "pdfViewerEnabled−true",
    "hardwareConcurrency−32",
    "xr−[object XRSystem]",
    "webdriver−false",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "getInstalledRelatedApps−function getInstalledRelatedApps() { [native code] }",
    "bluetooth−[object Bluetooth]"
]
document_key = ['_reactListeningo743lnnpvdg', 'location']
window_key = [
    "0",
    "window",
    "self",
    "document",
    "name",
    "location",
    "customElements",
    "history",
    "navigation",
    "locationbar",
    "menubar",
    "personalbar",
    "scrollbars",
    "statusbar",
    "toolbar",
    "status",
    "closed",
    "frames",
    "length",
    "top",
    "opener",
    "parent",
    "frameElement",
    "navigator",
    "origin",
    "external",
    "screen",
    "innerWidth",
    "innerHeight",
    "scrollX",
    "pageXOffset",
    "scrollY",
    "pageYOffset",
    "visualViewport",
    "screenX",
    "screenY",
    "outerWidth",
    "outerHeight",
    "devicePixelRatio",
    "clientInformation",
    "screenLeft",
    "screenTop",
    "styleMedia",
    "onsearch",
    "isSecureContext",
    "trustedTypes",
    "performance",
    "onappinstalled",
    "onbeforeinstallprompt",
    "crypto",
    "indexedDB",
    "sessionStorage",
    "localStorage",
    "onbeforexrselect",
    "onabort",
    "onbeforeinput",
    "onbeforematch",
    "onbeforetoggle",
    "onblur",
    "oncancel",
    "oncanplay",
    "oncanplaythrough",
    "onchange",
    "onclick",
    "onclose",
    "oncontentvisibilityautostatechange",
    "oncontextlost",
    "oncontextmenu",
    "oncontextrestored",
    "oncuechange",
    "ondblclick",
    "ondrag",
    "ondragend",
    "ondragenter",
    "ondragleave",
    "ondragover",
    "ondragstart",
    "ondrop",
    "ondurationchange",
    "onemptied",
    "onended",
    "onerror",
    "onfocus",
    "onformdata",
    "oninput",
    "oninvalid",
    "onkeydown",
    "onkeypress",
    "onkeyup",
    "onload",
    "onloadeddata",
    "onloadedmetadata",
    "onloadstart",
    "onmousedown",
    "onmouseenter",
    "onmouseleave",
    "onmousemove",
    "onmouseout",
    "onmouseover",
    "onmouseup",
    "onmousewheel",
    "onpause",
    "onplay",
    "onplaying",
    "onprogress",
    "onratechange",
    "onreset",
    "onresize",
    "onscroll",
    "onsecuritypolicyviolation",
    "onseeked",
    "onseeking",
    "onselect",
    "onslotchange",
    "onstalled",
    "onsubmit",
    "onsuspend",
    "ontimeupdate",
    "ontoggle",
    "onvolumechange",
    "onwaiting",
    "onwebkitanimationend",
    "onwebkitanimationiteration",
    "onwebkitanimationstart",
    "onwebkittransitionend",
    "onwheel",
    "onauxclick",
    "ongotpointercapture",
    "onlostpointercapture",
    "onpointerdown",
    "onpointermove",
    "onpointerrawupdate",
    "onpointerup",
    "onpointercancel",
    "onpointerover",
    "onpointerout",
    "onpointerenter",
    "onpointerleave",
    "onselectstart",
    "onselectionchange",
    "onanimationend",
    "onanimationiteration",
    "onanimationstart",
    "ontransitionrun",
    "ontransitionstart",
    "ontransitionend",
    "ontransitioncancel",
    "onafterprint",
    "onbeforeprint",
    "onbeforeunload",
    "onhashchange",
    "onlanguagechange",
    "onmessage",
    "onmessageerror",
    "onoffline",
    "ononline",
    "onpagehide",
    "onpageshow",
    "onpopstate",
    "onrejectionhandled",
    "onstorage",
    "onunhandledrejection",
    "onunload",
    "crossOriginIsolated",
    "scheduler",
    "alert",
    "atob",
    "blur",
    "btoa",
    "cancelAnimationFrame",
    "cancelIdleCallback",
    "captureEvents",
    "clearInterval",
    "clearTimeout",
    "close",
    "confirm",
    "createImageBitmap",
    "fetch",
    "find",
    "focus",
    "getComputedStyle",
    "getSelection",
    "matchMedia",
    "moveBy",
    "moveTo",
    "open",
    "postMessage",
    "print",
    "prompt",
    "queueMicrotask",
    "releaseEvents",
    "reportError",
    "requestAnimationFrame",
    "requestIdleCallback",
    "resizeBy",
    "resizeTo",
    "scroll",
    "scrollBy",
    "scrollTo",
    "setInterval",
    "setTimeout",
    "stop",
    "structuredClone",
    "webkitCancelAnimationFrame",
    "webkitRequestAnimationFrame",
    "chrome",
    "caches",
    "cookieStore",
    "ondevicemotion",
    "ondeviceorientation",
    "ondeviceorientationabsolute",
    "launchQueue",
    "documentPictureInPicture",
    "getScreenDetails",
    "queryLocalFonts",
    "showDirectoryPicker",
    "showOpenFilePicker",
    "showSaveFilePicker",
    "originAgentCluster",
    "onpageswap",
    "onpagereveal",
    "credentialless",
    "speechSynthesis",
    "onscrollend",
    "webkitRequestFileSystem",
    "webkitResolveLocalFileSystemURL",
    "sendMsgToSolverCS",
    "webpackChunk_N_E",
    "__next_set_public_path__",
    "next",
    "__NEXT_DATA__",
    "__SSG_MANIFEST_CB",
    "__NEXT_P",
    "_N_E",
    "regeneratorRuntime",
    "__REACT_INTL_CONTEXT__",
    "DD_RUM",
    "_",
    "filterCSS",
    "filterXSS",
    "__SEGMENT_INSPECTOR__",
    "__NEXT_PRELOADREADY",
    "Intercom",
    "__MIDDLEWARE_MATCHERS",
    "__STATSIG_SDK__",
    "__STATSIG_JS_SDK__",
    "__STATSIG_RERENDER_OVERRIDE__",
    "_oaiHandleSessionExpired",
    "__BUILD_MANIFEST",
    "__SSG_MANIFEST",
    "__intercomAssignLocation",
    "__intercomReloadLocation"
]


class ScriptSrcParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        global cached_scripts, cached_dpl, cached_time
        if tag == "script":
            attrs_dict = dict(attrs)
            if "src" in attrs_dict:
                src = attrs_dict["src"]
                cached_scripts.append(src)
                match = re.search(r"c/[^/]*/_", src)
                if match:
                    cached_dpl = match.group(0)
                    cached_time = int(time.time())


def get_data_build_from_html(html_content):
    global cached_scripts, cached_dpl, cached_time
    parser = ScriptSrcParser()
    parser.feed(html_content)
    if not cached_scripts:
        cached_scripts.append("https://chatgpt.com/backend-api/sentinel/sdk.js")
    if not cached_dpl:
        match = re.search(r'<html[^>]*data-build="([^"]*)"', html_content)
        if match:
            data_build = match.group(1)
            cached_dpl = data_build
            cached_time = int(time.time())
            logger.info(f"Found dpl: {cached_dpl}")


async def get_dpl(service):
    global cached_scripts, cached_dpl, cached_time
    if int(time.time()) - cached_time < 15 * 60:
        return True
    headers = service.base_headers.copy()
    cached_scripts = []
    cached_dpl = ""
    try:
        if conversation_only:
            return True
        r = await service.s.get(f"{service.host_url}/", headers=headers, timeout=5)
        r.raise_for_status()
        get_data_build_from_html(r.text)
        if not cached_dpl:
            raise Exception("No Cached DPL")
        else:
            return True
    except Exception as e:
        logger.info(f"Failed to get dpl: {e}")
        cached_dpl = None
        cached_time = int(time.time())
        return False


def get_parse_time():
    now = datetime.now(timezone(timedelta(hours=-5)))
    return now.strftime(timeLayout) + " GMT-0500 (Eastern Standard Time)"


def get_config(user_agent):
    config = [
        random.randint(1080, 1440+1080),
        get_parse_time(),
        4294705152,
        0,
        user_agent,
        random.choice(cached_scripts) if cached_scripts else "",
        cached_dpl,
        "en-US",
        "en-US,es-US,en,es",
        0,
        random.choice(navigator_key),
        random.choice(document_key),
        random.choice(window_key),
        time.perf_counter() * 1000,
        str(uuid.uuid4()),
        "",
        random.choice(cores),
        time.time() * 1000 - (time.perf_counter() * 1000),
    ]
    return config


def get_answer_token(seed, diff, config):
    start = time.time()
    answer, solved = generate_answer(seed, diff, config)
    end = time.time()
    logger.info(f'diff: {diff}, time: {int((end - start) * 1e6) / 1e3}ms, solved: {solved}')
    return "gAAAAAB" + answer, solved


def generate_answer(seed, diff, config, timeout_seconds=1.5):
    # 记录开始时间
    start_time = time.time()

    # 预处理静态部分
    diff_len = len(diff)
    seed_encoded = seed.encode()
    static_config_part1 = (json.dumps(config[:3], separators=(',', ':'), ensure_ascii=False)[:-1] + ',').encode()
    static_config_part2 = (',' + json.dumps(config[4:9], separators=(',', ':'), ensure_ascii=False)[1:-1] + ',').encode()
    static_config_part3 = (',' + json.dumps(config[10:], separators=(',', ':'), ensure_ascii=False)[1:]).encode()
    target_diff = bytes.fromhex(diff)

    # 设置默认返回值
    default_result = "wQ8Lk5FbGpA2NcR9dShT6gYjU7VxZ4D" + pybase64.b64encode(f'"{seed}"'.encode()).decode(), False

    for i in range(500000):
        if i % 500 == 0:
            # 每500次检查一次时间
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                logger.info(f"generate_answer: operation timed out after {timeout_seconds} seconds at iteration {i}.")
                return default_result

        # 准备动态部分
        dynamic_json_i = str(i).encode()
        dynamic_json_j = str(i >> 1).encode()
        final_json_bytes = static_config_part1 + dynamic_json_i + static_config_part2 + dynamic_json_j + static_config_part3

        # 计算哈希
        base_encode = pybase64.b64encode(final_json_bytes)
        hash_value = hashlib.sha3_512(seed_encoded + base_encode).digest()
        if hash_value[:diff_len] <= target_diff:
            return base_encode.decode(), True

    return default_result


def get_requirements_token(config):
    require, solved = generate_answer(format(random.random()), "0fffff", config)
    return 'gAAAAAC' + require


if __name__ == "__main__":
    # cached_scripts.append(
    #     "https://cdn.oaistatic.com/_next/static/cXh69klOLzS0Gy2joLDRS/_ssgManifest.js?dpl=453ebaec0d44c2decab71692e1bfe39be35a24b3")
    # cached_dpl = "453ebaec0d44c2decab71692e1bfe39be35a24b3"
    # cached_time = int(time.time())
    # for i in range(10):
    #     seed = format(random.random())
    #     diff = "000032"
    #     config = get_config("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome")
    #     answer = get_answer_token(seed, diff, config)
    cached_scripts.append(
        "https://cdn.oaistatic.com/_next/static/cXh69klOLzS0Gy2joLDRS/_ssgManifest.js?dpl=453ebaec0d44c2decab71692e1bfe39be35a24b3")
    cached_dpl = "prod-f501fe933b3edf57aea882da888e1a544df99840"
    config = get_config("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36")
    get_requirements_token(config)
