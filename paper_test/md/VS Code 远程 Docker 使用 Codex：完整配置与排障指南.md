# VS Code 远程 Docker 使用 Codex：完整配置与排障指南

> 适用场景：Windows 本地通过 VS Code Remote-SSH 连接远程 GPU 服务器，再用 Dev Containers 附加已有 Docker 容器，在容器中使用 OpenAI Codex IDE 扩展。远程容器无法直接访问 OpenAI，但 Windows 本地 HTTP 代理可以访问。
>
> 本文根据本次实际配置和成功登录经历整理。**不包含服务器密码、SSH 私钥或任何登录令牌**。IP、端口、容器 ID 是本次环境的示例值，容器重建或网络变化后需要重新核对。

## 1. 本次环境及最终效果

| 项目 | 本次实际配置 |
|---|---|
| Windows SSH 主机别名 | `CYX` |
| 远程宿主机 | `122.226.69.234:2241` |
| 宿主机用户名 | `zwq` |
| 宿主机提示符 | `zwq@fz-gpu:~$` |
| Docker 容器 ID | `0928358692bf` |
| 容器提示符 | `root@0928358692bf:/mnt/project#` |
| 宿主机项目路径 | `/home/zwq/project` |
| 容器项目路径 | `/mnt/project` |
| 项目挂载 | `bind /home/zwq/project -> /mnt/project` |
| 本次容器 IP | `172.17.0.7` |
| 本次 Docker 网关 | `172.17.0.1` |
| Windows HTTP 代理 | `127.0.0.1:7890`（本次已验证） |
| 远程 SSH 反向隧道 | 宿主机 `127.0.0.1:17890` |
| 容器可访问代理入口 | `172.17.0.1:17891` |
| 结果 | 容器终端 `curl` 返回 HTTP 200，Codex 侧边栏最终成功登录 |

连接示意：

```text
Windows 本地 VS Code + 本地 HTTP 代理 127.0.0.1:7890
  ├── VS Code Remote-SSH（CYX）→ 远程宿主机 fz-gpu
  │     └── Dev Containers → Docker 0928358692bf → /mnt/project
  └── SSH -R 反向隧道 → 宿主机 127.0.0.1:17890
                          └── Python 临时 TCP 转发
                              172.17.0.1:17891
                                └── 容器 172.17.0.7
                                    └── Codex 扩展访问 OpenAI
```

**两条线路要分清**：VS Code 负责远程开发/编辑容器文件；SSH 反向隧道 + Python 转发负责让容器里的 Codex 扩展访问 OpenAI。

## 2. VS Code 连接远程 Docker

### 2.1 Windows SSH 配置

编辑 Windows 本地文件 `C:\Users\zwq\.ssh\config`：

```sshconfig
Host CYX
    HostName 122.226.69.234
    Port 2241
    User zwq
    ServerAliveInterval 30
    ServerAliveCountMax 3
```

在 **Windows PowerShell** 验证：

```powershell
ssh CYX
```

成功后进入 `zwq@fz-gpu:~$`。如果 SSH 使用密码认证，PowerShell 会交互提示输入；不要把密码写进配置文件。

### 2.2 安装 VS Code 扩展并附加现有容器

在 Windows VS Code 扩展市场安装以下官方扩展：

- Microsoft **Remote - SSH**
- Microsoft **Dev Containers**
- OpenAI **Codex**

操作路径：

1. `Ctrl + Shift + P` → `Remote-SSH: Connect to Host...` → 选择 `CYX`，输入 SSH 密码。
2. 远程窗口连接成功，左下角会显示 `SSH: CYX`。
3. `Ctrl + Shift + P` → `Dev Containers: Attach to Running Container...`。
4. 选择正在运行的容器 `0928358692bf`（容器名称可能与 ID 不同）。
5. 新窗口左下角应显示 `Dev Container: ...`。
6. `File → Open Folder` → 打开 `/mnt/project`，或具体项目 `/mnt/project/Qwen3-8B-news`。
7. 在 VS Code 终端执行：

```bash
hostname
pwd
whoami
python --version
```

应核对 `hostname` 是否为 `0928358692bf`。**仅仅在 `SSH: CYX` 窗口的终端执行 `docker exec`，不等于整个 VS Code 窗口已经附加到容器。**

### 2.3 在容器窗口启用 Codex 扩展

在已经附加的 Dev Container 窗口中按 `Ctrl + Shift + X`，找到 OpenAI 发布的 Codex，检查是否提供 `Install in Dev Container`，必要时安装到远程容器环境。打开 Codex 侧边栏并登录 ChatGPT。

> 具体扩展安装位置以 VS Code 扩展页面显示为准。容器中的 Codex 扩展可能部署自己的运行组件，不应先假定必须手动安装容器级 Codex CLI 或 Node.js。

## 3. 为什么 PowerShell 能 SSH，而 Codex 桌面版先前连接失败

PowerShell 登录 `CYX` 时可以手动输入密码；之前独立 Codex 桌面 App 的 SSH 连接报 `Permission denied (publickey,password)`，说明其远程连接流程未完成认证。**这与 VS Code Dev Containers 的容器附加机制不同**。

本次选择 VS Code Remote-SSH + Dev Containers，复用交互式 SSH 登录，并通过 Docker Engine 附加容器。容器内无需新增 `sshd`。

此外，宿主机曾安装用户级 Codex CLI，路径为 `/home/zwq/.npm-global/bin/codex`、版本当时为 `0.169.0`；**这不是 VS Code 附加 Docker 的必要前提**。若需要验证宿主机安装情况，可用：

```powershell
ssh CYX "command -v codex && codex --version"
```

本次曾遇到非交互式 SSH 的 PATH 未包含 `~/.npm-global/bin`，后来修复。该事项与容器中 Codex IDE 扩展的网络问题是两回事。

## 4. 本次网络问题的原因

- Windows 本地可以访问 `https://chatgpt.com`。
- Docker 容器最初无法直接访问 OpenAI。
- 宿主机没有独立配置代理。
- 在容器中访问 `127.0.0.1:17890` 会得到 `Connection refused`，因为 Docker 容器的 `127.0.0.1` 是**容器自己的回环地址**，不是宿主机的。
- 需要分别打通 Windows → 宿主机 → Docker 容器，并让 **Codex 扩展进程**继承代理变量。

## 5. Windows 本地验证 HTTP 代理

在 **Windows PowerShell** 执行：

```powershell
curl.exe -I --proxy http://127.0.0.1:7890 --connect-timeout 5 --max-time 15 https://chatgpt.com
```

本次测试成功。`7890` 是已验证的本地端口，换电脑或代理客户端后需重新核对。仅有 Windows 浏览器可访问 OpenAI，并不意味着一定存在可用的 HTTP 代理监听端口。

## 6. 建立 Windows → 远程宿主机的 SSH 反向隧道

在 **Windows PowerShell 窗口 A** 启动（该窗口持续保持打开）：

```powershell
ssh -N -T -o ExitOnForwardFailure=yes -R 127.0.0.1:17890:127.0.0.1:7890 CYX
```

按提示输入 SSH 密码。如果命令保持运行且没有异常输出，通常表示隧道已建立。**不要把远程绑定地址改成 `0.0.0.0` 对外公开代理。**

在 **Windows PowerShell 窗口 B** 进入宿主机：

```powershell
ssh CYX
```

确认提示符为 `zwq@fz-gpu:~$`，然后执行：

```bash
curl -I --proxy http://127.0.0.1:17890 \
  --connect-timeout 5 --max-time 15 https://chatgpt.com
```

本次该步骤已成功。注意这是在**宿主机**测试，不能直接在容器里套用宿主机的 `127.0.0.1`。

## 7. 确认 Docker 网络及项目挂载

在 **宿主机 `zwq@fz-gpu`** 执行：

```bash
docker inspect 0928358692bf \
  --format '{{range .NetworkSettings.Networks}}Container IP: {{.IPAddress}} Gateway: {{.Gateway}}{{println}}{{end}}'
```

本次返回：

```text
Container IP: 172.17.0.7 Gateway: 172.17.0.1
```

检查项目挂载：

```bash
docker inspect 0928358692bf \
  --format '{{range .Mounts}}{{println .Type .Source "->" .Destination}}{{end}}'
```

返回：

```text
bind /home/zwq/project -> /mnt/project
```

检查 `socat`：

```bash
command -v socat
```

本次未安装，因此使用 Python 标准库搭建临时转发，不需要 `sudo apt install`。

## 8. 宿主机上运行 Python TCP 转发（无需安装软件）

**这段命令要在宿主机 `zwq@fz-gpu:~$` 运行，不能在 Docker 容器内运行。** 它只在 Docker 网关地址 `172.17.0.1:17891` 上监听，并仅接收来源 IP `172.17.0.7` 的连接，将流量转到宿主机回环端口 `127.0.0.1:17890`。容器重新创建后 IP 可能改变，请先重查。

在 **宿主机终端窗口 C** 执行：

```bash
python3 - <<'PY'
import socket
import socketserver
import select

LISTEN_HOST = "172.17.0.1"
LISTEN_PORT = 17891
ALLOWED_IP = "172.17.0.7"
TARGET_HOST = "127.0.0.1"
TARGET_PORT = 17890

class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        if self.client_address[0] != ALLOWED_IP:
            print("Rejected:", self.client_address, flush=True)
            return
        try:
            upstream = socket.create_connection(
                (TARGET_HOST, TARGET_PORT), timeout=10
            )
        except OSError as exc:
            print("Upstream error:", exc, flush=True)
            return
        with upstream:
            sockets = [self.request, upstream]
            while True:
                try:
                    readable, _, _ = select.select(sockets, [], [], 60)
                    if not readable:
                        continue
                    for src in readable:
                        data = src.recv(65536)
                        if not data:
                            return
                        dst = upstream if src is self.request else self.request
                        dst.sendall(data)
                except (OSError, ValueError):
                    return

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

with Server((LISTEN_HOST, LISTEN_PORT), Handler) as server:
    print(f"Listening on {LISTEN_HOST}:{LISTEN_PORT}", flush=True)
    print(f"Only allowing {ALLOWED_IP}", flush=True)
    server.serve_forever()
PY
```

正常启动时会打印：

```text
Listening on 172.17.0.1:17891
Only allowing 172.17.0.7
```

**安全边界**：这是用户进程，不写系统配置、不安装软件；但它会占用宿主机 Docker 网桥的端口，来源 IP 过滤不是强身份认证，也不能保证其他拥有该网络控制权的用户绝对无法使用。仅适合受信任网络，并遵守服务器管理员政策。

## 9. Docker 容器中测试代理

进入已连接的 VS Code Dev Container 终端，确认提示符类似：

```text
root@0928358692bf:/mnt/project#
```

先显式指定代理测试：

```bash
curl -I --proxy http://172.17.0.1:17891 \
  --connect-timeout 5 --max-time 15 https://chatgpt.com
```

然后在**当前终端临时**设置：

```bash
export HTTP_PROXY=http://172.17.0.1:17891
export HTTPS_PROXY=http://172.17.0.1:17891
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy="$NO_PROXY"
```

测试：

```bash
curl -I --max-time 15 https://chatgpt.com
```

本次最终观察到：

```text
HTTP/1.1 200 Connection established
HTTP/2 200
```

这表明容器当前终端的 HTTPS 请求成功通过 HTTP 代理访问 ChatGPT，但**不能仅凭这项测试证明 Codex 扩展的认证、模型请求、WebSocket 全部正常**。

## 10. 给 VS Code Codex 扩展配置代理（关键）

**为什么终端 `curl` 成功、侧边栏登录仍失败？** 在终端中执行的 `export` 只传给当前 Shell 的子进程，已经运行的 VS Code 扩展宿主进程不会自动获得这些变量。

操作步骤：

1. 确保当前是 Dev Container 窗口。
2. `Ctrl + Shift + P` → 搜索 `Dev Containers: Open Container Configuration File` 或 `Dev Containers: Open Attached Container Configuration File`。
3. 打开当前附加容器的配置文件，**保留原有属性**，合并如下 `remoteEnv`，不要覆盖整个 JSON：

```json
{
  "remoteEnv": {
    "HTTP_PROXY": "http://172.17.0.1:17891",
    "HTTPS_PROXY": "http://172.17.0.1:17891",
    "http_proxy": "http://172.17.0.1:17891",
    "https_proxy": "http://172.17.0.1:17891",
    "NO_PROXY": "localhost,127.0.0.1,::1",
    "no_proxy": "localhost,127.0.0.1,::1"
  }
}
```

4. 保存后重新附加**现有**容器，让远程扩展宿主进程重新启动。**不要点击 Rebuild Container**。
5. 打开 OpenAI Codex 扩展，重新进行账号登录。

本次经过容器终端联网、代理相关配置与排查后，**Codex 侧边栏最终登录成功**。具体哪一个调整单独起了决定性作用，现有记录不足以精确归因。

如果登录仍失败：

```bash
curl -I --connect-timeout 5 --max-time 15 https://auth.openai.com
curl -I --connect-timeout 5 --max-time 15 https://api.openai.com
```

HTTP `401/403/405` 也可能表明服务已返回响应，不一定是基础网络故障。检查 VS Code `View → Output` 中 Codex/OpenAI 相关日志。若浏览器认证完成但扩展仍显示未登录，需要进一步检查实际的 OAuth 回调端口与 VS Code `Ports` 转发，不要猜测或公开端口。日志、登录链接、令牌不要直接公开。

## 11. 常见错误与处理

| 症状 | 常见原因 | 处理方法 |
|---|---|---|
| PowerShell 可以 SSH，但独立 Codex App `Permission denied` | SSH 连接需要交互式密码 | 使用 VS Code Remote-SSH；若有授权再配置密钥 |
| 宿主机执行 `codex --version` 正常，但 `ssh CYX "codex --version"` 不行 | 非交互式 SSH PATH 不完整 | 检查用户级 PATH，不修改系统级目录 |
| 容器内 `docker: command not found` | 已在容器中，不具备 Docker CLI | 回到宿主机执行 `docker inspect` / `docker exec` |
| 容器访问 `127.0.0.1:17890` 被拒绝 | 容器回环地址不是宿主机 | 使用网关 `172.17.0.1:17891`，且要确保宿主机转发器在运行 |
| 宿主机代理可用、容器代理不可用 | Python 转发器未启动、IP/端口变化、防火墙阻断 | 查容器 IP/网关、监听信息和日志 |
| 容器 `curl` 成功但 Codex 登录失败 | 扩展宿主未继承代理、认证接口或 OAuth 回调问题 | 设置附加容器 `remoteEnv`，重新附加，查看日志 |
| Codex 突然无法访问 | Windows SSH 隧道或宿主机 Python 转发器退出 | 确保两个长期运行窗口都在 |
| 容器重启后网络失效 | 容器 IP 变化 | 重新执行 `docker inspect`，更新允许来源 IP 和代理配置 |

## 12. 关闭与恢复

本次代理链路由**两个长期运行进程**组成：

1. Windows PowerShell A：`ssh -N -T -R ... CYX`，按 `Ctrl + C` 结束。
2. 宿主机窗口 C：Python TCP 转发器，按 `Ctrl + C` 结束。

关闭任一进程后，容器的代理链路就可能中断。下次使用时按顺序启动：

```text
① Windows 本地代理可用
② Windows SSH 反向隧道运行
③ 宿主机 Python TCP 转发器运行
④ Docker 容器内 curl 测试通过
⑤ VS Code Dev Container / Codex 扩展运行
```

若不再需要代理，临时终端可执行 `unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy NO_PROXY no_proxy`。如果在附加容器配置中设置了 `remoteEnv`，需要删除相应字段并重新附加，才会撤销扩展环境中的配置。不要为了清理代理而重建或删除训练容器。

## 13. 服务器共享环境的注意事项

- 不要把任何 SSH 密码、私钥、OpenAI Token 写进文档、Git 仓库或 Codex 提示词。
- 不使用 `sudo npm install -g` 修改共享 Node.js 环境；宿主机若要安装 CLI，优先装在 `/home/zwq/.npm-global`。
- 不修改 `/etc/profile`、系统级 Docker 配置、CUDA/PyTorch 或共享用户环境。
- 本方案中的 Python 转发是临时用户进程；它虽不改变系统持久配置，但会开放一个 Docker 网桥监听端口，仍需遵守服务器权限与网络安全要求。
- VS Code 附加容器可能写入 VS Code Server 或扩展文件，**并非完全零写入**；避免在重要训练期间重建/重启容器。
- 真实训练前确认 GPU、Conda 环境及项目入口。首次使用 Codex 可让它仅分析目录，暂不修改代码或运行训练。

## 14. 首次使用 Codex 的建议提示词

```text
当前工作区位于远程 Docker 容器 0928358692bf 的 /mnt/project。
先执行 hostname、pwd、python --version 并检查当前项目目录。
请说明代码结构、主要模块和运行方式。
不要修改文件、安装依赖、删除内容、重启容器或启动训练任务，除非我明确授权。
```

---

**本次最终状态：**VS Code 已附加远程 Docker；Windows 代理、SSH 反向隧道、宿主机 Python TCP 转发、容器 curl 测试已成功；Codex IDE 扩展侧边栏已登录成功。
