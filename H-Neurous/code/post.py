import requests

url = "https://www.cnepaper.net/design/news/image/find"  # 注意：你测试的是 image/find
# 如果你想测 category/findPublic，就改 url 为下面这行：
# url = "https://www.cnepaper.net/design/news/category/findPublic"

payload = {"typeId": 1}
headers = {
    "Content-Type": "application/json",
    # 🔑 关键：添加从浏览器复制的 token
    "token": "__SET_VIA_ENVIRONMENT__"
}

try:
    response = requests.post(url, json=payload, headers=headers, timeout=10)
    print("状态码:", response.status_code)
    print("响应内容:")
    print(response.text)
except Exception as e:
    print("请求失败:", e)