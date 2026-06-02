---
paths:
  - "data_collection/**/*.py"
---

# 数据收集规则

## 教程输出目录格式

教程模态作为目录最外层分区（与 `data.tutorial_type` 配置对应）：

```
data_tutorial/{tutorial_type}/{benchmark}/{task_id}/tutorial/
```

各模态产物：

- **html**（`download_tutorials.py`）：`page.html`（图片 src 已改写为 `images/xxx.png`）+ `images/` + `metadata.json`
- **screenshot**（`download_via_playwright.py`）：`images/frame_001.png frame_002.png ...`（viewport 等高多帧滚动截图）+ `metadata.json`（必须含 `"content_type": "screenshot"`），**不写 page.html**
- **video**：预留，未实现

`metadata.json` 字段：
```json
{
  "task_id": "...",
  "instruction": "...",
  "source_url": "...",
  "content_type": "screenshot"
}
```
`content_type` 在 screenshot/video 类型下**必填**；html 类型下可选（缺失时由 loader 按调用方传入的 `tutorial_type` 落地）。

`tutorial_loader.py` 按调用方传入的 `tutorial_type` 强校验产物：html 必须有 `page*.html`，screenshot 必须有非空 `images/`，缺失直接抛 `FileNotFoundError`。

## URL 映射文件

- 放在 `data_collection/<benchmark>/` 下，命名自定（如 `urls.json`、`<benchmark>_urls.json`）
- 格式：JSON 数组，每项至少包含 `task_id`、`instruction`、`tutorial_url`
- `task_id` 必须与 benchmark 的任务 ID 一致

## 图片处理

- 下载图片到 `images/` 目录，保留原文件名
- 文件名冲突时追加 `_2`, `_3` 后缀
- 基于 MD5 内容哈希去重：相同内容跳过下载，返回已有文件名
- 跳过 tracking pixels（< 200B）和超大文件（> 10MB）
- HTML 中的 `<img src>` 改写为 `images/{local_name}` 相对路径

## 脚本结构

- html 入口：`data_collection/<benchmark>/download_tutorials.py`（`requests` + `BeautifulSoup`）
- screenshot 入口：`data_collection/<benchmark>/download_via_playwright.py`（`playwright.async_api`，需先 `pip install -r data_collection/requirements-playwright.txt && playwright install chromium`）
- 使用 `requests.Session` 复用连接，设置合理 User-Agent（仅 html）
- 支持重试（3 次）
- 页面内容提取：优先选择 `.entry-content`, `.post-content`, `article`, `main` 等语义标签
- 去除噪声：nav, aside, script, style, 广告等标签
- screenshot 脚本顶部常量集中放（`VIEWPORT_WIDTH/HEIGHT`、`MAX_FRAMES`、`SCROLL_RATIO`），便于 A/B 实验

## 添加新 Benchmark 数据收集

1. 创建 `data_collection/<benchmark>/` 目录
2. 准备 URL 映射 JSON 文件
3. 实现 `download_tutorials.py`，参考 `data_collection/osworld/download_tutorials.py`
4. 输出必须符合上述目录格式
