# CampusMind Evaluation V1（Frozen）

这是已经完成一轮人工边界复核的 Routing / Risk Benchmark V1。

## 放到项目里

把整个 `evaluation/` 目录复制到 `D:\CampusMind\`。

目录应为：

```text
D:\CampusMind
├── core
├── agents
├── ...
└── evaluation
    ├── EVALUATION_LABEL_SPEC_V1.md
    ├── CHANGELOG_V1.md
    ├── evaluate_routing_risk.py
    └── datasets
        └── routing_risk_v1.jsonl
```

## 第一步：只跑 dev baseline

本地 Python：

```powershell
cd D:\CampusMind
python evaluation\evaluate_routing_risk.py --split dev
```

Docker：

```powershell
docker compose run --rm campusmind python evaluation/evaluate_routing_risk.py --split dev
```

先看 dev 错例并做错误分析。

## test 暂时不要跑

test split 被脚本主动保护。等 dev 侧规则改完、代码冻结后，再执行：

```powershell
python evaluation\evaluate_routing_risk.py --split test --confirm-test
```



## API examples

### Health

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

### Chat

```powershell
$body = @{
    user_id = "demo_user"
    conversation_id = "demo_conv_1"
    message = "学校心理咨询中心怎么预约？"
} | ConvertTo-Json

$result = Invoke-RestMethod `
    -Uri "http://localhost:8000/chat" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body))

$result
```

### Trace

```powershell
$requestId = $result.request_id
Invoke-RestMethod -Uri "http://localhost:8000/traces/$requestId" -Method Get
```

