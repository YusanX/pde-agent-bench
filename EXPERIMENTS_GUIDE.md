### 实验：LLM
直接跑以下（第一次ground truth求解会慢一点）：
    python scripts/run_benchmark.py --agent qwen3-max
    python scripts/run_benchmark.py --agent claude-opus-4.5
    python scripts/run_benchmark.py --agent gpt-5.2
    python scripts/run_benchmark.py --agent gemini-3.0-pro

实验：Code/PDE agents
PDE agent：
    CodePDE：python scripts/run_benchmark.py --agent codepde

Code agent：
    Openhands：

    1.python scripts/prepare_agent_workspace.py --agent openhands
    2.uv tool install openhands --python 3.12 安装openhands CLI
    3.bash experiments/openhands.sh

    mini-swe-agent：
    1.python scripts/prepare_agent_workspace.py --agent mini-swe-agent
    2.pip install mini-swe-agent   安装mini-swe的 CLI
    3.bash experiments/openhands.sh   ## 记得把bash中的第一个RESULT_DIR 换成results中第一步生成的目录
    mini -m 支持的模型可以查看https://mini-swe-agent.com/latest/models/quickstart/
