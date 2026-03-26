"""MLAgentBench 运行入口。

本文件负责：
1) 解析命令行参数；
2) 初始化运行目录与全局 LLM/检索配置；
3) 根据 agent 名称实例化对应 Agent 并启动任务。
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com" # 设置 Hugging Face 镜像，加速模型加载（可选，根据需要调整或删除）。
import argparse
import sys
import datetime
import retrieval
import LLM
from environment import Environment
from agents.agent import Agent, SimpleActionAgent, ReasoningActionAgent
from agents.agent_research import ResearchAgent
from agents.dsagent import DSAgent
import high_level_actions
try:
    # AutoGPTAgent 是可选依赖：未安装相关包时不影响其他 agent 运行。
    from agents.agent_autogpt  import AutoGPTAgent
except:
    print("Failed to import AutoGPTAgent; Make sure you have installed the autogpt dependencies if you want to use it.")


def run(agent_cls, args):
    # 这一段做三件事：
    # - 创建并进入 Environment 上下文（负责任务环境生命周期）；
    # - 打印当前任务元信息（任务描述、可用动作、只读文件等）；
    # - 实例化并运行指定 Agent，拿到最终消息。
    with Environment(args) as env:

        print("=====================================")
        research_problem, benchmark_folder_name = env.get_task_description()
        print("Benchmark folder name: ", benchmark_folder_name)
        print("Research problem: ", research_problem)
        print("Lower level actions enabled: ", [action.name for action in env.low_level_actions])
        print("High level actions enabled: ", [action.name for action in env.high_level_actions])
        print("Read only files: ", env.read_only_files, file=sys.stderr)
        print("=====================================")  

        agent = agent_cls(args, env)
        final_message = agent.run(env)
        print("=====================================")
        print("Final message: ", final_message)

    # 环境退出后，将最终产物落盘（日志、轨迹、工作区快照等）。
    env.save("final")



if __name__ == "__main__":
    # ------------------------------
    # 参数定义区：所有 Agent 通用参数
    # ------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="airport_eta", help="task name") 
    parser.add_argument("--log-dir", type=str, default="./logs", help="log dir")
    parser.add_argument("--work-dir", type=str, default="./workspace", help="work dir")
    parser.add_argument("--max-steps", type=int, default=30, help="number of steps")
    parser.add_argument("--max-time", type=int, default=5* 60 * 60, help="max time")
    parser.add_argument("--device", type=int, default=0, help="device id")
    parser.add_argument("--python", type=str, default="python", help="python command")
    parser.add_argument("--interactive", action="store_true", help="interactive mode")
    parser.add_argument("--resume", type=str, default=None, help="resume from a previous run")
    parser.add_argument("--resume-step", type=int, default=0, help="the step to resume from")

    # ------------------------------
    # 参数定义区：Agent 与模型配置
    # ------------------------------
    parser.add_argument("--agent-type", type=str, default="DSAgent", help="agent type")
    parser.add_argument("--llm-name", type=str, default="deepseek-ai/DeepSeek-V3.2", help="llm name")
    parser.add_argument("--fast-llm-name", type=str, default="deepseek-ai/DeepSeek-V3.2", help="llm name")
    parser.add_argument("--edit-script-llm-name", type=str, default="deepseek-ai/DeepSeek-V3.2", help="llm name")
    parser.add_argument("--edit-script-llm-max-tokens", type=int, default=4000, help="llm max tokens")
    parser.add_argument("--agent-max-steps", type=int, default=50, help="max iterations for agent")

    # ------------------------------
    # 参数定义区：ResearchAgent 的动作/上下文控制
    # ------------------------------
    parser.add_argument("--actions-remove-from-prompt", type=str, nargs='+', default=["Execute Script", "Edit Script (AI)", "Reflection", "List Files", "Copy File", "Undo Edit Script", "Inspect Script Lines"])
    parser.add_argument("--actions-add-to-prompt", type=str, nargs='+', default=[], help="actions to add")
    parser.add_argument("--no-retrieval", default=True)        # w/ or w/o retrieval
    parser.add_argument("--valid-format-entires", type=str, nargs='+', default=None, help="valid format entries")
    parser.add_argument("--max-steps-in-context", type=int, default=3, help="max steps in context")
    parser.add_argument("--max-observation-steps-in-context", type=int, default=3, help="max observation steps in context")
    parser.add_argument("--max-retries", type=int, default=5, help="max retries")

    # ------------------------------
    # 参数定义区：LangChain Agent 配置
    # ------------------------------
    parser.add_argument("--langchain-agent", type=str, default="zero-shot-react-description", help="langchain agent")

    args = parser.parse_args()
    args.log_dir = args.log_dir + f"/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    args.work_dir = args.work_dir + f"/{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # 设置全局变量：
    # - token 统计输出目录；
    # - 检索重排模型名（与主模型一致）。
    LLM.STATISTICAL_DIR = args.log_dir
    retrieval.RANKING_MODEL = args.llm_name
    
    print(args, file=sys.stderr)
    
    # 当关闭检索或 agent 不是 ResearchAgent 时，移除依赖检索日志的高层动作，
    # 避免提示词中出现不可用工具。
    if args.no_retrieval or args.agent_type != "ResearchAgent":
        args.actions_remove_from_prompt.extend(["Retrieval from Research Log", "Append Summary to Research Log", "Reflection"])

    # 快速模型用于摘要/反思等辅助调用。
    LLM.FAST_MODEL = args.fast_llm_name

    # 根据字符串解析出 agent 类并启动主流程。
    run(getattr(sys.modules[__name__], args.agent_type), args)
    