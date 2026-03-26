"""DS-Agent 的核心工作流实现文件。"""
import os
import sys
import anthropic
from MLAgentBench.LLM import complete_text_fast, complete_text
from MLAgentBench.schema import Action
from MLAgentBench.low_level_actions import read_file
from .agent import Agent
from .utils import clean_log


class DSAgent(Agent):

    def __init__(self, args, env):
        # 调用父类初始化，完成通用 agent 参数与环境绑定。
        super().__init__(args, env)
        # 缓存当前研究问题描述，后续决策和日志可使用。
        self.research_problem = env._research_problem
        
    def run(self, env):
        # 全局动作步数（每次“规划”或“执行”都 +1）。
        step = 0
        # 实验轮次计数（每次真正执行实验都 +1）。
        experiment_step = 0
        
        # 运行日志初始内容：给出当前状态（尚无可用 baseline）。
        running_log = f"""
        [Initial State] Lack of a baseline model as a good starting point for the current research problem.
        """

        # 将初始状态写入主日志文件，便于后续追踪完整过程。
        with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
            f.write(f"Step {step}" + ":\n")
            f.write(running_log + "\n")
        
        # 主循环：当任务未结束且步数未超过上限（10）时，持续“规划 -> 执行 -> 记录”。
        while not env.is_final() and step < 10:
            # Develop the experiment plan (Retrieve -> RankRevise -> Reuse).
            # 1) 让环境基于当前 running_log 生成下一步实验计划。
            action = "Develop An Experiment Plan via CBR"
            action_input = {
                "experiment_log": running_log
            }
            plans = env.execute(Action(action, action_input))
            step += 1
                
            # 把“计划生成”的 observation 记录到主日志。
            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write(f"Step {step}" + ":\n")
                f.write(anthropic.AI_PROMPT + "\n" + f"Action: {action}" + "\nObservation:\n" + plans + "\n") 
                f.write("[CBR] Retrieval + rerank executed in planning action. See env_log/tool_logs for rerank details.\n")
                
            # Execute the experiment plan (Execute)
            # 2) 执行刚才生成的计划：默认操作 train.py，并保存回 train.py。
            action = "Execute the Experiment Plan"
            action_input = {
                "script_name": "train.py",
                "plan": plans,
                "save_name": "train.py"
            }

            # 执行后返回两部分：
            # - execution_log: 运行输出（指标、报错、训练信息等）
            # - diff: 本次代码改动差异（若执行失败可能为空）
            execution_log, diff = env.execute(Action(action, action_input))

            # 清洗执行日志，去掉噪声内容，方便后续总结。
            execution_log = clean_log(execution_log)

            # 更新计数器：总步数 +1，实验轮次 +1。
            step += 1
            experiment_step += 1

            # 把“计划执行”的 observation 写入主日志。
            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write(f"Step {step}" + ":\n")
                f.write(anthropic.AI_PROMPT + "\n" + f"Action: {action}" + "\nObservation:\n" + execution_log + "\n")
            
            # Write experiment logs (Log)
            # 3) 调用 LLM 把“计划 + 执行结果 + 代码差异”整理成结构化实验总结。
            log_content = self.revise_running_log(running_log, plans, execution_log, diff, log_file=os.path.join(self.log_dir, "tmp.txt"))

            # 追加到累计运行日志中，作为下一轮规划输入。
            running_log += f"\n{log_content}"

            # 将更新后的 running_log 持久化写入主日志。
            with open(os.path.join(self.log_dir , "main_log"), "a", 1) as f:
                f.write(f"Step {step}" + ":\n")
                f.write(running_log + "\n")

        # 退出循环后，返回结束原因：任务完成 or 达到最大步数。
        if env.is_final():
            return "Finished due to env.is_final() == True"
        else:
            return "Finished due to agent max steps reached"


    @staticmethod
    def revise_running_log(running_log, instructions, execution_log, diff, log_file=None):
        """根据本轮指令、执行日志与代码差异，生成可追加的运行日志摘要。"""

        # 构造总结提示词：
        # - 输入：上轮计划、执行观测、代码 diff、历史 running_log
        # - 输出：固定格式的 [Experiment Summary]/[Experiment Result]
        prompt = f"""Given instructions (what is expected to do), execution log (the experimental results) and the code difference (what is actually done and this will be nothing if the experiment failed) of last experiment on the research problem: 
        {instructions} 
        [Execution Log]:
        ```
        {execution_log}
        ```
        [Code Difference]:
        ```
        {diff}
        ```
        Here is the running log of your experiment:
        [Running Log]:
        ```
        {running_log}
        ```
        Summarize and append the progress of the last step to the running log in this format:
        [Experiment Summary]: According to the instructions and the code difference, summarize what was experimented in the last step objectively.
        [Experiment Result]: According to the execution log and the running log, summarize if the last step of experiment brings performance improvement objectively. Only report the performance if this is the first experiment result.
        Do not include any result that is guessed rather than directly confirmed by the observation. Do not include additional information or suggestions.
        """

        # 调用快速模型生成总结，并确保最终字符串以 [Experiment Summary]: 开头。
        log = "[Experiment Summary]:" + complete_text_fast(prompt, log_file=log_file).split("[Experiment Summary]:")[1]
        return log

    