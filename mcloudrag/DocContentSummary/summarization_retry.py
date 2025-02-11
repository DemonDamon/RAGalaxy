class SummarizationRetryHandler:
    def __init__(self, max_retries=2):
        self.max_retries = max_retries

    def check_missing_keywords(self, summary: str, key_points: list) -> list:
        """
        简易检测：若在摘要中找不到指定关键词，则返回缺失关键词列表
        """
        missing = []
        for kp in key_points:
            if kp not in summary:
                missing.append(kp)
        return missing

    def generate_retry_prompt(self, original_prompt: str, missing_points: list) -> str:
        """
        自动生成新的提示语，引导模型关注缺失要点
        """
        additional_instruction = f"请确保在摘要中包含以下关键信息: {', '.join(missing_points)}。"
        # 合并原始提示作上下文，再附加新指令
        retry_prompt = f"{original_prompt}\n{additional_instruction}"
        return retry_prompt

    def summarize_with_retry(self, text: str, key_points: list, summarize_func, original_prompt: str) -> str:
        """
        主函数：进行摘要生成并检查是否缺失关键词，不满足就重试。
        summarize_func: 传入实际的模型调用函数
        original_prompt: 第一次调用时用的提示语上下文
        """
        attempts = 0
        current_prompt = original_prompt
        final_summary = ""

        while attempts <= self.max_retries:
            # 1. 使用当前 prompt 调用模型生成摘要
            final_summary = summarize_func(text, current_prompt)

            # 2. 检查是否有缺失关键词
            missing = self.check_missing_keywords(final_summary, key_points)
            if len(missing) == 0:
                # 摘要合格，结束
                break
            else:
                # 如果还有尝试次数，则修改提示重试
                attempts += 1
                if attempts <= self.max_retries:
                    current_prompt = self.generate_retry_prompt(current_prompt, missing)
                else:
                    # 达到最大重试次数，直接返回最新的结果
                    break

        return final_summary