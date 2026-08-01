const labels: Record<string, string> = {
  add_column: "新增字段",
  clean: "无漂移对照",
  compound: "复合漂移",
  rename_column: "字段重命名",
  rename_table: "表重命名",
  replace_column: "字段替换",
  easy: "简单",
  medium: "中等",
  hard: "困难",
  multi_table_plain: "多表非限定通配符",
  multi_table_qualified: "多表限定通配符",
  single_table_plain: "单表非限定通配符",
  single_table_qualified: "单表限定通配符",
  submitted: "已提交答案",
  max_turns: "达到最大轮数",
  max_tool_calls: "达到工具调用上限",
  max_total_tokens: "达到 Token 上限",
  timeout: "执行超时",
  cancelled: "用户取消",
  service_restart: "服务重启",
  get_schema_version: "获取 Schema 版本",
  inspect_schema_diff: "检查 Schema 变更",
  execute_sql: "执行 SQL",
  search_business_knowledge: "检索业务知识",
  ask_user: "询问用户",
  submit_solution: "提交答案",
  success: "任务成功",
  valid: "提交有效",
  clarify: "消除歧义",
  efficient: "交互效率",
  execution: "执行成功",
  format: "格式正确",
  tool_cost: "工具成本",
  token_cost: "Token 成本",
  unsafe: "不安全操作",
  invalid: "无效操作",
  timed_out: "执行超时",
  budget_exhausted: "预算耗尽",
  service_error: "服务异常",
  task_failure: "任务失败",
};

export function zhLabel(value: string | null | undefined, fallback = "—"): string {
  if (!value) return fallback;
  return labels[value] ?? value.replaceAll("_", " ");
}

export function toolLabel(value: string): string {
  return labels[value] ?? value;
}

export function rewardLabel(value: string): string {
  return labels[value.replaceAll(" ", "_")] ?? value;
}

export function streamLabel(value: string): string {
  return {
    idle: "待机",
    connecting: "连接中",
    live: "实时",
    closed: "已关闭",
    error: "连接异常",
  }[value] ?? value;
}

export function experimentLabel(value: string): string {
  return {
    "Stage 7 frozen": "阶段 7 冻结基线",
    "Stage 8 SFT20": "阶段 8 SFT20",
    "GRPO step 5": "GRPO 第 5 步",
    "GRPO step 10": "GRPO 第 10 步",
    "Conservative step 2": "保守策略第 2 步",
    "Conservative step 4": "保守策略第 4 步",
    "Conservative step 6": "保守策略第 6 步",
  }[value] ?? value;
}
