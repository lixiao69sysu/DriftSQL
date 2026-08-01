import { describe, expect, it } from "vitest";

import { experimentLabel, streamLabel, toolLabel, zhLabel } from "./locale";

describe("中文可观测性标签", () => {
  it("翻译漂移、失败、工具、流状态和实验名称", () => {
    expect(zhLabel("rename_column")).toBe("字段重命名");
    expect(zhLabel("task_failure")).toBe("任务失败");
    expect(toolLabel("inspect_schema_diff")).toBe("检查 Schema 变更");
    expect(streamLabel("live")).toBe("实时");
    expect(experimentLabel("GRPO step 10")).toBe("GRPO 第 10 步");
  });
});
