"""
QA 对生成脚本
输入: 数据预处理后的 data.json (按页切分, 每条含 page_number / raw_text / char_count)
输出: qa_pairs.jsonl, 每行一条 QA 对, 供下游检索优化使用

用法:
    python generate_qa.py --input data.json --output qa_pairs.jsonl --doc-name 综合意外伤害保险条款

依赖:
    pip install openai tqdm
(也可换成任何兼容 OpenAI SDK 的模型,只需改 BASE_URL / MODEL)
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# 自动加载 .env 文件里的环境变量 (VOLC_API_KEY 等)
load_dotenv()

# ============== 配置区 ==============
# 火山引擎方舟 Coding Plan, 兼容 OpenAI SDK
# key 从 .env 文件加载, 不要写死在代码里
API_KEY = os.environ.get("VOLC_API_KEY")
BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
MODEL = "ark-code-latest"

# 每页生成多少个 QA 对
QA_PER_PAGE = 6
# 重试次数
MAX_RETRIES = 3
# 内容太短的页跳过 (避免给目录页/空白页出题)
MIN_CHARS = 100
# ===================================


SYSTEM_PROMPT = """你是一个专业的中文文档 QA 数据标注员。你的任务是基于给定的文档片段,生成高质量的「问题-答案」对,用于训练和评估检索系统。

## 严格遵守的规则
1. 答案必须能在原文中直接找到或直接推理得出,禁止编造、禁止引入外部知识。
2. 问题必须独立可理解:不要出现"本条款"、"上述"、"该条"、"如前所述"等依赖上下文的指代词。
3. 问题要使用通俗、用户视角的表达,避免照抄原文措辞。
4. 答案要简洁完整:能一句话说清楚的不写两句,但关键数字、条件、例外不能丢。
5. 覆盖多种问题类型,不要全是事实型。

## 问题类型 (尽量分散)
- factual:      事实型, 问"是什么/是谁/在哪"
- numerical:    数值型, 问具体数字、期限、比例
- conditional:  条件型, 问"在什么情况下"
- enumeration:  列举型, 问"包括哪些"
- negation:     否定/排除型, 问"哪些不属于/是否包括"
- procedural:   流程型, 问"如何申请/需要哪些材料"

## 输出格式
严格输出 JSON 数组,不要任何额外说明文字、不要 markdown 代码块。每个元素如下:
{
  "question": "...",
  "answer": "...",
  "question_type": "factual|numerical|conditional|enumeration|negation|procedural",
  "evidence": "原文中支持该答案的片段"
}
"""

USER_PROMPT_TEMPLATE = """文档名称: {doc_name}
当前页码: {page_number}

当前页内容:
```
{current_page_text}
```

相邻页上下文 (仅供理解跨页内容,不要单独基于这部分出题):
```
{neighbor_context}
```

请基于「当前页内容」生成 {n} 个 QA 对,类型尽量分散。直接输出 JSON 数组。
"""


def build_neighbor_context(pages: list, idx: int, window: int = 1) -> str:
    """取当前页前后各 window 页的内容作为上下文,缓解跨页切分的语义断裂问题"""
    parts = []
    for i in range(max(0, idx - window), min(len(pages), idx + window + 1)):
        if i == idx:
            continue
        parts.append(f"[第{pages[i]['page_number']}页]\n{pages[i]['raw_text']}")
    return "\n\n".join(parts) if parts else "(无)"


def extract_json_array(text: str):
    """从模型输出中提取 JSON 数组,容错处理 markdown 代码块、前后多余文字"""
    # 去掉 ```json ... ``` 这种包裹
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    # 找到第一个 [ 和最后一个 ]
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"模型输出中未找到 JSON 数组: {text[:200]}")
    return json.loads(text[start : end + 1])


def call_llm(client: OpenAI, system: str, user: str) -> str:
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,  # 略高,让问题表达更多样
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            wait = 2**attempt
            print(f"[warn] 第 {attempt+1} 次调用失败: {e}, {wait}s 后重试")
            time.sleep(wait)
    raise RuntimeError(f"调用 LLM 失败 {MAX_RETRIES} 次: {last_err}")


def generate_qa_for_page(
    client: OpenAI, page: dict, neighbor_context: str, doc_name: str, n: int
) -> list:
    user_prompt = USER_PROMPT_TEMPLATE.format(
        doc_name=doc_name,
        page_number=page["page_number"],
        current_page_text=page["raw_text"],
        neighbor_context=neighbor_context,
        n=n,
    )
    raw = call_llm(client, SYSTEM_PROMPT, user_prompt)
    try:
        items = extract_json_array(raw)
    except Exception as e:
        print(f"[warn] 第 {page['page_number']} 页 JSON 解析失败,跳过。错误: {e}")
        return []
    return items


def validate_qa(item: dict) -> bool:
    """基本字段校验,过滤掉残缺数据"""
    required = {"question", "answer", "question_type", "evidence"}
    if not required.issubset(item.keys()):
        return False
    if not all(isinstance(item[k], str) and item[k].strip() for k in required):
        return False
    # 过滤含有指代词的问题
    bad_refs = ["本条款", "本条", "上述", "该条", "如前所述", "前文", "该项"]
    if any(ref in item["question"] for ref in bad_refs):
        return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="输入的 data.json 路径")
    parser.add_argument("--output", required=True, help="输出的 qa_pairs.jsonl 路径")
    parser.add_argument(
        "--doc-name", required=True, help="文档名称,会写入 source_doc 字段"
    )
    parser.add_argument("--qa-per-page", type=int, default=QA_PER_PAGE)
    parser.add_argument(
        "--start-id", type=int, default=1, help="qa_id 起始编号,跨多文档时可调"
    )
    args = parser.parse_args()

    pages = json.loads(Path(args.input).read_text(encoding="utf-8"))
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    qa_id = args.start_id
    total_written = 0

    # 用 'a' 追加模式,挂了重跑也不会丢之前的
    with open(args.output, "a", encoding="utf-8") as fout:
        for idx, page in enumerate(tqdm(pages, desc="生成 QA")):
            if page.get("char_count", len(page["raw_text"])) < MIN_CHARS:
                continue

            neighbor = build_neighbor_context(pages, idx, window=1)
            items = generate_qa_for_page(
                client, page, neighbor, args.doc_name, args.qa_per_page
            )

            for item in items:
                if not validate_qa(item):
                    continue
                record = {
                    "qa_id": f"qa_{qa_id:05d}",
                    "question": item["question"].strip(),
                    "answer": item["answer"].strip(),
                    "question_type": item["question_type"].strip(),
                    "source_doc": args.doc_name,
                    "source_page": page["page_number"],
                    "source_chunk_id": f"{args.doc_name}_page_{page['page_number']}",
                    "evidence": item["evidence"].strip(),
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                qa_id += 1
                total_written += 1

    print(f"完成: 共写入 {total_written} 条 QA, 输出文件: {args.output}")


if __name__ == "__main__":
    main()
