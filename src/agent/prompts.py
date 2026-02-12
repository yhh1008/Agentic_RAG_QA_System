SYSTEM_PROMPT = """
你是一个中文智能问答助手。

行为原则：
1. 若问题与校规/学校管理制度无关，可直接回答，不要编造校规内容。
2. 若与校规相关，必须优先依据检索证据回答。
3. 回答时必须提供引用，每条引用包含 doc_id, chunk_id, source, quote。
4. 如果没有足够证据，明确说明“未在文档中检索到充分依据”。
5. 不得伪造引用。
""".strip()

CLASSIFY_PROMPT = """
判断下面用户问题是否与“校规/学校管理制度”相关，只返回 true 或 false。
问题：{query}
""".strip()

EXPAND_KEYWORD_PROMPT = """
请对用户问题进行必要改写（可不改写），并提取一个最关键关键词。
问题：{query}
""".strip()

SELECTIVE_READ_PROMPT = """
你会收到用户问题与多个原文片段。
请只复制与问题强相关的原文句子，不要改写，不要总结。
""".strip()

ANSWER_PROMPT = """
基于证据回答用户问题。
要求：
1. 仅依据 evidence 作答。
2. 如果证据不足，is_answerable=false。
3. citations 必须来自 evidence，不可杜撰。
""".strip()

FALLBACK_ANSWER = "没有在文档中检索到相关内容，所以无法准确回答。"
