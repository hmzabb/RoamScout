"""文本分块 + 结构化数据模板化。

- 语料 md：按 ## 二级标题切（语义边界），目标 400-600 字一块，超长续切并带 70 字
  overlap（约 10%）。为什么按标题不按固定字数？固定字数会把"大雁塔历史"和"大雁塔开放时间"
  切开重组，命中的块语义不完整；标题是作者标好的语义边界，优先信任它。
- 结构化记录（店铺/笔记/券）：一条记录 = 一个 chunk，不硬切。一条店铺信息是原子
  语义单元（几百字以内），切碎了反而丢信息。用模板把字段拼成自然中文——向量模型
  在自然语言上效果最好，字段裸拼（"name=xxx avgPrice=50"）会稀释语义。
"""
from __future__ import annotations # 延迟类型注解求值，允许前向引用

import logging
import re # 正则：HTML 标签清洗、句子边界切分
from dataclasses import dataclass, field

# ---- 分块参数 ----
CHUNK_MAX = 600   # 超过此长度的小节续切（A/B实验：corpus 小节平均 500 字，600 最优）
OVERLAP = 70       # 续切相邻块重叠字数，约 10%（A/B实验：20%=Prec52.2% vs 10%=Prec52.1%，选 10% 减少冗余）

_HTML_TAG_RE = re.compile(r"<[^>]+>")  # 笔记 是markdown形式 是富文本（含 <br/> 等标签）

logger = logging.getLogger(__name__)


def _is_table_block(text: str) -> bool:
    """检测文本块是否为表格（管道符分隔的多行结构）。

    规则：
      1. 连续 ≥2 行包含管道符（|）
      2. 管道行数占总非空行数的 ≥50%
      3. 不存在非管道、非空的"正文行"——防止"一段正文 + 表格 + 正文"
         这种混合内容被误判为纯表格块（导致 _split_table_with_header
         把普通文本当成表头拆分）

    为什么加比例约束？防止"一段正文里恰好有两行含 |"的误判。
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 2:
        return False

    pipe_lines = 0
    consecutive = 0
    max_consecutive = 0
    for line in lines:
        stripped = line.strip()
        if stripped.count("|") >= 1:
            pipe_lines += 1
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    if max_consecutive < 2:
        return False
    if pipe_lines / len(lines) < 0.5:
        return False
    # 如果存在不含管道符的非空行，说明是混合内容（正文+表格），
    # 不应走 _split_table_with_header（第一行可能不是表头）
    if pipe_lines != len(lines):
        return False
    return True


def _split_table_with_header(text: str) -> list[str]:
    """对管道符表格做带表头重复的拆分，每块 ≤ CHUNK_MAX。

    策略：第一行视为表头，后续数据行贪心装箱。每块都带上表头行，
    保证每个 chunk 都是自包含的语义单元——检索时无论命中哪一块，
    都能看到列名，embedding 也不会因文本过长而被稀释。

    为什么不像普通文本那样用 overlap？表格的 overlap 语义不对——
    上一块的尾部数据行带进下一块没有意义，表头才是真正需要的上下文。
    """
    lines = text.strip().split("\n")
    header = lines[0]
    data_lines = lines[1:]

    header_len = len(header) + 1  # +1 for \n
    if header_len >= CHUNK_MAX:
        # 极端情况：表头本身就超长，不退化为整表返回（至少表头完整）
        logger.warning("表格表头过长（%d 字符），整表作为一个 chunk 返回", header_len)
        return [text]

    blocks: list[str] = []
    buf_lines: list[str] = [header]
    buf_len = header_len

    for row in data_lines:
        row_len = len(row) + 1  # +1 for \n
        if buf_len + row_len > CHUNK_MAX and len(buf_lines) > 1:
            blocks.append("\n".join(buf_lines))
            buf_lines = [header]
            buf_len = header_len
        buf_lines.append(row)
        buf_len += row_len

    if len(buf_lines) > 1:  # 至少有一行数据才输出
        blocks.append("\n".join(buf_lines))

    return blocks if blocks else [text]


@dataclass
class Chunk:
    """待入库的最小单元：文本 + 元数据（metadata 用于过滤、统计、引用跳转）。"""

    text: str
    metadata: dict = field(default_factory=dict)


# ==================== 语料 md 分块 ====================

def _split_by_headings(md_text: str) -> list[tuple[str, str]]:
    """按 ## 二级标题切分，返回 [(小节标题, 正文)]；首个标题之前的内容归入「开篇」。"""
    lines = md_text.splitlines()
    sections: list[tuple[str, str]] = []
    current_title, current_body = "开篇", []
    for line in lines:
        if line.startswith("## "):
            if current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = line[3:].strip()
            current_body = []
        else:
            current_body.append(line)
    if current_body:
        sections.append((current_title, "\n".join(current_body).strip()))
    return [s for s in sections if s[1]]  # 丢掉空小节


def _split_long(text: str) -> list[str]:
    """超长文本按句子边界（句号/问号/感叹号/换行）切到 CHUNK_MAX 以内，
    相邻块重叠 OVERLAP 字。先按句子切碎再贪心装箱，避免把句子拦腰截断。

    表格保护：若文本为管道符分隔的表格块，走 _split_table_with_header
    做带表头重复的拆分——每块 ≤ CHUNK_MAX，且都带上表头行。
    既保证检索精度（chunk 大小合理），又保证语义完整（列名不丢失）。
    """
    if _is_table_block(text):
        pieces = _split_table_with_header(text)
        logger.info(
            "表格块（%d 字符）→ 带表头拆分为 %d 块，每块 ≤ %d 字符",
            len(text), len(pieces), CHUNK_MAX,
        )
        return pieces

    sentences = re.split(r"(?<=[。！？\n])", text)
    blocks: list[str] = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) > CHUNK_MAX and buf:#and buf防止第一轮就超限
            blocks.append(buf.strip())
            buf = buf[-OVERLAP:] + sent  # 重叠：上一块尾部 70 字带进下一块
        else:
            buf += sent
    if buf.strip():
        blocks.append(buf.strip())
    return blocks


def chunk_markdown(md_text: str, doc_title: str, file_name: str = "") -> list[Chunk]:
    """语料 md → 若干 Chunk。块内首行保留小节标题，检索命中时自带上下文。"""
    chunks: list[Chunk] = []
    for section_title, body in _split_by_headings(md_text):
        pieces = _split_long(body) if len(body) > CHUNK_MAX else [body]
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    text=f"{section_title}\n{piece}",
                    metadata={
                        "source": "corpus",
                        "doc": doc_title,
                        "file": file_name,
                        "section": section_title,
                        "part": i,
                    },
                )
            )
    return chunks


# ==================== 纯文本分块（PDF/Word 等无 Markdown 标题结构的文档） ====================

def chunk_plain_text(text: str, doc_title: str, file_name: str = "") -> list[Chunk]:
    """纯文本 → 若干 Chunk。按段落（空行）切分，超长段复用 _split_long 续切。

    PDF/Word 文档没有 Markdown 的 ## 标题结构，改为按段落语义边界切分。
    段落间以连续空行分隔，保留原文的语义分组。
    """
    chunks: list[Chunk] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    part = 0
    for para in paragraphs:
        if len(para) > CHUNK_MAX:
            pieces = _split_long(para)
            for piece in pieces:
                chunks.append(
                    Chunk(
                        text=piece,
                        metadata={
                            "source": "corpus",
                            "doc": doc_title,
                            "file": file_name,
                            "section": "正文",
                            "part": part,
                        },
                    )
                )
                part += 1
        else:
            chunks.append(
                Chunk(
                    text=para,
                    metadata={
                        "source": "corpus",
                        "doc": doc_title,
                        "file": file_name,
                        "section": "正文",
                        "part": part,
                    },
                )
            )
            part += 1
    return chunks


# ==================== 结构化数据模板（一条记录 = 一个 chunk） ====================

def _score_text(score) -> str:
    """Java 端 score ×10 存整数（4.7 分存 47），展示时除 10。"""
    return f"{score / 10:.1f}分" if score else "暂无评分"


def shop_to_text(shop: dict, type_name: str = "") -> str:
    """店铺 → 自然中文。type_name 由 ingest 传入（分类表 id→name 映射）。"""
    return (
        f"【{shop.get('name', '')}】是一家位于西安{shop.get('area', '')}商圈的"
        f"{type_name or '商铺'}店，地址：{shop.get('address', '')}。"
        f"人均消费约{shop.get('avgPrice') or '暂无'}元，"
        f"评分{_score_text(shop.get('score'))}，"
        f"月销量{shop.get('sold') or 0}单，累计评论{shop.get('comments') or 0}条。"
        f"营业时间：{shop.get('openHours') or '暂无'}。"
    )


def blog_to_text(blog: dict) -> str:
    """笔记 → 自然中文。content 去 HTML 标签；name 是 /blog/hot 填充的作者昵称。"""
    content = _HTML_TAG_RE.sub(" ", blog.get("content") or "").strip()
    return (
        f"用户{blog.get('name') or '网友'}发布探店笔记《{blog.get('title', '')}》："
        f"{content}"
        f"（点赞{blog.get('liked') or 0}，评论{blog.get('comments') or 0}）"
    )


def voucher_to_text(voucher: dict, shop_name: str = "") -> str:
    """优惠券 → 自然中文。

    库存三态（None/0/>0）区分；payValue=0 时表达为"免费领取"；
    rules 按 list 展开或按字符串原样输出；时间字段可选拼接。
    """

    pay, actual, stock = voucher.get("payValue"), voucher.get("actualValue"), voucher.get("stock")

    if stock is None:
        stock_text = "库存未知"
    elif stock == 0:
        stock_text = "已售罄"
    else:
        stock_text = f"库存{stock}张"

    begin_time = voucher.get("beginTime")
    end_time = voucher.get("endTime")
    time_parts = []
    if begin_time:
        time_parts.append(f"生效时间：{begin_time}")
    if end_time:
        time_parts.append(f"失效时间：{end_time}")
    time_text = "，".join(time_parts) if time_parts else "有效期暂无"

    if pay is not None and actual is not None:
        if pay == 0:
            deal = "免费领取"
        else:
            deal = f"{pay}元抵{actual}元"
    else:
        deal = ""

    rules_raw = voucher.get("rules")
    if rules_raw:
        if isinstance(rules_raw, list):
            rules_text = "；".join(rules_raw)
        else:
            rules_text = str(rules_raw)
    else:
        rules_text = "详见门店"

    return (
        f"{shop_name}店铺优惠券【{voucher.get('title', '')}】"
        f"{'，' + deal if deal else ''}"
        f"{'，' + voucher.get('subTitle', '') if voucher.get('subTitle') else ''}"
        f"。{stock_text}。{time_text}。"
        f"使用规则：{rules_text}。"
    )


# ==================== 递归字符切分（baseline，全格式通用） ====================

def _recursive_split(
    text: str,
    separators: list[str] | None = None,
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[str]:
    """纯递归字符切分：按 separators 优先级逐级切，贪心装箱 + 重叠。

    通过函数递归来实现"逐级切分"——当前级别切出的片段如果还是超长，
    就调用自己用下一级分隔符继续切。不需要显式的栈/队列数据结构，
    代码更短且语义直观。这就是"递归切分"名称的由来。

    参数：
      separators: 分隔符优先级列表，默认 ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]
      chunk_size: 目标块最大长度
      overlap: 相邻块重叠字符数

    实现要点：
      1. 用当前分隔符切分
      2. 贪心装箱：当前片上片 ≤ chunk_size → 合并
      3. 超长的片 → 递归用下一级分隔符继续切
      4. 相邻块尾部和首部 overlap 字重叠
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

    separator = separators[0]
    remaining_separators = separators[1:]

    if separator:
        pieces = text.split(separator)
    else:
        pieces = list(text)

    blocks: list[str] = []
    buf = ""
    for piece in pieces:
        if separator:
            piece_with_sep = piece + separator if piece != pieces[-1] else piece
        else:
            piece_with_sep = piece

        if len(buf) + len(piece_with_sep) <= chunk_size:
            buf += piece_with_sep
        else:
            if buf.strip():
                blocks.append(buf.strip())
            if len(piece_with_sep) > chunk_size and remaining_separators:
                sub_pieces = _recursive_split(
                    piece_with_sep, remaining_separators, chunk_size, overlap
                )
                blocks.extend(sub_pieces)
            else:
                buf = piece_with_sep

    if buf.strip():
        blocks.append(buf.strip())

    if overlap > 0 and len(blocks) > 1:
        overlapped: list[str] = [blocks[0]]
        for i in range(1, len(blocks)):
            prev_tail = blocks[i - 1][-overlap:] if len(blocks[i - 1]) >= overlap else blocks[i - 1]
            overlapped.append(prev_tail + blocks[i])
        return overlapped

    return blocks


def chunk_recursive(
    text: str,
    doc_title: str,
    file_name: str = "",
    chunk_size: int = 500,
    overlap: int = 100,
) -> list[Chunk]:
    """递归字符切分 → 若干 Chunk。全格式通用，不感知文档结构。

    这是最朴素的 baseline 策略——不管你是 Markdown 还是 PDF 还是 Word，
    全部当纯文本按固定字符数切。实现最快、最通用，但可能把紧邻的语义单元
    （如"门票价格"和"购票须知"）切到两个块里。
    """
    pieces = _recursive_split(text, chunk_size=chunk_size, overlap=overlap)
    chunks: list[Chunk] = []
    for i, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                text=piece,
                metadata={
                    "source": "corpus",
                    "doc": doc_title,
                    "file": file_name,
                    "section": "正文",
                    "part": i,
                },
            )
        )
    return chunks


# ==================== 父子文档切分 ====================

@dataclass
class ParentChunk:
    """父块：包含完整上下文的较大文本块，不做 embedding。

    检索时命中 Child → 通过 parent_id 取出对应的 ParentChunk → 把 Parent.text
    注入 LLM，确保 LLM 拿到完整的上下文窗口。
    """

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_parent_child(
    text: str,
    doc_title: str,
    file_name: str = "",
    child_size: int = 400,
    parent_strategy: str = "doc",
    overlap: int = 80,
) -> tuple[list[Chunk], dict[str, ParentChunk]]:
    """父子文档切分 → (Child Chunk 列表, parent_id → ParentChunk 映射)。

    Child 小颗粒（精确检索）→ 存入 Milvus 做 embedding 检索。
    Parent 大颗粒（完整上下文）→ 检索命中 Child 后，根据 parent_id 取出 Parent，
    将 Parent 文本注入 LLM prompt。

    parent_strategy:
      - "doc": 整篇文档作为一个 Parent（适合 corpus 中文档 ≤3000 字）
      - "fixed": 固定大小 Parent，按 parent_size 切分
      - "section": 按 ## 二级标题的自然小节作为 Parent
    """
    children: list[Chunk] = []
    parents: dict[str, ParentChunk] = {}
    parent_idx = 0

    if parent_strategy == "section":
        sections = _split_by_headings(text)
        for section_title, body in sections:
            section_text = f"{section_title}\n{body}"
            p_id = f"{doc_title}:p{parent_idx}"
            parents[p_id] = ParentChunk(
                id=p_id,
                text=section_text,
                metadata={"doc": doc_title, "file": file_name, "section": section_title},
            )

            pieces = _recursive_split(body, chunk_size=child_size, overlap=overlap)
            for i, piece in enumerate(pieces):
                children.append(
                    Chunk(
                        text=f"{section_title}\n{piece}",
                        metadata={
                            "source": "corpus",
                            "doc": doc_title,
                            "file": file_name,
                            "section": section_title,
                            "part": i,
                            "parent_id": p_id,
                        },
                    )
                )
            parent_idx += 1

    elif parent_strategy == "fixed":
        parent_size = int(child_size * 3)
        while text:
            p_text = text[:parent_size]
            text = text[parent_size:]
            p_id = f"{doc_title}:p{parent_idx}"
            parents[p_id] = ParentChunk(
                id=p_id,
                text=p_text,
                metadata={"doc": doc_title, "file": file_name, "section": "正文"},
            )

            pieces = _recursive_split(p_text, chunk_size=child_size, overlap=overlap)
            for i, piece in enumerate(pieces):
                children.append(
                    Chunk(
                        text=piece,
                        metadata={
                            "source": "corpus",
                            "doc": doc_title,
                            "file": file_name,
                            "section": "正文",
                            "part": i,
                            "parent_id": p_id,
                        },
                    )
                )
            parent_idx += 1

    else:
        p_id = f"{doc_title}:p0"
        parents[p_id] = ParentChunk(
            id=p_id,
            text=text,
            metadata={"doc": doc_title, "file": file_name, "section": "整篇"},
        )

        pieces = _recursive_split(text, chunk_size=child_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            children.append(
                Chunk(
                    text=piece,
                    metadata={
                        "source": "corpus",
                        "doc": doc_title,
                        "file": file_name,
                        "section": "正文",
                        "part": i,
                        "parent_id": p_id,
                    },
                )
            )

    return children, parents