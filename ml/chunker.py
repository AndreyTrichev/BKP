import os

CHUNK_MAX_CHARS = max(200, int(os.environ.get("CHUNK_MAX_CHARS", "900")))


def split_to_chunks(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    compact = text.strip()
    if not compact:
        return []
    if len(compact) <= max_chars:
        return [compact]

    paragraphs = compact.split("\n\n")
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

                                                                    
        if len(para) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            while len(para) > max_chars:
                cut = para[:max_chars].rfind(". ")
                if cut < max_chars // 2:
                    cut = max_chars
                else:
                    cut += 2                 
                chunks.append(para[:cut].strip())
                para = para[cut:].strip()
            if para:
                current = para
            continue

                                          
        candidate = (current + "\n\n" + para).strip() if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    return chunks
