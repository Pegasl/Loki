"""Route public progress and Markdown answers from model responses."""

from langchain_core.messages import message_chunk_to_message

from .progress import report_event


def text_content(content):
    """Extract public text blocks without exposing reasoning or tool arguments."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text")
        )
    return ""


class ResponseText:
    """Separate an optional leading progress block from verbatim Markdown."""

    OPEN = "<progress>"
    CLOSE = "</progress>"

    def __init__(self, commentary, answer, finish_commentary=lambda: None):
        self.commentary = commentary
        self.answer = answer
        self.finish_commentary = finish_commentary
        self.pending = ""
        self.mode = "start"
        self.text = ""

    def emit_answer(self, text):
        if text:
            self.text += text
            self.answer(text)

    def feed(self, text):
        if self.mode == "answer":
            self.emit_answer(text)
            return
        self.pending += text
        if self.mode == "start":
            candidate = self.pending.lstrip()
            if not candidate or self.OPEN.startswith(candidate):
                return
            if not candidate.startswith(self.OPEN):
                self.mode = "answer"
                self.emit_answer(self.pending)
                self.pending = ""
                return
            self.pending = candidate[len(self.OPEN):]
            self.mode = "progress"
        if self.CLOSE in self.pending:
            public, body = self.pending.split(self.CLOSE, 1)
            self.commentary(public)
            self.finish_commentary()
            self.mode, self.pending = "answer", ""
            self.emit_answer(body)
            return
        # Retain only a possible split closing tag.
        keep = 0
        for size in range(1, min(len(self.pending), len(self.CLOSE) - 1) + 1):
            if self.CLOSE.startswith(self.pending[-size:]):
                keep = size
        safe = self.pending[:-keep] if keep else self.pending
        self.pending = self.pending[-keep:] if keep else ""
        if safe:
            self.commentary(safe)

    def finish(self):
        if self.mode == "progress":
            raise ValueError("进度标签未闭合，回答未完成")
        if self.pending:
            self.emit_answer(self.pending)
            self.pending = ""
        self.mode = "answer"


def invoke(reporter, agent, model, messages):
    report_event(reporter, "model_started", agent)
    stream = agent == "ask" and getattr(reporter, "stream_models", False) is True
    project_answer = stream and callable(getattr(reporter, "answer_delta", None))
    public = ResponseText(
        lambda text: report_event(reporter, "commentary_delta", text),
        lambda text: report_event(reporter, "answer_delta", text) if project_answer else None,
        lambda: report_event(reporter, "commentary_finished"),
    )
    try:
        if not stream or not callable(getattr(model, "stream", None)):
            project_answer = False
            response = model.invoke(messages)
            if agent == "ask":
                public.feed(text_content(response.content))
        else:
            combined = None
            iterator = model.stream(messages)
            try:
                for chunk in iterator:
                    combined = chunk if combined is None else combined + chunk
                    public.feed(text_content(chunk.content))
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            if combined is None:
                raise ValueError("Model returned an empty stream")
            response = message_chunk_to_message(combined)
        if agent == "ask":
            public.finish()
            response.content = public.text
            if project_answer and public.text:
                response.additional_kwargs["loki_streamed_answer"] = public.text
        report_event(reporter, "model_finished", agent)
        return response
    except BaseException:
        report_event(reporter, "model_failed", agent)
        if project_answer and public.text:
            report_event(reporter, "answer_incomplete")
        raise
    finally:
        report_event(reporter, "commentary_finished")
