"""Stream public updates while keeping final JSON and tool arguments private."""

from langchain_core.messages import message_chunk_to_message

from .progress import report_event


class PublicUpdate:
    """Only an explicit leading <progress> block is user-facing commentary."""

    OPEN = "<progress>"
    CLOSE = "</progress>"

    def __init__(self, emit, finish=lambda: None):
        self.emit = emit
        self.finish = finish
        self.pending = ""
        self.mode = "start"

    def feed(self, text):
        if not isinstance(text, str) or self.mode == "done":
            return
        self.pending += text
        if self.mode == "start":
            self.pending = self.pending.lstrip()
            if self.OPEN.startswith(self.pending):
                return
            if not self.pending.startswith(self.OPEN):
                self.mode, self.pending = "done", ""
                return
            self.pending = self.pending[len(self.OPEN):]
            self.mode = "text"
        if self.CLOSE in self.pending:
            public, _ = self.pending.split(self.CLOSE, 1)
            self.emit(public)
            self.mode, self.pending = "done", ""
            self.finish()
            return
        # Retain only a possible split closing tag, not the whole text block.
        keep = 0
        for size in range(1, min(len(self.pending), len(self.CLOSE) - 1) + 1):
            if self.CLOSE.startswith(self.pending[-size:]):
                keep = size
        safe = self.pending[:-keep] if keep else self.pending
        self.pending = self.pending[-keep:] if keep else ""
        if safe:
            self.emit(safe)


def invoke(reporter, agent, model, messages):
    report_event(reporter, "model_started", agent)
    public = PublicUpdate(lambda text: report_event(reporter, "commentary_delta", text),
                          lambda: report_event(reporter, "commentary_finished"))
    stream = agent == "ask" and getattr(reporter, "stream_models", False) is True
    try:
        if not stream or not callable(getattr(model, "stream", None)):
            response = model.invoke(messages)
            if agent == "ask":
                public.feed(response.content)
        else:
            combined = None
            iterator = model.stream(messages)
            try:
                for chunk in iterator:
                    combined = chunk if combined is None else combined + chunk
                    content = chunk.content
                    if isinstance(content, list):
                        content = "".join(block if isinstance(block, str) else block.get("text", "")
                                          for block in content
                                          if isinstance(block, str) or (isinstance(block, dict) and block.get("type") == "text"))
                    public.feed(content)
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            if combined is None:
                raise ValueError("Model returned an empty stream")
            response = message_chunk_to_message(combined)
        report_event(reporter, "model_finished", agent)
        return response
    except BaseException:
        report_event(reporter, "model_failed", agent)
        raise
    finally:
        report_event(reporter, "commentary_finished")
