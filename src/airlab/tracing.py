import time, uuid
from contextlib import contextmanager
from .models import Trace, TraceSpan

class Tracer:
    def __init__(self):
        self.trace = Trace(trace_id=str(uuid.uuid4()))

    @contextmanager
    def span(self, name: str, **attributes):
        start = time.perf_counter() * 1000
        try:
            yield
        finally:
            end = time.perf_counter() * 1000
            self.trace.spans.append(TraceSpan(name, start, end, attributes))

    def annotate(self, **attributes):
        self.trace.attributes.update(attributes)
