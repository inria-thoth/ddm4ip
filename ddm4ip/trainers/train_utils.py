import contextlib
import time


def get_next_batch(loader, iterator):
    try:
        batch = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
    return batch, iterator


class Timer:
    def __init__(self, cpu_only: bool = False):
        self.cpu_only = cpu_only
        self._cpu_time = 0
        if not cpu_only:
            self._cuda_time = []

    @contextlib.contextmanager
    def measure(self):
        start_cpu = time.time()
        try:
            yield
        finally:
            self._cpu_time += time.time() - start_cpu
        # if not self.cpu_only:
        #     start_cuda = torch.cuda.Event(enable_timing=True)
        #     end_cuda = torch.cuda.Event(enable_timing=True)
        # start_cpu = time.time()
        # if not self.cpu_only:
        #     start_cuda.record() # type: ignore
        # try:
        #     yield
        # finally:
        #     if not self.cpu_only:
        #         end_cuda.record() # type: ignore
        #     self._cpu_time += time.time() - start_cpu
        #     if not self.cpu_only:
        #         self._cuda_time.append(lambda: start_cuda.elapsed_time(end_cuda) / 1000)
        #     return

    def cuda_time(self) -> float:
        return self.cpu_time()
        if self.cpu_only:
            raise ValueError()
        if len(self._cuda_time) == 0:
            return 0.0
        out = sum(ct() for ct in self._cuda_time)
        self._cuda_time = []
        return out

    def cpu_time(self) -> float:
        out = self._cpu_time
        self._cpu_time = 0
        return out
