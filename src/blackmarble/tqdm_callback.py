from typing import Callable, override

from tqdm.asyncio import tqdm_asyncio

ProgressCallback = Callable[[str, float], None]


class tqdm_callback(tqdm_asyncio):
    def __init__(
        self,
        *args,
        step_name: str | None = None,
        callback: Callable[[str, float], None] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.step_name = step_name
        self.callback = callback

    @override
    def update(self, n: float | None = 1) -> bool | None:
        if self.callback:
            name = str(self.step_name or self.desc or "default_step")
            self.callback(name, self.n / self.total)
        return super().update(n)
