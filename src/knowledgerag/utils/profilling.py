import asyncio
import time
from collections import namedtuple
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable

from loguru import logger

TimeitResult = namedtuple("TimeitResult", ["elapsed_time", "elapsed_time_str", "function_result"])


def timeit(func: Callable) -> Callable:
    """AI is creating summary for timeit

    Args:
        func (Callable): [description]

    Returns:
        Callable: [description]
    """

    @wraps(func)
    def wrapper(*args: list[Any], **kwargs: dict[str, Any]) -> TimeitResult:
        """AI is creating summary for wrapper

        Returns:
            TimeitResult: [description]
        """
        start_time = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.perf_counter()
            _time = end_time - start_time
            f_time = f"{_time:.4f} s"
        return TimeitResult(_time, f_time, result)

    return wrapper


def async_timeit(func):
    """AI is creating summary for async_timeit

    Args:
        func ([type]): [description]

    Returns:
        [type]: [description]
    """

    @wraps(func)
    async def wrapper(*args, **kwargs) -> TimeitResult:
        """AI is creating summary for wrapper

        Returns:
            TimeitResult: [description]
        """
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
        finally:
            _time = time.perf_counter() - start_time
            f_time = f"{_time:.4f} s"
        return TimeitResult(_time, f_time, result)

    return wrapper


def timeit_all(func) -> Callable:
    """AI is creating summary for timeit_all

    Args:
        func ([type]): [description]

    Returns:
        Callable: [description]
    """

    @contextmanager
    def wrapping_logic() -> None:
        """ """
        start_ts = time.time()
        yield
        dur = time.time() - start_ts
        logger.info(f"{func.__name__} took {dur:.4} seconds")

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        """ """
        match asyncio.iscoroutinefunction(func):
            case True:

                def tmp() -> Any:
                    with wrapping_logic():
                        return func(*args, **kwargs)

            case False:

                async def tmp() -> Any:
                    with wrapping_logic():
                        return await func(*args, **kwargs)

        return tmp()

    return wrapper


def timed(fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator log test start and end time of a function
    :param fn: Function to decorate
    :return: Decorated function
    Example:
    >>> @timed
    >>> def test_fn():
    >>>     time.sleep(1)
    >>> test_fn()
    """

    def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = fn(*args, **kwargs)
        duration = time.time() - start
        duration_str = get_duration_str(duration)
        return TimeitResult(duration, duration_str, result)

    async def wrapped_fn_async(*args: Any, **kwargs: Any) -> Any:
        start = time.time()
        result = await fn(*args, **kwargs)
        duration = time.time() - start
        duration_str = get_duration_str(start)
        return TimeitResult(duration, duration_str, result)

    match asyncio.iscoroutinefunction(fn):
        case True:
            return wrapped_fn_async
        case False:
            return wrapped_fn


def get_duration_str(duration: float) -> str:
    """Get human readable duration string from start time"""
    match duration:
        case _ if duration >= 1:
            duration_str = f"{duration:,.3f}s"
        case _ if duration >= 1e-3:
            duration_str = f"{round(duration * 1e3)}ms"
        case _ if duration >= 1e-6:
            duration_str = f"{round(duration * 1e6)}us"
        case _ if duration < 1e-6:
            duration_str = f"{duration * 1e9}ns"
    return duration_str
