"""Background-task controller for the Rivelero GUI.

This module provides the Qt bridge between long-running Rivelero operations
and ApplicationState.

Scientific functions remain independent of Qt. The TaskController executes
them on QThreadPool workers and translates their lifecycle into:

    ApplicationState.task
    Qt signals
    progress callbacks
    cooperative cancellation requests

The controller does not contain scientific calculations.

Cancellation
------------
Cancellation is cooperative. Rivelero must never forcibly terminate a Python
thread or GDAL operation. A running function can stop only when it reaches a
point where it checks the supplied CancellationToken.

Operations that do not yet support cancellation can still receive a
cancellation request. The GUI will enter CANCELLING state and the operation
will finish at the next supported boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
import traceback
from typing import Any, Callable
from uuid import uuid4

try:
    from PySide6.QtCore import (
        QObject,
        QRunnable,
        QThreadPool,
        Signal,
        Slot,
    )
except ImportError:
    try:
        from PyQt6.QtCore import (
            QObject,
            QRunnable,
            QThreadPool,
            pyqtSignal as Signal,
            pyqtSlot as Slot,
        )
    except ImportError as exc:
        raise ImportError(
            "Rivelero GUI requires PySide6 or PyQt6."
        ) from exc

from rivelero.gui.application_state import (
    ApplicationState,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# Exceptions / cancellation
# ---------------------------------------------------------------------------


class TaskCancelledError(RuntimeError):
    """Raised when a cooperative Rivelero task acknowledges cancellation."""


class CancellationToken:
    """Thread-safe cooperative cancellation token.

    Scientific/background operations may periodically call:

        token.raise_if_cancelled()

    or:

        if token.cancelled:
            ...

    The token does not forcibly terminate threads.
    """

    def __init__(self) -> None:
        self._event = Event()

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""

        return self._event.is_set()

    def cancel(self) -> None:
        """Request cooperative cancellation."""

        self._event.set()

    def raise_if_cancelled(self) -> None:
        """Raise TaskCancelledError if cancellation was requested."""

        if self.cancelled:
            raise TaskCancelledError(
                "Task cancellation was requested."
            )


# ---------------------------------------------------------------------------
# Task context
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskContext:
    """Context optionally supplied to a background operation.

    Parameters
    ----------
    task_id
        Unique identifier assigned by TaskController.

    cancellation_token
        Cooperative cancellation token.

    progress
        Callback accepting:

            processed
            total
            unit_id
            message

        All values except ``processed`` may be None where appropriate.
    """

    task_id: str

    cancellation_token: CancellationToken

    progress: Callable[
        [
            int,
            int | None,
            str | None,
            str | None,
        ],
        None,
    ]

    @property
    def cancelled(self) -> bool:
        """Whether cancellation has been requested."""

        return self.cancellation_token.cancelled

    def raise_if_cancelled(self) -> None:
        """Raise if cancellation has been requested."""

        self.cancellation_token.raise_if_cancelled()

    def report_progress(
        self,
        processed: int,
        total: int | None = None,
        unit_id: str | None = None,
        message: str | None = None,
    ) -> None:
        """Report progress through the TaskController."""

        self.progress(
            processed,
            total,
            unit_id,
            message,
        )


# ---------------------------------------------------------------------------
# Worker signals
# ---------------------------------------------------------------------------


class WorkerSignals(QObject):
    """Signals emitted by one Rivelero background worker."""

    progress = Signal(
        str,     # task_id
        int,     # processed
        object,  # total: int | None
        object,  # unit_id: str | None
        object,  # message: str | None
    )

    result = Signal(
        str,     # task_id
        object,  # result
    )

    error = Signal(
        str,     # task_id
        str,     # error message
        str,     # traceback
    )

    cancelled = Signal(
        str,     # task_id
        object,  # message
    )

    finished = Signal(
        str,     # task_id
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class TaskWorker(QRunnable):
    """Execute one callable on the global or supplied QThreadPool.

    The callable can optionally receive a ``task_context`` keyword argument.

    Examples
    --------
    Context-aware function::

        def build_something(*, task_context):
            task_context.report_progress(1, 10)
            task_context.raise_if_cancelled()
            return result

    Context-free function::

        def export_something(path):
            return export(path)

    ``inject_context`` controls whether TaskWorker supplies the context.
    """

    def __init__(
        self,
        *,
        task_id: str,
        function: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        cancellation_token: CancellationToken,
        inject_context: bool,
    ) -> None:
        super().__init__()

        if not callable(function):
            raise TypeError(
                "function must be callable."
            )

        self.task_id = task_id
        self.function = function
        self.args = args
        self.kwargs = dict(kwargs)

        self.cancellation_token = (
            cancellation_token
        )

        self.inject_context = bool(
            inject_context
        )

        self.signals = WorkerSignals()

        # QRunnable may be deleted automatically after execution.
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        """Execute the background operation."""

        try:
            self.cancellation_token.raise_if_cancelled()

            context = TaskContext(
                task_id=self.task_id,
                cancellation_token=(
                    self.cancellation_token
                ),
                progress=self._emit_progress,
            )

            kwargs = dict(
                self.kwargs
            )

            if self.inject_context:
                if "task_context" in kwargs:
                    raise ValueError(
                        "task_context is reserved by "
                        "TaskController."
                    )

                kwargs["task_context"] = context

            result = self.function(
                *self.args,
                **kwargs,
            )

            # A function that does not explicitly check the token may finish
            # after cancellation was requested. Treat that as cancelled
            # rather than publishing a result the user asked to abandon.
            if self.cancellation_token.cancelled:
                self.signals.cancelled.emit(
                    self.task_id,
                    "Task completed after cancellation "
                    "was requested; result was discarded.",
                )
                return

            self.signals.result.emit(
                self.task_id,
                result,
            )

        except TaskCancelledError as exc:
            self.signals.cancelled.emit(
                self.task_id,
                str(exc),
            )

        except Exception as exc:
            self.signals.error.emit(
                self.task_id,
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            )

        finally:
            self.signals.finished.emit(
                self.task_id
            )

    def _emit_progress(
        self,
        processed: int,
        total: int | None,
        unit_id: str | None,
        message: str | None,
    ) -> None:
        """Emit thread-safe progress."""

        self.signals.progress.emit(
            self.task_id,
            processed,
            total,
            unit_id,
            message,
        )


# ---------------------------------------------------------------------------
# Task controller
# ---------------------------------------------------------------------------


class TaskController(QObject):
    """Coordinate background work for the Rivelero GUI.

    Only one primary task is allowed at a time in this first implementation.

    This is deliberate. Operations such as rebuilding an SOF while changing
    the Environment or exporting another SOF concurrently would complicate
    scientific state consistency.

    Lightweight map rendering can remain outside this controller where
    appropriate.
    """

    task_started = Signal(
        str,     # task_id
        str,     # task_name
    )

    task_progress = Signal(
        str,     # task_id
        int,     # processed
        object,  # total
        object,  # unit_id
        object,  # message
    )

    task_result = Signal(
        str,     # task_id
        object,  # result
    )

    task_error = Signal(
        str,     # task_id
        str,     # message
        str,     # traceback
    )

    task_cancelled = Signal(
        str,     # task_id
        object,  # message
    )

    task_finished = Signal(
        str,     # task_id
    )

    def __init__(
        self,
        state: ApplicationState,
        *,
        thread_pool: QThreadPool | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(
            parent
        )

        if not isinstance(
            state,
            ApplicationState,
        ):
            raise TypeError(
                "state must be an ApplicationState."
            )

        self.state = state

        self.thread_pool = (
            QThreadPool.globalInstance()
            if thread_pool is None
            else thread_pool
        )

        if not isinstance(
            self.thread_pool,
            QThreadPool,
        ):
            raise TypeError(
                "thread_pool must be a QThreadPool or None."
            )

        self._active_worker: TaskWorker | None = None
        self._active_token: CancellationToken | None = None
        self._active_task_id: str | None = None

        # Track whether a terminal lifecycle signal has already been emitted.
        self._terminal_status_received = False

    # ------------------------------------------------------------------
    # Public state
    # ------------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """Whether the controller currently owns a running task."""

        return (
            self._active_task_id is not None
        )

    @property
    def active_task_id(self) -> str | None:
        """Identifier of the active task."""

        return self._active_task_id

    @property
    def cancellation_requested(self) -> bool:
        """Whether cancellation is pending."""

        return (
            self._active_token is not None
            and self._active_token.cancelled
        )

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(
        self,
        *,
        task_name: str,
        function: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        total: int | None = None,
        message: str | None = None,
        inject_context: bool = False,
        task_id: str | None = None,
    ) -> str:
        """Start one background task.

        Parameters
        ----------
        task_name
            Human-readable task name.

        function
            Callable executed on the QThreadPool.

        args, kwargs
            Arguments supplied to the callable.

        total
            Optional initial progress denominator.

        message
            Optional initial status message.

        inject_context
            If True, ``task_context=TaskContext(...)`` is added to the
            callable's keyword arguments.

        task_id
            Optional explicit identifier. Normally generated automatically.

        Returns
        -------
        str
            Unique task identifier.
        """

        if self.busy:
            raise RuntimeError(
                "Another Rivelero task is already running."
            )

        if self.state.task.busy:
            raise RuntimeError(
                "ApplicationState already reports a running task."
            )

        if not callable(
            function
        ):
            raise TypeError(
                "function must be callable."
            )

        if not isinstance(
            args,
            tuple,
        ):
            raise TypeError(
                "args must be a tuple."
            )

        if kwargs is None:
            kwargs = {}

        if not isinstance(
            kwargs,
            dict,
        ):
            raise TypeError(
                "kwargs must be a dictionary or None."
            )

        task_name = _required_string(
            "task_name",
            task_name,
        )

        if task_id is None:
            task_id = uuid4().hex
        else:
            task_id = _required_string(
                "task_id",
                task_id,
            )

        token = CancellationToken()

        worker = TaskWorker(
            task_id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            cancellation_token=token,
            inject_context=inject_context,
        )

        worker.signals.progress.connect(
            self._on_progress
        )

        worker.signals.result.connect(
            self._on_result
        )

        worker.signals.error.connect(
            self._on_error
        )

        worker.signals.cancelled.connect(
            self._on_cancelled
        )

        worker.signals.finished.connect(
            self._on_finished
        )

        self._active_worker = worker
        self._active_token = token
        self._active_task_id = task_id
        self._terminal_status_received = False

        self.state.start_task(
            task_id=task_id,
            task_name=task_name,
            total=total,
            message=message,
        )

        self.task_started.emit(
            task_id,
            task_name,
        )

        self.thread_pool.start(
            worker
        )

        return task_id

    # ------------------------------------------------------------------
    # Cancellation
    # ------------------------------------------------------------------

    def cancel(
        self,
    ) -> bool:
        """Request cancellation of the active task.

        Returns
        -------
        bool
            True when a cancellation request was issued, False when no task
            was running.
        """

        if not self.busy:
            return False

        if self._active_token is None:
            return False

        if self._active_token.cancelled:
            return True

        self._active_token.cancel()

        if self.state.task.status == TaskStatus.RUNNING:
            self.state.request_task_cancellation()

        return True

    # ------------------------------------------------------------------
    # SOF progress adapter
    # ------------------------------------------------------------------

    @staticmethod
    def sof_progress_adapter(
        task_context: TaskContext,
    ) -> Callable[
        [int, int, str],
        None,
    ]:
        """Adapt the SOF builder progress callback to TaskContext.

        ``build_survey_observability_field`` currently reports:

            processed
            total
            sampling_unit_id

        This adapter converts that into the GUI task progress protocol.

        Example
        -------
        A wrapper task can use::

            progress_callback = TaskController.sof_progress_adapter(
                task_context
            )

            result = build_survey_observability_field(
                ...,
                progress_callback=progress_callback,
            )
        """

        if not isinstance(
            task_context,
            TaskContext,
        ):
            raise TypeError(
                "task_context must be a TaskContext."
            )

        def callback(
            processed: int,
            total: int,
            sampling_unit_id: str,
        ) -> None:

            task_context.raise_if_cancelled()

            task_context.report_progress(
                processed=processed,
                total=total,
                unit_id=sampling_unit_id,
                # The builder reports each unit after it has been
                # processed, not while it is being computed.
                message=(
                    f"Processed {sampling_unit_id}"
                ),
            )

        return callback

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    @Slot(str, int, object, object, object)
    def _on_progress(
        self,
        task_id: str,
        processed: int,
        total: int | None,
        unit_id: str | None,
        message: str | None,
    ) -> None:
        """Receive progress on the GUI thread."""

        if not self._is_active(
            task_id
        ):
            return

        self.state.update_task_progress(
            processed=processed,
            total=total,
            unit_id=unit_id,
            message=message,
        )

        self.task_progress.emit(
            task_id,
            processed,
            total,
            unit_id,
            message,
        )

    @Slot(str, object)
    def _on_result(
        self,
        task_id: str,
        result: Any,
    ) -> None:
        """Receive a successful result."""

        if not self._is_active(
            task_id
        ):
            return

        if self._terminal_status_received:
            return

        self._terminal_status_received = True

        if self.state.task.status == TaskStatus.CANCELLING:
            self.state.mark_task_cancelled(
                message=(
                    "Task completed after cancellation "
                    "was requested; result was discarded."
                )
            )

            self.task_cancelled.emit(
                task_id,
                self.state.task.message,
            )

            return

        self.state.finish_task(
            message="Task completed."
        )

        self.task_result.emit(
            task_id,
            result,
        )

    @Slot(str, str, str)
    def _on_error(
        self,
        task_id: str,
        error_message: str,
        error_details: str,
    ) -> None:
        """Receive a worker failure."""

        if not self._is_active(
            task_id
        ):
            return

        if self._terminal_status_received:
            return

        self._terminal_status_received = True

        self.state.fail_task(
            error_message=error_message,
            error_details=error_details,
        )

        self.task_error.emit(
            task_id,
            error_message,
            error_details,
        )

    @Slot(str, object)
    def _on_cancelled(
        self,
        task_id: str,
        message: str | None,
    ) -> None:
        """Receive cooperative cancellation."""

        if not self._is_active(
            task_id
        ):
            return

        if self._terminal_status_received:
            return

        self._terminal_status_received = True

        self.state.mark_task_cancelled(
            message=(
                message
                or "Task cancelled."
            )
        )

        self.task_cancelled.emit(
            task_id,
            message,
        )

    @Slot(str)
    def _on_finished(
        self,
        task_id: str,
    ) -> None:
        """Finalize controller bookkeeping."""

        if not self._is_active(
            task_id
        ):
            return

        # Defensive fallback. Normally result/error/cancelled arrives first.
        if not self._terminal_status_received:

            self._terminal_status_received = True

            if (
                self._active_token is not None
                and self._active_token.cancelled
            ):
                if self.state.task.busy:
                    self.state.mark_task_cancelled(
                        message="Task cancelled."
                    )

                self.task_cancelled.emit(
                    task_id,
                    "Task cancelled.",
                )

            elif self.state.task.busy:
                self.state.fail_task(
                    error_message=(
                        "Task finished without returning a "
                        "result or error."
                    )
                )

                self.task_error.emit(
                    task_id,
                    (
                        "Task finished without returning "
                        "a result or error."
                    ),
                    "",
                )

        self._active_worker = None
        self._active_token = None
        self._active_task_id = None

        self.task_finished.emit(
            task_id
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_active(
        self,
        task_id: str,
    ) -> bool:
        """Whether a worker signal belongs to the active task."""

        return (
            self._active_task_id
            == task_id
        )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def make_progress_task(
    *,
    function: Callable[..., Any],
    kwargs: dict[str, Any],
) -> Callable[..., Any]:
    """Create a context-aware task for a ``progress_callback`` function.

    ``function`` must accept ``progress_callback(processed, total, unit_id)``,
    the protocol shared by the SOF builder and contribution analysis. The
    callback reports progress through TaskContext and raises
    TaskCancelledError at the next unit boundary once cancellation has been
    requested.

    The returned callable is suitable for::

        TaskController.start(..., function=task, inject_context=True)
    """

    if not callable(function):
        raise TypeError("function must be callable.")

    if not isinstance(kwargs, dict):
        raise TypeError("kwargs must be a dictionary.")

    kwargs_copy = dict(kwargs)

    if "progress_callback" in kwargs_copy:
        raise ValueError(
            "progress_callback is managed by make_progress_task."
        )

    def task(
        *,
        task_context: TaskContext,
    ) -> Any:

        task_context.raise_if_cancelled()

        progress_callback = (
            TaskController.sof_progress_adapter(
                task_context
            )
        )

        result = function(
            **kwargs_copy,
            progress_callback=progress_callback,
        )

        task_context.raise_if_cancelled()

        return result

    return task


def make_sof_build_task(
    *,
    build_function: Callable[..., Any],
    build_kwargs: dict[str, Any],
) -> Callable[..., Any]:
    """Create a context-aware SOF build callable.

    Thin wrapper around :func:`make_progress_task`, normally used with
    ``build_survey_observability_field``. This module deliberately does not
    import the builder, keeping TaskController reusable.
    """

    if not callable(build_function):
        raise TypeError("build_function must be callable.")

    if not isinstance(build_kwargs, dict):
        raise TypeError("build_kwargs must be a dictionary.")

    if "progress_callback" in build_kwargs:
        raise ValueError(
            "progress_callback is managed by make_sof_build_task."
        )

    return make_progress_task(
        function=build_function,
        kwargs=build_kwargs,
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _required_string(
    name: str,
    value: str,
) -> str:
    """Validate a required non-empty string."""

    if not isinstance(
        value,
        str,
    ):
        raise TypeError(
            f"{name} must be a string."
        )

    result = value.strip()

    if not result:
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return result