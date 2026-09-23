"""The single Matplotlib canvas used by every Rivelero Qt widget.

SafeFigureCanvas adds two lifecycle guarantees to FigureCanvasQTAgg:

* a deferred ``draw_idle`` that reaches an already deleted widget (a
  closed dialog, a replaced page) is skipped instead of raising;
* a canvas that is not visible does not render; it draws once when shown.
"""

from __future__ import annotations

# Import Rivelero's Qt binding *before* Matplotlib's Qt backend: Matplotlib
# binds to an already imported binding, otherwise it may pick another
# installed one (e.g. PyQt6) and mix two incompatible Qt bindings.
try:
    import PySide6.QtCore  # noqa: F401
except ImportError:
    import PyQt6.QtCore  # noqa: F401

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402

try:
    from shiboken6 import isValid as _qt_object_alive
except ImportError:  # PyQt6
    try:
        from PyQt6 import sip as _sip

        def _qt_object_alive(obj) -> bool:
            return not _sip.isdeleted(obj)
    except ImportError:
        def _qt_object_alive(obj) -> bool:
            return True


class SafeFigureCanvas(FigureCanvasQTAgg):
    """FigureCanvasQTAgg whose deferred redraw tolerates widget deletion.

    Matplotlib queues ``draw_idle`` with a zero-delay timer. If the widget is
    destroyed first (a closed dialog or discarded page), the queued callback
    would touch a deleted Qt object and raise; the redraw is simply skipped.

    Hidden canvases (pages or tabs not on screen) also skip the redraw and
    render once when they are next shown, so updating state never renders
    every map of the application.
    """

    _pending_draw = False

    def _draw_idle(self):
        if not _qt_object_alive(self):
            return
        if not self.isVisible():
            # Clear Matplotlib's own flag so later draw_idle calls still
            # queue; remember to draw when shown.
            self._draw_pending = False
            self._pending_draw = True
            return
        self._pending_draw = False
        super()._draw_idle()

    def showEvent(self, event):  # noqa: N802 - Qt API
        super().showEvent(event)
        if self._pending_draw:
            self._pending_draw = False
            self.draw_idle()
