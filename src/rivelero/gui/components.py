"""Reusable visual components for the Rivelero GUI.

This module contains presentation-only Qt widgets shared across the Rivelero
workflow pages.

Components in this module:

- contain no scientific calculations;
- do not own ApplicationState;
- do not call Rivelero engines;
- do not perform file I/O;
- do not launch background tasks.

They provide a consistent visual vocabulary for pages such as Survey, World,
Observability, Analysis & Design, and Output.

Styling is provided by rivelero.gui.theme through Qt dynamic properties.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

try:
    from PySide6.QtCore import Qt, Signal
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QFrame,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QSizePolicy,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )

except ImportError:
    try:
        from PyQt6.QtCore import Qt, pyqtSignal as Signal
        from PyQt6.QtGui import QFont
        from PyQt6.QtWidgets import (
            QFrame,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSizePolicy,
            QToolButton,
            QVBoxLayout,
            QWidget,
        )

    except ImportError as exc:
        raise ImportError(
            "Rivelero GUI requires PySide6 or PyQt6."
        ) from exc


from rivelero.gui.theme import (
    ADVANCED_LABEL,
    COMING_SOON_LABEL,
    EXPERIMENTAL_LABEL,
    SPACING,
    refresh_style,
)


# ---------------------------------------------------------------------------
# Status badge
# ---------------------------------------------------------------------------


class BadgeType(str, Enum):
    """Semantic badge variants used throughout Rivelero."""

    COMING_SOON = "comingSoon"
    EXPERIMENTAL = "experimental"
    ADVANCED = "advanced"


class StatusBadge(QLabel):
    """Small semantic label such as 'Experimental' or 'Coming soon'."""

    DEFAULT_LABELS = {
        BadgeType.COMING_SOON: COMING_SOON_LABEL,
        BadgeType.EXPERIMENTAL: EXPERIMENTAL_LABEL,
        BadgeType.ADVANCED: ADVANCED_LABEL,
    }

    def __init__(
        self,
        badge_type: BadgeType | str,
        *,
        text: str | None = None,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        if isinstance(
            badge_type,
            str,
        ):
            badge_type = BadgeType(
                badge_type
            )

        if not isinstance(
            badge_type,
            BadgeType,
        ):
            raise TypeError(
                "badge_type must be a BadgeType or valid string."
            )

        self.badge_type = badge_type

        self.setText(
            (
                self.DEFAULT_LABELS[
                    badge_type
                ]
                if text is None
                else str(text)
            )
        )

        self.setProperty(
            "badge",
            badge_type.value,
        )

        self.setSizePolicy(
            QSizePolicy.Policy.Maximum,
            QSizePolicy.Policy.Fixed,
        )

        self.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------


class PageHeader(QWidget):
    """Standard heading used at the top of a workflow page.

    The header can contain:

    - page title;
    - explanatory subtitle;
    - optional semantic badge;
    - optional right-side action widgets.
    """

    def __init__(
        self,
        title: str,
        description: str | None = None,
        *,
        badge: BadgeType | str | None = None,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        self._actions_layout: QHBoxLayout | None = None

        outer = QVBoxLayout(
            self
        )

        outer.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        outer.setSpacing(
            SPACING.sm
        )

        top_row = QHBoxLayout()

        top_row.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        top_row.setSpacing(
            SPACING.md
        )

        self.title_label = QLabel(
            title
        )

        self.title_label.setProperty(
            "pageTitle",
            True,
        )

        top_row.addWidget(
            self.title_label
        )

        if badge is not None:
            self.badge = StatusBadge(
                badge
            )

            top_row.addWidget(
                self.badge
            )

        else:
            self.badge = None

        top_row.addStretch(
            1
        )

        self._actions_layout = (
            QHBoxLayout()
        )

        self._actions_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        self._actions_layout.setSpacing(
            SPACING.sm
        )

        top_row.addLayout(
            self._actions_layout
        )

        outer.addLayout(
            top_row
        )

        self.description_label = QLabel()

        self.description_label.setWordWrap(
            True
        )

        self.description_label.setProperty(
            "pageDescription",
            True,
        )

        if description:
            self.description_label.setText(
                description
            )
            self.description_label.show()

        else:
            self.description_label.hide()

        outer.addWidget(
            self.description_label
        )

    def set_title(
        self,
        title: str,
    ) -> None:
        """Update page title."""

        self.title_label.setText(
            str(title)
        )

    def set_description(
        self,
        description: str | None,
    ) -> None:
        """Update or hide page description."""

        if description:
            self.description_label.setText(
                description
            )
            self.description_label.show()

        else:
            self.description_label.clear()
            self.description_label.hide()

    def add_action(
        self,
        widget: QWidget,
    ) -> None:
        """Add a widget to the right side of the page header."""

        if not isinstance(
            widget,
            QWidget,
        ):
            raise TypeError(
                "widget must be a QWidget."
            )

        self._actions_layout.addWidget(
            widget
        )


# ---------------------------------------------------------------------------
# Section header
# ---------------------------------------------------------------------------


class SectionHeader(QWidget):
    """Title and optional explanation for one page section."""

    def __init__(
        self,
        title: str,
        description: str | None = None,
        *,
        badge: BadgeType | str | None = None,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(
            SPACING.xs
        )

        title_row = QHBoxLayout()

        title_row.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        title_row.setSpacing(
            SPACING.sm
        )

        self.title_label = QLabel(
            title
        )

        self.title_label.setProperty(
            "sectionTitle",
            True,
        )

        title_row.addWidget(
            self.title_label
        )

        if badge is not None:
            self.badge = StatusBadge(
                badge
            )

            title_row.addWidget(
                self.badge
            )

        else:
            self.badge = None

        title_row.addStretch(
            1
        )

        layout.addLayout(
            title_row
        )

        self.description_label = QLabel()

        self.description_label.setWordWrap(
            True
        )

        self.description_label.setProperty(
            "secondaryText",
            True,
        )

        if description:
            self.description_label.setText(
                description
            )
            self.description_label.show()

        else:
            self.description_label.hide()

        layout.addWidget(
            self.description_label
        )


# ---------------------------------------------------------------------------
# Content card
# ---------------------------------------------------------------------------


class ContentCard(QFrame):
    """Standard Rivelero content container."""

    def __init__(
        self,
        *,
        title: str | None = None,
        description: str | None = None,
        badge: BadgeType | str | None = None,
        subtle: bool = False,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        self.setProperty(
            (
                "subtleCard"
                if subtle
                else "contentCard"
            ),
            True,
        )

        self._layout = QVBoxLayout(
            self
        )

        self._layout.setContentsMargins(
            SPACING.xl,
            SPACING.lg,
            SPACING.xl,
            SPACING.lg,
        )

        self._layout.setSpacing(
            SPACING.md
        )

        if title is not None:

            self.header = SectionHeader(
                title,
                description,
                badge=badge,
            )

            self._layout.addWidget(
                self.header
            )

        else:
            self.header = None

    @property
    def content_layout(
        self,
    ) -> QVBoxLayout:
        """Layout into which page-specific widgets should be inserted."""

        return self._layout

    def add_widget(
        self,
        widget: QWidget,
        *,
        stretch: int = 0,
        alignment=None,
    ) -> None:
        """Add a widget to the card."""

        if alignment is None:
            self._layout.addWidget(
                widget,
                stretch,
            )

        else:
            self._layout.addWidget(
                widget,
                stretch,
                alignment,
            )

    def add_layout(
        self,
        layout,
        *,
        stretch: int = 0,
    ) -> None:
        """Add a layout to the card."""

        self._layout.addLayout(
            layout,
            stretch,
        )

    def add_stretch(
        self,
        stretch: int = 1,
    ) -> None:
        """Add flexible vertical space."""

        self._layout.addStretch(
            stretch
        )


# ---------------------------------------------------------------------------
# Labeled value
# ---------------------------------------------------------------------------


class LabeledValue(QWidget):
    """Compact label/value pair for metadata and status summaries."""

    def __init__(
        self,
        label: str,
        value: str = "—",
        *,
        vertical: bool = False,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        if vertical:
            layout = QVBoxLayout(
                self
            )

        else:
            layout = QHBoxLayout(
                self
            )

        layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        layout.setSpacing(
            SPACING.sm
        )

        self.label_widget = QLabel(
            label
        )

        self.label_widget.setProperty(
            "secondaryText",
            True,
        )

        self.value_widget = QLabel(
            value
        )

        value_font = QFont()
        value_font.setBold(
            True
        )

        self.value_widget.setFont(
            value_font
        )

        if not vertical:
            layout.addWidget(
                self.label_widget
            )

            layout.addStretch(
                1
            )

            layout.addWidget(
                self.value_widget
            )

        else:
            layout.addWidget(
                self.label_widget
            )

            layout.addWidget(
                self.value_widget
            )

    @property
    def value(
        self,
    ) -> str:
        """Current displayed value."""

        return self.value_widget.text()

    def set_value(
        self,
        value,
    ) -> None:
        """Set displayed value."""

        self.value_widget.setText(
            "—"
            if value is None
            else str(value)
        )

    def set_label(
        self,
        label: str,
    ) -> None:
        """Set descriptive label."""

        self.label_widget.setText(
            str(label)
        )


# ---------------------------------------------------------------------------
# Empty state
# ---------------------------------------------------------------------------


class EmptyState(QFrame):
    """Friendly placeholder shown when required data are not yet available."""

    action_requested = Signal()

    def __init__(
        self,
        title: str,
        description: str,
        *,
        action_text: str | None = None,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        self.setProperty(
            "subtleCard",
            True,
        )

        layout = QVBoxLayout(
            self
        )

        layout.setContentsMargins(
            SPACING.xxl,
            SPACING.xxl,
            SPACING.xxl,
            SPACING.xxl,
        )

        layout.setSpacing(
            SPACING.md
        )

        layout.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.title_label = QLabel(
            title
        )

        title_font = QFont()
        title_font.setPointSize(
            13
        )
        title_font.setBold(
            True
        )

        self.title_label.setFont(
            title_font
        )

        self.title_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        layout.addWidget(
            self.title_label
        )

        self.description_label = QLabel(
            description
        )

        self.description_label.setWordWrap(
            True
        )

        self.description_label.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )

        self.description_label.setProperty(
            "secondaryText",
            True,
        )

        self.description_label.setMaximumWidth(
            520
        )

        layout.addWidget(
            self.description_label
        )

        if action_text is not None:

            self.action_button = QPushButton(
                action_text
            )

            self.action_button.setProperty(
                "primary",
                True,
            )

            self.action_button.setSizePolicy(
                QSizePolicy.Policy.Maximum,
                QSizePolicy.Policy.Fixed,
            )

            self.action_button.clicked.connect(
                self.action_requested.emit
            )

            layout.addWidget(
                self.action_button,
                0,
                Qt.AlignmentFlag.AlignCenter,
            )

        else:
            self.action_button = None


# ---------------------------------------------------------------------------
# Collapsible section
# ---------------------------------------------------------------------------


class CollapsibleSection(QWidget):
    """Expandable section used for Advanced and optional controls."""

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        *,
        description: str | None = None,
        expanded: bool = False,
        badge: BadgeType | str | None = None,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        self._expanded = bool(
            expanded
        )

        outer = QVBoxLayout(
            self
        )

        outer.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        outer.setSpacing(
            SPACING.sm
        )

        header = QFrame()

        header.setProperty(
            "subtleCard",
            True,
        )

        header_layout = QHBoxLayout(
            header
        )

        header_layout.setContentsMargins(
            SPACING.md,
            SPACING.sm,
            SPACING.md,
            SPACING.sm,
        )

        header_layout.setSpacing(
            SPACING.sm
        )

        self.toggle_button = QToolButton()

        self.toggle_button.setCheckable(
            True
        )

        self.toggle_button.setChecked(
            self._expanded
        )

        self.toggle_button.setArrowType(
            (
                Qt.ArrowType.DownArrow
                if self._expanded
                else Qt.ArrowType.RightArrow
            )
        )

        self.toggle_button.setAutoRaise(
            True
        )

        self.toggle_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )

        header_layout.addWidget(
            self.toggle_button
        )

        text_layout = QVBoxLayout()

        text_layout.setContentsMargins(
            0,
            0,
            0,
            0,
        )

        text_layout.setSpacing(
            SPACING.xxs
        )

        self.title_label = QLabel(
            title
        )

        title_font = QFont()
        title_font.setBold(
            True
        )

        self.title_label.setFont(
            title_font
        )

        text_layout.addWidget(
            self.title_label
        )

        if description:

            self.description_label = QLabel(
                description
            )

            self.description_label.setWordWrap(
                True
            )

            self.description_label.setProperty(
                "secondaryText",
                True,
            )

            text_layout.addWidget(
                self.description_label
            )

        else:
            self.description_label = None

        header_layout.addLayout(
            text_layout,
            1,
        )

        if badge is not None:

            self.badge = StatusBadge(
                badge
            )

            header_layout.addWidget(
                self.badge
            )

        else:
            self.badge = None

        outer.addWidget(
            header
        )

        self.content_widget = QWidget()

        self._content_layout = QVBoxLayout(
            self.content_widget
        )

        self._content_layout.setContentsMargins(
            SPACING.lg,
            SPACING.sm,
            SPACING.lg,
            SPACING.lg,
        )

        self._content_layout.setSpacing(
            SPACING.md
        )

        self.content_widget.setVisible(
            self._expanded
        )

        outer.addWidget(
            self.content_widget
        )

        self.toggle_button.toggled.connect(
            self.set_expanded
        )

    @property
    def content_layout(
        self,
    ) -> QVBoxLayout:
        """Layout for expandable content."""

        return self._content_layout

    @property
    def expanded(
        self,
    ) -> bool:
        """Whether the section is currently expanded."""

        return self._expanded

    def set_expanded(
        self,
        expanded: bool,
    ) -> None:
        """Expand or collapse the section."""

        expanded = bool(
            expanded
        )

        changed = (
            expanded
            != self._expanded
        )

        self._expanded = expanded

        if (
            self.toggle_button.isChecked()
            != expanded
        ):
            self.toggle_button.setChecked(
                expanded
            )

        self.toggle_button.setArrowType(
            (
                Qt.ArrowType.DownArrow
                if expanded
                else Qt.ArrowType.RightArrow
            )
        )

        self.content_widget.setVisible(
            expanded
        )

        if changed:
            self.toggled.emit(
                expanded
            )


# ---------------------------------------------------------------------------
# Action bar
# ---------------------------------------------------------------------------


class ActionBar(QFrame):
    """Horizontal action area used at the bottom of workflow sections.

    Secondary actions appear on the left. The primary workflow action is
    positioned on the right.
    """

    def __init__(
        self,
        *,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            parent
        )

        self.setProperty(
            "subtleCard",
            True,
        )

        layout = QHBoxLayout(
            self
        )

        layout.setContentsMargins(
            SPACING.md,
            SPACING.sm,
            SPACING.md,
            SPACING.sm,
        )

        layout.setSpacing(
            SPACING.sm
        )

        self._secondary_layout = (
            QHBoxLayout()
        )

        self._secondary_layout.setSpacing(
            SPACING.sm
        )

        self._primary_layout = (
            QHBoxLayout()
        )

        self._primary_layout.setSpacing(
            SPACING.sm
        )

        layout.addLayout(
            self._secondary_layout
        )

        layout.addStretch(
            1
        )

        layout.addLayout(
            self._primary_layout
        )

    def add_secondary_action(
        self,
        widget: QWidget,
    ) -> None:
        """Add a secondary action to the left side."""

        if not isinstance(
            widget,
            QWidget,
        ):
            raise TypeError(
                "widget must be a QWidget."
            )

        self._secondary_layout.addWidget(
            widget
        )

    def add_primary_action(
        self,
        widget: QWidget,
    ) -> None:
        """Add an action to the right side.

        QPushButtons are automatically given the Rivelero primary style.
        """

        if not isinstance(
            widget,
            QWidget,
        ):
            raise TypeError(
                "widget must be a QWidget."
            )

        if isinstance(
            widget,
            QPushButton,
        ):
            widget.setProperty(
                "primary",
                True,
            )

            refresh_style(
                widget
            )

        self._primary_layout.addWidget(
            widget
        )


# ---------------------------------------------------------------------------
# Feature card
# ---------------------------------------------------------------------------


class FeatureCard(ContentCard):
    """Card representing an implemented or future Rivelero capability.

    This is particularly useful on Analysis & Design, where the interface
    should communicate the future Rivelero roadmap without pretending that
    unfinished functionality is currently available.
    """

    action_requested = Signal()

    def __init__(
        self,
        title: str,
        description: str,
        *,
        action_text: str | None = None,
        badge: BadgeType | str | None = None,
        enabled: bool = True,
        parent: QWidget | None = None,
    ) -> None:

        super().__init__(
            title=title,
            description=description,
            badge=badge,
            parent=parent,
        )

        self._feature_enabled = bool(
            enabled
        )

        if action_text is not None:

            self.action_button = QPushButton(
                action_text
            )

            self.action_button.setEnabled(
                self._feature_enabled
            )

            self.action_button.setSizePolicy(
                QSizePolicy.Policy.Maximum,
                QSizePolicy.Policy.Fixed,
            )

            self.action_button.clicked.connect(
                self.action_requested.emit
            )

            self.content_layout.addWidget(
                self.action_button,
                0,
                Qt.AlignmentFlag.AlignLeft,
            )

        else:
            self.action_button = None

    @property
    def feature_enabled(
        self,
    ) -> bool:
        """Whether this feature is currently available."""

        return self._feature_enabled

    def set_feature_enabled(
        self,
        enabled: bool,
    ) -> None:
        """Enable or disable the feature action."""

        self._feature_enabled = bool(
            enabled
        )

        if self.action_button is not None:
            self.action_button.setEnabled(
                self._feature_enabled
            )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def make_primary_button(
    text: str,
    *,
    parent: QWidget | None = None,
) -> QPushButton:
    """Create a consistently styled primary-action button."""

    button = QPushButton(
        text,
        parent,
    )

    button.setProperty(
        "primary",
        True,
    )

    return button


def make_secondary_button(
    text: str,
    *,
    parent: QWidget | None = None,
) -> QPushButton:
    """Create a standard secondary-action button."""

    return QPushButton(
        text,
        parent,
    )


def make_danger_button(
    text: str,
    *,
    parent: QWidget | None = None,
) -> QPushButton:
    """Create a destructive-action button."""

    button = QPushButton(
        text,
        parent,
    )

    button.setProperty(
        "danger",
        True,
    )

    return button


def make_link_button(
    text: str,
    *,
    parent: QWidget | None = None,
) -> QPushButton:
    """Create a visually lightweight link-style action."""

    button = QPushButton(
        text,
        parent,
    )

    button.setProperty(
        "link",
        True,
    )

    button.setCursor(
        Qt.CursorShape.PointingHandCursor
    )

    return button


def add_widgets(
    layout,
    widgets: Iterable[QWidget],
) -> None:
    """Add several widgets to a Qt layout."""

    for widget in widgets:

        if not isinstance(
            widget,
            QWidget,
        ):
            raise TypeError(
                "All items must be QWidget instances."
            )

        layout.addWidget(
            widget
        )