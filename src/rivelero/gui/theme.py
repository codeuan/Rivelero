"""Visual theme and design tokens for the Rivelero GUI.

This module defines the shared visual language of Rivelero.

It contains:

- semantic colors;
- typography constants;
- spacing and sizing tokens;
- reusable Qt stylesheet generation;
- application-level theme application.

It deliberately contains no scientific logic and no application state.

GUI widgets and pages should consume these design tokens rather than
hard-coding colors, margins, font sizes, or status styles locally.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication
except ImportError:
    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QColor, QPalette
        from PyQt6.QtWidgets import QApplication
    except ImportError as exc:
        raise ImportError(
            "Rivelero GUI requires PySide6 or PyQt6."
        ) from exc


# ---------------------------------------------------------------------------
# Color system
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ColorPalette:
    """Semantic color palette for Rivelero."""

    # --------------------------------------------------------------
    # Application surfaces
    # --------------------------------------------------------------

    app_background: str = "#F6F7F8"
    surface: str = "#FFFFFF"
    surface_subtle: str = "#F9FAFA"
    surface_hover: str = "#F1F3F4"

    # --------------------------------------------------------------
    # Sidebar / navigation
    # --------------------------------------------------------------

    sidebar: str = "#20282D"
    sidebar_border: str = "#161C20"

    sidebar_hover: str = "#2D383E"
    sidebar_active: str = "#3A474E"

    sidebar_text: str = "#DCE3E6"
    sidebar_text_strong: str = "#FFFFFF"
    sidebar_muted: str = "#9CA8AE"

    # --------------------------------------------------------------
    # Text
    # --------------------------------------------------------------

    text_primary: str = "#202428"
    text_secondary: str = "#687177"
    text_muted: str = "#7B8489"
    text_disabled: str = "#A7AFB3"

    # --------------------------------------------------------------
    # Borders
    # --------------------------------------------------------------

    border: str = "#DFE4E7"
    border_strong: str = "#C8D0D4"
    divider: str = "#E8EBED"

    # --------------------------------------------------------------
    # Primary interaction
    # --------------------------------------------------------------

    primary: str = "#3F6F78"
    primary_hover: str = "#345E66"
    primary_pressed: str = "#294D54"
    primary_soft: str = "#E8F0F1"

    # --------------------------------------------------------------
    # Semantic state
    # --------------------------------------------------------------

    success: str = "#4E9B67"
    success_soft: str = "#E8F4EC"

    warning: str = "#C58A2A"
    warning_soft: str = "#FBF2E3"

    error: str = "#B85450"
    error_soft: str = "#F8E9E8"

    info: str = "#557FA3"
    info_soft: str = "#EAF0F5"

    experimental: str = "#7A65A8"
    experimental_soft: str = "#F0ECF7"

    # --------------------------------------------------------------
    # Scientific observability states
    # --------------------------------------------------------------
    #
    # These mirror OBSERVABILITY_STATE_COLORS in
    # rivelero.visualization.observability, which is the canonical,
    # colour-vision-deficiency-checked palette used by the maps.

    state_outside_domain: str = "#F2F2F2"
    state_invalid: str = "#6E6E6E"
    state_blind_spot: str = "#E07B39"
    state_observable: str = "#2A78D6"


COLORS = ColorPalette()


# ---------------------------------------------------------------------------
# Spacing system
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SpacingScale:
    """Shared spacing tokens in pixels."""

    xxs: int = 4
    xs: int = 6
    sm: int = 8
    md: int = 12
    lg: int = 16
    xl: int = 20
    xxl: int = 24
    xxxl: int = 32
    page: int = 38


SPACING = SpacingScale()


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SizeScale:
    """Shared GUI dimensions."""

    sidebar_width: int = 245
    header_height: int = 72

    minimum_window_width: int = 1180
    minimum_window_height: int = 760

    navigation_height: int = 46

    button_height: int = 36
    input_height: int = 34

    card_radius: int = 8
    control_radius: int = 6

    status_dot_width: int = 14

    logo_width: int = 190
    logo_height: int = 48


SIZES = SizeScale()


# ---------------------------------------------------------------------------
# Typography
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TypographyScale:
    """Typography tokens used by the GUI."""

    family: str = '"Segoe UI", "Arial", sans-serif'

    body_pt: int = 10
    body_large_pt: int = 11

    caption_pt: int = 8

    section_pt: int = 12

    page_title_pt: int = 22

    application_title_pt: int = 20


TYPOGRAPHY = TypographyScale()


# ---------------------------------------------------------------------------
# Semantic labels
# ---------------------------------------------------------------------------


COMING_SOON_LABEL = "Coming soon"
EXPERIMENTAL_LABEL = "Experimental"
ADVANCED_LABEL = "Advanced"


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------


def application_stylesheet() -> str:
    """Return the global Rivelero Qt stylesheet.

    Object names and dynamic properties are intentionally used for semantic
    styling so individual widgets do not need local stylesheets.
    """

    c = COLORS
    s = SIZES
    t = TYPOGRAPHY

    return f"""
    /* ================================================================
       APPLICATION
       ================================================================ */

    QMainWindow {{
        background: {c.app_background};
    }}

    QWidget {{
        font-family: {t.family};
        font-size: {t.body_pt}pt;
        color: {c.text_primary};
    }}

    QToolTip {{
        background: {c.sidebar};
        color: {c.sidebar_text_strong};
        border: 1px solid {c.sidebar_border};
        padding: 5px 7px;
    }}


    /* ================================================================
       HEADER
       ================================================================ */

    #AppHeader {{
        background: {c.surface};
        border-bottom: 1px solid {c.border};
    }}

    QLabel[headerSubtitle="true"] {{
        color: {c.text_secondary};
        font-size: {t.body_pt}pt;
    }}

    QLabel[projectName="true"] {{
        color: {c.text_primary};
        font-weight: 600;
    }}

    QLabel[dirtyIndicator="true"] {{
        color: {c.warning};
        font-size: 12pt;
    }}


    /* ================================================================
       SIDEBAR
       ================================================================ */

    #Sidebar {{
        background: {c.sidebar};
        border-right: 1px solid {c.sidebar_border};
    }}

    #Sidebar QLabel {{
        color: {c.sidebar_text};
    }}

    QLabel[sectionHeading="true"] {{
        color: {c.sidebar_muted};
        font-size: {t.caption_pt}pt;
        font-weight: 700;
    }}

    QLabel[sidebarFooter="true"] {{
        color: {c.sidebar_muted};
        font-size: {t.caption_pt}pt;
    }}

    QPushButton[navigation="true"] {{
        min-height: {s.navigation_height}px;

        background: transparent;

        color: {c.sidebar_text};

        border: none;
        border-radius: {s.control_radius}px;

        text-align: left;

        padding-left: 12px;
        padding-right: 12px;

        font-weight: 500;
    }}

    QPushButton[navigation="true"]:hover {{
        background: {c.sidebar_hover};
    }}

    QPushButton[navigation="true"]:checked {{
        background: {c.sidebar_active};
        color: {c.sidebar_text_strong};
        font-weight: 700;
    }}

    QPushButton[navigation="true"]:disabled {{
        color: {c.sidebar_muted};
    }}


    /* ================================================================
       READINESS / STATUS
       ================================================================ */

    QLabel[ready="false"] {{
        color: {c.text_disabled};
    }}

    #Sidebar QLabel[ready="false"] {{
        color: #738087;
    }}

    QLabel[ready="true"] {{
        color: {c.success};
    }}

    QLabel[statusWarning="true"] {{
        color: {c.warning};
    }}

    QLabel[statusSuccess="true"] {{
        color: #4E9B67;
        font-weight: 600;
    }}

    QLabel[statusError="true"] {{
        color: {c.error};
    }}

    QLabel[stepActive="true"] {{
        color: #3F6F78;
        font-weight: 700;
    }}

    QLabel[stepActive="false"] {{
        color: #7B8489;
    }}


    /* ================================================================
       PAGE TYPOGRAPHY
       ================================================================ */

    QLabel[pageTitle="true"] {{
        color: {c.text_primary};
        font-size: {t.page_title_pt}pt;
        font-weight: 700;
    }}

    QLabel[pageDescription="true"] {{
        color: {c.text_secondary};
        font-size: {t.body_large_pt}pt;
    }}

    QLabel[secondaryText="true"] {{
        color: {c.text_muted};
    }}

    QLabel[caption="true"] {{
        color: {c.text_muted};
        font-size: {t.caption_pt}pt;
    }}

    QLabel[sectionTitle="true"] {{
        color: {c.text_primary};
        font-size: {t.section_pt}pt;
        font-weight: 700;
    }}


    /* ================================================================
       CARDS
       ================================================================ */

    QFrame[contentCard="true"] {{
        background: {c.surface};
        border: 1px solid {c.border};
        border-radius: {s.card_radius}px;
    }}

    QFrame[subtleCard="true"] {{
        background: {c.surface_subtle};
        border: 1px solid {c.divider};
        border-radius: {s.card_radius}px;
    }}

    QFrame[successCard="true"] {{
        background: {c.success_soft};
        border: 1px solid {c.success};
        border-radius: {s.card_radius}px;
    }}

    QFrame[warningCard="true"] {{
        background: {c.warning_soft};
        border: 1px solid {c.warning};
        border-radius: {s.card_radius}px;
    }}

    QFrame[errorCard="true"] {{
        background: {c.error_soft};
        border: 1px solid {c.error};
        border-radius: {s.card_radius}px;
    }}

    QFrame[experimentalCard="true"] {{
        background: {c.experimental_soft};
        border: 1px solid {c.experimental};
        border-radius: {s.card_radius}px;
    }}


    /* ================================================================
       BUTTONS
       ================================================================ */

    QPushButton {{
        min-height: {s.button_height}px;

        background: {c.surface};

        border: 1px solid {c.border_strong};
        border-radius: {s.control_radius}px;

        padding: 0 14px;

        color: {c.text_primary};
        font-weight: 500;
    }}

    QPushButton:hover {{
        background: {c.surface_hover};
        border-color: {c.primary};
    }}

    QPushButton:pressed {{
        background: {c.primary_soft};
    }}

    QPushButton:disabled {{
        background: {c.surface_subtle};
        border-color: {c.border};
        color: {c.text_disabled};
    }}


    QPushButton[primary="true"] {{
        background: {c.primary};
        border: 1px solid {c.primary};
        color: white;
        font-weight: 700;
    }}

    QPushButton[primary="true"]:hover {{
        background: {c.primary_hover};
        border-color: {c.primary_hover};
    }}

    QPushButton[primary="true"]:pressed {{
        background: {c.primary_pressed};
        border-color: {c.primary_pressed};
    }}


    QPushButton[danger="true"] {{
        color: {c.error};
        border-color: {c.error};
    }}

    QPushButton[danger="true"]:hover {{
        background: {c.error_soft};
    }}


    QPushButton[link="true"] {{
        min-height: 0px;
        background: transparent;
        border: none;
        padding: 2px;
        color: {c.primary};
        text-align: left;
    }}

    QPushButton[link="true"]:hover {{
        color: {c.primary_hover};
        text-decoration: underline;
    }}


    /* ================================================================
       INPUTS
       ================================================================ */

    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QDateEdit,
    QDateTimeEdit {{
        min-height: {s.input_height}px;

        background: {c.surface};

        border: 1px solid {c.border_strong};
        border-radius: {s.control_radius}px;

        padding-left: 8px;
        padding-right: 8px;

        selection-background-color: {c.primary};
    }}

    QLineEdit:focus,
    QSpinBox:focus,
    QDoubleSpinBox:focus,
    QComboBox:focus,
    QDateEdit:focus,
    QDateTimeEdit:focus {{
        border: 1px solid {c.primary};
    }}

    QLineEdit:disabled,
    QSpinBox:disabled,
    QDoubleSpinBox:disabled,
    QComboBox:disabled {{
        background: {c.surface_subtle};
        color: {c.text_disabled};
    }}

    QTextEdit,
    QPlainTextEdit {{
        background: {c.surface};

        border: 1px solid {c.border_strong};
        border-radius: {s.control_radius}px;

        padding: 8px;

        selection-background-color: {c.primary};
    }}


    /* ================================================================
       TABLES
       ================================================================ */

    QTableView,
    QTableWidget {{
        background: {c.surface};

        alternate-background-color: {c.surface_subtle};

        border: 1px solid {c.border};

        gridline-color: {c.divider};

        selection-background-color: {c.primary_soft};
        selection-color: {c.text_primary};
        outline: none;
    }}

    /* Explicit item rules replace the platform's current-cell accent bar
       (the Windows accent colour) with the Rivelero selection colour. */
    QTableView::item:selected,
    QTableWidget::item:selected {{
        background: {c.primary_soft};
        color: {c.text_primary};
    }}

    QTableView::item:focus,
    QTableWidget::item:focus {{
        border: none;
        outline: none;
    }}

    QHeaderView::section {{
        background: {c.surface_subtle};

        border: none;
        border-bottom: 1px solid {c.border};
        border-right: 1px solid {c.divider};

        padding: 7px;

        color: {c.text_secondary};
        font-weight: 600;
    }}


    /* ================================================================
       CHECKBOX / RADIO
       ================================================================ */

    QCheckBox,
    QRadioButton {{
        spacing: 7px;
    }}

    QCheckBox:disabled,
    QRadioButton:disabled {{
        color: {c.text_disabled};
    }}


    /* ================================================================
       TABS
       ================================================================ */

    QTabWidget::pane {{
        border: 1px solid {c.border};
        background: {c.surface};
    }}

    QTabBar::tab {{
        background: {c.surface_subtle};

        border: 1px solid {c.border};
        border-bottom: none;

        padding: 8px 14px;

        color: {c.text_secondary};
    }}

    QTabBar::tab:selected {{
        background: {c.surface};
        color: {c.text_primary};
        font-weight: 600;
    }}


    /* ================================================================
       PROGRESS
       ================================================================ */

    QProgressBar {{
        min-height: 18px;

        background: {c.surface_subtle};

        border: 1px solid {c.border};
        border-radius: 5px;

        text-align: center;

        color: {c.text_secondary};
    }}

    QProgressBar::chunk {{
        background: {c.primary};
        border-radius: 4px;
    }}


    /* ================================================================
       SCROLL AREAS
       ================================================================ */

    QScrollArea {{
        border: none;
        background: transparent;
    }}

    QScrollArea > QWidget > QWidget {{
        background: transparent;
    }}


    /* ================================================================
       MENUS
       ================================================================ */

    QMenuBar {{
        background: {c.surface};
        border-bottom: 1px solid {c.border};
    }}

    QMenuBar::item {{
        padding: 6px 10px;
        background: transparent;
    }}

    QMenuBar::item:selected {{
        background: {c.surface_hover};
    }}

    QMenu {{
        background: {c.surface};
        border: 1px solid {c.border};
    }}

    QMenu::item {{
        padding: 7px 28px 7px 10px;
    }}

    QMenu::item:selected {{
        background: {c.primary_soft};
    }}

    /* Drop-down lists of combo boxes are separate popup windows; without
       explicit rules they inherit the operating-system palette, which is
       dark under Windows dark mode. */
    QComboBox QAbstractItemView {{
        background: {c.surface};
        color: {c.text_primary};
        border: 1px solid {c.border_strong};
        outline: none;
        selection-background-color: {c.primary_soft};
        selection-color: {c.text_primary};
    }}

    QComboBox QAbstractItemView::item {{
        min-height: 26px;
        padding: 2px 8px;
    }}

    QComboBox QAbstractItemView::item:hover {{
        background: {c.surface_hover};
    }}


    /* ================================================================
       STATUS BAR
       ================================================================ */

    QStatusBar {{
        background: {c.surface};

        border-top: 1px solid {c.border};

        color: {c.text_secondary};
    }}

    QLabel[statusBrand="true"] {{
        color: {c.text_secondary};
        font-weight: 700;
        padding-left: 12px;
    }}


    /* ================================================================
       BADGES
       ================================================================ */

    QLabel[badge="comingSoon"] {{
        background: {c.surface_subtle};
        color: {c.text_muted};

        border: 1px solid {c.border_strong};
        border-radius: 8px;

        padding: 2px 7px;

        font-size: {t.caption_pt}pt;
        font-weight: 600;
    }}

    QLabel[badge="experimental"] {{
        background: {c.experimental_soft};
        color: {c.experimental};

        border: 1px solid {c.experimental};
        border-radius: 8px;

        padding: 2px 7px;

        font-size: {t.caption_pt}pt;
        font-weight: 700;
    }}

    QLabel[badge="advanced"] {{
        background: {c.info_soft};
        color: {c.info};

        border: 1px solid {c.info};
        border-radius: 8px;

        padding: 2px 7px;

        font-size: {t.caption_pt}pt;
        font-weight: 600;
    }}
    """


# ---------------------------------------------------------------------------
# Theme application
# ---------------------------------------------------------------------------


def apply_theme(
    application: QApplication,
) -> None:
    """Apply the Rivelero theme to a QApplication."""

    if not isinstance(
        application,
        QApplication,
    ):
        raise TypeError(
            "application must be a QApplication."
        )

    # Rivelero's design is light-only. Qt otherwise follows the operating
    # system colour scheme, and under Windows dark mode unstyled surfaces
    # (combo popups, page backgrounds) turned dark behind dark text.
    hints = application.styleHints()
    if hasattr(hints, "setColorScheme"):
        hints.setColorScheme(Qt.ColorScheme.Light)

    application.setPalette(
        light_palette()
    )

    application.setStyleSheet(
        application_stylesheet()
    )


def light_palette() -> QPalette:
    """Return a light QPalette matching the Rivelero colour tokens."""

    c = COLORS
    palette = QPalette()

    roles = {
        QPalette.ColorRole.Window: c.app_background,
        QPalette.ColorRole.WindowText: c.text_primary,
        QPalette.ColorRole.Base: c.surface,
        QPalette.ColorRole.AlternateBase: c.surface_subtle,
        QPalette.ColorRole.Text: c.text_primary,
        QPalette.ColorRole.Button: c.surface,
        QPalette.ColorRole.ButtonText: c.text_primary,
        QPalette.ColorRole.Highlight: c.primary,
        QPalette.ColorRole.HighlightedText: c.surface,
        QPalette.ColorRole.ToolTipBase: c.sidebar,
        QPalette.ColorRole.ToolTipText: c.sidebar_text_strong,
        QPalette.ColorRole.PlaceholderText: c.text_muted,
        QPalette.ColorRole.Link: c.primary,
    }

    for role, value in roles.items():
        palette.setColor(role, QColor(value))

    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(
            QPalette.ColorGroup.Disabled,
            role,
            QColor(c.text_disabled),
        )

    return palette


# ---------------------------------------------------------------------------
# Semantic helpers
# ---------------------------------------------------------------------------


def refresh_style(
    widget,
) -> None:
    """Force Qt to re-evaluate dynamic stylesheet properties.

    Use after changing properties such as:

        ready
        badge
        primary
        danger
        contentCard
    """

    style = widget.style()

    style.unpolish(
        widget
    )

    style.polish(
        widget
    )

    widget.update()


def set_semantic_property(
    widget,
    name: str,
    value,
) -> None:
    """Set a Qt dynamic property and refresh its style."""

    widget.setProperty(
        name,
        value,
    )

    refresh_style(
        widget
    )