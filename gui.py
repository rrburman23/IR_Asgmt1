"""
File Name: gui.py
Description: PyQt6 GUI for Art Search.
- Layout: Clean Div-based Hierarchy (Name -> Artist -> Medium -> Description).
- Supports server-side pagination and spelling suggestion.
- /test and /evaluate display output inside the GUI (colored).
"""

from __future__ import annotations

import sys
import os
import time
import base64
import html
import requests
from typing import List, Optional

from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QApplication,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QLabel,
    QListWidget,
)
from PyQt6.QtCore import QThread, pyqtSignal, QUrl
from PyQt6.QtGui import QDesktopServices, QIcon


ICON_PATH = os.path.join(os.path.dirname(__file__), "icon.ico")


def _colorize_console_text_to_html(text: str) -> str:
    """
    Convert captured console text into HTML with colors.
    We escape HTML first, then apply replacements, so it's safe.
    """
    import re

    safe = html.escape(text)

    # Log tags
    safe = safe.replace(
        "[INFO]", "<span style='color:#3498db; font-weight:bold;'>[INFO]</span>"
    )
    safe = safe.replace(
        "[SUCCESS]", "<span style='color:#2ecc71; font-weight:bold;'>[SUCCESS]</span>"
    )
    safe = safe.replace(
        "[WARNING]", "<span style='color:#f39c12; font-weight:bold;'>[WARNING]</span>"
    )
    safe = safe.replace(
        "[ERROR]", "<span style='color:#e74c3c; font-weight:bold;'>[ERROR]</span>"
    )
    safe = safe.replace(
        "[CRITICAL]", "<span style='color:#e74c3c; font-weight:bold;'>[CRITICAL]</span>"
    )
    safe = safe.replace(
        "[CACHE]", "<span style='color:#9b59b6; font-weight:bold;'>[CACHE]</span>"
    )
    safe = safe.replace(
        "[CONFIG]", "<span style='color:#8e44ad; font-weight:bold;'>[CONFIG]</span>"
    )

    # PASS / FAIL
    safe = re.sub(
        r"\bPASS\b", "<span style='color:#2ecc71; font-weight:bold;'>PASS</span>", safe
    )
    safe = re.sub(
        r"\bFAIL\b", "<span style='color:#e74c3c; font-weight:bold;'>FAIL</span>", safe
    )

    # unittest-ish
    safe = re.sub(
        r"\bok\b", "<span style='color:#2ecc71; font-weight:bold;'>ok</span>", safe
    )
    safe = re.sub(
        r"\bFAILED\b",
        "<span style='color:#e74c3c; font-weight:bold;'>FAILED</span>",
        safe,
    )
    safe = re.sub(
        r"\bERROR\b",
        "<span style='color:#e74c3c; font-weight:bold;'>ERROR</span>",
        safe,
    )

    # headings (simple)
    safe = safe.replace(
        "Phase 1:", "<span style='color:#8ecaf7; font-weight:bold;'>Phase 1:</span>"
    )
    safe = safe.replace(
        "Phase 2:", "<span style='color:#8ecaf7; font-weight:bold;'>Phase 2:</span>"
    )
    safe = safe.replace(
        "FINAL SYSTEM PERFORMANCE REPORT",
        "<span style='color:#f1c40f; font-weight:bold;'>FINAL SYSTEM PERFORMANCE REPORT</span>",
    )

    # Traceback / AssertionError highlight
    safe = safe.replace(
        "Traceback (most recent call last):",
        "<span style='color:#e74c3c; font-weight:bold;'>Traceback (most recent call last):</span>",
    )
    safe = re.sub(
        r"\bAssertionError\b",
        "<span style='color:#e74c3c; font-weight:bold;'>AssertionError</span>",
        safe,
    )

    return safe


class EngineLoadThread(QThread):
    engine_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def run(self):
        try:
            from hybrid_search import ArtGallerySearchEngine
            import ingest_data

            engine = ArtGallerySearchEngine(ingest_data.OUTPUT_FILE)
            self.engine_ready.emit(engine)
        except Exception as e:
            self.error_occurred.emit(str(e))


class SystemTaskThread(QThread):
    """
    Runs /test or /evaluate in a background thread so GUI doesn't freeze.
    Captures stdout/stderr and sends text back to the UI.
    """

    finished_text = pyqtSignal(str)
    errored = pyqtSignal(str)

    def __init__(self, task: str, parent=None):
        super().__init__(parent)
        self.task = task

    def run(self):
        try:
            import io
            import contextlib
            import unittest

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                if self.task == "test":
                    import test_engine

                    suite = unittest.TestLoader().loadTestsFromModule(test_engine)
                    unittest.TextTestRunner(stream=buf, verbosity=2).run(suite)

                elif self.task == "evaluate":
                    import ingest_data
                    from evaluate_engine import run_evaluation_to_text, EvalConfig

                    # Use your configurable evaluator; adjust defaults here if you want
                    text = run_evaluation_to_text(
                        config=EvalConfig(
                            data_path=ingest_data.OUTPUT_FILE,
                            top_k=10,
                            k_rrf=60,
                            semantic_min_concept_hits=2,
                            semantic_min_hits_by_query={"mountain scenery": 1},
                        )
                    )
                    print(text, end="")

                else:
                    print(f"[ERROR] Unknown task: {self.task}")

            self.finished_text.emit(buf.getvalue())
        except Exception as e:
            self.errored.emit(str(e))


class ArtSearchGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tate Gallery | Semantic Search")
        self.setWindowIcon(QIcon(ICON_PATH))
        self.setMinimumSize(900, 600)
        self.resize(1150, 800)

        self.engine = None
        self.image_cache: dict[str, str] = {}

        # Pagination state
        self.current_query = ""
        self.current_page = 1
        self.per_page = 10
        self.total_pages = 1

        self.apply_styling()

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        self._setup_sidebar()
        self._setup_main_area()

        self.search_input.setDisabled(True)
        self.btn_search.setDisabled(True)

        self.engine_thread = EngineLoadThread()
        self.engine_thread.engine_ready.connect(self._on_engine_ready)
        self.engine_thread.error_occurred.connect(self._on_engine_error)
        self.engine_thread.start()

        self.task_thread: Optional[SystemTaskThread] = None

    def apply_styling(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #0f0f0f; }
            QWidget { font-family: 'Segoe UI', sans-serif; color: #ccc; }
            QLineEdit {
                padding: 12px; border-radius: 8px;
                background: #1a1a1a; color: white; border: 1px solid #333;
                font-size: 14px;
            }
            QPushButton {
                padding: 12px; border-radius: 8px;
                background: #2980b9; color: white; font-weight: bold;
            }
            QPushButton:hover { background: #3498db; }
            QPushButton#clearBtn {
                background: #222; color: #888; font-size: 11px; margin-top: 5px;
            }
            QPushButton#clearBtn:hover { background: #e74c3c; color: white; }
            QPushButton#navBtn {
                background: #191970; color: #fff; font-size: 11px; padding: 5px 12px;
            }
            QPushButton#navBtn:disabled { background: #222; color: #888; }
            QTextBrowser { background: #141414; border: 1px solid #222; border-radius: 8px; }
            QListWidget { background: #1a1a1a; border: none; border-radius: 8px; }
            QListWidget::item { padding: 10px; border-bottom: 1px solid #222; }
            QListWidget::item:selected { background: #2980b9; color: white; }
        """)

    def _setup_sidebar(self):
        sidebar = QVBoxLayout()
        sidebar.setContentsMargins(0, 0, 10, 0)
        lbl = QLabel("SEARCH HISTORY")
        lbl.setStyleSheet(
            "font-weight: bold; color: #555; font-size: 11px; margin-bottom: 5px;"
        )

        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._on_history_clicked)
        self.history_list.setFixedWidth(220)

        self.btn_clear = QPushButton("Clear History")
        self.btn_clear.setObjectName("clearBtn")
        self.btn_clear.clicked.connect(self.clear_history)

        sidebar.addWidget(lbl)
        sidebar.addWidget(self.history_list)
        sidebar.addWidget(self.btn_clear)
        self.main_layout.addLayout(sidebar)

    def _setup_main_area(self):
        content = QVBoxLayout()

        nav = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search the archives or type /help...")
        self.search_input.returnPressed.connect(self.perform_search)

        self.btn_search = QPushButton("Search")
        self.btn_search.clicked.connect(self.perform_search)

        self.status = QLabel("Initializing Search Engine...")
        self.status.setStyleSheet("color: #f39c12;")

        self.results_area = QTextBrowser()
        self.results_area.setOpenLinks(False)
        self.results_area.anchorClicked.connect(self._on_suggestion_clicked)

        nav.addWidget(self.search_input)
        nav.addWidget(self.btn_search)

        content.addLayout(nav)
        content.addWidget(self.status)
        content.addWidget(self.results_area)

        # Paging controls
        self.nav_area = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_prev.setObjectName("navBtn")
        self.btn_prev.clicked.connect(self.go_to_prev_page)

        self.btn_next = QPushButton("Next")
        self.btn_next.setObjectName("navBtn")
        self.btn_next.clicked.connect(self.go_to_next_page)

        self.btn_prev.setMinimumWidth(110)
        self.btn_next.setMinimumWidth(110)

        self.label_page = QLabel("Page 1 of 1")
        self.label_page.setStyleSheet(
            "color:#8ecaf7; font-size:13px; font-weight:bold; margin:0 12px;"
        )

        self.nav_area.addStretch(1)
        self.nav_area.addWidget(self.btn_prev)
        self.nav_area.addWidget(self.label_page)
        self.nav_area.addWidget(self.btn_next)
        self.nav_area.addStretch(1)

        self.nav_area.setSpacing(12)
        self.nav_area.setContentsMargins(0, 8, 0, 0)

        content.addLayout(self.nav_area)
        self.main_layout.addLayout(content)

        self.update_nav_controls(reset=True)

    def _on_engine_ready(self, engine):
        self.engine = engine
        self.search_input.setDisabled(False)
        self.btn_search.setDisabled(False)
        self.status.setText(f"Online | {len(engine.df)} Artworks Indexed")
        self.status.setStyleSheet("color: #2ecc71;")

    def _on_engine_error(self, err: str):
        self.status.setText(f"Failed: {err}")
        self.status.setStyleSheet("color: #e74c3c;")

    def _on_history_clicked(self, item):
        if item is not None:
            self.search_input.setText(item.text())
            self.current_query = item.text()
            self.current_page = 1
            self._run_search_and_display()

    def _on_suggestion_clicked(self, qurl: QUrl):
        url_str = qurl.toString()
        if url_str.startswith("img::"):
            QDesktopServices.openUrl(QUrl(url_str[5:]))
        else:
            self.search_input.setText(url_str)
            self.current_query = url_str
            self.current_page = 1
            self._run_search_and_display()

    def clear_history(self):
        self.history_list.clear()

    def update_nav_controls(self, reset: bool = False):
        need_paging = self.total_pages > 1
        self.btn_prev.setDisabled(self.current_page <= 1 or not need_paging)
        self.btn_next.setDisabled(
            self.current_page >= self.total_pages or not need_paging
        )
        self.label_page.setText(
            f"Page {self.current_page} of {self.total_pages}" if need_paging else ""
        )
        self.btn_prev.setVisible(need_paging)
        self.btn_next.setVisible(need_paging)
        self.label_page.setVisible(need_paging)
        if reset:
            self.current_page = 1
            self.total_pages = 1

    def go_to_next_page(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
            self._run_search_and_display()

    def go_to_prev_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self._run_search_and_display()

    def _get_img_b64(self, url: str) -> Optional[str]:
        if not url or len(url) < 10:
            return None
        if url in self.image_cache:
            return self.image_cache[url]
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=3)
            if r.status_code == 200:
                b64 = base64.b64encode(r.content).decode()
                self.image_cache[url] = b64
                return b64
        except Exception:
            pass
        return None

    def _display_help(self):
        html_txt = """
<h2 style='color:white;'>System Guide</h2><hr>
<p style='color:#3498db; font-size:16px;'>Available Slash Commands:</p>
<ul>
  <li style='margin-bottom:8px;'><b>/help</b> - Show this manual.</li>
  <li style='margin-bottom:8px;'><b>/exit</b> - Quit the application.</li>
  <li style='margin-bottom:8px;'><b>/test</b> - Run unit tests (shown in GUI, colored).</li>
  <li style='margin-bottom:8px;'><b>/evaluate</b> - Run performance eval (shown in GUI, colored).</li>
</ul>
"""
        self.results_area.setHtml(html_txt)

    def _show_preformatted_text(self, title: str, text: str):
        colored = _colorize_console_text_to_html(text)
        html_out = (
            f"<h2 style='color:white;'>{html.escape(title)}</h2><hr>"
            f"<pre style='color:#ddd; font-family: Consolas, monospace; font-size: 13px; "
            f"line-height:1.35; white-space:pre-wrap;'>{colored}</pre>"
        )
        self.results_area.setHtml(html_out)

    def _run_system_task(self, task: str):
        if self.task_thread is not None and self.task_thread.isRunning():
            return

        label = (
            "Running evaluation..." if task == "evaluate" else "Running unit tests..."
        )
        self.status.setText(label)
        self.status.setStyleSheet("color: #3498db;")
        QApplication.processEvents()

        self.task_thread = SystemTaskThread(task)
        self.task_thread.finished_text.connect(
            lambda out: self._on_task_done(task, out)
        )
        self.task_thread.errored.connect(self._on_task_error)
        self.task_thread.start()

    def _on_task_done(self, task: str, output: str):
        title = "Evaluation Output" if task == "evaluate" else "Unit Test Output"
        self._show_preformatted_text(title, output or "[No output captured]")
        self.status.setText("Task Completed. Engine Ready.")
        self.status.setStyleSheet("color: #2ecc71;")
        self.search_input.clear()

    def _on_task_error(self, err: str):
        self._show_preformatted_text("Task Error", err)
        self.status.setText("Task Failed.")
        self.status.setStyleSheet("color: #e74c3c;")

    def perform_search(self):
        query = self.search_input.text().strip()
        if not query or not self.engine:
            return

        if query.startswith("/"):
            cmd = query.lower()
            if cmd == "/exit":
                QApplication.quit()
                return
            if cmd == "/help":
                self._display_help()
                self.search_input.clear()
                return
            if cmd in ["/test", "/evaluate", "/eval"]:
                self._run_system_task("test" if cmd == "/test" else "evaluate")
                return

            self.results_area.setHtml(
                f"<div style='color:#e74c3c;'>[ERROR] Unknown command: {html.escape(query)}. "
                f"Valid commands are /help, /exit, /test, /evaluate.</div>"
            )
            return

        # Search history logic
        existing = []
        for i in range(self.history_list.count()):
            item = self.history_list.item(i)
            if item is not None:
                existing.append(item.text())
        if query not in existing:
            self.history_list.insertItem(0, query)

        self.current_query = query
        self.current_page = 1
        self._run_search_and_display()

    def _run_search_and_display(self):
        if not self.engine or not self.current_query:
            return
        start = time.perf_counter()
        data = self.engine.hybrid_search(
            self.current_query,
            page=self.current_page,
            per_page=self.per_page,
        )
        duration = time.perf_counter() - start
        self.total_pages = data.get("total_pages", 1)
        self.update_nav_controls(reset=False)
        self.display_results(
            data.get("results", []),
            self.current_query,
            duration,
            self.current_page,
            self.total_pages,
            data.get("suggestion"),
        )

    def display_results(
        self,
        results: List[dict],
        query: str,
        duration: float,
        page: int,
        total_pages: int,
        suggestion: Optional[str],
    ):
        suggestion_html = ""
        if suggestion:
            suggestion_html = (
                f"<div style='color:#f39c12; margin-bottom:10px;'>"
                f"Did you mean: <a href='{html.escape(suggestion)}' style='color:#3498db; text-decoration:none;'><b>{html.escape(suggestion)}</b></a>?</div>"
            )

        html_out = (
            f"{suggestion_html}"
            f"<h2 style='color:white;'>Results: '{html.escape(query)}' (Page {page} of {total_pages}, {duration:.3f}s)</h2><hr>"
        )

        if not results:
            html_out += "<h3 style='color:#888;'>No matches found.</h3>"

        for res in results:
            thumb_url = res.get("Thumbnail", "")
            b64 = self._get_img_b64(thumb_url) if thumb_url else None
            if b64:
                img_html = f"<a href='img::{thumb_url}'><img src='data:image/jpeg;base64,{b64}' width='140' style='border-radius:4px; cursor:pointer;'/></a>"
            else:
                img_html = "<div style='width:140px; height:100px; background:#222; border-radius:4px; text-align:center; padding-top:40px; color:#555;'>No Image</div>"

            year = res.get("Year", "")
            year_str = f" ({year})" if year and year != "Unknown Date" else ""

            title = res.get("Title", "Untitled")
            artist = res.get("Artist", "Unknown Artist")
            medium = res.get("Medium", "")
            desc = res.get("Description", "")
            reasons = str(res.get("Reasons", ""))
            score = str(res.get("Score", ""))

            row = "<table width='100%' style='margin-bottom:20px;'><tr>"
            row += f"<td width='160' valign='top'>{img_html}</td>"
            row += "<td valign='top'>"

            row += (
                f"<div style='font-size:18px; font-weight:bold; margin-bottom:2px;'>"
                f"<a href='img::{thumb_url}' style='color:#3498db; text-decoration:none;'>{html.escape(title)}</a>"
                f"<span style='color:#7f8c8d; font-size:14px; font-weight:normal;'>{html.escape(year_str)}</span></div>"
            )
            row += f"<div style='color:#ecf0f1; font-size:15px; font-weight:bold; margin-bottom:4px;'>{html.escape(artist)}</div>"
            row += f"<div style='color:#f1c40f; font-size:13px; font-style:italic; margin-bottom:6px;'>{html.escape(medium)}</div>"
            row += f"<div style='color:#cccccc; font-size:13px; margin-bottom:6px;'>{html.escape(desc)}</div>"
            row += f"<div style='color:#2ecc71; font-size:11px;'><b>Match:</b> {html.escape(reasons)} &nbsp;|&nbsp; <b>RRF Score:</b> {html.escape(score)}</div>"
            row += (
                "</td></tr></table><hr style='border:none; border-top:1px solid #222;'>"
            )

            html_out += row

        self.results_area.setHtml(html_out)
        self.label_page.setText(f"Page {page} of {total_pages}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(ICON_PATH))
    window = ArtSearchGUI()
    window.show()
    sys.exit(app.exec())
