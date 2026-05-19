"""Helpers for running code snippets from the XGBoost documentation."""

import atexit
import json
import subprocess
import tempfile
import uuid
from pathlib import Path

from docutils import nodes
from sphinx.application import Sphinx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NO_DOCTEST_CLASS = "no-doctest"
_R_SESSIONS: dict[str, "RSession"] = {}


class RSession:
    """A persistent R process for doctest snippets from one document."""

    def __init__(self) -> None:
        # pylint: disable=consider-using-with
        self._tmpdir = tempfile.TemporaryDirectory()
        self._proc = subprocess.Popen(
            ["R", "--vanilla", "--slave"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=PROJECT_ROOT,
            text=True,
            bufsize=1,
        )

    def run(self, code: str) -> None:
        """Run a snippet of code."""
        if self._proc.poll() is not None:
            raise RuntimeError(f"R process exited with code {self._proc.returncode}.")
        assert self._proc.stdin is not None

        path = Path(self._tmpdir.name) / "xgboost-doc-test.R"
        path.write_text(code, encoding="utf-8")
        # The marker is printed after the sourced snippet with its status code.  Python
        # reads R stdout until this marker, so each snippet can be checked without
        # restarting the R process.
        marker = f"__XGBOOST_R_DOCTEST_DONE_{uuid.uuid4().hex}__"
        command = rf""".__xgb_doctest_status <- 0L
tryCatch(
  source({json.dumps(str(path))}, local = globalenv(), echo = FALSE, keep.source = FALSE),
  error = function(e) {{
    .__xgb_doctest_status <<- 1L
    cat(conditionMessage(e), '\n', file = stderr())
  }}
)
cat('\n{marker}:', .__xgb_doctest_status, '\n', sep = '')
rm(.__xgb_doctest_status)
flush.console()
"""
        self._proc.stdin.write(command)
        self._proc.stdin.flush()

        output, failed = self._read_until_marker(marker)
        if not failed:
            return

        msg = ["R documentation snippet failed."]
        if output:
            msg.extend(["output:", output])
        raise RuntimeError("\n".join(msg))

    def close(self) -> None:
        """Close the R session."""
        if self._proc.poll() is None:
            # quit if R is still running.
            assert self._proc.stdin is not None
            self._proc.stdin.write("q('no', status = 0, runLast = FALSE)\n")
            self._proc.stdin.flush()
            self._proc.wait()
        self._tmpdir.cleanup()

    def _read_until_marker(self, marker: str) -> tuple[str, bool]:
        assert self._proc.stdout is not None
        output: list[str] = []
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                raise RuntimeError(
                    "R process exited before reporting snippet status. "
                    f"Exit code: {self._proc.poll()}."
                )
            if line.startswith(f"{marker}:"):
                return "".join(output), line.removeprefix(f"{marker}:").strip() != "0"
            output.append(line)


def run_r_code(session_key: str, code: str) -> None:
    """Run one R doctest snippet in a persistent document session."""
    session = _R_SESSIONS.get(session_key)
    if session is None:
        session = RSession()
        _R_SESSIONS[session_key] = session
    session.run(code)


def close_r_session(session_key: str) -> None:
    """Close the R session for a document if it was started."""
    session = _R_SESSIONS.pop(session_key, None)
    if session is not None:
        session.close()


def close_all_r_sessions() -> None:
    """Close all R doctest sessions."""
    for session_key in list(_R_SESSIONS):
        close_r_session(session_key)


def _normalise_language(language: str | None) -> str | None:
    if language is None:
        return None
    language = language.lower()
    if language in {"py", "python", "python3"}:
        return "python"
    if language in {"r", "s", "rlang"}:
        return "r"
    return None


def _mark_python_node(node: nodes.literal_block, docname: str) -> None:
    node["testnodetype"] = "testcode"
    node["groups"] = [docname]
    node["options"] = {}


def _mark_r_node(
    node: nodes.literal_block, docname: str, *, close_after: bool = False
) -> None:
    code = node.astext()
    test = [
        "from xgboost_doc_doctest import run_r_code as _xgb_run_r_code",
        f"_xgb_run_r_code({docname!r}, {code!r})",
    ]
    if close_after:
        test.extend(
            [
                "from xgboost_doc_doctest import close_r_session as _xgb_close_r_session",
                f"_xgb_close_r_session({docname!r})",
            ]
        )
    node["test"] = "\n".join(test)
    node["testnodetype"] = "testcode"
    node["groups"] = [docname]
    node["options"] = {}


def _has_class(node: nodes.Node, class_name: str) -> bool:
    return class_name in node.get("classes", [])


def _has_ancestor_class(node: nodes.Node, class_name: str) -> bool:
    parent = node.parent
    while parent is not None:
        if _has_class(parent, class_name):
            return True
        parent = parent.parent
    return False


def mark_doctest_nodes(app: Sphinx, doctree: nodes.document) -> None:
    """Mark code-tab blocks so ``sphinx.ext.doctest`` executes them."""
    docname = app.env.docname
    r_nodes: list[nodes.literal_block] = []
    for node in doctree.findall(nodes.literal_block):
        if "testnodetype" in node:
            continue
        if _has_class(node, NO_DOCTEST_CLASS) or _has_ancestor_class(
            node, NO_DOCTEST_CLASS
        ):
            continue
        if not _has_ancestor_class(node, "code-tab"):
            continue

        language = _normalise_language(node.get("language"))
        if language == "python":
            _mark_python_node(node, docname)
        elif language == "r":
            r_nodes.append(node)

    for node in r_nodes[:-1]:
        _mark_r_node(node, docname)
    if r_nodes:
        _mark_r_node(r_nodes[-1], docname, close_after=True)


def setup(app: Sphinx) -> dict[str, bool]:
    """Entry point for sphinx app setup."""
    app.connect("doctree-read", mark_doctest_nodes)
    atexit.register(close_all_r_sessions)
    return {
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
