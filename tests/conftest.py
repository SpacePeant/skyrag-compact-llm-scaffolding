"""Make `import skyrag` resolve to `src/skyrag/`, before any test module loads.

Without this, `import skyrag` resolves to the repository-root `skyrag/`
directory -- the upstream clone, which holds only README/LICENSE/CODE_OF_CONDUCT
and no `__init__.py`. Python treats such a directory as an implicit namespace
package, so the import succeeds and then every submodule lookup fails with
`No module named 'skyrag.llm'`.

`tests/test_skyrag.py` inserts `src/` itself, but `tests/test_openrouter.py`
imports at module scope with no such setup and is collected first
(`o` < `s`), so the namespace package landed in `sys.modules` before the
insert ever ran. Doing it here fixes both, because pytest imports conftest
before collecting anything.

`src` goes at the front rather than the end: a path entry containing a real
package wins over a namespace portion only if it is reached first.
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
else:
    sys.path.remove(str(SRC))
    sys.path.insert(0, str(SRC))
