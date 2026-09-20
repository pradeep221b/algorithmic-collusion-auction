"""Build the paper: pdflatex -> bibtex -> pdflatex x2, then summarise the log.

    py -3.14 paper/build.py            # from the repo root; writes paper/main.pdf

Finds MiKTeX/TeX Live on PATH or in the usual Windows install folders. Exit code 1 on any
LaTeX error, undefined reference, or undefined citation, so it doubles as a check.
"""
import os
import pathlib
import re
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
CANDIDATES = [
    pathlib.Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64",
    pathlib.Path(r"C:\Program Files\MiKTeX\miktex\bin\x64"),
    pathlib.Path(r"C:\texlive\2025\bin\windows"),
    pathlib.Path(r"C:\texlive\2024\bin\windows"),
]


def find(tool):
    if shutil.which(tool):
        return tool
    for d in CANDIDATES:
        exe = d / f"{tool}.exe"
        if exe.exists():
            return str(exe)
    sys.exit(f"{tool} not found. Install MiKTeX (winget install MiKTeX.MiKTeX) or TeX Live.")


def run(cmd):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, encoding="utf-8", errors="replace")


def main():
    pdflatex, bibtex = find("pdflatex"), find("bibtex")
    tex = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"]
    r = run(tex)
    if r.returncode:
        print(r.stdout[-3000:]); sys.exit("pdflatex failed on pass 1")
    r = run([bibtex, "main"])
    if r.returncode:
        print(r.stdout[-2000:]); sys.exit("bibtex failed")
    run(tex)
    r = run(tex)
    if r.returncode:
        print(r.stdout[-3000:]); sys.exit("pdflatex failed on final pass")

    log = (HERE / "main.log").read_text(encoding="utf-8", errors="replace")
    errors = [l for l in log.splitlines() if l.startswith("!")]
    undef_ref = re.findall(r"Reference `([^']+)' on page \d+ undefined", log)
    undef_cite = re.findall(r"Citation `([^']+)' on page \d+ undefined", log)
    multiply = re.findall(r"Label `([^']+)' multiply defined", log)
    overfull = len(re.findall(r"^Overfull \\hbox", log, flags=re.M))
    pages = re.search(r"Output written on main\.pdf \((\d+) pages", log)
    print(f"pages: {pages.group(1) if pages else '?'}  overfull hboxes: {overfull}")
    print(f"undefined refs: {sorted(set(undef_ref))}\nundefined cites: {sorted(set(undef_cite))}\nmultiply defined: {sorted(set(multiply))}")
    if errors or undef_ref or undef_cite or multiply:
        print("\n".join(errors[:20])); sys.exit(1)
    print("OK:", HERE / "main.pdf")


if __name__ == "__main__":
    main()
