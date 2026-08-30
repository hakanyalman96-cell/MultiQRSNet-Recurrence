"""
(Optional) Convert the manuscript + Online Resources (Markdown) to Word .docx,
with tables preserved. pypandoc_binary bundles pandoc, so no system install is needed.

Run:
    pip install pypandoc_binary -q
    python3 build_recurrence_docx.py

Place the .md files in the same folder.
"""
import os, pypandoc

FILES = [
    "Recurrence_Unipolar_Manuscript.md",
    "Online_Resource_1_TRIPOD_AI.md",
    "Online_Resource_2_Analysis_History.md",
]

def main():
    for md in FILES:
        if not os.path.exists(md):
            print("skip (not found):", md); continue
        dx = os.path.splitext(md)[0] + ".docx"
        pypandoc.convert_file(md, "docx", outputfile=dx,
                              extra_args=["--standalone", "--from", "gfm"])
        print("wrote", dx)

if __name__ == "__main__":
    main()
