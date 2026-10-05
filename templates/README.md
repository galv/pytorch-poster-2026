# Recovered conference poster template

`PyTorch Conf NA 2026 Poster Template 36x24_horizontal (1).pptx` is a
reconstruction from the supplied `../poster.pptx`, not a recovered original file.
It is a single 36 × 24 inch landscape slide with the conference background and
an editable generic title and author line. The body is blank.

The slide master, slide layout, background images, notes, presentation settings,
and remaining package parts were preserved byte for byte. The recovery removed
poster-specific slide shapes and hyperlinks, replaced the title and author text,
cleared author metadata, and reversed the hyperlink theme-color change described
in `../build_poster.py`. Original slide-level instructions or placeholders cannot
be recovered because the build script overwrote that slide XML.

The referenced upstream revision contains an earlier ReportLab PDF generator
and no PPTX template in `poster_examples`:
https://github.com/galv/pytorch/tree/89ee54e33a1d99c9ed5d69383377ad98f3cf9df0/poster_examples

The reconstructed file passed package, dimension, font-declaration, and Artifact
Tool import checks. It was exported with LibreOffice and visually inspected.
It has not been inspected in Microsoft PowerPoint.

`../build_poster.py` uses this file when its original parent-directory template
is absent. Building the poster replaces the generic title and blank body with
the generated poster content as before.
