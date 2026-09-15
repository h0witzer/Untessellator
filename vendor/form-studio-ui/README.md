# Form Studio UI kit (vendored)

An unmodified copy of `src/ui/` from [h0witzer/form-studio](https://github.com/h0witzer/form-studio):
the Onshape-native frame, toolbar, panel, menus, view cube, orbit camera and WebGL2 renderer
pieces, each measured from a live Part Studio. `docs/UI_KIT.md` and `docs/ONSHAPE_STYLE.md` in
that repository are the documentation.

| | |
| --- | --- |
| Source | `form-studio/src/ui/` at commit `ba0cf79` (the directory last changed in `4af13ba`, 2026-09-13) |
| Vendored | 2026-09-14 |
| Licence | **LGPL-2.1-or-later**, as each file's SPDX header states. The rest of stlToStep is MIT. |

## Rules for this folder

- **Do not edit these files.** Where stlToStep needs to look or behave differently (group colours,
  the unfitted mesh's faceted shading, highlight choices), the difference lives in stlToStep's own
  code under `js/studio/`, built from the kit's exports. That keeps the LGPL module a clean,
  replaceable copy of the upstream source.
- **To update**, copy `src/ui/*` from form-studio again and record the new commit above.
