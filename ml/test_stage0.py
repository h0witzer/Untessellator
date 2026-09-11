#!/usr/bin/env python3
"""
test_stage0.py - fixtures for the Stage 0 scanner.

Run:  python test_stage0.py

These exist because the scanner reads YAML with regexes instead of a parser, which is
a deliberate trade (see the module header of stage0_index.py) but only safe if the
failure modes are pinned down. Each test below corresponds to a way the histogram
could come out silently wrong rather than loudly broken:

  - BSpline is a valid type in BOTH the curve and surface taxonomies, so a global
    `type:` count would merge B-spline surfaces into B-spline curves.
  - A nested `type:` field (a Revolution's profile curve, say) would be counted as a
    patch in its own right.
  - Both dashed YAML list styles occur in the wild.
  - ABC models can have several parts; keying on the 8-digit id alone loses all but one.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stage0_index import (  # noqa: E402
    build_row,
    classify_sections,
    learn_section_roles,
    parse_identity,
    scan_file,
)

# A well-formed feature file in the style the ABC supplementary describes: a list of
# curves and a list of patches, each entry carrying a type plus type-specific params.
BASIC = """\
curves:
- type: Line
  sharp: true
  vert_indices:
  - 0
  - 1
- type: Circle
  sharp: true
  radius: 3.0
- type: BSpline
  sharp: false
  degree: 3
  poles:
  - - 0.0
    - 1.0
  - - 2.0
    - 3.0
surfaces:
- type: Plane
  location:
  - 0.0
  - 0.0
  - 0.0
  face_indices:
  - 0
  - 1
  - 2
- type: Torus
  max_radius: 10.0
  min_radius: 2.0
  face_indices:
  - 3
- type: BSpline
  u_degree: 3
  v_degree: 3
  face_indices:
  - 4
"""

# A Revolution patch whose profile curve is a nested object with its own `type:`.
# The Circle must NOT be counted as a patch.
NESTED = """\
surfaces:
- type: Revolution
  location:
  - 0.0
  - 0.0
  - 0.0
  curve:
    type: Circle
    radius: 2.0
  face_indices:
  - 0
- type: Cylinder
  radius: 5.0
  face_indices:
  - 1
"""

# The other legal YAML list style, with list items indented under the key.
INDENTED = """\
curves:
  - type: Line
    sharp: true
surfaces:
  - type: Cone
    radius: 1.0
  - type: Sphere
    radius: 4.0
"""

# A part whose only patches and only curves are B-splines: nothing in the file itself
# identifies which section is which. Must fall back to the learned role map.
AMBIGUOUS = """\
curves:
- type: BSpline
  degree: 3
surfaces:
- type: BSpline
  u_degree: 3
- type: Other
"""

# The REAL ABC layout, trimmed from
# 00000050_80d90bfdd2e74e709956122a_features_000.yml. Two properties of it broke an
# earlier version of the scanner and are the reason this fixture exists:
#
#   1. Keys are emitted in ALPHABETICAL order, so the dash carries `location:` and
#      `type:` arrives undashed two lines later - except in the third curve below,
#      whose remaining keys all sort after "type", putting it on the dash. Both are
#      the same entry level and must land in the same bucket.
#   2. Index arrays are flow-style and wrap across lines, so continuation lines are
#      indented and full of numbers. None may be mistaken for a key.
#
# The real file has 7 BSpline CURVES and 0 BSpline surfaces - the exact case a global
# `type:` count gets wrong, in the direction that invents spline surfaces that are
# not there.
REAL_LAYOUT = """\
curves:
- location: [0.0, -3.175, 0.0]
  radius: 17.5145949
  sharp: true
  type: Circle
  vert_indices: [1, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
    88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 1]
  vert_parameters: [0.0, 51.92715129900495, 103.8543025980099, 155.78145389701484,
    207.7086051960198, 259.63575649502474]
- direction: [0.0, 1.0, 0.0]
  location: [17.5145949, -3.175, 0.0]
  sharp: true
  type: Line
  vert_indices: [1, 2, 3]
- type: BSpline
  vert_indices: [4, 5, 6]
surfaces:
- coefficients: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.175]
  face_indices: [0, 1, 2, 3]
  location: [0.0, -3.175, 0.0]
  type: Plane
  vert_indices: [0, 1, 2]
  x_axis: [1.0, 0.0, 0.0]
- face_indices: [4, 5]
  location: [0.0, 0.0, 0.0]
  radius: 17.5145949
  type: Cylinder
  vert_indices: [3, 4, 5]
  z_axis: [0.0, 1.0, 0.0]
"""


class ScannerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, name: str, text: str) -> Path:
        p = self.root / name
        p.write_text(text, encoding="utf-8")
        return p

    # ---------------------------------------------------------------- sectioning

    def test_bspline_is_not_merged_across_sections(self):
        """The headline hazard: BSpline is valid as both a curve and a surface."""
        path = self.write("00000001_deadbeef_features_001.yml", BASIC)
        res = scan_file(path)
        surfaces, curves, flags = classify_sections(res["sections"])

        self.assertEqual(surfaces, {"Plane": 1, "Torus": 1, "BSpline": 1})
        self.assertEqual(curves, {"Line": 1, "Circle": 1, "BSpline": 1})
        self.assertEqual(flags, set())
        # If the sections were merged, BSpline would read as 2.
        self.assertEqual(surfaces["BSpline"], 1)

    def test_nested_type_is_not_counted_as_a_patch(self):
        path = self.write("00000002_deadbeef_features_001.yml", NESTED)
        res = scan_file(path)
        surfaces, _, _ = classify_sections(res["sections"])
        self.assertEqual(surfaces, {"Revolution": 1, "Cylinder": 1})
        self.assertNotIn("Circle", surfaces)

    def test_indented_list_style(self):
        path = self.write("00000003_deadbeef_features_001.yml", INDENTED)
        res = scan_file(path)
        surfaces, curves, _ = classify_sections(res["sections"])
        self.assertEqual(surfaces, {"Cone": 1, "Sphere": 1})
        self.assertEqual(curves, {"Line": 1})

    def test_real_abc_layout(self):
        """Alphabetical keys, undashed `type:`, wrapped flow-style arrays.

        Cross-checked against a full PyYAML parse of the published sample model
        00000050_80d90bfdd2e74e709956122a_features_000.yml: the scanner matched it
        exactly (curves BSpline 7 / Circle 47 / Line 52, surfaces Cylinder 23 /
        Plane 20, 77 sharp) while running 31x faster.
        """
        path = self.write("00000050_80d90bfd_features_000.yml", REAL_LAYOUT)
        res = scan_file(path)
        surfaces, curves, flags = classify_sections(res["sections"])

        self.assertEqual(sorted(res["sections"]), ["curves", "surfaces"])
        self.assertEqual(curves, {"Circle": 1, "Line": 1, "BSpline": 1})
        self.assertEqual(surfaces, {"Plane": 1, "Cylinder": 1})
        self.assertEqual(flags, set())
        # The whole point: a global count would put this BSpline in the surfaces bin.
        self.assertNotIn("BSpline", surfaces)

    def test_dashed_and_undashed_type_are_one_entry_level(self):
        """`- type:` at indent 0 and `  type:` at indent 2 are the same level.

        Regression: an earlier version preferred dashed lines outright, so a single
        entry whose keys happened to sort `type` first would discard every undashed
        entry in the section. On real ABC data that silently drops most patches.
        """
        path = self.write("00000099_deadbeef_features_000.yml", REAL_LAYOUT)
        res = scan_file(path)
        # The third curve is dashed (`- type: BSpline`), the first two are not.
        self.assertEqual(sum(res["sections"]["curves"].values()), 3)

    def test_wrapped_flow_arrays_are_not_parsed_as_keys(self):
        path = self.write("00000100_deadbeef_features_000.yml", REAL_LAYOUT)
        res = scan_file(path)
        self.assertEqual(sorted(res["sections"]), ["curves", "surfaces"])

    def test_sharp_curves_are_counted(self):
        path = self.write("00000004_deadbeef_features_001.yml", BASIC)
        res = scan_file(path)
        self.assertEqual(res["n_sharp"], 2)  # Line and Circle are sharp, BSpline is not

    # ---------------------------------------------------------------- role learning

    def test_roles_are_learned_from_unambiguous_types(self):
        self.write("00000005_deadbeef_features_001.yml", BASIC)
        self.write("00000006_deadbeef_features_001.yml", NESTED)
        roles, evidence = learn_section_roles(sorted(self.root.glob("*.yml")))
        self.assertEqual(roles.get("surfaces"), "surfaces")
        self.assertEqual(roles.get("curves"), "curves")
        self.assertNotIn("_conflict", evidence.get("surfaces", {}))

    def test_ambiguous_file_is_flagged_without_learned_roles(self):
        path = self.write("00000007_deadbeef_features_001.yml", AMBIGUOUS)
        res = scan_file(path)
        surfaces, curves, flags = classify_sections(res["sections"], learned={})
        self.assertIn("ambiguous_sections", flags)
        self.assertEqual(surfaces, {})
        self.assertEqual(curves, {})

    def test_ambiguous_file_resolves_with_learned_roles(self):
        path = self.write("00000008_deadbeef_features_001.yml", AMBIGUOUS)
        res = scan_file(path)
        roles = {"surfaces": "surfaces", "curves": "curves"}
        surfaces, curves, flags = classify_sections(res["sections"], learned=roles)
        self.assertEqual(flags, set())
        self.assertEqual(surfaces, {"BSpline": 1, "Other": 1})
        self.assertEqual(curves, {"BSpline": 1})

    # ---------------------------------------------------------------- identity

    def test_multi_part_models_get_distinct_keys(self):
        a = parse_identity(Path("00000050_80d90bfdd2e74e709a56641e_features_001.yml"))
        b = parse_identity(Path("00000050_80d90bfdd2e74e709a56641e_features_002.yml"))
        self.assertEqual(a, ("00000050", "1"))
        self.assertEqual(b, ("00000050", "2"))
        self.assertNotEqual(a, b)  # same model, different parts: must not collide

    def test_identity_falls_back_on_unexpected_names(self):
        self.assertEqual(parse_identity(Path("00001234_something_else.yml"))[0], "00001234")
        self.assertEqual(parse_identity(Path("oddball.yml")), ("oddball", "0"))

    # ---------------------------------------------------------------- rows

    def test_single_primitive_is_flagged(self):
        path = self.write("00000009_deadbeef_features_001.yml",
                          "surfaces:\n- type: Plane\n  face_indices:\n  - 0\n")
        row = build_row(scan_file(path), self.root, {})
        self.assertEqual(row["n_patches"], 1)
        self.assertIn("single_primitive", row["flags"])

    def test_no_surfaces_is_flagged(self):
        path = self.write("00000010_deadbeef_features_001.yml", "curves:\n- type: Line\n")
        row = build_row(scan_file(path), self.root, {})
        self.assertEqual(row["n_patches"], 0)
        self.assertIn("no_surfaces", row["flags"])

    def test_empty_file_does_not_crash(self):
        path = self.write("00000011_deadbeef_features_001.yml", "")
        row = build_row(scan_file(path), self.root, {})
        self.assertEqual(row["n_patches"], 0)
        self.assertIn("no_sections", row["flags"])

    def test_row_columns_are_populated(self):
        path = self.write("00000012_deadbeef_features_001.yml", BASIC)
        row = build_row(scan_file(path), self.root, {})
        self.assertEqual(row["s_plane"], 1)
        self.assertEqual(row["s_torus"], 1)
        self.assertEqual(row["s_bspline"], 1)
        self.assertEqual(row["c_bspline"], 1)
        self.assertEqual(row["n_patches"], 3)
        self.assertEqual(row["n_curves"], 3)
        self.assertEqual(row["flags"], "")

    def test_struct_hash_matches_structural_twins(self):
        a = self.write("00000013_deadbeef_features_001.yml", BASIC)
        b = self.write("00000014_cafebabe_features_001.yml", BASIC)
        c = self.write("00000015_deadbeef_features_001.yml", NESTED)
        ha = build_row(scan_file(a), self.root, {})["struct_hash"]
        hb = build_row(scan_file(b), self.root, {})["struct_hash"]
        hc = build_row(scan_file(c), self.root, {})["struct_hash"]
        self.assertEqual(ha, hb)
        self.assertNotEqual(ha, hc)


if __name__ == "__main__":
    unittest.main(verbosity=2)
