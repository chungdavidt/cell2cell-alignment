#!/usr/bin/env python3
"""
Checks on orientation.py's letter algebra and the graph builder's mirror guard.

stdlib only and no display, so it runs anywhere — the derivation is pure, which
is the reason it lives in orientation.py rather than inside the GUI. Does not
touch orientation.json: the round-trip test writes to a temp dir.

    python3 tests/test_orientation.py
"""
import ast, itertools, json, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import orientation as o

fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)

# ---------------------------------------------------------------- 1: truth table
print("1. truth table, 4 x 2 x 2 x 2 = 32 inputs")
combos = []
for ant in o.EDGES:
    for med in o.perpendicular_edges(ant):
        for hemi in o.HEMISPHERES:
            for first in o.FIRST_SECTIONS:
                combos.append((ant, med, hemi, first))
check("32 input combinations", len(combos) == 32, f"got {len(combos)}")

codes = {c: o.derive_code(*c) for c in combos}
check("every code well-formed (3 distinct anatomical axes)",
      all(o.validate_code(v) == v for v in codes.values()))
check("every code has one letter per axis pair",
      all({frozenset((l, o.opposite(l))) for l in v} ==
          {frozenset('AP'), frozenset('RL'), frozenset('DV')} for v in codes.values()))
check("z is always D or V (sections cut axial)",
      all(v[0] in 'DV' for v in codes.values()))
check("y and x are never both from the same pair",
      all(v[1] not in (v[2], o.opposite(v[2])) for v in codes.values()))

# Collisions are expected and correct: flipping BOTH medial_edge and hemisphere
# names the same anatomical direction (left hemisphere with medial to the right
# == right hemisphere with medial to the left).
distinct = set(codes.values())
check("32 inputs -> 16 distinct codes", len(distinct) == 16, f"got {len(distinct)}")
bad_collision = []
for a, b in itertools.combinations(combos, 2):
    if codes[a] == codes[b]:
        same_plane = a[0] == b[0] and a[3] == b[3]
        both_flipped = a[1] != b[1] and a[2] != b[2]
        if not (same_plane and both_flipped):
            bad_collision.append((a, b, codes[a]))
check("collisions only where medial edge AND hemisphere both flip",
      not bad_collision, str(bad_collision[:2]))

# ---------------------------------------------------------------- 2: worked example
print("2. worked example")
code = o.derive_code('top', 'right', 'left', 'dorsal')
check("anterior=top, medial=right, hemisphere=left, first=dorsal -> VPR",
      code == 'VPR', f"got {code}  ({o.describe(code)})")

# ---------------------------------------------------------------- 3: mirror pair
print("3. mirror: flipping only medial_edge")
for ant, first in itertools.product(o.EDGES, o.FIRST_SECTIONS):
    e1, e2 = o.perpendicular_edges(ant)
    for hemi in o.HEMISPHERES:
        a = o.derive_code(ant, e1, hemi, first)
        b = o.derive_code(ant, e2, hemi, first)
        axis = 1 if o.EDGE_AXIS[e1] == 'y' else 2
        others = [i for i in range(3) if i != axis]
        if not (o.handedness(a) == -o.handedness(b)
                and all(a[i] == b[i] for i in others)
                and a[axis] == o.opposite(b[axis])):
            check(f"mirror {ant}/{hemi}/{first}", False, f"{a} vs {b}")
            break
    else:
        continue
    break
else:
    check("flipping medial_edge flips handedness and only the lateral letter", True)

# ---------------------------------------------------------------- 4: hemisphere swap
print("4. hemisphere swap")
ok = True
for ant, first in itertools.product(o.EDGES, o.FIRST_SECTIONS):
    for med in o.perpendicular_edges(ant):
        a = o.derive_code(ant, med, 'left', first)
        b = o.derive_code(ant, med, 'right', first)
        pairs = [(x, y) for x, y in zip(a, b)]
        swapped = [(x, y) for x, y in pairs if x != y]
        unchanged = [(x, y) for x, y in pairs if x == y]
        if not (len(swapped) == 1 and set(swapped[0]) == {'R', 'L'} and len(unchanged) == 2):
            ok = False
            check(f"hemisphere swap {ant}/{med}/{first}", False, f"{a} vs {b}")
            break
    if not ok:
        break
check("hemisphere swap maps R<->L and leaves the other two letters alone", ok)

# ---------------------------------------------------------------- 5: round trip
print("5. save / load round trip")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "orientation.json"
    entry_a = o.make_entry('top', 'right', 'left', 'dorsal',
                           image='/data/slice22_subslice_mScarlet_cellmask.tif')
    o.save(path, 'barseq_subslice', entry_a, subject='BY95')
    loaded = o.load(path)
    check("round trip identical",
          loaded['modalities']['barseq_subslice'] == entry_a)
    check("subject stamped", loaded.get('subject') == 'BY95')

    entry_b = o.make_entry('bottom', 'left', 'left', 'ventral', image='/data/invivo.tif')
    o.save(path, 'invivo_ref', entry_b)
    loaded = o.load(path)
    check("second modality added without disturbing the first",
          loaded['modalities']['barseq_subslice'] == entry_a
          and loaded['modalities']['invivo_ref'] == entry_b
          and loaded.get('subject') == 'BY95')

    entry_c = o.make_entry('left', 'top', 'right', 'dorsal', image='/data/slice22.tif')
    o.save(path, 'barseq_subslice', entry_c)
    loaded = o.load(path)
    check("re-running a modality replaces its entry, others untouched",
          loaded['modalities']['barseq_subslice'] == entry_c
          and loaded['modalities']['invivo_ref'] == entry_b)
    check("codes() reads both", o.codes(path) ==
          {'barseq_subslice': entry_c['code'], 'invivo_ref': entry_b['code']})
    check("raw answers stored, not just the code",
          all(k in entry_a for k in
              ('anterior_edge', 'medial_edge', 'hemisphere', 'first_section',
               'image', 'assigned')))
    check("json is readable text", isinstance(json.loads(path.read_text()), dict))

# ---------------------------------------------------------------- 6: agree + builder
print("6. mirrored pair, and the builder's guard")
mirrored = (o.derive_code('top', 'right', 'left', 'dorsal'),     # VPR
            o.derive_code('top', 'left', 'left', 'dorsal'))      # VPL
check("mirrored pair disagrees", not o.agree(*mirrored), f"{mirrored}")
check("a volume agrees with itself", o.agree(mirrored[0], mirrored[0]))
# The same section rotated 90° clockwise in the display: different code, same
# handedness — agreement is about chirality, not about matching letters.
rotated = o.derive_code('right', 'bottom', 'left', 'dorsal')
check("agreement is not equality",
      rotated != 'VPR' and o.agree('VPR', rotated), f"VPR vs {rotated}")
check("disagreeing_pair finds it",
      o.disagreeing_pair({'invivo_ref': mirrored[0],
                          'barseq_subslice': mirrored[1]}) ==
      ('barseq_subslice', 'invivo_ref'))
check("disagreeing_pair returns None when all agree",
      o.disagreeing_pair({'a': 'VPR', 'b': 'VPR'}) is None)

# The builder imports castalign and numpy, neither installed on WSL, so exec its
# two orientation functions from source rather than importing the module.
src = (ROOT / "alignment" / "subslice_graph_builder.py").read_text()
tree = ast.parse(src)
wanted = {'check_orientation_handedness', 'load_orientation_codes'}
ns = {'orientation': o, 'print': print}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        exec(compile(ast.Module([node], []), '<builder>', 'exec'), ns)
check("builder defines both orientation functions", wanted <= set(ns))

guard = ns['check_orientation_handedness']
guard({'invivo_ref': 'VPR', 'ex_vivo_block': 'VPR'}, ['invivo_ref', 'ex_vivo_block'])
guard({}, ['invivo_ref'])
guard({'invivo_ref': 'VPR'}, ['invivo_ref', 'ex_vivo_block'])
guard({'invivo_ref': 'VPR', 'barseq_subslice': 'VPL'}, ['invivo_ref'])  # not configured
check("guard passes on agreement, missing and unconfigured codes", True)
try:
    guard({'invivo_ref': 'VPR', 'barseq_subslice': 'VPL'},
          ['invivo_ref', 'barseq_subslice'])
    check("guard raises on a mirror", False)
except ValueError as e:
    check("guard raises on a mirror", 'handedness disagreement' in str(e).lower())

# ---------------------------------------------------------------- extra: input guards
print("extra: input guards and the single-plane path")
for bad in [('top', 'bottom', 'left', 'dorsal'),      # medial parallel to anterior
            ('top', 'right', 'middle', 'dorsal'),     # bad hemisphere
            ('front', 'right', 'left', 'dorsal'),     # bad edge
            ('top', 'right', 'left', 'up')]:          # bad first_section
    try:
        o.derive_code(*bad)
        check(f"rejects {bad}", False)
    except ValueError:
        check(f"rejects {bad}", True)
for bad_code in ['VP', 'VPQ', 'VPA', 'DVR']:
    try:
        o.validate_code(bad_code)
        check(f"rejects code {bad_code}", False)
    except ValueError:
        check(f"rejects code {bad_code}", True)
single = {c[:3]: o.derive_code(*c[:3]) for c in combos}
mismatch = [c for c in combos if single[c[:3]] != codes[c]]
check("derived z equals the answered z on exactly the non-mirrored inputs",
      all((single[c[:3]] == codes[c]) ==
          (o.handedness(codes[c]) == o.CONSISTENT_HANDEDNESS) for c in combos),
      f"{len(mismatch)} of 32 differ — the ones whose answers describe a mirror")
check("derived-z codes all have the consistent handedness",
      all(o.handedness(v) == o.CONSISTENT_HANDEDNESS for v in single.values()))

# ---------------------------------------------------------------- 6: per-section records
print("6. per-section records — save_sections, majority, handedness groups")

def section(ant, med, hemi, first='dorsal'):
    return o.make_entry(ant, med, hemi, first)

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / o.ORIENTATION_FILENAME

    # 20 sections mounted alike, one mounted face-down: medial on the other
    # edge, which is a mirror, not a rotation.
    uniform = {n: section('top', 'right', 'left') for n in range(20, 40)}
    flipped = section('top', 'left', 'left')
    entries = dict(uniform); entries[31] = flipped

    o.save_sections(path, 'barseq_subslice', entries, subject='TEST')
    record = json.loads(path.read_text())
    entry = record['modalities']['barseq_subslice']

    check("subject stamped", record.get('subject') == 'TEST')
    check("per_section flag set", entry.get('per_section') is True)
    check("every section stored", entry['n_sections'] == 20, str(entry['n_sections']))
    check("section keys are strings", all(isinstance(k, str) for k in entry['sections']))
    check("modality code is the majority", entry['code'] == 'VPR', entry['code'])
    check("n_agree counts the majority", entry['n_agree_with_code'] == 19,
          str(entry['n_agree_with_code']))

    # The whole point of the modality-level code staying populated: every
    # existing reader, the graph builder included, keeps working unchanged.
    check("codes() still returns one code per modality",
          o.codes(path) == {'barseq_subslice': 'VPR'}, str(o.codes(path)))

    per_section = o.section_codes(path, 'barseq_subslice')
    check("section_codes() keys are ints", all(isinstance(k, int) for k in per_section))
    check("section_codes() finds the deviant", per_section[31] == 'VPL',
          per_section.get(31))
    check("section_codes() on an unassigned modality is empty",
          o.section_codes(path, 'invivo_ref') == {})

    groups = o.handedness_groups(per_section)
    minority = groups[1] if len(groups[1]) < len(groups[-1]) else groups[-1]
    check("handedness_groups isolates the flipped section", minority == [31], str(minority))
    check("groups partition every section",
          len(groups[1]) + len(groups[-1]) == 20)

    # A rotated section shares handedness with its neighbours; only a mirror
    # does not. This is the distinction the report leans on.
    # Rotating (anterior=top, medial=right) a quarter turn clockwise puts
    # anterior on the right and medial on the bottom -- (right, top) would be a
    # reflection, which is the distinction being tested.
    rotated = o.derive_code('right', 'bottom', 'left', 'dorsal')
    check("a 90-degree mounting difference is a different code", rotated != 'VPR', rotated)
    check("...but the same handedness", o.agree(rotated, 'VPR'), rotated)
    check("a face-down mounting is the opposite handedness", not o.agree('VPL', 'VPR'))
    check("and a reflection dressed as a rotation is caught",
          not o.agree(o.derive_code('right', 'top', 'left', 'dorsal'), 'VPR'))

    # Re-saving one section must not drop the rest.
    entries[31] = section('top', 'right', 'left')
    o.save_sections(path, 'barseq_subslice', entries, subject='TEST')
    after = o.section_codes(path, 'barseq_subslice')
    check("re-saving keeps every section", len(after) == 20, str(len(after)))
    check("corrected section takes the new code", after[31] == 'VPR', after[31])
    check("now unanimous", len(set(after.values())) == 1)

    # A second modality must survive a per-section write to the first.
    o.save(path, 'invivo_ref', o.make_entry('top', 'right', 'left', 'dorsal'))
    o.save_sections(path, 'barseq_subslice', entries, subject='TEST')
    check("other modalities untouched by save_sections",
          set(o.codes(path)) == {'barseq_subslice', 'invivo_ref'}, str(o.codes(path)))

    check("sort_sections orders numerically",
          o.sort_sections(['10', '2', '33']) == ['2', '10', '33'],
          str(o.sort_sections(['10', '2', '33'])))
    check("majority_code on an empty map is None", o.majority_code({}) is None)

print()
print("FAILURES:", fails if fails else "none")
sys.exit(1 if fails else 0)
