"""
简单测试：Substrate + substrate_replicate 应与 Substrate_replicate222 一致，
且 replicate 仅作用于首轮 model.xyz，后续轮次不再放大。
"""
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "SingleRunInputScript"))
from utility import parse_substrate_replicate, prepare_initial_model_xyz, replicate_xy  # noqa: E402

SUBSTRATE = ROOT / "Substrate"
REFERENCE = ROOT / "Substrate_replicate222"


def parse_lattice_box(header_line):
    m = re.search(r'Lattice="([^"]*)"', header_line)
    vals = list(map(float, m.group(1).split()))
    return vals[0], vals[4], vals[8]


def load_xyz_summary(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    n_atoms = int(lines[0].strip())
    box_x, box_y, box_z = parse_lattice_box(lines[1])
    atom_lines = [ln for ln in lines[2:] if ln.strip()]
    return {
        "n_atoms": n_atoms,
        "n_lines": len(atom_lines),
        "box": (box_x, box_y, box_z),
        "header": lines[1],
        "first_atom": atom_lines[0] if atom_lines else "",
    }


def test_parse_substrate_replicate():
    assert parse_substrate_replicate(None) == (1, 1)
    assert parse_substrate_replicate(1) == (1, 1)
    assert parse_substrate_replicate(2) == (2, 2)
    assert parse_substrate_replicate([2, 3]) == (2, 3)
    print("  parse_substrate_replicate: OK")


def test_replicate_matches_reference():
    for name in ("100.xyz", "110.xyz", "111.xyz"):
        src = SUBSTRATE / name
        ref = REFERENCE / name
        assert src.is_file(), f"缺少 {src}"
        assert ref.is_file(), f"缺少参考 {ref}"

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / name
            info = replicate_xy(str(src), str(out), nx=2, ny=2)
            got = load_xyz_summary(out)
            expected = load_xyz_summary(ref)

            assert info["n_out"] == expected["n_atoms"], name
            assert got["n_atoms"] == expected["n_atoms"], name
            for i, (a, b) in enumerate(zip(got["box"], expected["box"])):
                assert abs(a - b) < 1e-3, f"{name} box[{i}] {a} != {b}"
            assert got["first_atom"] == expected["first_atom"], name
        print(f"  replicate 2x2 {name}: {info['n_in']} -> {info['n_out']} atoms, box OK")


def test_prepare_initial_model_xyz_no_double_replicate():
    src = SUBSTRATE / "100.xyz"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        out1 = tmp / "round1" / "model.xyz"
        out1.parent.mkdir()
        info1 = prepare_initial_model_xyz(str(src), str(out1), substrate_replicate=2)
        s1 = load_xyz_summary(out1)

        # 模拟第二轮：基于已有 model.xyz 直接 copy（replicate=1 或不设）
        out2 = tmp / "round2" / "model.xyz"
        out2.parent.mkdir()
        info2 = prepare_initial_model_xyz(str(out1), str(out2), substrate_replicate=None)
        s2 = load_xyz_summary(out2)

        assert info1["replicated"] is True
        assert info2["replicated"] is False
        assert s1["n_atoms"] == s2["n_atoms"]
        assert s1["box"] == s2["box"]
        print(
            f"  no double replicate: round1={s1['n_atoms']} atoms, "
            f"round2 unchanged={s2['n_atoms']} atoms"
        )


def main():
    print("test_substrate_replicate")
    test_parse_substrate_replicate()
    test_replicate_matches_reference()
    test_prepare_initial_model_xyz_no_double_replicate()
    print("ALL PASSED")


if __name__ == "__main__":
    main()
