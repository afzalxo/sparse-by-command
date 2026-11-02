"""Generate per-command and dense model-spec files from the extractor output.

Reads  <data-dir>/model_spec.txt  and  <data-dir>/task_masks.txt  (both produced by
extract_data_for_fpga.py) and writes, into <spec-dir>:
  - spec_cmd{0..5}.txt   (per-command conv tile masks; used by open_loop_tests)
  - model_spec_dense.txt (all tiles active)

The conv-layer weight/bias/golden paths in the spec are rewritten to <data-name>/...
so the host finds them relative to the run directory.
"""
import argparse, os

COMMANDS = ["follow_lane", "turn_left", "turn_right",
            "chlane_left", "chlane_right", "brake"]
OC_PAR = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="fpga_data",
                    help="dir holding model_spec.txt + task_masks.txt (extractor output)")
    ap.add_argument("--spec-dir", default=None,
                    help="output dir for the generated specs (default: <data-dir>/../specs)")
    ap.add_argument("--data-name", default=None,
                    help="path prefix the specs should reference for the .bin files "
                         "(default: basename of --data-dir)")
    args = ap.parse_args()

    data_dir = args.data_dir
    spec_dir = args.spec_dir or os.path.join(os.path.dirname(os.path.abspath(data_dir)), "specs")
    data_name = args.data_name or os.path.basename(os.path.normpath(data_dir))
    os.makedirs(spec_dir, exist_ok=True)

    base_lines = open(os.path.join(data_dir, "model_spec.txt")).readlines()
    # normalize the data path prefix (extractor writes fpga_data_carla/)
    base_lines = [l.replace("fpga_data_carla/", data_name + "/") for l in base_lines]

    task_masks = [list(map(int, l.split())) for l in
                  open(os.path.join(data_dir, "task_masks.txt")) if l.strip()]

    conv_idx = [i for i, l in enumerate(base_lines) if l.startswith("conv_")]
    assert len(conv_idx) == 8, f"Expected 8 conv layers, got {len(conv_idx)}"

    def write_spec(path, masks_row, keep_branch=None):
        # keep_branch=c -> keep only the b{c}_* head lines (drop the other 5 branches);
        # keep_branch=None -> keep all branches (dense spec).
        out_lines = []
        conv_set = set(conv_idx)
        for i, ln in enumerate(base_lines):
            name = ln.split()[0]
            if name[:1] == "b" and name[1:2].isdigit():        # a branch layer bJ_*
                if keep_branch is not None and not name.startswith(f"b{keep_branch}_"):
                    continue
            if i in conv_set:
                p = ln.split()
                p[10], p[11] = str(masks_row[conv_idx.index(i)]), "0"   # lo mask, hi=0
                ln = " ".join(p) + "\n"
            out_lines.append(ln)
        open(path, "w").writelines(out_lines)

    # dense spec (all tiles active, all branches)
    dense = [(1 << ((int(base_lines[li].split()[4]) + OC_PAR - 1) // OC_PAR)) - 1 for li in conv_idx]
    write_spec(os.path.join(spec_dir, "model_spec_dense.txt"), dense)
    print(f"wrote model_spec_dense.txt (dense masks: {dense})")

    # per-command specs (each keeps only its own command branch b{c})
    for c, masks_row in enumerate(task_masks):
        out = os.path.join(spec_dir, f"spec_cmd{c}.txt")
        write_spec(out, masks_row, keep_branch=c)
        print(f"wrote spec_cmd{c}.txt ({COMMANDS[c]}, active tiles/layer: {[bin(m).count('1') for m in masks_row]})")
    print("Done.")


if __name__ == "__main__":
    main()
