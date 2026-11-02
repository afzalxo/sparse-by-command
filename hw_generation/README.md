# hw_generation — RTL + bitstream build

SystemVerilog RTL for the tiled INT8 CNN accelerator and the scripts that package
it into a Xilinx `.xo` and link it into a U50 `.xclbin`.

```
hw_generation/
├── rtl/                 22 .sv/.v source files (the design in the .xclbin)
├── build_scripts/
│   ├── build_xo_vla.sh      RTL -> vla_accel.xo   (Vivado, gen_xo_vla.tcl)
│   ├── gen_xo_vla.tcl       Vivado packaging TCL
│   ├── link_xo_vla_accel.py .xo -> .xclbin        (Vitis v++, --config opts.cfg)
│   ├── gen_xclbin.sh        one-shot wrapper (xo + link)
│   └── opts.cfg             P&R directives (OPT/PHYS_OPT/ROUTE = Explore, etc.)
├── build_bitstream.sh   convenience wrapper that emits Table-3 reports
└── reports/             utilization + timing summary from a build
```

## Design

- Tiled INT8 conv accelerator: `OC_PAR = IC_PAR = 16`, `PP_PAR = 8`
  → 2048 INT8 MACs/cycle. HBM masters: `m_axi_gmem`, `m_axi_fo_gmem`, `m_axi_wgmem`.
- Top: `vla_accel_top.sv`. Orchestration: `tile_manager.sv`. Compute:
  `conv_accelerator → vector_compute_unit → processing_element → mac_lane`.
- **Sparse-by-command**: `tile_manager.sv` skips masked oc/ic tiles; the
  `S_ZERO_MASKED` pass zeroes masked oc-tile output slots (via `output_dma.sv`'s
  `zero_fill`) so no stale data reaches the fusion.

## Build (needs Vivado + Vitis 2024.2, XRT, a U50 platform; ~2–3 h)

```bash
cd build_scripts
source /path/to/Vitis/2024.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
bash gen_xclbin.sh        # -> link_u50_hw_2025/.../vla_accel.xclbin
```

The linker passes `--config opts.cfg`; the kernel closes at **261.7 MHz**
(auto-scaled from a 300 MHz target). Copy the resulting `.xclbin` over
`../open_loop_tests/vla_accel.xclbin` to test it.

> Note: `link_xo_vla_accel.py` and `gen_xo_vla.tcl` contain site-specific paths
> (platform, project root) that you may need to adjust for your machine.
