# This needs to do 3 things:
# 1. Be able to make just the host
# 2. Be able to make individual xo's
# 3. Be able to build xclbin from the xo's

import os
import datetime

cur_dir = os.getcwd()
xf_proj_root = os.path.abspath(os.path.join(cur_dir))

make_what = "host_api"
target = "hw"
target_clk_mhz = 300
num_kernels = 1

platform="/opt/xilinx/platforms/xilinx_u50_gen3x16_xdma_5_202210_1/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm"
platform_shortname = platform.split("/")[-1].split(".")[0].split("_")[1]
num_hls_jobs = 8
platform_name = platform.split("/")[-1].split(".")[0]

# Check if XILINX_XRT environment variable is set
if "XILINX_XRT" not in os.environ:
    print("Please set the XILINX_XRT environment variable to the XRT installation path")
    exit(1)

if "XILINX_HLS" not in os.environ:
    print("Please set the XILINX_HLS environment variable to the HLS installation path")
    exit(1)

if "XILINX_VITIS" not in os.environ:
    print("Please set the XILINX_VITIS environment variable to the Vitis installation path")
    exit(1)

if "XILINX_VIVADO" not in os.environ:
    print("Please set the XILINX_VIVADO environment variable to the Vivado installation path")
    exit(1)

xilinx_xrt = os.environ["XILINX_XRT"]
xilinx_hls = os.environ["XILINX_HLS"]
xilinx_vitis = os.environ["XILINX_VITIS"]
xilinx_vivado = os.environ["XILINX_VIVADO"]

# Table 2 uses the single-inference host built from main_carla.cpp.
host_executable = "app_single.exe"

cxxflags = ""
ldflags = ""

gcc = "g++"

# Single-inference host (main_carla.cpp) + OpenCL wrapper + srcs.
host_srcs = [os.path.join(xf_proj_root, "main_carla.cpp"),
             os.path.join(xf_proj_root, "xcl2.cpp"),
             os.path.join(xf_proj_root, "srcs/*")]

cxxflags += " -fmessage-length=0 -I{0}/include -I{1}/include -std=c++14 -O3 -Wall -Wno-unknown-pragmas -Wno-unused-label".format(xilinx_xrt, xilinx_hls)
cxxflags += " -I {0} -O3".format(xf_proj_root)

ldflags += "-pthread -L{0}/lib -L{1}/lnx64/tools/fpo_v7_1 -Wl,--as-needed -lOpenCL -lxrt_coreutil -lgmp -lmpfr -lIp_floating_point_v7_1_bitacc_cmodel".format(xilinx_xrt, xilinx_hls)

macros = ""

# Compile the host binary into ./app_single.exe (in the host/ directory).
compile_cmd = "{0} -o {1} {2} {3} {4} {5}".format(
    gcc, os.path.join(xf_proj_root, host_executable),
    " ".join(host_srcs), macros, cxxflags, ldflags)
print("Compiling host binary:\n{0}".format(compile_cmd))
rc = os.system(compile_cmd)
if rc == 0:
    print("\nBuilt " + os.path.join(xf_proj_root, host_executable))
else:
    print("\nHost build failed (exit %d)." % rc)
    exit(1)
