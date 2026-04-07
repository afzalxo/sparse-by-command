import os
import datetime
import argparse

# Get arguments using argparse
argparser = argparse.ArgumentParser(description="Make script")
argparser.add_argument("--fpga", type=str, default="u50", help="FPGA platform to target")
argparser.add_argument("--target", type=str, default="hw_emu", help="Target platform (hw_emu or hw)")
argparser.add_argument("--target_clk_mhz", type=int, default=300, help="Target clock frequency in MHz")
parsed_args = argparser.parse_args()

cur_dir = os.getcwd()
xf_proj_root = os.path.abspath(os.path.join(cur_dir))

exp_dir_custom_tag = ""
make_what = "xclbin"
target = parsed_args.target
target_clk_mhz = parsed_args.target_clk_mhz
num_kernels = 1
kernel_names = ["vla_accel"]
out_xclbin_name = "vla_accel"

if parsed_args.fpga == "u50":
    platform = "/opt/xilinx/platforms/xilinx_u50_gen3x16_xdma_5_202210_1/xilinx_u50_gen3x16_xdma_5_202210_1.xpfm"
elif parsed_args.fpga == "u55c":
    platform = "/opt/xilinx/platforms/xilinx_u55c_gen3x16_xdma_3_202210_1/xilinx_u55c_gen3x16_xdma_3_202210_1.xpfm"
elif parsed_args.fpga == "u250":
    platform = "/opt/xilinx/platforms/xilinx_u250_gen3x16_xdma_4_1_202210_1/xilinx_u250_gen3x16_xdma_4_1_202210_1.xpfm"
elif parsed_args.fpga == "u280":
    platform = "/opt/xilinx/platforms/xilinx_u280_gen3x16_xdma_1_202211_1/xilinx_u280_gen3x16_xdma_1_202211_1.xpfm"
else:
    print("Invalid FPGA platform specified")
    exit(1)

platform_shortname = platform.split("/")[-1].split(".")[0].split("_")[1]
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

vppflags = ""

build_dir_postfix = ""
temp_dir_postfix = ""

build_dir = "build_dir.{0}.{1}".format(target, platform_name)
if build_dir_postfix:
    build_dir += "." + build_dir_postfix

temp_dir = "_x_temp.{0}.{1}".format(target, platform_name)
if temp_dir_postfix:
    temp_dir += "." + temp_dir_postfix

# cur_date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
cur_date_time = "2025"
experiment_dir = "link_{0}_{1}_".format(
        platform_shortname,
        target
    ) + cur_date_time

if exp_dir_custom_tag:
    experiment_dir += "_" + exp_dir_custom_tag

# make build directory in experiment directory
build_dir = os.path.join(experiment_dir, build_dir)
xos_dir = os.path.join(xf_proj_root, experiment_dir)
temp_dir = os.path.join(experiment_dir, temp_dir)
os.makedirs(build_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

vpp = "v++"

vppflags += "-t {0} --platform {1} --save-temps".format(target, platform)

emconfig_cmd = "emconfigutil --platform {0} --od {1}".format(platform, build_dir)
os.system(emconfig_cmd)

all_vpp_connectivity_flags = ""
vpp_connectivity_flags = ""
vpp_connectivity_flags += f" --connectivity.nk vla_accel_top:1:vla_accel_top_0"

# 2. Memory Connections (--connectivity.sp)
vpp_connectivity_flags += f" --connectivity.sp vla_accel_top_0.m_axi_gmem:HBM[0:2]"
vpp_connectivity_flags += f" --connectivity.sp vla_accel_top_0.m_axi_fo_gmem:HBM[1:2]"
vpp_connectivity_flags += f" --connectivity.sp vla_accel_top_0.m_axi_wgmem:HBM[3]"

# For clarity, print the final generated flags
print(vpp_connectivity_flags)
all_vpp_connectivity_flags += vpp_connectivity_flags

print("Vitis Linker Flags:")
print(vpp_connectivity_flags)

if make_what == "xclbin" or make_what == "all":
    # Link the kernels
    kernel_xos = ""
    kernel_link_report = os.path.join(temp_dir, "link_report")
    for i in range(num_kernels):
        kernel_name = kernel_names[i]
        kernel_xo = os.path.join(xos_dir, kernel_name + ".xo")
        kernel_xos += kernel_xo + " "
        # Adding additional connectivity flags for SLR assignment for testing
        # slr_idx = 0 if i < 2 else 1
        # all_vpp_connectivity_flags += " --connectivity.slr {0}_0:SLR{1}".format(kernel_name, slr_idx)
    rkernel_xos = kernel_xos.split()[::-1]
    kernel_xos = " ".join(rkernel_xos)
    macros = ""
    # Add -g --debug.chipscope vla_accel_top_0 for Chipscope debug
    vpp_cmd_link = "{0} -l -g {1} {2} {3} --vivado.impl.jobs 8 --temp_dir {4} --report_dir {5} --optimize 3 -R 2 --kernel_frequency {6} -o {7} {8}".format(vpp, macros, vppflags, all_vpp_connectivity_flags, temp_dir, kernel_link_report, target_clk_mhz, os.path.join(build_dir, out_xclbin_name + ".xclbin"), kernel_xos)
    print("Linking Kernels:\n{0}".format(vpp_cmd_link))
    os.system(vpp_cmd_link)
