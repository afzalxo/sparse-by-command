# Build the open-loop single-image host: main_open_loop.cpp -> app_open_loop.exe
# Requires XILINX_XRT and XILINX_HLS to be set (source the XRT + Vitis settings).
import os

root = os.path.abspath(os.path.dirname(__file__))
for v in ("XILINX_XRT", "XILINX_HLS"):
    if v not in os.environ:
        print("Please set %s (source /opt/xilinx/xrt/setup.sh and the Vitis settings)." % v)
        raise SystemExit(1)

xrt = os.environ["XILINX_XRT"]
hls = os.environ["XILINX_HLS"]
exe = os.path.join(root, "app_open_loop.exe")

srcs = [os.path.join(root, "main_open_loop.cpp"),
        os.path.join(root, "xcl2.cpp"),
        os.path.join(root, "srcs/*")]

cxxflags = ("-fmessage-length=0 -I{0}/include -I{1}/include -std=c++14 -O3 "
            "-Wall -Wno-unknown-pragmas -Wno-unused-label -I {2} -O3").format(xrt, hls, root)
ldflags = ("-pthread -L{0}/lib -L{1}/lnx64/tools/fpo_v7_1 -Wl,--as-needed "
           "-lOpenCL -lxrt_coreutil -lgmp -lmpfr "
           "-lIp_floating_point_v7_1_bitacc_cmodel").format(xrt, hls)

cmd = "g++ -o {0} {1} {2} {3}".format(exe, " ".join(srcs), cxxflags, ldflags)
print("Compiling:\n" + cmd)
rc = os.system(cmd)
if rc == 0:
    print("\nBuilt " + exe)
else:
    print("\nBuild failed (exit %d)." % rc)
    raise SystemExit(1)
