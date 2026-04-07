set kernel_name vla_accel_top
set xo_name vla_accel.xo
set ip_dir ./build/tmp0
set path_to_xo_folder ./build/xo
set path_to_xo ./build/xo/$xo_name

create_project -force kernel_pack ./build/tmp0/tmp_kernel_pack
add_files -norecurse [glob ../srcs/*.sv ../srcs/*.v]

update_compile_order -fileset sources_1
ipx::package_project -root_dir $ip_dir -vendor user.org -library user -taxonomy /UserIP -import_files

ipx::unload_core $path_to_xo_folder/component.xml

set_property ipi_drc {ignore_freq_hz false} [ipx::current_core]
set_property sdx_kernel true [ipx::current_core]
set_property sdx_kernel_type rtl [ipx::current_core]
set_property vitis_drc {ctrl_protocol user_managed} [ipx::current_core]
set_property ipi_drc {ignore_freq_hz true} [ipx::current_core]

ipx::associate_bus_interfaces -busif s_axi_control -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axi_gmem -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axi_wgmem -clock ap_clk [ipx::current_core]
ipx::associate_bus_interfaces -busif m_axi_fo_gmem -clock ap_clk [ipx::current_core]

# Adding registers
ipx::add_register heap_addr [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]
ipx::add_register weights_addr [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]
ipx::add_register buff_a_addr [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]
ipx::add_register buff_b_addr [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]
# CTRL register is required for ap_ctrl_hs control protocol, it needs to be at 0x00 and 32 bits
ipx::add_register CTRL [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]

# Setting properties for registers
set_property address_offset 0x00 [ipx::get_registers CTRL -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property size 32 [ipx::get_registers CTRL -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]

set_property address_offset 0x10 [ipx::get_registers heap_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property size 64 [ipx::get_registers heap_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
ipx::add_register_parameter ASSOCIATED_BUSIF [ipx::get_registers heap_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property value m_axi_gmem [ipx::get_register_parameters ASSOCIATED_BUSIF -of_objects [ipx::get_registers heap_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]]

set_property address_offset 0x18 [ipx::get_registers buff_a_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property size 64 [ipx::get_registers buff_a_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
ipx::add_register_parameter ASSOCIATED_BUSIF [ipx::get_registers buff_a_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property value m_axi_fo_gmem [ipx::get_register_parameters ASSOCIATED_BUSIF -of_objects [ipx::get_registers buff_a_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]]

set_property address_offset 0x20 [ipx::get_registers buff_b_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property size 64 [ipx::get_registers buff_b_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
ipx::add_register_parameter ASSOCIATED_BUSIF [ipx::get_registers buff_b_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property value m_axi_fo_gmem [ipx::get_register_parameters ASSOCIATED_BUSIF -of_objects [ipx::get_registers buff_b_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]]

set_property address_offset 0x28 [ipx::get_registers weights_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property size 64 [ipx::get_registers weights_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
ipx::add_register_parameter ASSOCIATED_BUSIF [ipx::get_registers weights_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]
set_property value m_axi_wgmem [ipx::get_register_parameters ASSOCIATED_BUSIF -of_objects [ipx::get_registers weights_addr -of_objects [ipx::get_address_blocks reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::current_core]]]]]


ipx::add_bus_parameter FREQ_TOLERANCE_HZ [ipx::get_bus_interfaces ap_clk -of_objects [ipx::current_core]]
set_property value -1 [ipx::get_bus_parameters FREQ_TOLERANCE_HZ -of_objects [ipx::get_bus_interfaces ap_clk -of_objects [ipx::current_core]]]

set_property core_revision 3 [ipx::current_core]
ipx::update_source_project_archive -component [ipx::current_core]
ipx::create_xgui_files [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::check_integrity -kernel -xrt [ipx::current_core]

ipx::save_core [ipx::current_core]
package_xo -xo_path $path_to_xo -kernel_name $kernel_name -ip_directory $ip_dir -ctrl_protocol ap_ctrl_hs

ipx::check_integrity -quiet -kernel -xrt [ipx::current_core]
ipx::archive_core $ip_dir/user.org_user_vla_accel_top_1.0.zip [ipx::current_core]

exit
