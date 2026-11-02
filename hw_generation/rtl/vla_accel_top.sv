`default_nettype none

module vla_accel_top #(
    parameter IC_PAR = 16,
    parameter OC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8,

    parameter MAX_IMG_WIDTH = 256,
    parameter TILE_HEIGHT = 8,
    
    parameter C_M_AXI_GMEM_ADDR_WIDTH = 64,
    parameter C_M_AXI_GMEM_DATA_WIDTH = 512,

    parameter C_S_AXI_ADDR_WIDTH = 8,
    parameter C_S_AXI_DATA_WIDTH = 32

) (
    input  wire                                         ap_clk,
    input  wire                                         ap_rst_n,
    input  wire  [C_S_AXI_ADDR_WIDTH-1:0]            	s_axi_control_awaddr,
    input  wire                                         s_axi_control_awvalid,
    output wire                                         s_axi_control_awready,
    input  wire  [C_S_AXI_DATA_WIDTH-1:0]             	s_axi_control_wdata,
    input  wire  [C_S_AXI_DATA_WIDTH/8-1:0]           	s_axi_control_wstrb,
    input  wire                                         s_axi_control_wvalid,
    output wire                                         s_axi_control_wready,
    output wire  [1:0]                                	s_axi_control_bresp,
    output wire                                         s_axi_control_bvalid,
    input  wire                                         s_axi_control_bready,
    input  wire  [C_S_AXI_ADDR_WIDTH-1:0]             	s_axi_control_araddr,
    input  wire                                         s_axi_control_arvalid,
    output wire                                         s_axi_control_arready,
    output wire  [C_S_AXI_DATA_WIDTH-1:0]             	s_axi_control_rdata,
    output wire  [1:0]                                	s_axi_control_rresp,
    output wire                                         s_axi_control_rvalid,
    input  wire                                         s_axi_control_rready,

    // AXI Master Interface for Global Memory Reads for Feature Map (and
    // Instructions)
    output wire       					m_axi_gmem_arvalid,
    input  wire                                         m_axi_gmem_arready,
    output wire  [C_M_AXI_GMEM_ADDR_WIDTH-1:0]          m_axi_gmem_araddr,
    output wire  [7:0]                                  m_axi_gmem_arlen,
    output wire  [2:0]                                  m_axi_gmem_arsize,
    output wire  [1:0]                                  m_axi_gmem_arburst,
    input  wire                                         m_axi_gmem_rvalid,
    output wire                                         m_axi_gmem_rready,
    input  wire  [C_M_AXI_GMEM_DATA_WIDTH-1:0]          m_axi_gmem_rdata,
    input  wire       					m_axi_gmem_rlast,

    // AXI Master Interface for Global Memory Reads for Weights
    output wire       					m_axi_wgmem_arvalid,
    input  wire                                         m_axi_wgmem_arready,
    output wire  [C_M_AXI_GMEM_ADDR_WIDTH-1:0]          m_axi_wgmem_araddr,
    output wire  [7:0]                                  m_axi_wgmem_arlen,
    output wire  [2:0]                                  m_axi_wgmem_arsize,
    output wire  [1:0]                                  m_axi_wgmem_arburst,
    input  wire                                         m_axi_wgmem_rvalid,
    output wire                                         m_axi_wgmem_rready,
    input  wire  [C_M_AXI_GMEM_DATA_WIDTH-1:0]          m_axi_wgmem_rdata,
    input  wire       					m_axi_wgmem_rlast,

    // AXI Master Interface for Global Memory Writes for Output Feature Map
    output wire       					m_axi_fo_gmem_awvalid,
    input  wire                                         m_axi_fo_gmem_awready,
    output wire  [C_M_AXI_GMEM_ADDR_WIDTH-1:0]          m_axi_fo_gmem_awaddr,
    output wire  [7:0]                                  m_axi_fo_gmem_awlen,
    output wire  [2:0]                                  m_axi_fo_gmem_awsize,
    output wire  [1:0]                                  m_axi_fo_gmem_awburst,
    output wire                                         m_axi_fo_gmem_wvalid,
    input  wire                                         m_axi_fo_gmem_wready,
    output wire  [C_M_AXI_GMEM_DATA_WIDTH-1:0]          m_axi_fo_gmem_wdata,
    output wire  [C_M_AXI_GMEM_DATA_WIDTH/8-1:0]        m_axi_fo_gmem_wstrb,
    output wire                                         m_axi_fo_gmem_wlast,
    input  wire  [1:0]                                	m_axi_fo_gmem_bresp,
    input  wire                                         m_axi_fo_gmem_bvalid,
    output wire                                         m_axi_fo_gmem_bready
); 

    // Start, ready and done signals
    wire start_process_internal; // Pulse from controller to start core logic
    wire process_done_internal; // Signal from core logic to controller
    reg ap_idle_reg = 1'b1;
    reg ap_done_reg;
    reg ap_ready_reg;
    wire ap_idle = ap_idle_reg;
    wire ap_ready = ap_ready_reg;
    always @(posedge ap_clk or negedge ap_rst_n) begin
	if (!ap_rst_n) begin
            ap_idle_reg <= 1'b1;
	    ap_done_reg <= 1'b0;
	    ap_ready_reg <= 1'b0;
	end else begin
	    if (start_process_internal) begin
		ap_idle_reg <= 1'b0;
	    end else if (process_done_internal) begin
		ap_idle_reg <= 1'b1;
	    end
	    ap_ready_reg <= process_done_internal;
	    ap_done_reg <= process_done_internal;
	end
    end

    wire        sched_arvalid;
    wire [63:0] sched_araddr;  // Offset from Scheduler
    wire [7:0]  sched_arlen;
    wire [2:0]  sched_arsize;
    wire [1:0]  sched_arburst;
    wire        sched_rready;

    // --- Internal AXI Signals for Feature Map Reader ---
    // (Ensure your burst_axi_read_master connects to these, not top-level directly)
    wire        feat_arvalid;
    wire [63:0] feat_araddr;   // Physical address from Tile Manager logic
    wire [7:0]  feat_arlen;
    wire [2:0]  feat_arsize;
    wire [1:0]  feat_arburst;
    wire        feat_rready;

    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] heap_base_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_input_addr_w;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_reg_a_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_reg_b_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_foutput_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_weight_input_addr_base;
    wire [1:0] cfg_input_bank, cfg_output_bank;

    wire [63:0] cfg_input_offset;
    wire [63:0] cfg_output_offset;
    wire [63:0] cfg_weight_offset;
    wire [63:0] cfg_bias_offset;

    wire [15:0] img_width, img_height, img_channels;
    wire [15:0] out_channels;
    wire [4:0] quant_shift;
    wire is_conv;
    wire is_memcpy;
    wire is_gap;
    wire bias_en;

    wire [63:0] base_hbm0 = heap_base_addr;
    wire [63:0] base_hbm1 = hbm_reg_a_addr;
    wire [63:0] base_hbm2 = hbm_reg_b_addr;
    wire [63:0] base_hbm3 = hbm_weight_input_addr_base;

    wire [1:0] wb_dma_active;

    wire [63:0] current_input_base  = (cfg_input_bank  == 2'b00) ? base_hbm0 :
	    			      (cfg_input_bank  == 2'b01) ? base_hbm1 :
				      (cfg_input_bank  == 2'b10) ? base_hbm2 : base_hbm0;
    wire [63:0] current_output_base = (cfg_output_bank == 2'b00) ? base_hbm0 : 
	    			      (cfg_output_bank == 2'b01) ? base_hbm1 :
				      (cfg_output_bank == 2'b10) ? base_hbm2 : base_hbm0;

    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_vec_read_addr_base = current_input_base + cfg_input_offset;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_vec_read_addr;

    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_fm_read_addr = current_input_base + cfg_input_offset + (hbm_input_addr_w << 6);  // * 64 since data width is 512 bits = 64 bytes per burst

    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_winput_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_binput_addr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_read_b_addr = hbm_weight_input_addr_base + cfg_bias_offset + (hbm_binput_addr << 6);
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_read_w_addr = hbm_weight_input_addr_base + cfg_weight_offset + (hbm_winput_addr << 6);
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_read_wb_addr = (wb_dma_active == 2'b01) ? hbm_read_w_addr :
	    						  (wb_dma_active == 2'b10) ? hbm_read_b_addr : 64'd0;

    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] hbm_write_f_addr = current_output_base + cfg_output_offset + (hbm_foutput_addr << 6);

    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] memcpy_awaddr;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] memcpy_awaddr_base = current_output_base + cfg_output_offset;

    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] gap_src_addr_base = current_input_base + cfg_input_offset;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] gap_araddr;

    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] gap_dst_addr_base = current_output_base + cfg_output_offset;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] gap_awaddr;

    vla_accel_control_s_axi #(
        .C_S_AXI_ADDR_WIDTH(C_S_AXI_ADDR_WIDTH),
        .C_S_AXI_DATA_WIDTH(C_S_AXI_DATA_WIDTH)
    ) axi_controller_inst (
        .ACLK(ap_clk),
        .ARESET(ap_rst_n), // Provide active high reset
        .ACLK_EN(1'b1), // Assuming clock enable is always high
        .AWADDR(s_axi_control_awaddr), 
	.AWVALID(s_axi_control_awvalid),
	.AWREADY(s_axi_control_awready),
        .WDATA(s_axi_control_wdata), 
	.WSTRB(s_axi_control_wstrb), 
	.WVALID(s_axi_control_wvalid), 
	.WREADY(s_axi_control_wready),
        .BRESP(s_axi_control_bresp),
	.BVALID(s_axi_control_bvalid),
	.BREADY(s_axi_control_bready),
        .ARADDR(s_axi_control_araddr),
	.ARVALID(s_axi_control_arvalid),
	.ARREADY(s_axi_control_arready),
        .RDATA(s_axi_control_rdata),
	.RRESP(s_axi_control_rresp),
	.RVALID(s_axi_control_rvalid),
	.RREADY(s_axi_control_rready),
	.reg_heap_addr(heap_base_addr),
	.reg_buff_a_addr(hbm_reg_a_addr),
	.reg_buff_b_addr(hbm_reg_b_addr),
	.reg_weight_input_addr(hbm_weight_input_addr_base),
        .user_start(start_process_internal), // Controller generates start pulse
        .user_done(process_done_internal),     // Core logic signals completion
	.user_idle(ap_idle),
	.user_ready(ap_ready)
    );

    wire read_master_data_valid;
    wire read_master_rready;
    wire hbm_read_fm_start;
    wire hbm_load_done;
    wire [C_M_AXI_GMEM_DATA_WIDTH-1:0] read_master_data_out;
    wire [15:0] feature_map_words;
    wire feat_data_ready;

    burst_axi_read_master #(
	.AXI_ADDR_WIDTH(C_M_AXI_GMEM_ADDR_WIDTH),
	.AXI_DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH)
    ) hbm_feat_read_master_inst (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .start(hbm_read_fm_start),
	// .cmd_ready(cmd_ready),
	// .data_ready(feat_data_ready),
	.base_address(hbm_fm_read_addr),
        .m_axi_arvalid(feat_arvalid),
        .m_axi_arready(m_axi_gmem_arready),
        .m_axi_araddr(feat_araddr),
        .m_axi_arlen(feat_arlen),
        .m_axi_arsize(feat_arsize),
	.m_axi_arburst(feat_arburst),
        .m_axi_rvalid(m_axi_gmem_rvalid),
        .m_axi_rready(feat_rready),
        .m_axi_rdata(m_axi_gmem_rdata),
        .m_axi_rlast(m_axi_gmem_rlast),

	.total_words_to_read(feature_map_words),
        .data_valid(read_master_data_valid),
	.data_ready(read_master_rready),
        .data_out(read_master_data_out),
        .done(hbm_load_done)
    );

    wire memcpy_done;
    wire memcpy_arvalid;
    wire [31:0] memcpy_length_bytes = {16'd0, out_channels};
    wire [7:0]  memcpy_arlen;
    wire [2:0]  memcpy_arsize;
    wire [1:0]  memcpy_arburst;
    wire        memcpy_rready;

    wire memcpy_awvalid, memcpy_wvalid;
    wire [7:0]  memcpy_awlen;
    wire [2:0]  memcpy_awsize;
    wire [1:0]  memcpy_awburst;
    wire        memcpy_wready;
    wire [C_M_AXI_GMEM_DATA_WIDTH-1:0] memcpy_wdata;
    wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] memcpy_wstrb;
    wire        memcpy_wlast;
    wire        memcpy_bready;

    memcpy_unit #(
	.ADDR_WIDTH(C_M_AXI_GMEM_ADDR_WIDTH),
	.DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH)
    ) memcpy_unit_inst (
	.clk(ap_clk),
	.rst_n(ap_rst_n),
	.start(memcpy_start),

	.src_addr_base(hbm_vec_read_addr_base),
	.dst_addr_base(memcpy_awaddr_base),
	.length_bytes(memcpy_length_bytes),

	.m_axi_arvalid(memcpy_arvalid),
	.m_axi_arready(m_axi_gmem_arready),
	.m_axi_araddr(hbm_vec_read_addr),
	.m_axi_arlen(memcpy_arlen),
	.m_axi_arsize(memcpy_arsize),
	.m_axi_arburst(memcpy_arburst),
	.m_axi_rvalid(m_axi_gmem_rvalid),
	.m_axi_rready(memcpy_rready),
	.m_axi_rdata(m_axi_gmem_rdata),

	.m_axi_awvalid(memcpy_awvalid),
	.m_axi_awready(m_axi_fo_gmem_awready),
	.m_axi_awaddr(memcpy_awaddr),
	.m_axi_awlen(memcpy_awlen),
	.m_axi_awsize(memcpy_awsize),
	.m_axi_awburst(memcpy_awburst),
	.m_axi_wvalid(memcpy_wvalid),
	.m_axi_wready(m_axi_fo_gmem_wready),
	.m_axi_wdata(memcpy_wdata),
	.m_axi_wstrb(memcpy_wstrb),
	.m_axi_wlast(memcpy_wlast),
	.m_axi_bvalid(m_axi_fo_gmem_bvalid),
	.m_axi_bready(memcpy_bready),
	.m_axi_bresp(m_axi_fo_gmem_bresp),
	.done(memcpy_done)
    );

    wire gap_done;
    wire gap_arvalid;
    wire [7:0]  gap_arlen;
    wire [2:0]  gap_arsize;
    wire [1:0]  gap_arburst;
    wire        gap_rready;

    wire gap_awvalid, gap_wvalid;
    wire [7:0]  gap_awlen;
    wire [2:0]  gap_awsize;
    wire [1:0]  gap_awburst;
    wire        gap_wready;
    wire [C_M_AXI_GMEM_DATA_WIDTH-1:0] gap_wdata;
    wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] gap_wstrb;
    wire        gap_wlast;
    wire        gap_bready;

    gap_unit #(
	.IC_PAR(IC_PAR),
	.PP_PAR(PP_PAR),
	.HBM_DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH),
        .ADDR_WIDTH(C_M_AXI_GMEM_ADDR_WIDTH),
	.TILE_HEIGHT(TILE_HEIGHT)
    ) gap_unit_inst (
	.clk(ap_clk),
	.rst_n(ap_rst_n),
	.start(gap_start),
	.src_addr_base(gap_src_addr_base),
	.dst_addr_base(gap_dst_addr_base),
	.img_width(img_width),
	.img_height(img_height),
	.img_channels(img_channels),
	.log2_mem_tile_height(log2_mem_tile_height),
	.quant_shift(quant_shift),
	.m_axi_arvalid(gap_arvalid),
	.m_axi_arready(m_axi_gmem_arready),
	.m_axi_araddr(gap_araddr),
	.m_axi_arlen(gap_arlen),
	.m_axi_arsize(gap_arsize),
	.m_axi_arburst(gap_arburst),
	.m_axi_rvalid(m_axi_gmem_rvalid),
	.m_axi_rready(gap_rready),
	.m_axi_rdata(m_axi_gmem_rdata),

	.m_axi_awvalid(gap_awvalid),
	.m_axi_awready(m_axi_fo_gmem_awready),
	.m_axi_awaddr(gap_awaddr),
	.m_axi_awlen(gap_awlen),
	.m_axi_awsize(gap_awsize),
	.m_axi_awburst(gap_awburst),
	.m_axi_wvalid(gap_wvalid),
	.m_axi_wready(m_axi_fo_gmem_wready),
	.m_axi_wdata(gap_wdata),
	.m_axi_wstrb(gap_wstrb),
	.m_axi_wlast(gap_wlast),
	.m_axi_bvalid(m_axi_fo_gmem_bvalid),
	.m_axi_bready(gap_bready),
	.m_axi_bresp(m_axi_fo_gmem_bresp),
	.done(gap_done)
    );

    wire read_master_wb_data_valid;
    wire hbm_wread_start;
    wire hbm_bread_start;
    wire hbm_wb_read_start = (wb_dma_active == 2'b01) ? hbm_wread_start :
	    		     (wb_dma_active == 2'b10) ? hbm_bread_start : 1'b0;
    wire hbm_wb_load_done;
    wire hbm_fwrite_start;
    wire [C_M_AXI_GMEM_DATA_WIDTH-1:0] read_master_wb_data_out;
    wire [15:0] weight_words;
    wire [15:0] bias_words;
    wire [15:0] wb_words = (wb_dma_active == 2'b01) ? weight_words :
	    	    	   (wb_dma_active == 2'b10) ? bias_words : 16'd0;

    wire cmd_ready;

    burst_axi_read_master #(
	.AXI_ADDR_WIDTH(C_M_AXI_GMEM_ADDR_WIDTH),
	.AXI_DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH)
    ) hbm_weights_read_master_inst (
        .clk(ap_clk),
        .rst_n(ap_rst_n),
        .start(hbm_wb_read_start),
	// .cmd_ready(),
	// .data_ready(1'b1),
	.base_address(hbm_read_wb_addr),
        .m_axi_arvalid(m_axi_wgmem_arvalid),
        .m_axi_arready(m_axi_wgmem_arready),
        .m_axi_araddr(m_axi_wgmem_araddr),
        .m_axi_arlen(m_axi_wgmem_arlen),
        .m_axi_arsize(m_axi_wgmem_arsize),
	.m_axi_arburst(m_axi_wgmem_arburst),
        .m_axi_rvalid(m_axi_wgmem_rvalid),
        .m_axi_rready(m_axi_wgmem_rready),
        .m_axi_rdata(m_axi_wgmem_rdata),
        .m_axi_rlast(m_axi_wgmem_rlast),
	.total_words_to_read(wb_words),
        .data_valid(read_master_wb_data_valid),
	.data_ready(1'b1),
        .data_out(read_master_wb_data_out),
        .done(hbm_wb_load_done)
    );

    wire [15:0] output_feature_map_words;
    wire [C_M_AXI_GMEM_DATA_WIDTH-1:0] write_master_fmap_data_in;
    wire write_master_fmap_data_valid;
    wire write_master_fmap_data_ready;
    wire hbm_fwrite_done;

    wire fm_output_awvalid;
    wire [C_M_AXI_GMEM_ADDR_WIDTH-1:0] fm_output_awaddr;
    wire [7:0]  fm_output_awlen;
    wire [2:0]  fm_output_awsize;
    wire [1:0]  fm_output_awburst;
    wire fm_output_wvalid;
    wire [C_M_AXI_GMEM_DATA_WIDTH-1:0] fm_output_wdata;
    wire [C_M_AXI_GMEM_DATA_WIDTH/8-1:0] fm_output_wstrb;
    wire fm_output_wlast;
    wire fm_output_bready;

    /*
    assign m_axi_fo_gmem_awvalid = fm_output_awvalid | memcpy_awvalid | gap_awvalid;
    assign m_axi_fo_gmem_awaddr  = memcpy_awvalid ? memcpy_awaddr : 
	    			   gap_awvalid    ? gap_awaddr    : 
				   fm_output_awaddr;
    assign m_axi_fo_gmem_awlen   = memcpy_awvalid ? memcpy_awlen  : 
	    			   gap_awvalid    ? gap_awlen     :
				   fm_output_awlen;
    assign m_axi_fo_gmem_awsize  = memcpy_awvalid ? memcpy_awsize : 
	    			   gap_awvalid    ? gap_awsize    : 
				   fm_output_awsize;
    assign m_axi_fo_gmem_awburst = memcpy_awvalid ? memcpy_awburst : 
	    			   gap_awvalid    ? gap_awburst    :
				   fm_output_awburst;

    assign m_axi_fo_gmem_wvalid  = fm_output_wvalid | memcpy_wvalid | gap_wvalid;
    assign m_axi_fo_gmem_wdata   = memcpy_wvalid ? memcpy_wdata : 
	    			   gap_wvalid    ? gap_wdata    :
				   fm_output_wdata;
    assign m_axi_fo_gmem_wstrb   = memcpy_wvalid ? memcpy_wstrb : 
	    			   gap_wvalid    ? gap_wstrb    :
				   fm_output_wstrb;
    assign m_axi_fo_gmem_wlast   = memcpy_wvalid ? memcpy_wlast :
	    			   gap_wvalid    ? gap_wlast    :
				   fm_output_wlast;
    assign m_axi_fo_gmem_bready  = fm_output_bready | memcpy_bready | gap_bready;
    */

    burst_axi_write_master #(
	.AXI_ADDR_WIDTH(C_M_AXI_GMEM_ADDR_WIDTH),
	.AXI_DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH)
    ) hbm_output_wr_master_inst (
	.clk(ap_clk),
	.rst_n(ap_rst_n),
	.start(hbm_fwrite_start),
	.base_address(hbm_write_f_addr),
	.total_words_to_write(output_feature_map_words),
	.data_in(write_master_fmap_data_in),
	.data_valid(write_master_fmap_data_valid),
	.data_ready(write_master_fmap_data_ready),
	.m_axi_awvalid(fm_output_awvalid),
	.m_axi_awready(m_axi_fo_gmem_awready),
	.m_axi_awaddr(fm_output_awaddr),
	.m_axi_awlen(fm_output_awlen),
	.m_axi_awsize(fm_output_awsize),
	.m_axi_awburst(fm_output_awburst),
	.m_axi_wvalid(fm_output_wvalid),
	.m_axi_wready(m_axi_fo_gmem_wready),
	.m_axi_wdata(fm_output_wdata),
	.m_axi_wstrb(fm_output_wstrb),
	.m_axi_wlast(fm_output_wlast),
	.m_axi_bvalid(m_axi_fo_gmem_bvalid),
	.m_axi_bready(fm_output_bready),
	.m_axi_bresp(m_axi_fo_gmem_bresp),
	.done(hbm_fwrite_done)
    );

    wire exec_start;
    wire tm_start = (is_memcpy | is_gap) ? 0 : exec_start;
    wire tm_done;
    wire memcpy_start = is_memcpy ? exec_start : 0;
    wire gap_start = is_gap ? exec_start : 0;
    wire exec_done  = is_memcpy ? memcpy_done  : 
	    	      is_gap    ? gap_done     :
		      tm_done;
    wire [63:0] sched_fetch_addr = heap_base_addr + sched_araddr;

    assign m_axi_gmem_arvalid = sched_arvalid | feat_arvalid | memcpy_arvalid | gap_arvalid;
    assign m_axi_gmem_araddr = sched_arvalid ? sched_fetch_addr :
	    		       feat_arvalid ? feat_araddr :  //hbm_fm_read_addr :
			       gap_arvalid ? gap_araddr :
			       hbm_vec_read_addr;
    assign m_axi_gmem_arlen   = sched_arvalid ? sched_arlen  : 
	    			feat_arvalid ? feat_arlen :
				gap_arvalid ? gap_arlen :
				memcpy_arlen;
    assign m_axi_gmem_arsize  = sched_arvalid ? sched_arsize : 
	    			feat_arvalid ? feat_arsize :
				gap_arvalid ? gap_arsize :
				memcpy_arsize;
    assign m_axi_gmem_arburst = sched_arvalid ? sched_arburst :
	    			feat_arvalid ? feat_arburst :
				gap_arvalid ? gap_arburst :
				memcpy_arburst;
    assign m_axi_gmem_rready  = sched_rready | feat_rready | memcpy_rready | gap_rready;

    wire relu_en;
    wire [1:0] stride;
    wire [2:0] log2_mem_tile_height;
    wire is_sparse;
    wire [127:0] ic_tile_mask;
    wire [127:0] oc_tile_mask;
    wire flatten;

//////////////
    reg[31:0] total_cycles;
    reg [31:0] compute_cycles;
    reg[31:0] memory_cycles;
    reg[31:0] stall_cycles;       // Neither compute nor memory active
    reg[31:0] overlap_cycles;     // Both compute AND memory active
    reg[31:0] output_dma_cycles;  // Output write channel active
    reg[31:0] mem_stall_cycles;   // DMA busy but no AXI signals (waiting for HBM response)

    wire mem_active = m_axi_gmem_arvalid | m_axi_gmem_rvalid |
                      m_axi_wgmem_arvalid | m_axi_wgmem_rvalid |
                      m_axi_fo_gmem_awvalid | m_axi_fo_gmem_wvalid;

    wire output_write_active = m_axi_fo_gmem_awvalid | m_axi_fo_gmem_wvalid;

    // DMA modules are busy (started but not done)
    wire feat_dma_busy = hbm_feat_read_master_inst.busy;
    wire wt_dma_busy   = hbm_weights_read_master_inst.busy;
    wire any_dma_busy  = feat_dma_busy | wt_dma_busy;
    // AXI read channel signals actually active
    wire mem_read_active = m_axi_gmem_arvalid | m_axi_gmem_rvalid |
                           m_axi_wgmem_arvalid | m_axi_wgmem_rvalid;

    wire acc_start_top = tile_manager_inst.acc_start;
    wire acc_done_top  = tile_manager_inst.acc_done;
    reg acc_active;

    always @(posedge ap_clk or negedge ap_rst_n) begin
        if (!ap_rst_n) acc_active <= 0;
        else if (acc_start_top) acc_active <= 1;
        else if (acc_done_top) acc_active <= 0;
    end

    always @(posedge ap_clk or negedge ap_rst_n) begin
        if (!ap_rst_n) begin
            total_cycles <= 0;
            compute_cycles <= 0;
            memory_cycles <= 0;
            stall_cycles <= 0;
            overlap_cycles <= 0;
            output_dma_cycles <= 0;
            mem_stall_cycles <= 0;
        end else begin
            if (start_process_internal) begin
                total_cycles <= 0;
                compute_cycles <= 0;
                memory_cycles <= 0;
                stall_cycles <= 0;
                overlap_cycles <= 0;
                output_dma_cycles <= 0;
                mem_stall_cycles <= 0;
            end else if (!ap_idle) begin
                total_cycles <= total_cycles + 1;
                if (acc_active) compute_cycles <= compute_cycles + 1;
                if (mem_active) memory_cycles <= memory_cycles + 1;
                if (!acc_active && !mem_active) stall_cycles <= stall_cycles + 1;
                if (acc_active && mem_active) overlap_cycles <= overlap_cycles + 1;
                if (output_write_active) output_dma_cycles <= output_dma_cycles + 1;
                if (any_dma_busy && !mem_read_active) mem_stall_cycles <= mem_stall_cycles + 1;
            end
        end
    end

    // --- FSM to dump performance counters at end of processing ---
    localparam PERF_IDLE = 0, PERF_AW = 1, PERF_W = 2, PERF_B = 3;
    reg [1:0] perf_state;
    reg perf_awvalid_reg;
    reg perf_wvalid_reg;
    reg perf_bready_reg;
    reg top_done;
    wire sched_done;
    
    always @(posedge ap_clk or negedge ap_rst_n) begin
        if (!ap_rst_n) begin
            perf_state <= PERF_IDLE;
            perf_awvalid_reg <= 0;
            perf_wvalid_reg <= 0;
            perf_bready_reg <= 0;
            top_done <= 0;
        end else begin
            top_done <= 0; // one cycle pulse
            case (perf_state)
                PERF_IDLE: begin
                    if (sched_done) begin
                        perf_awvalid_reg <= 1;
                        perf_state <= PERF_AW;
                    end
                end
                PERF_AW: begin
                    if (perf_awvalid_reg && m_axi_fo_gmem_awready) begin
                        perf_awvalid_reg <= 0;
                        perf_wvalid_reg <= 1;
                        perf_state <= PERF_W;
                    end
                end
                PERF_W: begin
                    if (perf_wvalid_reg && m_axi_fo_gmem_wready) begin
                        perf_wvalid_reg <= 0;
                        perf_bready_reg <= 1;
                        perf_state <= PERF_B;
                    end
                end
                PERF_B: begin
                    if (perf_bready_reg && m_axi_fo_gmem_bvalid) begin
                        perf_bready_reg <= 0;
                        top_done <= 1;
                        perf_state <= PERF_IDLE;
                    end
                end
            endcase
        end
    end
    
    assign process_done_internal = top_done;

    // Hardcoded to write into the last 64 bytes of Buf A (256 * 256 * 64 - 64)
    wire[C_M_AXI_GMEM_ADDR_WIDTH-1:0] perf_awaddr = hbm_reg_a_addr + 32'd4194240;

    assign m_axi_fo_gmem_awvalid = fm_output_awvalid | memcpy_awvalid | gap_awvalid | perf_awvalid_reg;
    assign m_axi_fo_gmem_awaddr  = perf_awvalid_reg ? perf_awaddr :
                                   memcpy_awvalid ? memcpy_awaddr : 
	    			   gap_awvalid    ? gap_awaddr    : 
				   fm_output_awaddr;
    assign m_axi_fo_gmem_awlen   = perf_awvalid_reg ? 8'd0 :
                                   memcpy_awvalid ? memcpy_awlen  : 
	    			   gap_awvalid    ? gap_awlen     :
				   fm_output_awlen;
    assign m_axi_fo_gmem_awsize  = perf_awvalid_reg ? 3'b110 :
                                   memcpy_awvalid ? memcpy_awsize : 
	    			   gap_awvalid    ? gap_awsize    : 
				   fm_output_awsize;
    assign m_axi_fo_gmem_awburst = perf_awvalid_reg ? 2'b01 :
                                   memcpy_awvalid ? memcpy_awburst : 
	    			   gap_awvalid    ? gap_awburst    :
				   fm_output_awburst;

    assign m_axi_fo_gmem_wvalid  = fm_output_wvalid | memcpy_wvalid | gap_wvalid | perf_wvalid_reg;
    assign m_axi_fo_gmem_wdata   = perf_wvalid_reg ? { 256'd0, mem_stall_cycles, output_dma_cycles, overlap_cycles, stall_cycles, 32'hDEADBEEF, memory_cycles, compute_cycles, total_cycles } :
                                   memcpy_wvalid ? memcpy_wdata : 
	    			   gap_wvalid    ? gap_wdata    :
				   fm_output_wdata;
    assign m_axi_fo_gmem_wstrb   = perf_wvalid_reg ? 64'hFFFFFFFFFFFFFFFF :
                                   memcpy_wvalid ? memcpy_wstrb : 
	    			   gap_wvalid    ? gap_wstrb    :
				   fm_output_wstrb;
    assign m_axi_fo_gmem_wlast   = perf_wvalid_reg ? 1'b1 :
                                   memcpy_wvalid ? memcpy_wlast :
	    			   gap_wvalid    ? gap_wlast    :
				   fm_output_wlast;
    assign m_axi_fo_gmem_bready  = fm_output_bready | memcpy_bready | gap_bready | perf_bready_reg;

//////////////

    instruction_scheduler #(
	.ADDR_WIDTH(C_M_AXI_GMEM_ADDR_WIDTH),
	.DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH)
    ) instr_scheduler_inst (
	.clk(ap_clk),
	.rst_n(ap_rst_n),
	.start(start_process_internal),
	.base_addr(heap_base_addr),
	.done(sched_done),  //process_done_internal),

	.m_axi_arvalid(sched_arvalid),
	.m_axi_arready(m_axi_gmem_arready),
	.m_axi_araddr(sched_araddr),
	.m_axi_arlen(sched_arlen),
	.m_axi_arsize(sched_arsize),
	.m_axi_arburst(sched_arburst),
	.m_axi_rvalid(m_axi_gmem_rvalid),
	.m_axi_rready(sched_rready),
	.m_axi_rdata(m_axi_gmem_rdata),

	.exec_start(exec_start),
	.exec_done(exec_done),
	.cfg_input_addr(cfg_input_offset),
	.cfg_output_addr(cfg_output_offset),
	.cfg_weight_addr(cfg_weight_offset),
	.cfg_bias_addr(cfg_bias_offset),
	.cfg_img_width(img_width),
	.cfg_img_height(img_height),
	.cfg_in_channels(img_channels),
	.cfg_out_channels(out_channels),
	.cfg_quant_shift(quant_shift),
	.cfg_is_conv(is_conv),
	.cfg_is_memcpy(is_memcpy),
	.cfg_is_gap(is_gap),
	.cfg_bias_en(bias_en),
	.cfg_relu_en(relu_en),
	.cfg_stride(stride),
	.cfg_flatten(flatten),
	.cfg_log2_mem_tile_height(log2_mem_tile_height),
	.cfg_is_sparse(is_sparse),
	.cfg_ic_tile_mask(ic_tile_mask),
	.cfg_oc_tile_mask(oc_tile_mask),
	.cfg_input_bank(cfg_input_bank),
	.cfg_output_bank(cfg_output_bank)
    );

    (* max_fanout = 20 *) reg [15:0] img_width_strips_reg;
    
    always @(posedge ap_clk) begin
        if (img_width < PP_PAR) begin 
            img_width_strips_reg <= 1;
        end
        else begin 
            img_width_strips_reg <= (img_width + PP_PAR - 1) / PP_PAR;
	end
    end
    
    wire [15:0] img_width_strips = img_width_strips_reg;

    tile_manager #(
	.IC_PAR(IC_PAR),
	.OC_PAR(OC_PAR),
	.PP_PAR(PP_PAR),
	.DATA_WIDTH(DATA_WIDTH),
	.MAX_IMG_WIDTH(MAX_IMG_WIDTH),
	.TILE_HEIGHT(TILE_HEIGHT),
	.GMEM_DATA_WIDTH(C_M_AXI_GMEM_DATA_WIDTH)
    ) tile_manager_inst (
	.clk(ap_clk),
	.rst_n(ap_rst_n),
	.start(tm_start),

	// .cmd_ready(cmd_ready),
	// .feat_data_ready(feat_data_ready),
	.full_img_width_strips(img_width_strips),
	.full_img_width(img_width),
	.full_img_height(img_height),
	.full_img_channels(img_channels),
	.output_channels(out_channels),
	.feature_map_words(feature_map_words),
	.weight_words(weight_words),
	.bias_words(bias_words),
	.fmap_out_words(output_feature_map_words),
	.quant_shift(quant_shift),
	.relu_en(relu_en),
	.stride(stride),
	.flatten(flatten),
	.is_conv(is_conv),
	.bias_en(bias_en),
	.log2_mem_tile_height(log2_mem_tile_height),
	.is_sparse(is_sparse),
	.ic_tile_mask(ic_tile_mask),
	.oc_tile_mask(oc_tile_mask),
	// FMap HBM interface	
	.hbm_data_in(read_master_data_out),
	.hbm_addr(hbm_input_addr_w),
	.hbm_ren(hbm_read_fm_start),
	.hbm_rvalid(read_master_data_valid),
	.hbm_rready(read_master_rready),
	// Weights HBM interface
	.hbm_data_in_w(read_master_wb_data_out),
	.hbm_addr_w(hbm_winput_addr),
	.hbm_ren_w(hbm_wread_start),
	.hbm_rvalid_w(read_master_wb_data_valid),
	// Bias HBM interface
	.hbm_data_in_b(read_master_wb_data_out),
	.hbm_addr_b(hbm_binput_addr),
	.hbm_ren_b(hbm_bread_start),
	.hbm_rvalid_b(read_master_wb_data_valid),
	.wb_dma_active(wb_dma_active),
	// Output FMap HBM interface
	.hbm_fmap_out_data(write_master_fmap_data_in),
	.hbm_fmap_out_addr(hbm_foutput_addr),
	.hbm_fmap_out_wen(hbm_fwrite_start),
	.hbm_fmap_wvalid(write_master_fmap_data_valid),
	.hbm_fmap_out_ready(write_master_fmap_data_ready),
	.hbm_fmap_out_done(hbm_fwrite_done),
	.done(tm_done)
    );

endmodule
