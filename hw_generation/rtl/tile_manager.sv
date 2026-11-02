`timescale 1ns / 1ps

module tile_manager #(
    parameter IC_PAR = 16,
    parameter OC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8,
    parameter MAX_IMG_WIDTH = 256,
    parameter TILE_HEIGHT = 4, // Height of strip to process per pass
    parameter GMEM_DATA_WIDTH = 512,
    parameter ACC_WIDTH = 28,
    parameter BIAS_WIDTH = 32,
    parameter HBM_SIZE = 65536
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    input wire [15:0] full_img_width_strips,
    input wire [15:0] full_img_width,
    input wire [15:0] full_img_height,
    input wire [15:0] full_img_channels,

    input wire [15:0] output_channels,
    output wire [15:0] feature_map_words,
    output wire [15:0] weight_words,
    output wire [15:0] bias_words,
    output wire [15:0] fmap_out_words,
    
    input wire [4:0] quant_shift,
    input wire relu_en,
    input wire [1:0] stride,
    input wire flatten,
    input wire is_conv,
    input wire bias_en,
    input wire [2:0] log2_mem_tile_height,
    input wire is_sparse,
    input wire [127:0] ic_tile_mask,
    input wire [127:0] oc_tile_mask,
    // Input FMap DMA Interface
    input wire [PP_PAR*IC_PAR*DATA_WIDTH-1:0] hbm_data_in,
    output wire [63:0] hbm_addr,
    output wire hbm_ren,
    input wire hbm_rvalid,
    output wire hbm_rready,
    
    // Weights DMA Interface
    input wire [OC_PAR*IC_PAR*DATA_WIDTH-1:0] hbm_data_in_w,
    output wire [63:0] hbm_addr_w,
    output wire hbm_ren_w,
    input wire hbm_rvalid_w,

    // Bias DMA Interface
    input wire [OC_PAR*BIAS_WIDTH-1:0] hbm_data_in_b,
    output wire [63:0] hbm_addr_b,
    output wire hbm_ren_b,
    input wire hbm_rvalid_b,
    // Weights or Bias DMA active
    output wire [1:0] wb_dma_active,  // 00: None, 01: Weights, 10: Bias
    
    // Output FMap DMA Interface
    output wire [PP_PAR*OC_PAR*ACC_WIDTH-1:0] hbm_fmap_out_data,
    output wire [63:0] hbm_fmap_out_addr,
    output wire hbm_fmap_out_wen,
    output wire hbm_fmap_wvalid,
    input wire hbm_fmap_out_ready,
    input wire hbm_fmap_out_done,
    // Status
    output reg done
);

    wire [15:0] num_tiles_y = (full_img_height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    wire [15:0] num_tiles_ic = full_img_channels / IC_PAR;
    wire [15:0] num_tiles_oc = output_channels / OC_PAR;
    reg  [15:0] fetch_tile_ic;
    reg  [15:0] compute_tile_ic;
    reg  [15:0] fetch_tile_oc;
    reg  [15:0] compute_tile_oc;
    
    // --- INTERNAL URAMs ---
    // Input Buffer: Stores Tile + Halo
    // Size: (TILE_HEIGHT + 2) * MAX_IMG_WIDTH
    // Make sure tools infer URAM here.
    localparam INPUT_BANK_SIZE = (TILE_HEIGHT+2)*MAX_IMG_WIDTH/PP_PAR;
    localparam WEIGHT_BANK_SIZE = 9;

    (* ram_style = "bram" *)
    reg [PP_PAR*IC_PAR*DATA_WIDTH-1:0] uram_input [0:(2 * INPUT_BANK_SIZE) - 1];

    (* ram_style = "bram" *)
    reg [OC_PAR*IC_PAR*DATA_WIDTH-1:0] weights_bram [0:(2 * WEIGHT_BANK_SIZE) - 1];

    (* ram_style = "distributed" *)
    reg signed [OC_PAR*BIAS_WIDTH-1:0] bias_buffer [0:1];
    
    reg [127:0] active_oc_mask;
    reg [127:0] active_ic_mask;

    (* max_fanout = 16 *) reg fetch_bank;
    (* max_fanout = 16 *) reg compute_bank;
    (* max_fanout = 16 *) reg bias_fetch_bank;
    (* max_fanout = 16 *) reg bias_compute_bank;
    (* max_fanout = 16 *) reg has_fetch;

    reg [15:0] current_tile_y;
    
    reg [15:0] rows_remaining;
    reg [15:0] active_height;

    // DMA Signals
    reg dma_start;
    reg weights_dma_start;
    wire dma_done;
    wire dma_w_done;
    reg dma_done_r;
    reg dma_w_done_r;
    wire [15:0] dma_uram_addr;
    wire [15:0] dma_w_bram_addr;
    wire [15:0] dma_uram_fmap_out_addr;
    wire [PP_PAR*IC_PAR*DATA_WIDTH-1:0] dma_wdata;
    wire [IC_PAR*OC_PAR*DATA_WIDTH-1:0] dma_w_wdata;
    wire [PP_PAR*OC_PAR*ACC_WIDTH-1:0] dma_fmap_out_wdata;
    wire dma_wen;
    wire dma_w_wen;
    wire dma_uram_ren;
    
    reg output_dma_start;
    wire output_dma_done;
    // Accelerator Signals
    reg acc_start;
    wire acc_done;
    reg acc_done_r;
    wire acc_din_ready;
    reg acc_din_valid;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] acc_din_data;
    wire signed [PP_PAR-1:0][OC_PAR-1:0][27:0] acc_dout_data;
    wire acc_dout_valid;
    
    wire [1:0] k_x;
    wire [1:0] k_y;
    // wire signed [OC_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] acc_weights_data = weights_bram[(compute_bank * WEIGHT_BANK_SIZE) + k_y*3 + k_x];
    reg signed [OC_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] acc_weights_data;
    always @(posedge clk) begin
        acc_weights_data <= weights_bram[(compute_bank * WEIGHT_BANK_SIZE) + k_y*3 + k_x];
    end
    wire acc_weights_req;
    wire acc_weights_ack;
    // Streamer Signals
    reg [15:0] stream_ptr;
    reg [15:0] stream_ptr_ulimit;

    reg acc_out_bank;
    reg dma_out_bank;
    reg [15:0] dma_tile_oc;
    reg [OC_PAR*BIAS_WIDTH-1:0] dma_bias_data;
    reg output_dma_busy;
    reg dma_zero_fill;      // FIX: drive output_dma zero_fill for masked oc-tiles
    reg [15:0] zero_oc;     // FIX: masked-tile zeroing loop counter
    reg [127:0] zero_mask;  // FIX: shift-register copy of oc_tile_mask for zeroing pass
    reg [5:0] tile_manager_state;

    tiled_dma #(
        .IC_PAR(IC_PAR),
	.PP_PAR(PP_PAR),
	.DATA_WIDTH(DATA_WIDTH),
	.TILE_HEIGHT(TILE_HEIGHT),
	.HBM_DATA_WIDTH(GMEM_DATA_WIDTH)
    ) dma (
        .clk(clk), .rst_n(rst_n), .start(dma_start),
        .img_width(full_img_width),
        .img_height(full_img_height),
	.img_channels(full_img_channels),
	.feature_map_words(feature_map_words),
        .tile_y_index(current_tile_y),
	.tile_ic_index(fetch_tile_ic),
	.active_height(active_height),
	.log2_mem_tile_height(log2_mem_tile_height),
        .hbm_data_in(hbm_data_in),
        .hbm_addr(hbm_addr),
        .hbm_ren(hbm_ren),
	.hbm_rvalid(hbm_rvalid),
	.hbm_rready(hbm_rready),
        .uram_addr(dma_uram_addr),
        .uram_wdata(dma_wdata),
        .uram_wen(dma_wen),
        .done(dma_done)
    );

    weights_dma #(
        .IC_PAR(IC_PAR),
	.OC_PAR(OC_PAR),
	.DATA_WIDTH(DATA_WIDTH),
	.HBM_DATA_WIDTH(GMEM_DATA_WIDTH)
    ) weights_dma (
        .clk(clk), .rst_n(rst_n), .start(weights_dma_start),
        .oc(output_channels),
        .ic(full_img_channels),
	.weight_words(weight_words),
	.ic_tile(fetch_tile_ic),
        .oc_tile(fetch_tile_oc),
	.is_conv(is_conv),
	.hbm_data_in(hbm_data_in_w),
        .hbm_addr(hbm_addr_w),
        .hbm_ren(hbm_ren_w),
	.hbm_rvalid(hbm_rvalid_w),
        .bram_addr(dma_w_bram_addr),
        .bram_wdata(dma_w_wdata),
        .bram_wen(dma_w_wen),
        .done(dma_w_done)
    );

    wire dma_b_done;
    reg bias_dma_start;
    wire [15:0] dma_b_bram_addr;
    wire [OC_PAR*BIAS_WIDTH-1:0] dma_b_wdata;
    wire dma_b_wen;

    bias_dma #(
	.OC_PAR(OC_PAR),
	.HBM_DATA_WIDTH(GMEM_DATA_WIDTH),
	.BIAS_WIDTH(32)
    ) bias_dma (
        .clk(clk), .rst_n(rst_n), .start(bias_dma_start),
        .oc(output_channels),
	.bias_words(bias_words),
        .oc_tile(fetch_tile_oc),
	.hbm_data_in(hbm_data_in_b),
        .hbm_addr(hbm_addr_b),
        .hbm_ren(hbm_ren_b),
	.hbm_rvalid(hbm_rvalid_b),
        .bias_bram_addr(dma_b_bram_addr),
        .bias_data(dma_b_wdata),
        .bias_wen(dma_b_wen),
        .done(dma_b_done)
    );

    reg fetch_found_first_ic;  
    reg comp_is_first_ic;
    reg fetch_is_first_ic;
    // Controls reset of pointers
    wire acc_clear_ptr = acc_start;
    // Controls accumulation mode (0 for first IC tile, 1 for subsequent)
    wire acc_accumulate = !comp_is_first_ic;

    localparam RES_ACCUM_DEPTH = TILE_HEIGHT * MAX_IMG_WIDTH / PP_PAR;
    localparam RES_ACCUM_ADDR_WIDTH = $clog2(RES_ACCUM_DEPTH);

    result_accumulator #(
        .IC_PAR(IC_PAR), .OC_PAR(OC_PAR), .PP_PAR(PP_PAR), .ACC_WIDTH(ACC_WIDTH),
        .DEPTH(RES_ACCUM_DEPTH), .ADDR_WIDTH(RES_ACCUM_ADDR_WIDTH)
    ) res_acc (
        .clk(clk), .rst_n(rst_n),
	.comp_out_bank(acc_out_bank),
	.dma_out_bank(dma_out_bank),
        .clear_ptr(acc_clear_ptr),
        .accumulate_en(acc_accumulate),
        .acc_in_data(acc_dout_data),
        .acc_in_valid(acc_dout_valid),
        .stride(stride),
        .dma_raddr(dma_uram_fmap_out_addr[RES_ACCUM_ADDR_WIDTH-1:0]), // Truncate to ADDR_WIDTH
        .dma_ren(dma_uram_ren),
        .dma_rdata_packed(dma_fmap_out_wdata)
    );

    wire [OC_PAR*BIAS_WIDTH-1:0] bias_buffer_w = bias_buffer[bias_compute_bank];

    output_dma #(
	.OC_PAR(OC_PAR),
	.PP_PAR(PP_PAR),
	.DATA_WIDTH(DATA_WIDTH),
	.ACCUM_WIDTH(ACC_WIDTH),
	.TILE_HEIGHT(TILE_HEIGHT),
	.HBM_DATA_WIDTH(GMEM_DATA_WIDTH)
    ) out_dma (
	.clk(clk), .rst_n(rst_n), .start(output_dma_start),
	.zero_fill(dma_zero_fill),
	.img_width(full_img_width),  //full_img_width_strips * PP_PAR),
	.oc_channels(output_channels),
	.tile_y_index(current_tile_y),
	.tile_oc_index(dma_tile_oc),
	.active_height(active_height),
	.quant_shift(quant_shift),
	.relu_en(relu_en),
	.stride(stride),
	.flatten(flatten),
	.write_count(fmap_out_words),
	.uram_addr(dma_uram_fmap_out_addr),
	.uram_rdata(dma_fmap_out_wdata),
	.uram_ren(dma_uram_ren),
	.bias_rdata(dma_bias_data),
	.start_write(hbm_fmap_out_wen),
	.write_base_addr(hbm_fmap_out_addr),
	.write_data(hbm_fmap_out_data),
	.write_valid(hbm_fmap_wvalid),
	.write_ready(hbm_fmap_out_ready),
	.write_done(hbm_fmap_out_done),
	.done(output_dma_done)
    );
 
    conv_accelerator #(
        .IC_PAR(IC_PAR), .OC_PAR(OC_PAR), .PP_PAR(PP_PAR), 
        .DATA_WIDTH(DATA_WIDTH), .MAX_IMG_WIDTH(MAX_IMG_WIDTH)
    ) acc (
        .clk(clk), .rst_n(rst_n), .start(acc_start),
        .img_width_strips(full_img_width_strips),
        .img_height(active_height + 2),
	.img_channels(full_img_channels),
        .weights(acc_weights_data),
	.k_x(k_x), .k_y(k_y),
	.stride(stride),
	.is_conv(is_conv),
	.weight_req(acc_weights_req),
	.weight_ack(acc_weights_ack),
        .din_data(acc_din_data),
        .din_valid(acc_din_valid),
        .din_ready(acc_din_ready),
        .dout_data(acc_dout_data),
        .dout_valid(acc_dout_valid),
        .done(acc_done)
    );

    (* max_fanout = 20 *) reg [15:0] init_ptr;
    (* max_fanout = 20 *) reg init_active;

    // --- URAM WRITE LOGIC (DMA -> URAM) ---
    always @(posedge clk) begin
	if (init_active) begin
	    uram_input[init_ptr - 1] <= 0;
	end else if (dma_wen) begin
            uram_input[(fetch_bank * INPUT_BANK_SIZE) + dma_uram_addr-1] <= dma_wdata;
        end
	if (dma_w_wen) begin
	    weights_bram[(fetch_bank * WEIGHT_BANK_SIZE) + dma_w_bram_addr-1] <= dma_w_wdata;
	end
	if (dma_b_wen) begin
	    bias_buffer[bias_fetch_bank] <= dma_b_wdata;
	end
    end

    always @(posedge clk) begin
        acc_din_data <= uram_input[(compute_bank * INPUT_BANK_SIZE) + stream_ptr];
    end

    reg [31:0] compute_active;
    
    reg [1:0] wb_dma_active_r;
    assign wb_dma_active = wb_dma_active_r;
   

    integer i;
    
    localparam S_IDLE = 0;
    localparam S_CALC_CONSTS_0 = 1;
    localparam S_CALC_CONSTS_1 = 2;
    localparam S_INIT_URAM = 3;

    // Prologue States
    localparam S_SEARCH_PROLOGUE = 4;
    localparam S_FETCH_BIAS = 5;
    localparam S_FETCH_FMW = 6;
    localparam S_PROLOGUE_DONE = 7;

    // Pipeline States
    localparam S_SEARCH_NEXT = 8;
    localparam S_PIPELINE_RUN = 9;
    localparam S_PIPELINE_FETCH_BIAS = 10;
    localparam S_PIPELINE_WAIT = 11;

    // localparam S_WRITE_OUTPUT_DMA = 12;
    localparam S_NEXT_Y = 12;
    localparam S_WAIT_OUTPUT_DMA = 13;
    localparam S_WAIT_LAST_OUTPUT = 14;
    // FIX: zero masked oc-tile output slots (matches prebatch; preserves compute-skip)
    localparam S_ZERO_MASKED = 15;
    localparam S_ZERO_WAIT   = 16;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tile_manager_state <= S_IDLE;
            current_tile_y <= 0;
            dma_start <= 0;
            acc_start <= 0;
	    acc_done_r <= 0;
            acc_din_valid <= 0;
            stream_ptr <= 0;
            done <= 0;
            init_active <= 0;
            init_ptr <= 0;
            dma_done_r <= 0;
            dma_w_done_r <= 0;
            output_dma_start <= 0;
            dma_zero_fill <= 0;
            zero_oc <= 0;
            zero_mask <= 0;
            rows_remaining <= 0;
            active_height <= 0;
            stream_ptr_ulimit <= 0;
            bias_dma_start <= 0;
            wb_dma_active_r <= 0;
            
            // Pipeline resets
            fetch_tile_oc <= 0;
            fetch_tile_ic <= 0;
            compute_tile_oc <= 0;
            compute_tile_ic <= 0;
            has_fetch <= 0;
            fetch_bank <= 0;
	    bias_fetch_bank <= 0;
            compute_bank <= 1;
	    bias_compute_bank <= 1;
            comp_is_first_ic <= 0;
            fetch_is_first_ic <= 0;
            fetch_found_first_ic <= 0;
	    compute_active <= 0;
            
        end else begin
            dma_start <= 0;
            weights_dma_start <= 0;
            bias_dma_start <= 0;
            output_dma_start <= 0;
            acc_start <= 0;
	    if (output_dma_done) output_dma_busy <= 0;
            
            case (tile_manager_state)
                S_IDLE: begin
                    done <= 0;
                    wb_dma_active_r <= 0;
		    compute_active <= 0;
		    output_dma_busy <= 0;
		    acc_out_bank <= 0;
		    dma_out_bank <= 1;
                    if (start) begin
			has_fetch <= 0;
                        init_active <= 0;
                        init_ptr <= 0;
                        current_tile_y <= 0;
                        stream_ptr_ulimit <= 0;
                        tile_manager_state <= S_CALC_CONSTS_0;
                    end
                end

                S_CALC_CONSTS_0: begin
                    if ((full_img_height - (current_tile_y * TILE_HEIGHT)) < TILE_HEIGHT) begin
                        active_height <= (full_img_height - (current_tile_y * TILE_HEIGHT));
                    end else begin
                        active_height <= TILE_HEIGHT;
                    end
                    tile_manager_state <= S_CALC_CONSTS_1;
                end

                S_CALC_CONSTS_1: begin
                    stream_ptr_ulimit <= (active_height + 2) * full_img_width_strips;
                    tile_manager_state <= S_INIT_URAM;
                end

                S_INIT_URAM: begin
                    init_active <= 1;
                    // Clear BOTH banks
                    if (init_ptr == 2 * INPUT_BANK_SIZE) begin
                        init_active <= 0;
                        init_ptr <= 0;
                        
			active_oc_mask <= oc_tile_mask; 
			active_ic_mask <= ic_tile_mask;
                        // Setup Prologue variables
                        fetch_tile_oc <= 0;
                        fetch_tile_ic <= 0;
                        fetch_found_first_ic <= 0;
                        fetch_bank <= 0;
			bias_fetch_bank <= 0;
                        compute_bank <= 1;
			bias_compute_bank <= 0;
                        // FIX: run masked-oc-tile zeroing pass before compute
                        zero_oc <= 0;
                        zero_mask <= oc_tile_mask;
                        tile_manager_state <= S_ZERO_MASKED;
                    end else begin
                        init_ptr <= init_ptr + 1;
                    end
                end

                // FIX: zero masked oc-tile output slots (prebatch-equivalent).
                // Compute (MAC) is still skipped for masked tiles; we only add a
                // cheap zero writeback so downstream (memcpy->fusion, next-layer
                // reads) sees 0 instead of stale ping-pong garbage. Runs per Y-tile
                // BEFORE active compute, so no double-buffer interaction.
                S_ZERO_MASKED: begin
                    dma_zero_fill <= 0;
                    if (!is_sparse || zero_oc >= num_tiles_oc) begin
                        tile_manager_state <= S_SEARCH_PROLOGUE;
                    end else if (!zero_mask[0]) begin
                        // masked oc-tile: fire a zero-write at its natural slot
                        output_dma_start <= 1;
                        dma_zero_fill    <= 1;
                        dma_tile_oc      <= zero_oc;
                        tile_manager_state <= S_ZERO_WAIT;
                    end else begin
                        // active oc-tile: leave it for the compute pass
                        zero_oc   <= zero_oc + 1;
                        zero_mask <= zero_mask >> 1;
                    end
                end

                S_ZERO_WAIT: begin
                    output_dma_start <= 0;
                    if (output_dma_done) begin
                        dma_zero_fill <= 0;
                        zero_oc   <= zero_oc + 1;
                        zero_mask <= zero_mask >> 1;
                        tile_manager_state <= S_ZERO_MASKED;
                    end
                end

                S_SEARCH_PROLOGUE: begin
		    /*
                    if (fetch_tile_oc >= num_tiles_oc) begin
                        // No valid OCs in this entire layer/mask. Go to next Y.
                        tile_manager_state <= S_NEXT_Y;
                    end else if (is_sparse && !oc_tile_mask[fetch_tile_oc]) begin
                        fetch_tile_oc <= fetch_tile_oc + 1;
                        fetch_tile_ic <= 0;
                        fetch_found_first_ic <= 0;
                    end else if (fetch_tile_ic >= num_tiles_ic) begin
                        fetch_tile_oc <= fetch_tile_oc + 1;
                        fetch_tile_ic <= 0;
                        fetch_found_first_ic <= 0;
                    end else if (is_sparse && !ic_tile_mask[fetch_tile_ic]) begin
                        fetch_tile_ic <= fetch_tile_ic + 1;
                    end else begin
                        // Valid Tile Found
                        fetch_is_first_ic <= !fetch_found_first_ic;
                        fetch_found_first_ic <= 1;
                        has_fetch <= 1;
                        
                        if (!fetch_found_first_ic) begin
                            bias_dma_start <= 1;
                            wb_dma_active_r <= 2'b10;
                            tile_manager_state <= S_FETCH_BIAS;
                        end else begin
                            dma_start <= 1;
                            weights_dma_start <= 1;
                            wb_dma_active_r <= 2'b01;
                            tile_manager_state <= S_FETCH_FMW;
                        end
                    end
		    */
		    if (fetch_tile_oc >= num_tiles_oc) begin
                        // No valid OCs in this entire layer/mask. Go to next Y.
                        has_fetch <= 0;
                        if (tile_manager_state == S_SEARCH_PROLOGUE) tile_manager_state <= S_NEXT_Y;
                        else tile_manager_state <= S_PIPELINE_RUN;
                    end else if (is_sparse && !active_oc_mask[0]) begin // CHANGED
                        fetch_tile_oc <= fetch_tile_oc + 1;
                        active_oc_mask <= active_oc_mask >> 1;          // CHANGED
                        fetch_tile_ic <= 0;
                        active_ic_mask <= ic_tile_mask;                 // CHANGED
                        fetch_found_first_ic <= 0;
                    end else if (fetch_tile_ic >= num_tiles_ic) begin
                        fetch_tile_oc <= fetch_tile_oc + 1;
                        active_oc_mask <= active_oc_mask >> 1;          // CHANGED
                        fetch_tile_ic <= 0;
                        active_ic_mask <= ic_tile_mask;                 // CHANGED
                        fetch_found_first_ic <= 0;
                    end else if (is_sparse && !active_ic_mask[0]) begin // CHANGED
                        fetch_tile_ic <= fetch_tile_ic + 1;
                        active_ic_mask <= active_ic_mask >> 1;          // CHANGED
                    end else begin
                        // Valid Tile Found
                        fetch_is_first_ic <= !fetch_found_first_ic;
                        fetch_found_first_ic <= 1;
                        has_fetch <= 1;
                        
                        if (!fetch_found_first_ic) begin
                            bias_dma_start <= 1;
                            wb_dma_active_r <= 2'b10;
                            tile_manager_state <= S_FETCH_BIAS;
                        end else begin
                            dma_start <= 1;
                            weights_dma_start <= 1;
                            wb_dma_active_r <= 2'b01;
                            tile_manager_state <= S_FETCH_FMW;
                        end
                    end

                end

                S_FETCH_BIAS: begin
                    if (dma_b_done) begin
                        dma_start <= 1;
                        weights_dma_start <= 1;
                        wb_dma_active_r <= 2'b01;
                        tile_manager_state <= S_FETCH_FMW;
                    end
                end
                
                S_FETCH_FMW: begin
		    dma_start <= 0;
		    weights_dma_start <= 0;
                    if (dma_done) dma_done_r <= 1;
		    if (dma_w_done) begin 
		        dma_w_done_r <= 1;
			wb_dma_active_r <= 2'b00;
		    end
                    if ((dma_done || dma_done_r) && (dma_w_done || dma_w_done_r)) begin
			dma_done_r <= 0;
			dma_w_done_r <= 0;
                        tile_manager_state <= S_PROLOGUE_DONE;
                    end
                end
                
                S_PROLOGUE_DONE: begin
                    // Transfer fetch data to compute pointers
                    compute_tile_oc <= fetch_tile_oc;
                    compute_tile_ic <= fetch_tile_ic;
                    comp_is_first_ic <= fetch_is_first_ic;

		    bias_compute_bank <= bias_fetch_bank;
		    bias_fetch_bank <= ~bias_fetch_bank;
                    
                    fetch_tile_ic <= fetch_tile_ic + 1; // Advance search
		    active_ic_mask <= active_ic_mask >> 1;
                    tile_manager_state <= S_SEARCH_NEXT;
                end

                // --- PIPELINE PHASE ---
                S_SEARCH_NEXT: begin
		    /*
                    if (fetch_tile_oc >= num_tiles_oc) begin
                        // No more data to fetch. Just run the pipeline to empty it.
                        has_fetch <= 0;
                        tile_manager_state <= S_PIPELINE_RUN;
                    end else if (is_sparse && !oc_tile_mask[fetch_tile_oc]) begin
                        fetch_tile_oc <= fetch_tile_oc + 1;
                        fetch_tile_ic <= 0;
                        fetch_found_first_ic <= 0;
                    end else if (fetch_tile_ic >= num_tiles_ic) begin
                        // Hit end of ICs for this OC. 
                        // Stop searching. The NEXT fetch will be for a new OC.
                        // We must compute current OC and write it out before we can fetch
                        // too far ahead, but fetching the NEXT OC is fine because it goes into the unused bank!
                        fetch_tile_oc <= fetch_tile_oc + 1;
                        fetch_tile_ic <= 0;
                        fetch_found_first_ic <= 0;
                    end else if (is_sparse && !ic_tile_mask[fetch_tile_ic]) begin
                        fetch_tile_ic <= fetch_tile_ic + 1;
                    end else begin
                        // Found next tile
                        fetch_is_first_ic <= !fetch_found_first_ic;
                        fetch_found_first_ic <= 1;
                        has_fetch <= 1;
                        tile_manager_state <= S_PIPELINE_RUN;
                    end
		    */
		    if (fetch_tile_oc >= num_tiles_oc) begin
                        // No valid OCs in this entire layer/mask. Go to next Y.
                        has_fetch <= 0;
                        if (tile_manager_state == S_SEARCH_PROLOGUE) tile_manager_state <= S_NEXT_Y;
                        else tile_manager_state <= S_PIPELINE_RUN;
                    end else if (is_sparse && !active_oc_mask[0]) begin // CHANGED
                        fetch_tile_oc <= fetch_tile_oc + 1;
                        active_oc_mask <= active_oc_mask >> 1;          // CHANGED
                        fetch_tile_ic <= 0;
                        active_ic_mask <= ic_tile_mask;                 // CHANGED
                        fetch_found_first_ic <= 0;
                    end else if (fetch_tile_ic >= num_tiles_ic) begin
                        fetch_tile_oc <= fetch_tile_oc + 1;
                        active_oc_mask <= active_oc_mask >> 1;          // CHANGED
                        fetch_tile_ic <= 0;
                        active_ic_mask <= ic_tile_mask;                 // CHANGED
                        fetch_found_first_ic <= 0;
                    end else if (is_sparse && !active_ic_mask[0]) begin // CHANGED
                        fetch_tile_ic <= fetch_tile_ic + 1;
                        active_ic_mask <= active_ic_mask >> 1;          // CHANGED
                    end else begin
                        // Found next tile
                        fetch_is_first_ic <= !fetch_found_first_ic;
                        fetch_found_first_ic <= 1;
                        has_fetch <= 1;
                        tile_manager_state <= S_PIPELINE_RUN;
                    end
                end

                S_PIPELINE_RUN: begin
                    // Swap Banks
                    compute_bank <= fetch_bank;
                    fetch_bank <= ~fetch_bank;

                    // Start Compute for 'comp' pointers
                    acc_start <= 1;
		    acc_done_r <= 0;
                    stream_ptr <= 0;
                    
                    // Prepare for Fetch for 'fetch' pointers
                    dma_done_r <= 0;
                    dma_w_done_r <= 0;
                    
                    if (has_fetch) begin
                        if (fetch_is_first_ic) begin
                            bias_dma_start <= 1;
                            wb_dma_active_r <= 2'b10;
                            tile_manager_state <= S_PIPELINE_FETCH_BIAS;
                        end else begin
                            dma_start <= 1;
                            weights_dma_start <= 1;
                            wb_dma_active_r <= 2'b01;
                            tile_manager_state <= S_PIPELINE_WAIT;
                        end
                    end else begin
                        dma_done_r <= 1;
                        dma_w_done_r <= 1;
                        tile_manager_state <= S_PIPELINE_WAIT;
                    end
                end

                S_PIPELINE_FETCH_BIAS: begin
		    bias_dma_start <= 0;
                    // Stream data to accelerator while waiting for bias
                    if (stream_ptr < stream_ptr_ulimit) begin
                        acc_din_valid <= 1;
                        // acc_din_data <= uram_input[(compute_bank * INPUT_BANK_SIZE) + stream_ptr];
                        if (acc_din_ready) stream_ptr <= stream_ptr + 1;
                    end else begin
                        acc_din_valid <= 0;
                    end
                    
                    if (dma_b_done) begin
                        dma_start <= 1;
                        weights_dma_start <= 1;
                        wb_dma_active_r <= 2'b01;
                        tile_manager_state <= S_PIPELINE_WAIT;
                    end
                end

		/*
                S_PIPELINE_WAIT: begin
		    dma_start <= 0;
		    weights_dma_start <= 0;
                    // Stream data to accelerator
                    if (stream_ptr < stream_ptr_ulimit) begin
                        acc_din_valid <= 1;
                        // acc_din_data <= uram_input[(compute_bank * INPUT_BANK_SIZE) + stream_ptr];
                        if (acc_din_ready) stream_ptr <= stream_ptr + 1;
                    end else begin
                        acc_din_valid <= 0;
                    end

                    // Wait for DMAs
                    if (has_fetch) begin
                        if (dma_done) dma_done_r <= 1;
			if (dma_w_done) begin 
			    dma_w_done_r <= 1;
			    wb_dma_active_r <= 2'b00;
			end
                    end

                    // Barrier
                    // if (dma_done_r && dma_w_done_r && acc_done) begin
		    if ((!has_fetch || ((dma_done || dma_done_r) && (dma_w_done || dma_w_done_r))) && (acc_done || acc_done_r)) begin
                        // Was this the LAST IC for this OC?
                        // If there is no next fetch, OR the next fetch belongs to a different OC,
                        // then we must write the output.
                        if (!has_fetch || (fetch_tile_oc != compute_tile_oc)) begin
                            output_dma_start <= 1;
                            tile_manager_state <= S_WRITE_OUTPUT;
                        end else begin
                            // Same OC, shift pointers and loop
                            compute_tile_oc <= fetch_tile_oc;
                            compute_tile_ic <= fetch_tile_ic;
                            comp_is_first_ic <= fetch_is_first_ic;

			    if (fetch_is_first_ic) begin
                                bias_compute_bank <= bias_fetch_bank;
                                bias_fetch_bank <= ~bias_fetch_bank;
                            end
                            
                            fetch_tile_ic <= fetch_tile_ic + 1;
			    active_ic_mask <= active_ic_mask >> 1;
                            tile_manager_state <= S_SEARCH_NEXT;
                        end
			acc_done_r <= 0;
			dma_done_r <= 0;
			dma_w_done_r <= 0;
                    end
                end

                S_WRITE_OUTPUT: begin
		    output_dma_start <= 0;
                    acc_din_valid <= 0;
                    if (output_dma_done) begin
                        if (!has_fetch) begin
                            tile_manager_state <= S_NEXT_Y;
                        end else begin
                            // Output written, we can immediately run the next OC
                            // that we already fetched!
                            compute_tile_oc <= fetch_tile_oc;
                            compute_tile_ic <= fetch_tile_ic;
                            comp_is_first_ic <= fetch_is_first_ic;

			    if (fetch_is_first_ic) begin
                                bias_compute_bank <= bias_fetch_bank;
                                bias_fetch_bank <= ~bias_fetch_bank;
                            end
                            
                            fetch_tile_ic <= fetch_tile_ic + 1;
			    active_ic_mask <= active_ic_mask >> 1;
                            tile_manager_state <= S_SEARCH_NEXT;
                        end
                    end
                end
		*/
		S_PIPELINE_WAIT: begin
		            dma_start <= 0;
		            weights_dma_start <= 0;
                    
                    // Stream data to accelerator
                    if (stream_ptr < stream_ptr_ulimit) begin
                        acc_din_valid <= 1;
                        // acc_din_data <= uram_input[(compute_bank * INPUT_BANK_SIZE) + stream_ptr];
                        if (acc_din_ready) stream_ptr <= stream_ptr + 1;
                    end else begin
                        acc_din_valid <= 0;
                    end

                    if (has_fetch) begin
                        if (dma_done) dma_done_r <= 1;
			if (dma_w_done) begin
			    dma_w_done_r <= 1;
			    wb_dma_active_r <= 2'b00;
			end
                    end

                    // Barrier
		    if ((!has_fetch || ((dma_done || dma_done_r) && (dma_w_done || dma_w_done_r))) && (acc_done || acc_done_r)) begin
                        // Was this the LAST IC for this OC?
                        if (!has_fetch || (fetch_tile_oc != compute_tile_oc)) begin
                            // We must write the output! Check if the DMA is still busy with the LAST tile.
                            if (output_dma_busy && !output_dma_done) begin
                                tile_manager_state <= S_WAIT_OUTPUT_DMA; // Bank is full, stall.
                            end else begin
                                // DMA is free! Fire it instantly and loop back to compute.
                                output_dma_start <= 1;
                                output_dma_busy <= 1;
                                dma_tile_oc <= compute_tile_oc;
                                dma_bias_data <= bias_buffer[bias_compute_bank];
                                dma_out_bank <= acc_out_bank;
                                acc_out_bank <= ~acc_out_bank; // Ping-Pong Compute Bank!
                                
                                if (!has_fetch) begin
                                    tile_manager_state <= S_WAIT_LAST_OUTPUT;
                                end else begin
                                    compute_tile_oc <= fetch_tile_oc;
                                    compute_tile_ic <= fetch_tile_ic;
                                    comp_is_first_ic <= fetch_is_first_ic;

                                    if (fetch_is_first_ic) begin
                                        bias_compute_bank <= bias_fetch_bank;
                                        bias_fetch_bank <= ~bias_fetch_bank;
                                    end
                                    
                                    fetch_tile_ic <= fetch_tile_ic + 1;
                                    active_ic_mask <= active_ic_mask >> 1;
                                    tile_manager_state <= S_SEARCH_NEXT;
                                end
                            end
                        end else begin
                            // Same OC, shift pointers and loop instantly
                            compute_tile_oc <= fetch_tile_oc;
                            compute_tile_ic <= fetch_tile_ic;
                            comp_is_first_ic <= fetch_is_first_ic;

                            if (fetch_is_first_ic) begin
                                bias_compute_bank <= bias_fetch_bank;
                                bias_fetch_bank <= ~bias_fetch_bank;
                            end
                            
                            fetch_tile_ic <= fetch_tile_ic + 1;
                            active_ic_mask <= active_ic_mask >> 1;
                            tile_manager_state <= S_SEARCH_NEXT;
                        end
			            acc_done_r <= 0;
			            dma_done_r <= 0;
			            dma_w_done_r <= 0;
                    end
                end

                S_WAIT_OUTPUT_DMA: begin
                    // We only land here if the compute engine outran the DMA (both banks full)
                    if (output_dma_done) begin
                        output_dma_start <= 1;
                        output_dma_busy <= 1;
                        dma_tile_oc <= compute_tile_oc;
                        dma_bias_data <= bias_buffer[bias_compute_bank];
                        dma_out_bank <= acc_out_bank;
                        acc_out_bank <= ~acc_out_bank;
                        
                        if (!has_fetch) begin
                            tile_manager_state <= S_WAIT_LAST_OUTPUT;
                        end else begin
                            compute_tile_oc <= fetch_tile_oc;
                            compute_tile_ic <= fetch_tile_ic;
                            comp_is_first_ic <= fetch_is_first_ic;

                            if (fetch_is_first_ic) begin
                                bias_compute_bank <= bias_fetch_bank;
                                bias_fetch_bank <= ~bias_fetch_bank;
                            end
                            
                            fetch_tile_ic <= fetch_tile_ic + 1;
                            active_ic_mask <= active_ic_mask >> 1;
                            tile_manager_state <= S_SEARCH_NEXT;
                        end
                    end
                end

                S_WAIT_LAST_OUTPUT: begin
                    acc_din_valid <= 0;
                    // Drain the very last tile of the layer
                    if (output_dma_done) begin
                        tile_manager_state <= S_NEXT_Y;
                    end
                end

                S_NEXT_Y: begin
                    if (current_tile_y + 1 >= num_tiles_y) begin
                        done <= 1;
                        tile_manager_state <= S_IDLE;
                    end else begin
                        current_tile_y <= current_tile_y + 1;
                        tile_manager_state <= S_CALC_CONSTS_0;
                    end
                end
            endcase
	    if (acc_done) acc_done_r <= 1; 
        end
    end

endmodule
