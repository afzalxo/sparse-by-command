`timescale 1ns / 1ps

module gap_unit #(
    parameter DATA_WIDTH = 8,
    parameter IC_PAR = 16, 
    parameter PP_PAR = 8,
    parameter HBM_DATA_WIDTH = 512,
    parameter ADDR_WIDTH = 64,
    parameter TILE_HEIGHT = 4
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Configuration
    input wire [63:0] src_addr_base,
    input wire [63:0] dst_addr_base,
    input wire [15:0] img_width,
    input wire [15:0] img_height,
    input wire [15:0] img_channels,
    input wire [2:0]  log2_mem_tile_height,
    input wire [4:0]  quant_shift,

    // AXI Read Master
    output reg m_axi_arvalid,
    input wire m_axi_arready,
    output reg [ADDR_WIDTH-1:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst,
    input wire m_axi_rvalid,
    output reg m_axi_rready,
    input wire [HBM_DATA_WIDTH-1:0] m_axi_rdata,

    // AXI Write Master
    output reg m_axi_awvalid,
    input wire m_axi_awready,
    output reg [ADDR_WIDTH-1:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst,
    output reg m_axi_wvalid,
    input wire m_axi_wready,
    output reg [HBM_DATA_WIDTH-1:0] m_axi_wdata,
    output wire [HBM_DATA_WIDTH/8-1:0] m_axi_wstrb,
    output wire m_axi_wlast,
    input wire m_axi_bvalid,
    output reg m_axi_bready,
    input wire [1:0] m_axi_bresp,

    output reg done
);

    // --- CONSTANTS ---
    localparam CHUNK_BITS = IC_PAR * DATA_WIDTH; // 128 bits
    localparam CHUNKS_PER_BEAT = HBM_DATA_WIDTH / CHUNK_BITS; // 4
    
    // Calculate how many HBM beats constitute one PP_PAR strip
    // (8 pixels * 128 bits) / 512 bits = 2 beats
    localparam BEATS_PER_STRIP = (PP_PAR * CHUNK_BITS) / HBM_DATA_WIDTH;

    // AXI Defaults
    assign m_axi_arlen = 8'd0; 
    assign m_axi_arsize = 3'b110; 
    assign m_axi_arburst = 2'b01;
    assign m_axi_awlen = 8'd0;
    assign m_axi_awsize = 3'b110;
    assign m_axi_awburst = 2'b01;
    assign m_axi_wstrb = {(HBM_DATA_WIDTH/8){1'b1}};
    assign m_axi_wlast = 1'b1;

    /*
    wire [31:0] stride_row_words = img_width / CHUNKS_PER_BEAT;
    wire [31:0] stride_ic_tile_words = stride_row_words << log2_mem_tile_height;
    wire [31:0] stride_height_tile_words = stride_ic_tile_words * (img_channels / IC_PAR);
    */

    reg [31:0] r_stride_row_words;
    reg [31:0] r_stride_ic_tile_words;
    reg [31:0] r_stride_height_tile_words;
    reg [63:0] tile_oc_offset;
    reg [63:0] tile_ht_offset;
    
    reg [15:0] r_num_h_tiles_minus_1;
    wire [15:0] num_oc_tiles = img_channels / IC_PAR;
    // wire [15:0] num_h_tiles = (img_height + (1 << log2_mem_tile_height) - 1) >> log2_mem_tile_height;
    wire [15:0] rows_per_tile = 1 << log2_mem_tile_height;

    // --- STATE MACHINE ---
    localparam S_IDLE       = 0;
    localparam S_INIT_TILE  = 1;
    localparam S_READ_REQ   = 2;
    localparam S_READ_ACK   = 3;
    localparam S_READ_WAIT  = 4;
    localparam S_ACCUM_REAL = 5;
    localparam S_ACCUM      = 6;
    localparam S_WRITE_REQ  = 7;
    localparam S_WRITE_ACK  = 8;
    localparam S_WRITE_DATA = 9;
    localparam S_WRITE_DATA_ACK = 10;
    localparam S_WRITE_RESP = 11;
    localparam S_DONE       = 12;
    localparam S_CALC_0      = 13;
    localparam S_CALC_1      = 14;
    localparam S_CALC_2      = 15;
    localparam S_CALC_3      = 16;
    localparam S_CALC_4      = 17;

    reg [4:0] state;
    reg [15:0] cnt_oc, cnt_ht, cnt_row, cnt_col;
    (* ram_style = "registers" *) reg signed [31:0] acc [0:IC_PAR-1];
    reg [63:0] curr_read_addr;
    reg [63:0] curr_write_addr;
    reg [63:0] src_addr_base_r;  // Registered copy to break timing path
    
    // --- AVERAGING LOGIC ---
    logic [HBM_DATA_WIDTH-1:0] averaged_data;
    always_comb begin
	averaged_data = 0;
	for (integer i = 0; i < IC_PAR; i = i + 1) begin
    	    automatic logic signed [31:0] shifted;
	    automatic logic signed [31:0] bias;
	    automatic logic signed [32:0] biased_acc;

	    bias = (quant_shift > 0) ? (1 << (quant_shift - 1)) : 0;
	    biased_acc = $signed(acc[i]) + $signed({1'b0, bias});

	    shifted = biased_acc >>> quant_shift;

	    // shifted = acc[i] >>> quant_shift;
	    if (shifted > 127) shifted = 127;
	    else if (shifted < -128) shifted = -128;
	    averaged_data[i*8 +: 8] = shifted[7:0];
        end
    end

    // Counter for padding beats
    reg [3:0] write_beat_cnt;
    reg [3:0] beat_cntr;
    reg [HBM_DATA_WIDTH-1:0] rdata_latch;

    integer i, c;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 0;
            m_axi_arvalid <= 0;
            m_axi_rready <= 0;
            m_axi_awvalid <= 0;
            m_axi_wvalid <= 0;
            m_axi_bready <= 0;
	    rdata_latch <= 0;
            cnt_oc <= 0;
	    beat_cntr <= 0;
            write_beat_cnt <= 0;
	    r_stride_row_words <= 0;
	    r_stride_ic_tile_words <= 0;
	    r_stride_height_tile_words <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        cnt_oc <= 0;
			beat_cntr <= 0;
			write_beat_cnt <= 0;
                        curr_write_addr <= dst_addr_base;
			rdata_latch <= 0;
			src_addr_base_r <= src_addr_base;  // Register base address
                        state <= S_INIT_TILE;
                    end
                end

                // 1. Initialize for a new Output Channel Tile
                S_INIT_TILE: begin
		    for (i = 0; i < IC_PAR; i = i + 1) begin acc[i] <= 0; end
                    cnt_ht <= 0; cnt_row <= 0; cnt_col <= 0;
                    // curr_read_addr <= src_addr_base + (cnt_oc * stride_ic_tile_words * 64);
                    // state <= S_READ_REQ;
		    state <= S_CALC_0;
                end

		S_CALC_0: begin
		    r_stride_row_words <= (img_width / CHUNKS_PER_BEAT);
		    r_stride_ic_tile_words <= (img_width / CHUNKS_PER_BEAT) << log2_mem_tile_height;
		    r_num_h_tiles_minus_1 <= ((img_height + (16'd1 << log2_mem_tile_height) - 1) >> log2_mem_tile_height) - 1;
		    state <= S_CALC_1;
		end

		S_CALC_1: begin
		    r_stride_height_tile_words <= r_stride_ic_tile_words * (img_channels / IC_PAR);
		    tile_oc_offset <= cnt_oc * r_stride_ic_tile_words * 64;
		    state <= S_CALC_2;
		end

		S_CALC_2: begin
		    tile_ht_offset <= cnt_ht * r_stride_height_tile_words * 64;
		    state <= S_CALC_3;
		end

		S_CALC_3: begin
		    curr_read_addr <= src_addr_base_r + tile_oc_offset;
		    state <= S_CALC_4;
		end

		S_CALC_4: begin
		    curr_read_addr <= curr_read_addr + tile_ht_offset;
		    state <= S_READ_REQ;
		end

                // 2. Read Request
                S_READ_REQ: begin
                    m_axi_araddr <= curr_read_addr;
                    m_axi_arvalid <= 1;
		    state <= S_READ_ACK;
                end

		S_READ_ACK: begin
		    if (m_axi_arready) begin
			m_axi_arvalid <= 0;
			m_axi_rready <= 1;
			state <= S_READ_WAIT;
		    end
		end

                // 3. Read Data
                S_READ_WAIT: begin
                    if (m_axi_rvalid) begin
			rdata_latch <= m_axi_rdata;
                        m_axi_rready <= 0;
			beat_cntr <= 0;
                        state <= S_ACCUM_REAL;
                    end
                end

		S_ACCUM_REAL: begin
		    if (beat_cntr >= 4) begin
			beat_cntr <= 0;
			state <= S_ACCUM;
		    end else begin
                        if ((cnt_col * 4 + beat_cntr) < img_width) begin
                            for (c = 0; c < IC_PAR; c = c + 1) begin
                                acc[c] <= acc[c] + $signed(rdata_latch[beat_cntr*128 + c*8 +: 8]);
                            end
			end
			beat_cntr <= beat_cntr + 1;
		    end
		end

                // 4. Accumulate (Same as before)
                S_ACCUM: begin
                    if ((cnt_col + 1) * 4 >= img_width) begin
                        cnt_col <= 0;
                        if (cnt_row == rows_per_tile - 1) begin
                            cnt_row <= 0;
                            // if (cnt_ht == num_h_tiles - 1) begin
                            if (cnt_ht == r_num_h_tiles_minus_1) begin
				cnt_ht <= 0;
                                state <= S_WRITE_REQ;
                            end else begin
                                cnt_ht <= cnt_ht + 1;
                                // curr_read_addr <= src_addr_base + 
                                //                  ((cnt_ht + 1) * stride_height_tile_words * 64) + 
                                //                   (cnt_oc * stride_ic_tile_words * 64);
                                // state <= S_READ_REQ;
				state <= S_CALC_0;
                            end
                        end else begin
                            cnt_row <= cnt_row + 1;
                            curr_read_addr <= curr_read_addr + 64;
                            state <= S_READ_REQ;
                        end
                    end else begin
                        cnt_col <= cnt_col + 1;
                        curr_read_addr <= curr_read_addr + 64;
                        state <= S_READ_REQ;
                    end
                end

                // 5. Write Result (Averaging & Padding)
		S_WRITE_REQ: begin
		    m_axi_wdata <= (write_beat_cnt == 0) ? averaged_data : 512'd0;
                    m_axi_awaddr <= curr_write_addr;
                    m_axi_awvalid <= 1;
		    state <= S_WRITE_ACK;

                end

		S_WRITE_ACK: begin
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 0;
                        state <= S_WRITE_DATA;
                    end
		end

                S_WRITE_DATA: begin
                    m_axi_wvalid <= 1;
		    state <= S_WRITE_DATA_ACK;
                end

		S_WRITE_DATA_ACK: begin
                    if (m_axi_wready) begin
                        m_axi_wvalid <= 0;
                        m_axi_bready <= 1;
                        state <= S_WRITE_RESP;
                    end
		end

		S_WRITE_RESP: begin
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 0;
                        curr_write_addr <= curr_write_addr + 64; 
                        
                        if (write_beat_cnt < BEATS_PER_STRIP - 1) begin
                            write_beat_cnt <= write_beat_cnt + 1;
                            state <= S_WRITE_REQ; // Loop back to write padding
                        end else begin
                            write_beat_cnt <= 0;
                            if (cnt_oc == num_oc_tiles - 1) begin
				m_axi_awaddr <= 0;
				m_axi_wdata <= 0;
				curr_write_addr <= 0;
                                state <= S_DONE;
                            end else begin
                                cnt_oc <= cnt_oc + 1;
                                state <= S_INIT_TILE;
                            end
                        end
                    end
                end
               
                S_DONE: begin
                    done <= 1;
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
