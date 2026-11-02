`timescale 1ns / 1ps

module bias_dma #(
    parameter OC_PAR = 16,
    parameter HBM_DATA_WIDTH = 512,
    parameter ADDR_WIDTH = 64,  // HBM Address width
    parameter BIAS_WIDTH = 32,
    parameter BIAS_TILE_BITS = OC_PAR * BIAS_WIDTH,
    parameter BEATS_PER_BIAS_TILE = (BIAS_TILE_BITS >= HBM_DATA_WIDTH)
                                  ? (BIAS_TILE_BITS / HBM_DATA_WIDTH) : 1
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Configuration
    input wire [15:0] oc,
    output wire [15:0] bias_words,
    input wire [15:0] oc_tile,  //  Current OC tile

    input wire [HBM_DATA_WIDTH-1:0] hbm_data_in,
    output reg [ADDR_WIDTH-1:0] hbm_addr,
    output reg hbm_ren,
    input wire hbm_rvalid,
    
    (* max_fanout = 20 *) output reg [15:0] bias_bram_addr,
    output reg [OC_PAR*BIAS_WIDTH-1:0] bias_data,
    output reg bias_wen,

    output wire done
);

    localparam BEATS_PER_BIAS_BLOCK = (BIAS_TILE_BITS >= HBM_DATA_WIDTH)
                                    ? (BIAS_TILE_BITS / HBM_DATA_WIDTH) : 1;
    // State Machine
    localparam S_IDLE = 0;
    localparam S_READ = 1;
    localparam S_WAIT = 2;
    localparam S_DONE = 3;
    
    reg [2:0] state;
    reg [15:0] beat_count;
    reg done_r;
    assign done = done_r;
    
    wire [31:0] tile_start_index = oc_tile * BEATS_PER_BIAS_TILE;
    assign bias_words = BEATS_PER_BIAS_TILE;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done_r <= 0;
            hbm_ren <= 0;
            bias_wen <= 0;
	    beat_count <= 0;
        end else begin
            hbm_ren <= 0;
            bias_wen <= 0;
            done_r <= 0;
            
            case (state)
                S_IDLE: begin
                    if (start) begin
			beat_count <= 0;
                        bias_bram_addr <= 0;
                        state <= S_READ;
                    end
                end
                
                S_READ: begin
		    hbm_addr <= tile_start_index;
		    hbm_ren <= 1; 
		    state <= S_WAIT;
		end

		S_WAIT: begin
		    if (hbm_rvalid) begin
			// Write to bram_wdata at the correct position
			// We need to write BEATS_PER_KERNEL_PIXEL beats for
			// each kernel pixel, then enable bram_wen
		        if (BIAS_TILE_BITS >= HBM_DATA_WIDTH)
		            bias_data[(beat_count % BEATS_PER_BIAS_TILE) * HBM_DATA_WIDTH +: HBM_DATA_WIDTH] <= hbm_data_in;
		        else
		            bias_data[0 +: BIAS_TILE_BITS] <= hbm_data_in[BIAS_TILE_BITS-1:0];
			if ((beat_count + 1) % BEATS_PER_BIAS_TILE == 0) begin
		            bias_wen <= 1;
                            bias_bram_addr <= bias_bram_addr + 1;
			end

			if (beat_count < bias_words - 1) begin
			    beat_count <= beat_count + 1;
			    state <= S_WAIT;
			end
			if (beat_count == bias_words - 1) begin 
			    state <= S_DONE;
			    beat_count <= 0;
			    hbm_ren <= 0;
			end
	            end
                end
                
                S_DONE: begin
                    done_r <= 1;
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
