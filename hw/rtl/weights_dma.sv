`timescale 1ns / 1ps

module weights_dma #(
    parameter IC_PAR = 16,
    parameter OC_PAR = 16,
    parameter DATA_WIDTH = 8,
    parameter HBM_DATA_WIDTH = 512,
    // Ensure IC_PAR * OC_PAR is divisible by HBM_DATA_WIDTH
    parameter BEATS_PER_KERNEL_PIXEL = (IC_PAR * OC_PAR * DATA_WIDTH + HBM_DATA_WIDTH - 1) / HBM_DATA_WIDTH,
    parameter ADDR_WIDTH = 64,  // HBM Address width
    parameter BIAS_WIDTH = 32,
    parameter BRAM_DEPTH = 3 * 3
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Configuration
    input wire [15:0] oc,
    input wire [15:0] ic,
    output wire [15:0] weight_words,
    input wire [15:0] ic_tile,   // Current IC tile index
    input wire [15:0] oc_tile,  //  Current OC tile
    input wire is_conv,

    // HBM Interface (Simulated as a large array port)
    input wire [HBM_DATA_WIDTH-1:0] hbm_data_in,
    output reg [ADDR_WIDTH-1:0] hbm_addr,
    output reg hbm_ren,
    input wire hbm_rvalid,
    
    // BRAM Interface (Write Port)
    (* max_fanout = 20 *) output reg [15:0] bram_addr,
    output reg [IC_PAR*OC_PAR*DATA_WIDTH-1:0] bram_wdata,
    (* max_fanout = 20 *) output reg bram_wen,

    output wire done
);

    // State Machine
    localparam S_IDLE = 0;
    localparam S_READ = 1;
    localparam S_WAIT = 2;
    localparam S_DONE = 3;
    
    reg [2:0] state;
    reg [15:0] beat_count;
    reg done_r;
    assign done = done_r;
    
    // Each tile has 3 x 3 IC_PAR x OC_PAR weights and weights are stored as:
    // [OC/OC_PAR][IC/IC_PAR][9][OC_PAR][IC_PAR] in HBM
    // Calculate start index for the current tile at [oc_tile][ic_tile]
    wire [31:0] kernel_size = is_conv ? 9 : 1;
    wire [31:0] tile_start_index = ic_tile * ((kernel_size * OC_PAR * IC_PAR) / (HBM_DATA_WIDTH / 8)) + 
	    		 	   oc_tile * (((ic / IC_PAR) * kernel_size * OC_PAR * IC_PAR) / (HBM_DATA_WIDTH / 8));
    assign weight_words = (kernel_size * OC_PAR * IC_PAR) / (HBM_DATA_WIDTH / 8);
    // Need BEATS_PER_KERNEL_PIXEL words per kernel pixel

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done_r <= 0;
            hbm_ren <= 0;
            bram_wen <= 0;
	    beat_count <= 0;
        end else begin
            hbm_ren <= 0;
            bram_wen <= 0;
            done_r <= 0;
            
            case (state)
                S_IDLE: begin
                    if (start) begin
			beat_count <= 0;
                        bram_addr <= is_conv ? 0 : 4;
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
		        bram_wdata[(beat_count % BEATS_PER_KERNEL_PIXEL) * HBM_DATA_WIDTH +: HBM_DATA_WIDTH] <= hbm_data_in; 
			if ((beat_count + 1) % BEATS_PER_KERNEL_PIXEL == 0) begin
		            bram_wen <= 1;
			    // if (is_conv) begin
                            bram_addr <= bram_addr + 1;
			    // end else begin
			    // end
			end

			if (beat_count < weight_words - 1) begin
			    beat_count <= beat_count + 1;
			    state <= S_WAIT;
			end
			if (beat_count == weight_words - 1) begin 
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
