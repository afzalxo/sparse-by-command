`timescale 1ns / 1ps

module tiled_dma #(
    parameter IC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8,
    parameter TILE_HEIGHT = 4,
    parameter INPUT_TILE_HEIGHT = TILE_HEIGHT + 2,  // Top and bottom halo
    parameter HBM_DATA_WIDTH = 512,  // Ensure matches bus width
    parameter BEATS_PER_URAM_IDX = (IC_PAR * PP_PAR) / (HBM_DATA_WIDTH / DATA_WIDTH),
    parameter ADDR_WIDTH = 64,  // HBM Address width
    parameter URAM_DEPTH = 4096
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Configuration
    input wire [15:0] img_width,      // Full Image Width
    input wire [15:0] img_height,     // Full Image Height
    input wire [15:0] img_channels,   // Number of Channels
    input wire [15:0] tile_y_index,   // Current vertical tile index
    input wire [15:0] tile_ic_index,  // Current IC tile index
    input wire [15:0] active_height,
    input wire [2:0] log2_mem_tile_height, // (0=1, 1=2, 2=4)

    output wire [15:0] feature_map_words,
    
    // HBM Interface (Simulated as a large array port)
    input wire [HBM_DATA_WIDTH-1:0] hbm_data_in,
    output reg [ADDR_WIDTH-1:0] hbm_addr,
    output reg hbm_ren,
    input wire hbm_rvalid,
    output reg hbm_rready,
    
    // URAM Interface (Write Port)
    (* max_fanout = 20 *) output reg [15:0] uram_addr,
    (* max_fanout = 32 *) output reg [PP_PAR*IC_PAR*DATA_WIDTH-1:0] uram_wdata,
    (* max_fanout = 32 *) output reg uram_wen,
    
    output wire done
);

    localparam BITS_PER_VECTOR = IC_PAR * DATA_WIDTH;
    localparam VECTORS_PER_HBM_WORD = HBM_DATA_WIDTH / BITS_PER_VECTOR;  // 4

    // wire [31:0] stride_row_words = img_width / VECTORS_PER_HBM_WORD;  // 8 / 4 = 2
    reg [31:0] r_stride_row_words;
    
    // Size of one IC Tile Block (contains TILE_HEIGHT rows)
    // wire [31:0] stride_ic_tile_words = stride_row_words << log2_mem_tile_height;
    reg [31:0] r_stride_ic_tile_words;
    
    // Size of one Height Tile Block (contains all IC tiles for this height)
    // This is the jump required to go from Height Tile N to Height Tile N+1
    // wire [31:0] stride_height_tile_words = stride_ic_tile_words * (img_channels / IC_PAR);
    reg [31:0] r_stride_height_tile_words;

    // B. Row Iterator Logic
    // -------------------------------------
    // We need to load: [Start_Row, End_Row]
    // Start = Top Halo (-1 relative to tile)
    // End   = Bottom Halo (+TILE_HEIGHT relative to tile)
    wire signed [15:0] abs_start_row = (tile_y_index * TILE_HEIGHT) - 1;
    wire signed [15:0] abs_end_row   = (tile_y_index * TILE_HEIGHT) + active_height;
    
    // Current Row Iterator
    reg signed [15:0] curr_row;
    
    // C. Dynamic Address Calculation for 'curr_row'
    // -------------------------------------
    // Determine which physical Height Tile block this row belongs to.
    // This handles the Halo case where we read a row from the Next/Prev tile.
    // wire [15:0] target_height_tile = curr_row >>> log2_mem_tile_height;
    wire [15:0] row_mask = (16'd1 << log2_mem_tile_height) - 1;
    // wire [15:0] row_in_tile = curr_row & row_mask;
    reg [15:0] r_row_in_tile;
    reg [15:0] r_target_height_tile;

    reg [63:0] tile_h_offset;
    reg [63:0] tile_ic_offset;
    reg [63:0] tile_row_offset;
    
    // Final Base Address for the current row
    // wire [63:0] curr_row_base_addr = (target_height_tile * r_stride_height_tile_words) + 
    //                                (tile_ic_index      * r_stride_ic_tile_words) + 
    //                                (row_in_tile        * r_stride_row_words);
    reg [63:0] r_curr_row_base_addr;

    // assign feature_map_words = r_stride_row_words;
    reg [15:0] feature_map_words_reg;
    assign feature_map_words = feature_map_words_reg;

    // State Machine
    localparam S_IDLE = 0;
    localparam S_CALC_0 = 1;
    localparam S_CALC_1 = 2;
    localparam S_CALC_2 = 3;
    localparam S_CHECK_BOUNDS = 4;
    localparam S_READ = 5;
    localparam S_PAD = 6;
    localparam S_DONE = 7;
    
    reg [3:0] tiled_dma_state;
    assign hbm_rready = (tiled_dma_state == S_READ);

    reg [15:0] beat_count;
    reg [15:0] row_beat_count;
    reg signed [15:0] segment_rows_reg;
    reg [15:0] req_count;
    reg [7:0] uram_beat_count;
    reg [15:0] pad_count;
    reg done_reg;
    assign done = done_reg;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            tiled_dma_state <= S_IDLE;
            done_reg <= 0;
            hbm_ren <= 0;
	    hbm_addr <= 0;
            uram_wen <= 0;
	    uram_addr <= 0;
	    curr_row <= 0;
	    beat_count <= 0;
	    row_beat_count <= 0;
	    uram_beat_count <= 0;
	    uram_wdata <= 0;
	    req_count <= 0;
	    r_stride_ic_tile_words <= 0;
	    r_stride_height_tile_words <= 0;
	    r_curr_row_base_addr <= 0;
	    r_stride_row_words <= 0;
        end else begin
            hbm_ren <= 0;
            uram_wen <= 0;
            done_reg <= 0;
            
            case (tiled_dma_state)
                S_IDLE: begin
                    if (start) begin
			curr_row <= abs_start_row;
                        uram_addr <= 0;
			uram_beat_count <= 0;
			// tiled_dma_state <= S_CHECK_BOUNDS;
			tiled_dma_state <= S_CALC_0;
                    end
                end

		S_CALC_0: begin
		    r_stride_row_words <=  (img_width + VECTORS_PER_HBM_WORD - 1) / VECTORS_PER_HBM_WORD;
		    r_stride_ic_tile_words <= ((img_width + VECTORS_PER_HBM_WORD - 1) / VECTORS_PER_HBM_WORD) << log2_mem_tile_height;
		    r_stride_height_tile_words <= (((img_width + VECTORS_PER_HBM_WORD - 1) / VECTORS_PER_HBM_WORD) << log2_mem_tile_height) * (img_channels / IC_PAR);
		    r_target_height_tile <= curr_row >>> log2_mem_tile_height;
		    r_row_in_tile <= curr_row & row_mask;
		    tiled_dma_state <= S_CALC_1;
		end

		/*
		S_CALC_1: begin
		    tile_h_offset <= r_target_height_tile * r_stride_height_tile_words;
		    tile_ic_offset <= tile_ic_index * r_stride_ic_tile_words;
		    tile_row_offset <= r_row_in_tile * r_stride_row_words;
		    tiled_dma_state <= S_CALC_2;
		end
		*/
		S_CALC_1: begin
		    tile_h_offset <= r_target_height_tile * r_stride_height_tile_words;
		    tile_ic_offset <= tile_ic_index * r_stride_ic_tile_words;
		    tile_row_offset <= r_row_in_tile * r_stride_row_words;
		    
		    begin
                        automatic logic signed[15:0] rows_per_tile = (16'd1 << log2_mem_tile_height);
                        automatic logic signed [15:0] max_row_in_tile = (r_target_height_tile * rows_per_tile) + rows_per_tile - 1;
                        automatic logic signed [15:0] end_row = (abs_end_row < img_height - 1) ? abs_end_row : (img_height - 1);
                        if (max_row_in_tile < end_row) end_row = max_row_in_tile;
                        segment_rows_reg <= end_row - curr_row + 1;
                    end
		    tiled_dma_state <= S_CALC_2;
		end

		/*S_CALC_2: begin
		    // r_curr_row_base_addr <= (r_target_height_tile * r_stride_height_tile_words) +
		    //	   		    (tile_ic_index      * r_stride_ic_tile_words) + 
		    // 			    (r_row_in_tile        * r_stride_row_words);
		    r_curr_row_base_addr <= tile_h_offset + tile_ic_offset + tile_row_offset;
		    tiled_dma_state <= S_CHECK_BOUNDS;
		end
		*/
		S_CALC_2: begin
		    r_curr_row_base_addr <= tile_h_offset + tile_ic_offset + tile_row_offset;
		    feature_map_words_reg <= segment_rows_reg * r_stride_row_words;
		    tiled_dma_state <= S_CHECK_BOUNDS;
		end
		/*
		S_CHECK_BOUNDS: begin
                    if (curr_row > abs_end_row) begin
                        tiled_dma_state <= S_DONE;
                    end else begin
                        if (curr_row >= 0 && curr_row < img_height) begin
                            // hbm_addr <= curr_row_base_addr;
			    hbm_addr <= r_curr_row_base_addr;
			    hbm_ren <= 1;
                            beat_count <= 0;
                            uram_beat_count <= 0;
                            // Transition to REQ tiled_dma_state to assert REN next cycle
                            tiled_dma_state <= S_READ; 
                        end else begin
                            // PADDING ROW
                            uram_addr <= uram_addr + ((img_width + PP_PAR - 1) / PP_PAR);
                            curr_row <= curr_row + 1;
                            // tiled_dma_state <= S_CHECK_BOUNDS;
                            tiled_dma_state <= S_CALC_0;
                        end
                    end
                end
		*/
		S_CHECK_BOUNDS: begin
                    if (curr_row > abs_end_row) begin
                        tiled_dma_state <= S_DONE;
                    end else begin
                        if (curr_row >= 0 && curr_row < img_height) begin
			    hbm_addr <= r_curr_row_base_addr;
			    hbm_ren <= 1; // Assert start
                            beat_count <= 0;
                            row_beat_count <= 0;
                            uram_beat_count <= 0;
                            tiled_dma_state <= S_READ;
                        end else begin
                            // PADDING ROW: write explicit zeros
                            uram_wdata <= 0;
                            pad_count <= (img_width + PP_PAR - 1) / PP_PAR;
                            tiled_dma_state <= S_PAD;
                        end
                    end
                end

		/*
                S_READ: begin
                    if (hbm_rvalid) begin
                        uram_wdata[uram_beat_count * HBM_DATA_WIDTH +: HBM_DATA_WIDTH] <= hbm_data_in;
                        
                        // if (uram_beat_count == BEATS_PER_URAM_IDX - 1) begin
			if (uram_beat_count == BEATS_PER_URAM_IDX - 1 || beat_count == r_stride_row_words - 1) begin
                            uram_wen <= 1;
                            uram_addr <= uram_addr + 1;
			    // TODO: This is new to deal with edge cases where
			    // input is small
			    for (int i = 0; i < BEATS_PER_URAM_IDX; i++) begin
                                if (i > uram_beat_count) begin
                                    uram_wdata[i * HBM_DATA_WIDTH +: HBM_DATA_WIDTH] <= 0;
                                end
                            end
                            uram_beat_count <= 0;
                        end else begin
                            uram_beat_count <= uram_beat_count + 1;
                        end
                        
                        if (beat_count == r_stride_row_words - 1) begin
                            curr_row <= curr_row + 1;
                            hbm_ren <= 0;
                            beat_count <= 0;
                            // uram_beat_count <= 0;
                            tiled_dma_state <= S_CALC_0;
                        end else begin
                            beat_count <= beat_count + 1;
			    hbm_ren <= 1;
                            hbm_addr <= hbm_addr + 1; 
                        end
                    end
                end
		*/
		S_READ: begin
                    if (hbm_rvalid) begin
                        uram_wdata[uram_beat_count * HBM_DATA_WIDTH +: HBM_DATA_WIDTH] <= hbm_data_in;
                        
                        // Trigger URAM write on URAM_WORD boundary OR end of physical row
			if (uram_beat_count == BEATS_PER_URAM_IDX - 1 || row_beat_count == r_stride_row_words - 1) begin
                            uram_wen <= 1;
                            uram_addr <= uram_addr + 1;
			    for (int i = 0; i < BEATS_PER_URAM_IDX; i++) begin
                                if (i > uram_beat_count) begin
                                    uram_wdata[i * HBM_DATA_WIDTH +: HBM_DATA_WIDTH] <= 0;
                                end
                            end
                            uram_beat_count <= 0;
                        end else begin
                            uram_beat_count <= uram_beat_count + 1;
                        end
                        
                        // Track beats per physical row
                        if (row_beat_count == r_stride_row_words - 1) begin
                            row_beat_count <= 0;
                        end else begin
                            row_beat_count <= row_beat_count + 1;
                        end

                        // Track total beats for the entire segment burst
                        if (beat_count == feature_map_words_reg - 1) begin
                            curr_row <= curr_row + segment_rows_reg; // Jump by entire segment
                            hbm_ren <= 0;
                            beat_count <= 0;
                            tiled_dma_state <= S_CALC_0;
                        end else begin
                            beat_count <= beat_count + 1;
                            // NOTE: Removed `hbm_ren <= 1` and `hbm_addr <= hbm_addr + 1` from here!
                        end
                    end
                end
		S_PAD: begin
                    uram_wen <= 1;
                    uram_addr <= uram_addr + 1;
                    pad_count <= pad_count - 1;
                    if (pad_count == 1) begin
                        curr_row <= curr_row + 1;
                        tiled_dma_state <= S_CALC_0;
                    end
                end

                S_DONE: begin
                    done_reg <= 1;
                    tiled_dma_state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
