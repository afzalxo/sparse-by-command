`timescale 1ns / 1ps

module flatten_unit #(
    parameter DATA_WIDTH = 8,
    parameter CHANNELS_PAR = 16, // Matches OC_PAR of prev layer
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
    input wire [1:0]  stride, // Stride of the PREVIOUS layer (to calculate padding)

    // AXI Read Master (Connect to Mux)
    output reg m_axi_arvalid,
    input wire m_axi_arready,
    output reg [ADDR_WIDTH-1:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst,
    input wire m_axi_rvalid,
    output reg m_axi_rready,
    input wire [HBM_DATA_WIDTH-1:0] m_axi_rdata,
    input wire m_axi_rlast,

    // AXI Write Master (Connect to Mux)
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

    output reg done
);

    // --- CONSTANTS ---
    assign m_axi_arlen = 8'd0; // Single beat reads
    assign m_axi_arsize = 3'b110; // 64 bytes
    assign m_axi_arburst = 2'b01;
    
    assign m_axi_awlen = 8'd0; // Single beat writes
    assign m_axi_awsize = 3'b110;
    assign m_axi_awburst = 2'b01;
    assign m_axi_wstrb = {(HBM_DATA_WIDTH/8){1'b1}};
    assign m_axi_wlast = 1'b1; // Always last for single beat

    // --- ADDRESS CALCULATION LOGIC ---
    // We need to iterate through the Source Memory in the exact order Tile Manager wrote it.
    // Order: [Height_Tile] -> [OC_Tile] -> [Row] -> [Strip]
    
    // 1. Dimensions
    wire [15:0] out_width = img_width / stride;
    wire [15:0] out_height = img_height / stride;
    
    // 2. Loop Bounds
    wire [15:0] num_h_tiles = (img_height + TILE_HEIGHT - 1) / TILE_HEIGHT;
    wire [15:0] num_c_tiles = (img_channels + CHANNELS_PAR - 1) / CHANNELS_PAR;
    wire [15:0] num_w_strips = (img_width + PP_PAR - 1) / PP_PAR;
    
    // 3. Valid Pixels per Strip
    // If stride=2, PP_PAR=8, we have 4 valid pixels per strip.
    // If stride=1, we have 8.
    // Note: This assumes Width is aligned. If Width=4 and PP=8, valid is 4.
    wire [3:0] valid_pixels_per_strip = (out_width < (PP_PAR/stride)) ? out_width : (PP_PAR/stride);
    
    // Pixel Size
    localparam PIXEL_BITS = CHANNELS_PAR * DATA_WIDTH; // 128 bits
    
    // --- STATE MACHINE ---
    localparam S_IDLE = 0;
    localparam S_CALC_ADDR = 1;
    localparam S_READ_REQ = 2;
    localparam S_READ_WAIT = 3;
    localparam S_PACK = 4;
    localparam S_WRITE_REQ = 5;
    localparam S_WRITE_DATA = 6;
    localparam S_WRITE_RESP = 7;
    localparam S_DONE = 8;

    reg [3:0] state;
    
    // Loop Counters
    reg [15:0] cnt_ht, cnt_ct, cnt_row, cnt_ws;
    
    // Packing Buffer
    reg [HBM_DATA_WIDTH-1:0] pack_buffer;
    reg [9:0] pack_ptr; // Bit pointer (0 to 512)

    // Address Tracking
    reg [63:0] curr_read_addr;
    reg [63:0] curr_write_addr;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 0;
            m_axi_arvalid <= 0;
            m_axi_rready <= 0;
            m_axi_awvalid <= 0;
            m_axi_wvalid <= 0;
            m_axi_bready <= 0;
            pack_ptr <= 0;
        end else begin
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        cnt_ht <= 0; cnt_ct <= 0; cnt_row <= 0; cnt_ws <= 0;
                        pack_ptr <= 0;
                        pack_buffer <= 0;
                        curr_write_addr <= dst_addr_base;
                        state <= S_CALC_ADDR;
                    end
                end

                // 1. Calculate Read Address for current Strip
                S_CALC_ADDR: begin
                    // Address Math:
                    // We need to reconstruct the Tiled Address.
                    // Stride_Row = 64 bytes (1 HBM word) * num_w_strips
                    // Stride_CT = Stride_Row * Active_Height_In_Tile
                    // Stride_HT = Stride_CT * Num_C_Tiles
                    
                    // To save logic, we can just increment a pointer IF the memory is contiguous.
                    // BUT, the hardware writes padding rows if TILE_HEIGHT > Active_Height.
                    // However, output_dma skips invalid rows.
                    // So the memory IS contiguous!
                    // We can just increment read address by 64 bytes every loop.
                    
                    if (cnt_ht == 0 && cnt_ct == 0 && cnt_row == 0 && cnt_ws == 0)
                        curr_read_addr <= src_addr_base;
                    else
                        curr_read_addr <= curr_read_addr + 64; // Next 512-bit word

                    state <= S_READ_REQ;
                end

                // 2. Read Strip from HBM
                S_READ_REQ: begin
                    m_axi_araddr <= curr_read_addr;
                    m_axi_arvalid <= 1;
                    if (m_axi_arready) begin
                        m_axi_arvalid <= 0;
                        m_axi_rready <= 1;
                        state <= S_READ_WAIT;
                    end
                end

                S_READ_WAIT: begin
                    if (m_axi_rvalid) begin
                        m_axi_rready <= 0;
                        // We have data in m_axi_rdata.
                        // It contains 'valid_pixels_per_strip' valid pixels starting at LSB.
                        // The rest is padding.
                        state <= S_PACK;
                    end
                end

                // 3. Pack Valid Pixels into Buffer
                S_PACK: begin
                    // Append valid bits to pack_buffer
                    // Note: This is a simplified behavioral description. 
                    // In real HW, you might need a loop or muxes.
                    // Assuming valid_pixels is small (4), we can do it in one cycle or loop.
                    
                    integer i;
                    reg [HBM_DATA_WIDTH-1:0] temp_buffer;
                    reg [9:0] temp_ptr;
                    
                    temp_buffer = pack_buffer;
                    temp_ptr = pack_ptr;
                    
                    for (i = 0; i < valid_pixels_per_strip; i = i + 1) begin
                        // Extract pixel i from rdata
                        // Append to temp_buffer at temp_ptr
                        temp_buffer[temp_ptr +: PIXEL_BITS] = m_axi_rdata[i*PIXEL_BITS +: PIXEL_BITS];
                        temp_ptr = temp_ptr + PIXEL_BITS;
                    end
                    
                    pack_buffer <= temp_buffer;
                    pack_ptr <= temp_ptr;
                    
                    // Check if buffer is full (>= 512 bits)
                    if (temp_ptr >= 512) begin
                        state <= S_WRITE_REQ;
                    end else begin
                        // Not full, fetch next strip
                        // Update Loop Counters
                        if (cnt_ws == num_w_strips - 1) begin
                            cnt_ws <= 0;
                            // Calculate active height for this tile
                            // (Same logic as verify_output)
                            // ... simplified for brevity ...
                            if (cnt_row == (TILE_HEIGHT/stride) - 1) begin // Approx
                                cnt_row <= 0;
                                if (cnt_ct == num_c_tiles - 1) begin
                                    cnt_ct <= 0;
                                    if (cnt_ht == num_h_tiles - 1) begin
                                        // Done reading all inputs!
                                        // Flush remaining buffer
                                        state <= S_WRITE_REQ; 
                                    end else begin
                                        cnt_ht <= cnt_ht + 1;
                                        state <= S_CALC_ADDR;
                                    end
                                end else begin
                                    cnt_ct <= cnt_ct + 1;
                                    state <= S_CALC_ADDR;
                                end
                            end else begin
                                cnt_row <= cnt_row + 1;
                                state <= S_CALC_ADDR;
                            end
                        end else begin
                            cnt_ws <= cnt_ws + 1;
                            state <= S_CALC_ADDR;
                        end
                    end
                end

                // 4. Write Packed Data to HBM
                S_WRITE_REQ: begin
                    m_axi_awaddr <= curr_write_addr;
                    m_axi_awvalid <= 1;
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 0;
                        state <= S_WRITE_DATA;
                    end
                end

                S_WRITE_DATA: begin
                    m_axi_wdata <= pack_buffer[0 +: 512]; // Write bottom 512 bits
                    m_axi_wvalid <= 1;
                    if (m_axi_wready) begin
                        m_axi_wvalid <= 0;
                        m_axi_bready <= 1;
                        state <= S_WRITE_RESP;
                        
                        // Shift buffer down
                        // If we had 600 bits, we wrote 512, now we have 88 left.
                        if (pack_ptr >= 512) begin
                            pack_buffer <= pack_buffer >> 512;
                            pack_ptr <= pack_ptr - 512;
                        end else begin
                            // Flushing partial data
                            pack_ptr <= 0;
                        end
                        
                        curr_write_addr <= curr_write_addr + 64;
                    end
                end

                S_WRITE_RESP: begin
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 0;
                        
                        // If we still have data (flush case) or we were just flushing
                        if (pack_ptr > 0) begin
                            // Write remaining
                            state <= S_WRITE_REQ;
                        end else if (cnt_ht == num_h_tiles - 1 && cnt_ct == num_c_tiles - 1 /*...*/) begin
                            // Truly done
                            state <= S_DONE;
                        end else begin
                            // Go back to reading
                            state <= S_CALC_ADDR;
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
