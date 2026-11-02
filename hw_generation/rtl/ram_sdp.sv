`timescale 1ns / 1ps

module ram_sdp #(
    parameter DATA_WIDTH = 16*28, // 16 channels * 28 bits
    parameter ADDR_WIDTH = 6,   // Depth 64
    parameter RAM_STYLE  = "ultra" // "block" or "ultra"
)(
    input wire clk,
    
    // Port A: Write
    input wire [ADDR_WIDTH-1:0] waddr,
    input wire [DATA_WIDTH-1:0] wdata,
    input wire wen,
    
    // Port B: Read
    input wire [ADDR_WIDTH-1:0] raddr,
    input wire ren,
    output reg [DATA_WIDTH-1:0] rdata
);

    (* ram_style = RAM_STYLE *)
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];

    always @(posedge clk) begin
        if (wen) begin
            mem[waddr] <= wdata;
        end
    end

    always @(posedge clk) begin
        if (ren) begin
            rdata <= mem[raddr];
        end
    end

endmodule
