`timescale 1ns / 1ps

module memcpy_unit #(
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 512
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    // Configuration
    input wire [ADDR_WIDTH-1:0] src_addr_base,
    input wire [ADDR_WIDTH-1:0] dst_addr_base,
    input wire [31:0] length_bytes, // Total bytes to copy

    // AXI Read Master (Source)
    output reg m_axi_arvalid,
    input wire m_axi_arready,
    output reg [ADDR_WIDTH-1:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst,
    input wire m_axi_rvalid,
    output reg m_axi_rready,
    input wire [DATA_WIDTH-1:0] m_axi_rdata,

    // AXI Write Master (Destination)
    output reg m_axi_awvalid,
    input wire m_axi_awready,
    output reg [ADDR_WIDTH-1:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst,
    output reg m_axi_wvalid,
    input wire m_axi_wready,
    output reg [DATA_WIDTH-1:0] m_axi_wdata,
    output wire [DATA_WIDTH/8-1:0] m_axi_wstrb,
    output wire m_axi_wlast,
    input wire m_axi_bvalid,
    output reg m_axi_bready,
    input wire [1:0] m_axi_bresp,

    output reg done
);

    // --- AXI CONSTANTS (Single Beat) ---
    assign m_axi_arlen = 8'd0;
    assign m_axi_arsize = 3'b110; // 64 bytes
    assign m_axi_arburst = 2'b01;
    
    assign m_axi_awlen = 8'd0;
    assign m_axi_awsize = 3'b110;
    assign m_axi_awburst = 2'b01;
    assign m_axi_wstrb = {(DATA_WIDTH/8){1'b1}};
    assign m_axi_wlast = 1'b1;

    // --- STATE MACHINE ---
    localparam S_IDLE       = 0;
    localparam S_READ_REQ   = 1;
    localparam S_READ_WAIT  = 2;
    localparam S_WRITE_REQ  = 3;
    localparam S_WRITE_DATA = 4;
    localparam S_WRITE_RESP = 5;
    localparam S_DONE       = 6;
    localparam S_READ_ACK   = 7;
    localparam S_WRITE_ACK  = 8;
    localparam S_WRITE_DATA_ACK = 9;

    reg [3:0] state;
    reg [31:0] bytes_copied;
    reg [DATA_WIDTH-1:0] data_buffer;
    reg [ADDR_WIDTH-1:0] curr_src;
    reg [ADDR_WIDTH-1:0] curr_dst;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 0;
            m_axi_arvalid <= 0;
            m_axi_rready <= 0;
            m_axi_awvalid <= 0;
            m_axi_wvalid <= 0;
            m_axi_bready <= 0;
            bytes_copied <= 0;
            curr_src <= 0;
            curr_dst <= 0;
        end else begin
            // Default
            done <= 0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        curr_src <= src_addr_base;
                        curr_dst <= dst_addr_base;
                        bytes_copied <= 0;
                        state <= S_READ_REQ;
                    end
                end

                // 1. Read Request
                S_READ_REQ: begin
                    m_axi_araddr <= curr_src;
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

                // 2. Read Data
                S_READ_WAIT: begin
                    if (m_axi_rvalid) begin
                        data_buffer <= m_axi_rdata;
                        m_axi_rready <= 0;
                        state <= S_WRITE_REQ;
                    end
                end

                // 3. Write Request
                S_WRITE_REQ: begin
                    m_axi_awaddr <= curr_dst;
                    m_axi_awvalid <= 1;
		    state <= S_WRITE_ACK;
                end

		S_WRITE_ACK: begin
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 0;
                        state <= S_WRITE_DATA;
                    end
		end

                // 4. Write Data
                S_WRITE_DATA: begin
                    m_axi_wdata <= data_buffer;
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

                // 5. Write Response & Loop Check
                S_WRITE_RESP: begin
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 0;
                        
                        // Update Pointers (64 bytes per word)
                        curr_src <= curr_src + 64;
                        curr_dst <= curr_dst + 64;
                        bytes_copied <= bytes_copied + 64;
                        
                        if (bytes_copied + 64 >= length_bytes) begin
                            state <= S_DONE;
                        end else begin
                            state <= S_READ_REQ;
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
