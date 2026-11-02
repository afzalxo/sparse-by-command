`default_nettype none
/*
 * burst_axi_write_master
 * Fully pipelined to allow overlapping AW, W, and B phases.
 */
module burst_axi_write_master #(
    parameter AXI_ADDR_WIDTH = 64,
    parameter AXI_DATA_WIDTH = 512,
    parameter MAX_BURST_LEN  = 64,
    parameter MAX_OUTSTANDING = 8
) (
    input wire clk,
    input wire rst_n, 

    // --- Control Interface ---
    input wire start, 
    input wire[AXI_ADDR_WIDTH-1:0] base_address,
    input wire [15:0]              total_words_to_write,

    // --- Data Input Interface (FIFO-like) ---
    input wire[AXI_DATA_WIDTH-1:0]  data_in,
    input wire                      data_valid, 
    output wire                     data_ready, 

    // --- AXI4 Master Write Interface ---
    output reg                            m_axi_awvalid,
    input wire                            m_axi_awready,
    output reg[AXI_ADDR_WIDTH-1:0]       m_axi_awaddr,
    output reg[7:0]                      m_axi_awlen,
    output wire [2:0]                     m_axi_awsize,
    output wire [1:0]                     m_axi_awburst,

    output wire                           m_axi_wvalid,
    input wire                            m_axi_wready,
    output wire [AXI_DATA_WIDTH-1:0]      m_axi_wdata,
    output wire [AXI_DATA_WIDTH/8-1:0]    m_axi_wstrb,
    output wire                           m_axi_wlast,

    input wire                            m_axi_bvalid,
    output wire                           m_axi_bready,
    input wire [1:0]                      m_axi_bresp,

    output reg done 
);

    assign m_axi_awsize  = $clog2(AXI_DATA_WIDTH/8);
    assign m_axi_awburst = 2'b01; 
    assign m_axi_wstrb   = {(AXI_DATA_WIDTH/8){1'b1}};
    assign m_axi_bready  = 1'b1;

    reg busy;
    reg [31:0] aw_words_left;
    reg [31:0] w_words_left;
    reg [31:0] b_bursts_left;
    reg [7:0]  outstanding_bursts;
    reg[7:0]  w_beat_count;
    
    reg [31:0] aw_issued_count;
    reg [31:0] w_issued_count;

    wire safe_start = start && !busy && (total_words_to_write > 0);
    
    wire aw_fire = m_axi_awvalid && m_axi_awready;
    wire w_fire  = m_axi_wvalid && m_axi_wready;
    wire b_fire  = m_axi_bvalid && m_axi_bready;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 0;
            done <= 0;
        end else begin
            done <= 0;
            if (start && !busy) begin
                if (total_words_to_write > 0) busy <= 1;
                else done <= 1; // Immediately done if 0 words
            end else if (busy && b_fire && (b_bursts_left == 1)) begin
                busy <= 0;
                done <= 1;
            end
        end
    end

    wire [31:0] next_aw_burst_size = (aw_words_left > MAX_BURST_LEN) ? MAX_BURST_LEN : aw_words_left;
    wire can_issue_aw = (outstanding_bursts < MAX_OUTSTANDING) && (aw_words_left > 0);
    
    // AW Channel
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awvalid <= 0;
            m_axi_awaddr  <= 0;
            m_axi_awlen   <= 0;
            aw_words_left <= 0;
        end else if (safe_start) begin
            m_axi_awvalid <= 0;
            m_axi_awaddr  <= base_address;
            aw_words_left <= total_words_to_write;
        end else begin
            if (can_issue_aw && !m_axi_awvalid) begin
                m_axi_awvalid <= 1;
                m_axi_awlen   <= next_aw_burst_size - 1;
            end else if (aw_fire) begin
                m_axi_awvalid <= 0;
                m_axi_awaddr  <= m_axi_awaddr + ((m_axi_awlen + 1) * (AXI_DATA_WIDTH/8));
                aw_words_left <= aw_words_left - (m_axi_awlen + 1);
            end
        end
    end

    // Outstanding Bursts Counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) outstanding_bursts <= 0;
        else if (safe_start) outstanding_bursts <= 0;
        else outstanding_bursts <= outstanding_bursts + aw_fire - b_fire;
    end
    
    // B Bursts Left Counter
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) b_bursts_left <= 0;
        else if (safe_start) b_bursts_left <= (total_words_to_write + MAX_BURST_LEN - 1) / MAX_BURST_LEN;
        else if (b_fire) b_bursts_left <= b_bursts_left - 1;
    end
    
    // W Channel Logic
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_issued_count <= 0;
            w_issued_count <= 0;
        end else if (safe_start) begin
            aw_issued_count <= 0;
            w_issued_count <= 0;
        end else begin
            if (aw_fire) aw_issued_count <= aw_issued_count + 1;
            if (w_fire && m_axi_wlast) w_issued_count <= w_issued_count + 1;
        end
    end

    reg [31:0] latched_w_burst_size;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            latched_w_burst_size <= 0;
        end else if (safe_start) begin
            latched_w_burst_size <= (total_words_to_write > MAX_BURST_LEN) ? MAX_BURST_LEN : total_words_to_write;
        end else if (w_fire && m_axi_wlast) begin
            latched_w_burst_size <= ((w_words_left - 1) > MAX_BURST_LEN) ? MAX_BURST_LEN : (w_words_left - 1);
        end
    end

    wire w_allowed = (aw_issued_count > w_issued_count) || aw_fire;

    assign m_axi_wvalid = data_valid && w_allowed && (w_words_left > 0);
    assign data_ready   = m_axi_wready && w_allowed && (w_words_left > 0);
    assign m_axi_wdata  = data_in;
    assign m_axi_wlast  = (w_beat_count == latched_w_burst_size - 1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            w_words_left <= 0;
            w_beat_count <= 0;
        end else if (safe_start) begin
            w_words_left <= total_words_to_write;
            w_beat_count <= 0;
        end else if (w_fire) begin
            w_words_left <= w_words_left - 1;
            if (m_axi_wlast) w_beat_count <= 0;
            else w_beat_count <= w_beat_count + 1;
        end
    end
endmodule
`default_nettype wire
