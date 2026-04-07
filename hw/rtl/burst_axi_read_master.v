`default_nettype none
/*
 * Pipelined burst_axi_read_master
 * Decouples AR and R channels to allow Multiple Outstanding Transactions.
 * Hides HBM round-trip latency.
 */
module burst_axi_read_master #(
    parameter AXI_ADDR_WIDTH = 64,
    parameter AXI_DATA_WIDTH = 512,
    parameter MAX_BURST_LEN  = 64,
    parameter MAX_OUTSTANDING = 8  // Number of requests to run ahead
) (
    input wire clk,
    input wire rst_n,

    // --- Control Interface ---
    input wire start,
    input wire [AXI_ADDR_WIDTH-1:0] base_address,

    // --- AXI4 Master Read Interface ---
    output reg                            m_axi_arvalid,
    input wire                            m_axi_arready,
    output reg[AXI_ADDR_WIDTH-1:0]        m_axi_araddr,
    output reg [7:0]                      m_axi_arlen,
    output wire [2:0]                     m_axi_arsize,
    output wire [1:0]                     m_axi_arburst, 

    input wire                            m_axi_rvalid,
    output wire                           m_axi_rready,
    input wire[AXI_DATA_WIDTH-1:0]        m_axi_rdata,
    input wire                            m_axi_rlast,

    // --- Data Output Interface ---
    input  wire [15:0]                    total_words_to_read,
    output wire                           data_valid,
    input  wire 			  data_ready,
    output wire[AXI_DATA_WIDTH-1:0]       data_out,
    output reg                            done
);

    assign m_axi_arsize  = $clog2(AXI_DATA_WIDTH/8);
    assign m_axi_arburst = 2'b01; // INCR burst type
    
    // Internal State
    reg [31:0] ar_words_left;
    reg[31:0] r_words_left;
    reg [7:0]  outstanding_bursts;

    // --- 0. Busy Flag & Start Protection (THE FIX) ---
    // This perfectly mimics the original FSM's IDLE state protection.
    // It prevents spurious 'start' pulses from downstream DMAs 
    // from resetting the counters in the middle of an active transaction.
    reg busy;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            busy <= 0;
        end else begin
            if (start && !busy) begin
                busy <= 1; // Lock the doors
            end else if (m_axi_rvalid && m_axi_rready && (r_words_left == 1)) begin
                busy <= 0; // Unlock when the very last word arrives
            end
        end
    end

    // Only accept a start if we are completely idle
    wire safe_start = start && !busy;

    // We are always ready to receive data if a read is active.
    assign m_axi_rready  = (r_words_left > 0) && data_ready; 
    
    assign data_out      = m_axi_rdata;
    assign data_valid    = m_axi_rvalid && m_axi_rready;

    wire [31:0] next_burst_size = (ar_words_left > MAX_BURST_LEN) ? MAX_BURST_LEN : ar_words_left;
    wire can_issue = (outstanding_bursts < MAX_OUTSTANDING) && (ar_words_left > 0);

    wire ar_fire = m_axi_arvalid && m_axi_arready;
    wire r_last_fire = m_axi_rvalid && m_axi_rready && m_axi_rlast;

    // --- 1. Address Channel (AR) Process ---
    wire [31:0] ar_words_after_fire = ar_words_left - (m_axi_arlen + 1);
    wire can_issue_after_fire = (outstanding_bursts + 1 < MAX_OUTSTANDING) && (ar_words_after_fire > 0);
    wire [31:0] next_burst_after_fire = (ar_words_after_fire > MAX_BURST_LEN) ? MAX_BURST_LEN : ar_words_after_fire;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_arvalid <= 0;
            m_axi_araddr  <= 0;
            m_axi_arlen   <= 0;
            ar_words_left <= 0;
        end else if (safe_start) begin
            m_axi_arvalid <= 0;
            m_axi_araddr  <= base_address;
            ar_words_left <= total_words_to_read;
        end else begin
            if (ar_fire) begin
                m_axi_araddr  <= m_axi_araddr + ((m_axi_arlen + 1) * (AXI_DATA_WIDTH/8));
                ar_words_left <= ar_words_after_fire;
                if (can_issue_after_fire) begin
                    // Back-to-back: keep arvalid high, issue next burst immediately
                    m_axi_arvalid <= 1;
                    m_axi_arlen   <= next_burst_after_fire - 1;
                end else begin
                    m_axi_arvalid <= 0;
                end
            end else if (can_issue && !m_axi_arvalid) begin
                m_axi_arvalid <= 1;
                m_axi_arlen   <= next_burst_size - 1;
            end
        end
    end

    // --- 2. Outstanding Transaction Counter ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) outstanding_bursts <= 0;
        else if (safe_start) outstanding_bursts <= 0;
        else outstanding_bursts <= outstanding_bursts + ar_fire - r_last_fire;
    end

    // --- 3. Data Channel (R) Process ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            r_words_left <= 0;
            done <= 0;
        end else begin
            done <= 0;
            if (safe_start) begin
                r_words_left <= total_words_to_read;
            end else if (m_axi_rvalid && m_axi_rready) begin
                r_words_left <= r_words_left - 1;
                if (r_words_left == 1) begin
                    done <= 1; // Pulse done when the very last word is received
                end
            end
        end
    end

endmodule
`default_nettype wire
