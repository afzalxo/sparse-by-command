`timescale 1ns / 1ps

module result_accumulator #(
    parameter IC_PAR = 16,
    parameter OC_PAR = 16,
    parameter PP_PAR = 8,
    parameter ACC_WIDTH = 28,
    parameter DEPTH = 64,
    parameter ADDR_WIDTH = $clog2(DEPTH) // log2(64)
)(
    input wire clk,
    input wire rst_n,
    
    input wire comp_out_bank,
    input wire dma_out_bank,
    // Control
    input wire clear_ptr,      // Reset internal pointers (Start of Tile)
    input wire accumulate_en,  // 0 = Overwrite (First IC), 1 = Add (Next ICs)

    // Accelerator Interface (Push)
    input wire signed [PP_PAR-1:0][OC_PAR-1:0][ACC_WIDTH-1:0] acc_in_data,
    input wire acc_in_valid,
    input wire [1:0] stride,
    
    // DMA Interface (Pull)
    input wire [ADDR_WIDTH-1:0] dma_raddr,
    input wire dma_ren,
    output wire [PP_PAR*OC_PAR*ACC_WIDTH-1:0] dma_rdata_packed
);

    localparam LANE_WIDTH = OC_PAR * ACC_WIDTH;

    // --- Internal Signals ---
    reg [ADDR_WIDTH-1:0] acc_ptr;
    reg pipe_out_bank;
    reg pipe2_out_bank;
    reg pipe3_out_bank;
    
    // Pipeline Registers (Delay 1 cycle for RAM Read Latency)
    reg [ADDR_WIDTH-1:0] pipe_addr;
    reg signed [PP_PAR-1:0][OC_PAR-1:0][ACC_WIDTH-1:0] pipe_data;
    reg pipe_valid;
    reg pipe_acc_en;
    reg [1:0] pipe_stride;

    // RAM Signals
    wire [PP_PAR-1:0] ram_wen;
    wire [ADDR_WIDTH:0] ram_waddr;
    wire [LANE_WIDTH-1:0] ram_wdata [0:PP_PAR-1];
    
    wire ram_ren;
    wire [LANE_WIDTH-1:0] ram_rdata [0:PP_PAR-1];

    // --- 1. Input Address Logic ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_ptr <= 0;
        end else if (clear_ptr) begin
            acc_ptr <= 0;
        end else if (acc_in_valid) begin
            acc_ptr <= acc_ptr + 1;
        end
    end

    // --- 2. Pipeline Stage (Cycle 0 -> 1) ---
    // Existing pipeline registers
    always @(posedge clk) begin
	pipe_out_bank <= comp_out_bank;
        pipe_addr     <= acc_ptr;
        pipe_data     <= acc_in_data;
        pipe_valid    <= acc_in_valid;
        pipe_acc_en   <= accumulate_en;
        pipe_stride   <= stride;
    end

    // --- Pipeline Stage 2 (Cycle 1 -> 2) ---
    // Delay signals to match URAM Read Latency
    reg [ADDR_WIDTH-1:0] pipe2_addr;
    reg signed [PP_PAR-1:0][OC_PAR-1:0][ACC_WIDTH-1:0] pipe2_data;
    reg pipe2_valid;
    reg pipe2_acc_en;
    reg [1:0] pipe2_stride;

    always @(posedge clk) begin
	pipe2_out_bank <= pipe_out_bank;
        pipe2_addr     <= pipe_addr;
        pipe2_data     <= pipe_data;
        pipe2_valid    <= pipe_valid;
        pipe2_acc_en   <= pipe_acc_en;
        pipe2_stride   <= pipe_stride;
    end

    // --- 3. Packing Logic (Moved to Stage 2) ---
    // Use pipe2 signals
    /*
    reg signed [PP_PAR-1:0][OC_PAR-1:0][ACC_WIDTH-1:0] packed_pipe_data;
    reg [PP_PAR-1:0] packed_wen_mask;
    integer i;
    reg [3:0] valid_idx;

    always @(*) begin
        packed_pipe_data = 0;
        packed_wen_mask = 0;
        valid_idx = 0;
        for (i = 0; i < PP_PAR; i = i + 1) begin
            if (i % pipe2_stride == 0) begin
                packed_pipe_data[valid_idx] = pipe2_data[i];
                packed_wen_mask[valid_idx] = 1'b1;
                valid_idx = valid_idx + 1;
            end
        end
    end

    // Read Address comes from Stage 1 (pipe_addr) or DMA
    // Write Address comes from Stage 2 (pipe2_addr)
    
    // Mux Read Address: DMA takes priority
    // Note: When accumulating, we read at pipe_addr (Cycle 1).
    // The data arrives at rdata in Cycle 2.
    assign ram_raddr = dma_ren ? dma_raddr : pipe_addr;
    
    // Read Enable: DMA or Valid Input (Stage 1)
    assign ram_ren   = dma_ren || pipe_valid;

    genvar p;
    generate
        for (p = 0; p < PP_PAR; p = p + 1) begin : GEN_RAMS
            
            // Write Enable: Valid (Stage 2) AND Mask
            assign ram_wen[p] = pipe2_valid && packed_wen_mask[p];
            assign ram_waddr  = pipe2_addr;

            // ALU: Add or Overwrite
            // Inputs: ram_rdata (Available in Cycle 2 from Read at Cycle 1)
            //         packed_pipe_data (From Stage 2)
            reg [LANE_WIDTH-1:0] write_val_packed;
            always @(*) begin
                for (integer j = 0; j < OC_PAR; j = j + 1) begin
                    reg signed [ACC_WIDTH-1:0] old_val;
                    reg signed [ACC_WIDTH-1:0] new_val;
                    reg signed [ACC_WIDTH-1:0] sum;
                    
                    old_val = ram_rdata[p][j*ACC_WIDTH +: ACC_WIDTH];
                    new_val = packed_pipe_data[p][j];
                    
                    if (pipe2_acc_en) sum = old_val + new_val;
                    else              sum = new_val;
                    
                    write_val_packed[j*ACC_WIDTH +: ACC_WIDTH] = sum;
                end
            end
            assign ram_wdata[p] = write_val_packed;

            // RAM Instance
            ram_sdp #(
                .DATA_WIDTH(LANE_WIDTH),
                .ADDR_WIDTH(ADDR_WIDTH),
                .RAM_STYLE("ultra")
            ) mem (
                .clk(clk),
                .waddr(ram_waddr),
                .wdata(ram_wdata[p]),
                .wen(ram_wen[p]),
                .raddr(ram_raddr),
                .ren(ram_ren),
                .rdata(ram_rdata[p])
            );
            
            // Output Masking (Use pipe2_stride or just stride? DMA is separate)
            // For DMA read, we don't use the pipeline stride.
            // We use the 'stride' input directly? No, DMA reads are burst.
            // The masking logic for DMA read should use the current 'stride' input
            // assuming DMA happens when accelerator is idle/done.
	    assign dma_rdata_packed[p*LANE_WIDTH +: LANE_WIDTH] = ram_rdata[p];
                
        end
    endgenerate
    */

    reg signed[PP_PAR-1:0][OC_PAR-1:0][ACC_WIDTH-1:0] packed_pipe_data;
    reg [PP_PAR-1:0] packed_wen_mask;
    integer i;
    reg [3:0] valid_idx;

    always @(*) begin
        packed_pipe_data = 0;
        packed_wen_mask = 0;
        valid_idx = 0;
        for (i = 0; i < PP_PAR; i = i + 1) begin
            if (i % pipe2_stride == 0) begin
                packed_pipe_data[valid_idx] = pipe2_data[i];
                packed_wen_mask[valid_idx] = 1'b1;
                valid_idx = valid_idx + 1;
            end
        end
    end

    // --- NEW: Pipeline Stage 3 (Cycle 2 -> 3) ---
    // This pipeline stage severs the critical path between URAM Read -> Adder -> URAM Write
    reg [ADDR_WIDTH-1:0] pipe3_addr;
    reg pipe3_valid;
    reg pipe3_acc_en;
    reg [PP_PAR-1:0] pipe3_wen_mask;
    reg signed [PP_PAR-1:0][OC_PAR-1:0][ACC_WIDTH-1:0] pipe3_packed_data;

    // Registers to capture the RAM output safely
    reg [LANE_WIDTH-1:0] r_ram_rdata[0:PP_PAR-1];

    always @(posedge clk) begin
        // Pipeline the control and data signals
	pipe3_out_bank <= pipe2_out_bank;
        pipe3_addr <= pipe2_addr;
        pipe3_valid <= pipe2_valid;
        pipe3_acc_en <= pipe2_acc_en;
        pipe3_wen_mask <= packed_wen_mask;
        pipe3_packed_data <= packed_pipe_data;

        // Capture the URAM read data into registers
        for (integer p_idx = 0; p_idx < PP_PAR; p_idx = p_idx + 1) begin
            r_ram_rdata[p_idx] <= ram_rdata[p_idx];
        end
    end

    // Mux Read Address: DMA takes priority
    // Read at pipe_addr (Cycle 1). Data arrives at rdata in Cycle 2.
    wire [ADDR_WIDTH:0] ram_raddr = dma_ren ? {dma_out_bank, dma_raddr} : {pipe_out_bank, pipe_addr};
    
    // Read Enable: DMA or Valid Input (Stage 1)
    assign ram_ren   = dma_ren || pipe_valid;

    genvar p;
    generate
        for (p = 0; p < PP_PAR; p = p + 1) begin : GEN_RAMS
            
            // Write Enable: Valid (Stage 3) AND Mask
            assign ram_waddr  = {pipe3_out_bank, pipe3_addr};
            assign ram_wen[p] = pipe3_valid && pipe3_wen_mask[p];

            // ALU: Add or Overwrite
            // Inputs: r_ram_rdata (Available in Cycle 3 from Register)
            //         pipe3_packed_data (From Stage 3)
            reg [LANE_WIDTH-1:0] write_val_packed;
            always @(*) begin
                for (integer j = 0; j < OC_PAR; j = j + 1) begin
                    reg signed [ACC_WIDTH-1:0] old_val;
                    reg signed [ACC_WIDTH-1:0] new_val;
                    reg signed [ACC_WIDTH-1:0] sum;
                    
                    old_val = r_ram_rdata[p][j*ACC_WIDTH +: ACC_WIDTH];
                    new_val = pipe3_packed_data[p][j];
                    
                    if (pipe3_acc_en) sum = old_val + new_val;
                    else              sum = new_val;
                    
                    write_val_packed[j*ACC_WIDTH +: ACC_WIDTH] = sum;
                end
            end
            assign ram_wdata[p] = write_val_packed;

            // RAM Instance
            ram_sdp #(
                .DATA_WIDTH(LANE_WIDTH),
                .ADDR_WIDTH(ADDR_WIDTH + 1),
                .RAM_STYLE("ultra")
            ) mem (
                .clk(clk),
                .waddr(ram_waddr),
                .wdata(ram_wdata[p]),
                .wen(ram_wen[p]),
                .raddr(ram_raddr),
                .ren(ram_ren),
                .rdata(ram_rdata[p])
            );
            
            // Output Masking
            // DMA reads bypass the pipeline and read directly from ram_rdata (Cycle 2)
	    assign dma_rdata_packed[p*LANE_WIDTH +: LANE_WIDTH] = ram_rdata[p];
                
        end
    endgenerate

endmodule
