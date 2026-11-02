`timescale 1ns / 1ps

module mac_lane #(
    parameter IC_PAR = 16,      
    parameter DATA_WIDTH = 8,   
    parameter ACC_WIDTH = 28    
)(
    input wire clk,
    input wire rst_n,
    input wire en,          
    input wire clear_acc,   
    input wire valid_in,    
    
    input wire signed [IC_PAR-1:0][DATA_WIDTH-1:0] activations,
    input wire signed [IC_PAR-1:0][DATA_WIDTH-1:0] weights,
    
    (* use_dsp = "no" *) output reg signed [ACC_WIDTH-1:0] result,
    output reg valid_out
);

    // --- DYNAMIC PIPELINE DEPTH CALCULATION ---
    // Depth = 1 (Input Reg) + 1 (Product) + log2(IC_PAR) (Adder Tree)
    // Increased by 1 due to input registering
    localparam TREE_DEPTH = $clog2(IC_PAR);
    localparam PIPE_DEPTH = 2 + TREE_DEPTH; 
    
    (* keep = "true" *) reg [PIPE_DEPTH-1:0] valid_pipe;
    (* keep = "true" *) reg [PIPE_DEPTH-1:0] clear_pipe;

    // --- CONTROL PIPELINE ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipe <= 0;
            clear_pipe <= 0;
        end else begin
            // Shift in new control signals
            // Note: en controls the input registers, so valid/clear must propagate
            // We assume valid_in aligns with activations at the input.
            valid_pipe <= {valid_pipe[PIPE_DEPTH-2:0], en ? valid_in : 1'b0};
            clear_pipe <= {clear_pipe[PIPE_DEPTH-2:0], clear_acc};
        end
    end

    wire valid_delayed = valid_pipe[PIPE_DEPTH-1];
    wire clear_delayed = clear_pipe[PIPE_DEPTH-1];
    
    // --- DATA PIPELINE ---
    
    // Stage 0: Input Registers (TIMING FIX)
    // Breaks the path from Window Sequencer Mux to Multiplier
    reg signed [DATA_WIDTH-1:0] act_reg [0:IC_PAR-1];
    reg signed [DATA_WIDTH-1:0] wgt_reg [0:IC_PAR-1];
    
    integer i;
    always @(posedge clk) begin
        if (en) begin
            for (i=0; i<IC_PAR; i=i+1) begin
                act_reg[i] <= activations[i];
                wgt_reg[i] <= weights[i];
            end
        end
        // If !en, hold value (or zero if you prefer, but hold is fine for DSP)
    end

    // Stage 1: Products
    // Multiplies the REGISTERED inputs
    /*
    (* use_dsp = "yes" *) reg signed [15:0] products [0:IC_PAR-1];
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i<IC_PAR; i=i+1) products[i] <= 0;
        end else begin
            // Always run multiplier, control logic handles validity downstream
            // or use 'en' to gate power if needed.
            // Using 'en' here matches previous logic style.
            // Note: act_reg is already gated by 'en' above.
            for (i=0; i<IC_PAR; i=i+1) 
                products[i] <= $signed(act_reg[i]) * $signed(wgt_reg[i]);
        end
    end

    // --- PARAMETERIZED ADDER TREE ---
    reg signed [ACC_WIDTH-1:0] tree [0:TREE_DEPTH][0:IC_PAR-1];

    // Connect Stage 0 to Products
    always @(*) begin
        for (i = 0; i < IC_PAR; i = i + 1) begin
            tree[0][i] = $signed(products[i]);
        end
    end

    // Generate Reduction Stages
    genvar s, k;
    generate
        for (s = 0; s < TREE_DEPTH; s = s + 1) begin : STAGE
            localparam COUNT = IC_PAR >> (s + 1);
            for (k = 0; k < COUNT; k = k + 1) begin : ADDER
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        tree[s+1][k] <= 0;
                    end else begin
                        tree[s+1][k] <= tree[s][2*k] + tree[s][2*k+1];
                    end
                end
            end
        end
    endgenerate

    wire signed [ACC_WIDTH-1:0] dot_product = tree[TREE_DEPTH][0];

    // --- ACCUMULATOR ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result <= 0;
            valid_out <= 0;
        end else begin
            valid_out <= valid_delayed;
            if (clear_delayed) begin
                result <= (valid_delayed) ? dot_product : 0;
            end else if (valid_delayed) begin
                result <= result + dot_product;
            end
        end
    end
    */
    // Stage 1: Products (NO ASYNC RESET)
    (* use_dsp = "yes" *) reg signed [15:0] products [0:IC_PAR-1];
    always @(posedge clk) begin
        for (i=0; i<IC_PAR; i=i+1) 
            products[i] <= $signed(act_reg[i]) * $signed(wgt_reg[i]);
    end

    // --- PARAMETERIZED ADDER TREE (NO ASYNC RESET) ---
    (* use_dsp = "no" *) reg signed [ACC_WIDTH-1:0] tree [0:TREE_DEPTH][0:IC_PAR-1];

    always @(*) begin
        for (i = 0; i < IC_PAR; i = i + 1) begin
            tree[0][i] = $signed(products[i]);
        end
    end

    genvar s, k;
    generate
        for (s = 0; s < TREE_DEPTH; s = s + 1) begin : STAGE
            localparam COUNT = IC_PAR >> (s + 1);
            for (k = 0; k < COUNT; k = k + 1) begin : ADDER
                always @(posedge clk) begin
                    tree[s+1][k] <= tree[s][2*k] + tree[s][2*k+1];
                end
            end
        end
    endgenerate

    wire signed[ACC_WIDTH-1:0] dot_product = tree[TREE_DEPTH][0];

    // --- ACCUMULATOR (SEPARATED CONTROL AND DATA) ---
    // Control gets the async reset
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_out <= 0;
        end else begin
            valid_out <= valid_delayed;
        end
    end

    // Data path gets no reset (it initializes itself when clear_delayed is high)
    always @(posedge clk) begin
        if (clear_delayed) begin
            result <= (valid_delayed) ? dot_product : 0;
        end else if (valid_delayed) begin
            result <= result + dot_product;
        end
    end

endmodule
