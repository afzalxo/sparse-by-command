`timescale 1ns / 1ps

module processing_element #(
    parameter IC_PAR = 16,
    parameter OC_PAR = 16,
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    input wire en,
    input wire clear_acc,
    input wire valid_in,
    
    // Input: 1 Pixel (16 Channels) - Packed
    input wire signed [IC_PAR-1:0][DATA_WIDTH-1:0] pixel_in,
    
    // Weights: 16 Filters x 16 Channels - Packed 3D Array
    // [Filter_Index][Channel_Index][Bit_Index]
    input wire signed [OC_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] weights_in,
    
    // Output: 16 Output Channels - Packed
    output wire signed [OC_PAR-1:0][27:0] pixel_out,
    output wire valid_out
);

    wire [OC_PAR-1:0] valid_outs;
    assign valid_out = valid_outs[0];

    /*
    genvar k;
    generate
        for (k = 0; k < OC_PAR; k = k + 1) begin : GEN_LANES
            mac_lane #(
                .IC_PAR(IC_PAR),
                .DATA_WIDTH(DATA_WIDTH)
            ) lane_inst (
                .clk(clk),
                .rst_n(rst_n),
                .en(en),
                .clear_acc(clear_acc),
		.valid_in(valid_in),
                .activations(pixel_in),      // Broadcast same pixel to all lanes
                .weights(weights_in[k]),     // Slice the k-th filter (Returns 2D packed)
                .result(pixel_out[k]),
		.valid_out(valid_outs[k])
            );
        end
    endgenerate
    */
    genvar k;
    generate
        for (k = 0; k < OC_PAR; k = k + 1) begin : GEN_LANES
            // Level 2 Pipeline: Local copy for this specific Lane/DSP
            (* keep = "true" *) reg signed [IC_PAR-1:0][DATA_WIDTH-1:0] lane_pixel;
            (* keep = "true" *) reg signed [IC_PAR-1:0][DATA_WIDTH-1:0] lane_weight;
            (* keep = "true" *) reg lane_valid;
            (* keep = "true" *) reg lane_en;
            (* keep = "true" *) reg lane_clear;

            always @(posedge clk) begin
                lane_pixel <= pixel_in;
                lane_weight <= weights_in[k];
                lane_valid <= valid_in;
                lane_en <= en;
                lane_clear <= clear_acc;
            end

            mac_lane #(
                .IC_PAR(IC_PAR),
                .DATA_WIDTH(DATA_WIDTH)
            ) lane_inst (
                .clk(clk),
                .rst_n(rst_n),
                .en(lane_en),
                .clear_acc(lane_clear),
		.valid_in(lane_valid),
                .activations(lane_pixel),
                .weights(lane_weight),
                .result(pixel_out[k]),
		.valid_out(valid_outs[k])
            );
        end
    endgenerate

endmodule
