`timescale 1ns / 1ps

module conv_accelerator #(
    parameter IC_PAR = 16,
    parameter OC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8,
    parameter MAX_IMG_WIDTH = 256
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    input wire [15:0] img_width_strips,
    input wire [15:0] img_height,
    input wire [15:0] img_channels,
    
    input wire signed [OC_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] weights,
    output wire [1:0] k_x,
    output wire [1:0] k_y,
    input wire [1:0] stride,
    input wire is_conv,
    output wire weight_req,
    input wire weight_ack,
    
    input wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] din_data,
    input wire din_valid,
    output wire din_ready,
    
    output wire signed [PP_PAR-1:0][OC_PAR-1:0][27:0] dout_data,
    output wire dout_valid,
    output wire done
);

    wire lb_shift_en, seq_load, cu_en, cu_clear;
    wire [15:0] col_idx, row_idx;
    
    wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] row_0, row_1, row_2;
    wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] seq_out;
    
    // NEW: Wire for valid out (unused but good for debug)
    wire cu_valid_out;

    wire conv_controller_dout_valid;
    assign dout_valid = is_conv ? conv_controller_dout_valid : cu_valid_out;

    conv_controller #(
        .MAX_IMG_WIDTH(MAX_IMG_WIDTH),
	.IC_PAR(IC_PAR)
    ) ctrl (
        .clk(clk), .rst_n(rst_n), .start(start),
        .img_width_strips(img_width_strips), .img_height(img_height),
        .din_valid(din_valid), .din_ready(din_ready),
	.num_ic_tiles(img_channels / IC_PAR),
	.weight_req(weight_req),
	.weight_ack(weight_ack),
        .lb_shift_en(lb_shift_en), .seq_load(seq_load),
        .k_x(k_x), .k_y(k_y),
	.stride(stride),
	.is_conv(is_conv),
        .cu_en(cu_en), .cu_clear(cu_clear),
        .col_idx(col_idx), .row_idx(row_idx),
        .dout_valid(conv_controller_dout_valid), .done(done)
    );

    line_buffer #(
        .IC_PAR(IC_PAR), .PP_PAR(PP_PAR), .DATA_WIDTH(DATA_WIDTH), .MAX_IMG_WIDTH(MAX_IMG_WIDTH)
    ) lb (
        .clk(clk), .rst_n(rst_n), .shift_en(lb_shift_en), 
        .img_width_strips(img_width_strips),
        .data_in(din_data),
        .row_0(row_0), .row_1(row_1), .row_2(row_2)
    );

    window_sequencer #(
        .IC_PAR(IC_PAR), .PP_PAR(PP_PAR), .DATA_WIDTH(DATA_WIDTH)
    ) seq (
        .clk(clk), .rst_n(rst_n),
        .kernel_y(k_y), .kernel_x(k_x), .load_new_data(seq_load),
        .col_idx(col_idx), .img_width_strips(img_width_strips),
        .row_idx(row_idx), .img_height(img_height),
        .lb_row_0(row_0), .lb_row_1(row_1), .lb_row_2(row_2),
        .pixels_out(seq_out)
    );

    reg r_cu_en;
    reg r_cu_clear;
    reg r_valid_in;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] r_seq_out;
    // reg signed [OC_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] r_weights;

    always @(posedge clk) begin
	/*if (!rst_n) begin
	    r_cu_en <= 0;
	    r_cu_clear <= 0;
	    r_valid_in <= 0;
	    r_seq_out <= 0;
	    // r_weights <= 0;
	end else begin
	*/
	    r_cu_en <= cu_en;
	    r_cu_clear <= cu_clear;
	    r_valid_in <= cu_en;
	    r_seq_out <= seq_out;
	    // r_weights <= weights;
    end

    vector_compute_unit #(
        .IC_PAR(IC_PAR), .OC_PAR(OC_PAR), .PP_PAR(PP_PAR), .DATA_WIDTH(DATA_WIDTH)
    ) cu (
        .clk(clk), .rst_n(rst_n),
	.en(r_cu_en),
	.clear_acc(r_cu_clear),
        .valid_in(r_valid_in),
        .img_vector_in(r_seq_out),
        .weights_in(weights),
        .img_vector_out(dout_data),
        .valid_out(cu_valid_out)
    );

endmodule
