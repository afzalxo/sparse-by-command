`timescale 1ns / 1ps

module window_sequencer #(
    parameter IC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8
)(
    input wire clk,
    input wire rst_n,
    
    input wire [1:0] kernel_y, 
    input wire [1:0] kernel_x, 
    input wire load_new_data,  
    
    input wire [15:0] col_idx,          
    input wire [15:0] img_width_strips, 
    input wire [15:0] row_idx,          
    input wire [15:0] img_height,       
    
    input wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] lb_row_0,
    input wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] lb_row_1,
    input wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] lb_row_2,
    
    output wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] pixels_out
);

    // --- REGISTERS ---
    // Current Strip (Being processed)
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] curr_strip_0;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] curr_strip_1;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] curr_strip_2;

    // Next Strip (Look-ahead buffer)
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] next_strip_0;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] next_strip_1;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] next_strip_2;

    // Previous Last Pixel (For left overlap)
    reg signed [IC_PAR-1:0][DATA_WIDTH-1:0] prev_last_pixel_0;
    reg signed [IC_PAR-1:0][DATA_WIDTH-1:0] prev_last_pixel_1;
    reg signed [IC_PAR-1:0][DATA_WIDTH-1:0] prev_last_pixel_2;
    
    // --- LATCH LOGIC (Pipeline Shift) ---
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Reset all registers...
            curr_strip_0 <= 0; next_strip_0 <= 0; prev_last_pixel_0 <= 0;
            curr_strip_1 <= 0; next_strip_1 <= 0; prev_last_pixel_1 <= 0;
            curr_strip_2 <= 0; next_strip_2 <= 0; prev_last_pixel_2 <= 0;
        end else if (load_new_data) begin
            // 1. Save Tail of Current (Old) Strip
            prev_last_pixel_0 <= curr_strip_0[PP_PAR-1];
            prev_last_pixel_1 <= curr_strip_1[PP_PAR-1];
            prev_last_pixel_2 <= curr_strip_2[PP_PAR-1];

            // 2. Shift Next -> Current
            curr_strip_0 <= next_strip_0;
            curr_strip_1 <= next_strip_1;
            curr_strip_2 <= next_strip_2;

            // 3. Load New Input -> Next
            next_strip_0 <= lb_row_0;
            next_strip_1 <= lb_row_1;
            next_strip_2 <= lb_row_2;
        end
    end

    // --- MUX LOGIC ---
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] selected_curr;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] selected_next;
    reg signed [IC_PAR-1:0][DATA_WIDTH-1:0] selected_prev_pixel;
    
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] pixels_out_comb;
    (* max_fanout = 20 *) reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] pixels_out_reg;

    always @(posedge clk) begin
        pixels_out_reg <= pixels_out_comb;
    end
    // assign pixels_out = pixels_out_comb;
    assign pixels_out = pixels_out_comb;
    // 1. Select Row (Vertical Padding)
    always @(*) begin
        selected_curr = 0; selected_next = 0; selected_prev_pixel = 0;

        case (kernel_y)
            2'd0: begin // TOP
                selected_curr = curr_strip_0; 
                selected_next = next_strip_0;
                selected_prev_pixel = prev_last_pixel_0; 
            end
            2'd1: begin // CENTER
                selected_curr = curr_strip_1; 
                selected_next = next_strip_1;
                selected_prev_pixel = prev_last_pixel_1; 
            end
            2'd2: begin // BOTTOM
                selected_curr = curr_strip_2; 
                selected_next = next_strip_2;
                selected_prev_pixel = prev_last_pixel_2; 
            end
        endcase
    end

    // 2. Shift (Horizontal Padding & Lookahead)
    integer p;
    always @(*) begin
        for (p = 0; p < PP_PAR; p = p + 1) begin
            case (kernel_x)
                2'd0: begin // Left
                    if (p == 0) begin
                        if (col_idx == 0) pixels_out_comb[p] = 0; 
                        else              pixels_out_comb[p] = selected_prev_pixel;
                    end else begin
                        pixels_out_comb[p] = selected_curr[p-1];
                    end
                end
                
                2'd1: begin // Center
                    pixels_out_comb[p] = selected_curr[p];
                end
                
                2'd2: begin // Right
                    if (p == PP_PAR-1) begin
                        // LOOKAHEAD LOGIC
                        if (col_idx == img_width_strips - 1) begin
                            pixels_out_comb[p] = 0; // Image Boundary
                        end else begin
                            // Use the first pixel of the NEXT strip
                            pixels_out_comb[p] = selected_next[0]; 
                        end
                    end else begin
                        pixels_out_comb[p] = selected_curr[p+1];
                    end
                end
                default: pixels_out_comb[p] = 0;
            endcase
        end
    end

endmodule
