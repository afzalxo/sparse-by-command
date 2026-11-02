`timescale 1ns / 1ps

module line_buffer #(
    parameter IC_PAR = 16,
    parameter PP_PAR = 8,
    parameter DATA_WIDTH = 8,
    parameter MAX_IMG_WIDTH = 256
)(
    input wire clk,
    input wire rst_n,
    input wire shift_en,
    input wire [15:0] img_width_strips,
    input wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] data_in,
    
    // Outputs are now driven by internal registers, stable during compute
    output wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] row_0,
    output wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] row_1,
    output wire signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] row_2
);

    /*
    localparam RAM_DEPTH = MAX_IMG_WIDTH / PP_PAR;
    
    (* ram_style = "distributed" *)
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] ram_0 [0:RAM_DEPTH-1];
    (* ram_style = "distributed" *)
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] ram_1 [0:RAM_DEPTH-1];
    
    (* max_fanout = 20 *) reg [15:0] wr_ptr;
    (* max_fanout = 20 *) reg [15:0] rd_ptr;

    // --- OUTPUT REGISTERS ---
    // These hold the 3x3 window data stable for the Sequencer
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] r_row_0;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] r_row_1;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] r_row_2;

    assign row_0 = r_row_0;
    assign row_1 = r_row_1;
    assign row_2 = r_row_2;

    // Initialize RAM (Simulation only)
    integer i;
    initial begin
        for (i = 0; i < RAM_DEPTH; i = i + 1) begin
            ram_0[i] = 0;
            ram_1[i] = 0;
        end
    end

    (* max_fanout = 20 *) reg local_sync_rst_n;
    always @(posedge clk) local_sync_rst_n <= rst_n;

    // reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] temp_ram_1;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            r_row_0 <= 0;
            r_row_1 <= 0;
            r_row_2 <= 0;
	    // temp_ram_1 <= 0;
        end else if (shift_en) begin
            // --- 1. READ PHASE (Capture History) ---
            // Latch the current input
            r_row_2 <= data_in;
            
            // Latch the OLD data from RAM before we overwrite it
            // This ensures row_1 gets the data from the PREVIOUS image row
            r_row_1 <= ram_1[rd_ptr];
	    // temp_ram_1 <= ram_1[rd_ptr];
	    // r_row_1 <= temp_ram_1;
            
            // Latch the OLDER data
            r_row_0 <= ram_0[rd_ptr];

	    if (local_sync_rst_n) begin
                // --- 2. WRITE PHASE (Update History) ---
                // Write new input to RAM 1 (to become row_1 next time we visit this column)
                ram_1[wr_ptr] <= data_in;
                
                // Move what was in RAM 1 to RAM 0 (to become row_0 next time)
                // Note: In Verilog non-blocking assignments, ram_1[rd_ptr] on the RHS
                // refers to the value at the START of the clock cycle (the old value).
                // So this correctly moves the data down the line.
                ram_0[wr_ptr] <= ram_1[rd_ptr];
	    end
            
            // --- 3. POINTER UPDATE ---
            if (wr_ptr == img_width_strips - 1) begin
                wr_ptr <= 0;
                rd_ptr <= 0;
            end else begin
                wr_ptr <= wr_ptr + 1;
                rd_ptr <= rd_ptr + 1;
            end
        end
    end
    */
    localparam RAM_DEPTH = MAX_IMG_WIDTH / PP_PAR;
    localparam PTR_W = $clog2(RAM_DEPTH);
    
    // CHANGED: Force Block RAM
    (* ram_style = "distributed" *)
    reg signed[PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] ram_0 [0:RAM_DEPTH-1];
    (* ram_style = "distributed" *)
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] ram_1 [0:RAM_DEPTH-1];

    (* max_fanout = 20 *) reg local_sync_rst_n;
    always @(posedge clk) local_sync_rst_n <= rst_n;

    (* max_fanout = 20 *) reg [15:0] wr_ptr;
    (* max_fanout = 20 *) reg [15:0] rd_ptr;
    reg shift_en_d1;
    reg local_sync_rst_n_d1;
    // NEW: Pipeline registers for delayed BRAM write
    reg [15:0] wr_ptr_d1;

    // --- OUTPUT REGISTERS ---
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] r_row_0;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] r_row_1;
    reg signed [PP_PAR-1:0][IC_PAR-1:0][DATA_WIDTH-1:0] r_row_2;

    assign row_0 = r_row_0;
    assign row_1 = r_row_1;
    assign row_2 = r_row_2;

    always @(posedge clk) begin
        if (shift_en) begin
            if (local_sync_rst_n) begin
                ram_1[wr_ptr[PTR_W-1:0]] <= data_in;
            end
        end
        
        if (shift_en_d1 && local_sync_rst_n_d1) begin
            ram_0[wr_ptr_d1[PTR_W-1:0]] <= r_row_1;
        end
    end

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
            r_row_0 <= 0;
            r_row_1 <= 0;
            r_row_2 <= 0;
            
            wr_ptr_d1 <= 0;
            shift_en_d1 <= 0;
            local_sync_rst_n_d1 <= 0;
        end else begin
            // Shift pipeline registers
            shift_en_d1 <= shift_en;
            local_sync_rst_n_d1 <= local_sync_rst_n;

            if (shift_en) begin
                // --- 1. READ PHASE (Capture History) ---
                r_row_2 <= data_in;
                // r_row_1 <= ram_1[rd_ptr];
		r_row_1 <= ram_1[rd_ptr[PTR_W-1:0]];
                // r_row_0 <= ram_0[rd_ptr];
		r_row_0 <= ram_0[rd_ptr[PTR_W-1:0]];

                // --- 2. WRITE PHASE (Update History) ---
		/*
                if (local_sync_rst_n) begin
                    // ram_1[wr_ptr] <= data_in;
                    ram_1[wr_ptr[PTR_W-1:0]] <= data_in;
                end
		*/
                
                // Save the pointer for the delayed write
                wr_ptr_d1 <= wr_ptr;
                
                // --- 3. POINTER UPDATE ---
                if (wr_ptr == img_width_strips - 1) begin
                    wr_ptr <= 0;
                    rd_ptr <= 0;
                end else begin
                    wr_ptr <= wr_ptr + 1;
                    rd_ptr <= rd_ptr + 1;
                end
            end
            
            // --- 4. PIPELINED WRITE TO RAM 0 ---
            // Because r_row_1 captured ram_1 on the previous shift_en, 
            // we safely write it into ram_0 one cycle later!
	    /*
            if (shift_en_d1 && local_sync_rst_n_d1) begin
                // ram_0[wr_ptr_d1] <= r_row_1;
                ram_0[wr_ptr_d1[PTR_W-1:0]] <= r_row_1;
            end
	    */
        end
    end

endmodule
