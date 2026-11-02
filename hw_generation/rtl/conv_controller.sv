`timescale 1ns / 1ps

module conv_controller #(
    parameter MAX_IMG_WIDTH = 256,
    parameter IC_PAR = 16,
    parameter LATENCY = 5 + $clog2(IC_PAR)
)(
    input wire clk,
    input wire rst_n,
    input wire start,
    
    input wire [15:0] img_width_strips,
    input wire [15:0] img_height,
    
    input wire din_valid,
    output reg din_ready,

    // (0, IC, IC_PAR) loop
    input wire [15:0] num_ic_tiles,
    output reg weight_req,
    input wire weight_ack,
    
    (* max_fanout = 20 *) output reg lb_shift_en,
    (* max_fanout = 20 *) output reg seq_load,
    (* max_fanout = 20 *) output reg [1:0] k_x,
    (* max_fanout = 20 *) output reg [1:0] k_y,
    (* max_fanout = 20 *) input wire [1:0] stride,
    (* max_fanout = 20 *) input wire is_conv,
    (* max_fanout = 20 *) output reg cu_en,
    (* max_fanout = 20 *) output reg cu_clear,
    
    output reg [15:0] col_idx,
    output reg [15:0] row_idx,
    
    output reg dout_valid,
    output reg done
);

    localparam S_IDLE        = 0;
    localparam S_PRIME_ROW   = 1; // Load Row 0
    localparam S_PRIME_SEQ_1 = 2; // Load Row 1, Strip 0
    localparam SEQ_LOAD_0    = 8; // Intermediate state to pulse seq_load
    localparam S_PRIME_SEQ_2 = 3; // Load Row 1, Strip 1
    localparam SEQ_LOAD_1    = 9; // Intermediate state to pulse seq_load
    localparam S_COMPUTE     = 4;
    localparam S_WAIT        = 5;
    localparam S_FETCH_NEXT  = 6; // Load Next Strip
    localparam S_FLUSH       = 10; // Flush Remaining Data
    localparam S_DONE        = 7;
    localparam S_SWITCH_IC   = 11;
    localparam S_SKIP_COMPUTE = 12;
    localparam S_FETCH_WAIT  = 13;
    localparam S_FETCH_PULSE_2 = 14;

    reg [3:0] state;
    reg [3:0] kernel_step;
    reg [15:0] load_col_cnt;
    reg [15:0] load_row_cnt;
    reg [3:0] pipe_flush_cnt;

    reg [15:0] height_limit; 
    reg row_is_valid;
    // assign k_x = (!is_conv) ? 2'd1 : (kernel_step % 3);
    // assign k_y = (!is_conv) ? 2'd1 : (kernel_step / 3);
    //
    wire [15:0] next_row_idx = row_idx + 1;
    
    // Helper for validity check (Optimized Modulo)
    // If stride=1, always true. If stride=2, check LSB is 0.
    wire next_row_mod_ok = (stride == 1) || (next_row_idx[0] == 0);
    wire next_row_in_bounds = (next_row_idx < height_limit);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            din_ready <= 0;
            lb_shift_en <= 0;
            seq_load <= 0;
            cu_en <= 0;
            cu_clear <= 1;
            dout_valid <= 0;
            done <= 0;
            col_idx <= 0;
            row_idx <= 0;
            load_col_cnt <= 0;
            load_row_cnt <= 0;
            kernel_step <= 0;
            pipe_flush_cnt <= 0;
	    weight_req <= 0;
	    height_limit <= 0;
	    row_is_valid <= 1;
	    k_x <= 0;
	    k_y <= 0;
        end else begin
            // Default Pulses
            lb_shift_en <= 0;
            seq_load <= 0;
            dout_valid <= 0;
            
            case (state)
                S_IDLE: begin
                    done <= 0;
                    if (start) begin
                        load_col_cnt <= 0;
                        load_row_cnt <= 0;
                        col_idx <= 0;
                        row_idx <= 0;
			height_limit <= img_height - 2;
			row_is_valid <= 1;
                        state <= S_PRIME_ROW;
                    end
                end

                // --- 1. PRIME VERTICAL (Row -1, 0) ---
                // Only shift Line Buffer. Do NOT pulse Sequencer.
                S_PRIME_ROW: begin
                    din_ready <= 1;
		    lb_shift_en <= 1;
                    if (din_valid && din_ready) begin
                        din_ready <= 0;
			lb_shift_en <= 0;
                        
                        if (load_col_cnt == img_width_strips - 1) begin
                            load_col_cnt <= 0;
			    if (load_row_cnt == 1) begin
                                load_row_cnt <= 2; 
                                state <= S_PRIME_SEQ_1;
                            end else begin
                                load_row_cnt <= load_row_cnt + 1;
                            end
                        end else begin
                            load_col_cnt <= load_col_cnt + 1;
                        end
                    end
                end

                // --- 2. PRIME HORIZONTAL 1 (Row 1, Strip 0) ---
                // Load Strip -> Pulse Sequencer (Fills 'next')
                S_PRIME_SEQ_1: begin
                    din_ready <= 1;
		    lb_shift_en <= 1;
                    if (din_valid && din_ready) begin
                        din_ready <= 0;
			lb_shift_en <= 0;
                        
                        if (load_col_cnt == img_width_strips - 1) begin
                            load_col_cnt <= 0;
                            load_row_cnt <= load_row_cnt + 1;
                        end else begin
                            load_col_cnt <= load_col_cnt + 1;
                        end

                        // If image is only 1 strip wide, we are ready. Else load 2nd strip.
                        if (img_width_strips == 1) state <= SEQ_LOAD_0;  // S_COMPUTE;
                        else state <= SEQ_LOAD_0;
                    end
                end

		SEQ_LOAD_0: begin
		    seq_load <= 1;
		    if (img_width_strips == 1) begin
			state <= SEQ_LOAD_1;
		    end else begin
		        state <= S_PRIME_SEQ_2;
		    end
		end

                // --- 3. PRIME HORIZONTAL 2 (Row 1, Strip 1) ---
                // Load Strip -> Pulse Sequencer (Fills 'curr', Refills 'next')
                S_PRIME_SEQ_2: begin
                    din_ready <= 1;
		    lb_shift_en <= 1;
                    if (din_valid && din_ready) begin
                        din_ready <= 0;
			lb_shift_en <= 0;
                        
                        if (load_col_cnt == img_width_strips - 1) begin
                            load_col_cnt <= 0;
                            load_row_cnt <= load_row_cnt + 1;
                        end else begin
                            load_col_cnt <= load_col_cnt + 1;
                        end

                        cu_clear <= 1;
                        kernel_step <= 0;
                        state <=  SEQ_LOAD_1;
                    end
                end

		SEQ_LOAD_1: begin
		    seq_load <= 1;
		    state <= S_COMPUTE;
		end

                // --- 4. COMPUTE ---
                S_COMPUTE: begin
		    if (!is_conv) begin
			cu_en <= 1;
			cu_clear <= 1;
			kernel_step <= 0;
			k_x <= 1;
			k_y <= 1;
			state <= S_WAIT;
		    end else begin
                        if (kernel_step < 9) begin
                            cu_en <= 1;
		            // Clear CU only at first kernel step of first IC tile
		            if (kernel_step == 0) cu_clear <= 1;
		            else cu_clear <= 0;

			    k_x <= kernel_step % 3;
			    k_y <= kernel_step / 3;
                            kernel_step <= kernel_step + 1;
                        end else begin
			    k_x <= 0;
			    k_y <= 0;
		            kernel_step <= 0;
                            cu_en <= 0;
                            pipe_flush_cnt <= 0;
                            state <= S_WAIT;
                        end
	    	    end
                end
                
                // --- 5. WAIT & UPDATE ---
                S_WAIT: begin
		    cu_en <= 0;
		    cu_clear <= 0;
                    if (pipe_flush_cnt == LATENCY) begin
                            dout_valid <= 1;
                            // Update Output Coordinate
                            if (col_idx == img_width_strips - 1) begin
                                col_idx <= 0;
                                row_idx <= row_idx + 1;
				row_is_valid <= next_row_in_bounds && next_row_mod_ok;
                            end else begin
                                col_idx <= col_idx + 1;
                            end
                            
                            // Check for completion
                            if (row_idx == img_height - 1 && col_idx == img_width_strips - 1) begin
                                state <= S_DONE;
                            end else if (load_row_cnt >= img_height) begin
			        state <= S_FLUSH;
		            end else begin
                                state <= S_FETCH_NEXT;
			    end
			// end
                    end else begin
                        pipe_flush_cnt <= pipe_flush_cnt + 1;
                    end
                end

                // --- 6. FETCH NEXT STRIP ---
                // Load 1 Strip -> Pulse Sequencer -> Go Compute
                S_FETCH_NEXT: begin
                    // If we have loaded everything, we just pulse sequencer (flush)
                    if (load_row_cnt >= img_height) begin
                        // End of image flush logic (simplified for now)
                        // Just pulse seq to push the last data through
			state <= S_FLUSH;
                    end else begin
                        din_ready <= 1;
			lb_shift_en <= 1;
                        if (din_valid && din_ready) begin
                            din_ready <= 0;
			    lb_shift_en <= 0;
                            
                            if (load_col_cnt == img_width_strips - 1) begin
                                load_col_cnt <= 0;
                                load_row_cnt <= load_row_cnt + 1;
                            end else begin
                                load_col_cnt <= load_col_cnt + 1;
                            end
			    state <= S_FETCH_WAIT;
                        end
                    end
                end

		S_FETCH_WAIT: begin
		    seq_load <= 1; // Pulse 1: LB -> Next
                    
                    if (img_width_strips == 1) begin
                        // Single Strip: Need 2nd pulse to move Next -> Curr
                        state <= S_FETCH_PULSE_2;
                    end else begin
                        // Multi Strip: Pipeline is fine
			// if (row_idx < height_limit && ((stride == 1) || row_idx[0] == 0)) begin
                        if (row_is_valid) begin
                            cu_clear <= 1;
                            kernel_step <= 0;
                            state <= S_COMPUTE;
                        end else begin
                            state <= S_SKIP_COMPUTE;
                        end
                    end
		end

		S_FETCH_PULSE_2: begin
                    seq_load <= 1; // Pulse 2: Next -> Curr
                    
		    // if (row_idx < height_limit && ((stride == 1) || row_idx[0] == 0)) begin
                    if (row_is_valid) begin
                        cu_clear <= 1;
                        kernel_step <= 0;
                        state <= S_COMPUTE;
                    end else begin
                        state <= S_SKIP_COMPUTE;
                    end
                end


		S_SKIP_COMPUTE: begin
                    // Mimics S_WAIT but without waiting for pipeline flush
                    // because we didn't put anything in the pipeline!
                    // Update Output Coordinate
                    if (col_idx == img_width_strips - 1) begin
                        col_idx <= 0;
                        row_idx <= row_idx + 1;
			row_is_valid <= next_row_in_bounds && next_row_mod_ok;
                    end else begin
                        col_idx <= col_idx + 1;
                    end
                    
                    // Check for completion
                    if (row_idx == img_height - 1 && col_idx == img_width_strips - 1) begin
                        state <= S_DONE;
                    end else if (load_row_cnt >= img_height) begin
                        state <= S_FLUSH;
                    end else begin
                        state <= S_FETCH_NEXT;
                    end
                end


		S_FLUSH: begin
		    lb_shift_en <= 1;
		    if (lb_shift_en) begin
			lb_shift_en <= 0;
		        seq_load <= 1;
		        if (load_col_cnt == img_width_strips - 1) begin
		    	    load_col_cnt <= 0;
		    	    // Increment row count to indicate flushing is complete
		    	    load_row_cnt <= load_row_cnt + 1; 
		        end else begin
		    	    load_col_cnt <= load_col_cnt + 1;
		        end
			if (img_width_strips == 1) state <= S_FETCH_PULSE_2;
			else begin
			    // if (row_idx < height_limit && ((stride == 1) || row_idx[0] == 0)) begin
			    if (row_is_valid) begin
                                cu_clear <= 1;
                                kernel_step <= 0;
                                state <= S_COMPUTE;
                            end else begin
                                state <= S_SKIP_COMPUTE;
                            end
			end
		    end
		end

                S_DONE: begin
                    done <= 1;
                    if (!start) state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
