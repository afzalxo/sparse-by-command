`timescale 1ns / 1ps

module instruction_scheduler #(
    parameter ADDR_WIDTH = 64,
    parameter DATA_WIDTH = 512
)(
    input wire clk,
    input wire rst_n,
    
    // --- Host Control Interface ---
    input wire start,                 // Pulse to start the instruction stream
    input wire [ADDR_WIDTH-1:0] base_addr, // Address of the first instruction
    output reg done,                  // Asserted when HALT opcode is encountered
    
    // --- AXI4 Read Master Interface (Lite version for Instructions) ---
    // We share the GMEM port, so we need standard AXI signals
    output reg m_axi_arvalid,
    input wire m_axi_arready,
    output reg [ADDR_WIDTH-1:0] m_axi_araddr,
    output wire [7:0] m_axi_arlen,    // Always 0 (Single beat)
    output wire [2:0] m_axi_arsize,   // Always 6 (64 bytes)
    output wire [1:0] m_axi_arburst,  // INCR
    
    input wire m_axi_rvalid,
    output reg m_axi_rready,
    input wire [DATA_WIDTH-1:0] m_axi_rdata,
    
    // --- Tile Manager Control Interface ---
    output reg exec_start,
    input wire exec_done,
    
    // --- Configuration Outputs (Driven to Tile Manager) ---
    (* max_fanout = 20 *) output reg [ADDR_WIDTH-1:0] cfg_input_addr,
    (* max_fanout = 20 *) output reg [ADDR_WIDTH-1:0] cfg_output_addr,
    (* max_fanout = 20 *) output reg [ADDR_WIDTH-1:0] cfg_weight_addr,
    (* max_fanout = 20 *) output reg [ADDR_WIDTH-1:0] cfg_bias_addr,
    output reg [15:0] cfg_img_width,
    output reg [15:0] cfg_img_height,
    output reg [15:0] cfg_in_channels,
    output reg [15:0] cfg_out_channels,
    (* max_fanout = 20 *) output reg [4:0]  cfg_quant_shift,
    output reg        cfg_is_conv,
    output reg	      cfg_is_memcpy,
    output reg	      cfg_is_gap,
    output reg        cfg_bias_en,
    (* max_fanout = 20 *) output reg 	      cfg_relu_en,
    output reg [1:0]  cfg_stride,
    output reg 	      cfg_flatten,
    output reg [2:0]  cfg_log2_mem_tile_height,
    output reg        cfg_is_sparse,
    output reg [127:0] cfg_ic_tile_mask,
    output reg [127:0] cfg_oc_tile_mask,
    output reg [1:0]  cfg_input_bank,
    output reg [1:0]  cfg_output_bank
);

    // --- AXI Constants ---
    assign m_axi_arlen = 8'd0;       // Single beat (64 bytes)
    assign m_axi_arsize = 3'b110;    // 64 bytes
    assign m_axi_arburst = 2'b01;    // INCR

    // --- State Machine ---
    localparam S_IDLE        = 0;
    localparam S_FETCH_REQ   = 1; // Send AR
    localparam S_FETCH_WAIT  = 2; // Wait for R
    localparam S_DECODE      = 3; // Latch values
    localparam S_EXECUTE     = 4; // Trigger Tile Manager
    localparam S_WAIT_TM     = 5; // Wait for Tile Manager Done
    localparam S_NEXT_INSTR  = 6; // Increment PC
    localparam S_DONE        = 7;
    localparam S_FETCH_ACK   = 8;

    reg [3:0] instruction_scheduler_state;
    reg [ADDR_WIDTH-1:0] pc; // Program Counter (Current Instruction Address)
    reg [63:0] cc_counter;
    
    // Instruction Register
    reg [DATA_WIDTH-1:0] instr_reg;
    wire [7:0] opcode = instr_reg[327:320];

    // Opcode Definitions
    localparam OP_CONV   = 8'h01;
    localparam OP_GEMM   = 8'h02;
    localparam OP_MEMCPY = 8'h03;
    localparam OP_GAP    = 8'h04;
    localparam OP_HALT   = 8'hFF;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            instruction_scheduler_state <= S_IDLE;
            done <= 0;
            m_axi_arvalid <= 0;
            m_axi_rready <= 0;
            m_axi_araddr <= 0;
            exec_start <= 0;
            pc <= 0;
	    cc_counter <= 0;
            // Config defaults
            cfg_input_addr <= 0;
            cfg_output_addr <= 0;
            cfg_weight_addr <= 0;
	    cfg_bias_addr <= 0;
            cfg_img_width <= 0;
            cfg_img_height <= 0;
            cfg_in_channels <= 0;
            cfg_out_channels <= 0;
            cfg_quant_shift <= 0;
            cfg_is_conv <= 0;
	    cfg_is_memcpy <= 0;
	    cfg_is_gap <= 0;
	    cfg_relu_en <= 0;
	    cfg_input_bank <= 0;
	    cfg_output_bank <= 0;
	    cfg_stride <= 2'b01;
	    cfg_flatten <= 0;
	    cfg_is_sparse <= 0;
	    cfg_ic_tile_mask <= {128{1'b1}};
	    cfg_oc_tile_mask <= {128{1'b1}};
        end else begin
            // Default Pulses
            exec_start <= 0;
            done <= 0;
	    cc_counter <= cc_counter + 1;
            case (instruction_scheduler_state)
                S_IDLE: begin
                    if (start) begin
                        pc <= base_addr;
                        instruction_scheduler_state <= S_FETCH_REQ;
			cc_counter <= 0;

		        cfg_is_sparse <= 0;
		        cfg_ic_tile_mask <= {128{1'b1}};
		        cfg_oc_tile_mask <= {128{1'b1}};
                    end
                end

                // 1. Request Instruction
                S_FETCH_REQ: begin
                    m_axi_araddr <= pc;
                    m_axi_arvalid <= 1;
		    instruction_scheduler_state <= S_FETCH_ACK;
                end

		S_FETCH_ACK: begin
		    if (m_axi_arready) begin
                        m_axi_arvalid <= 0; // Deassert
                        m_axi_rready <= 1;  // Ready to receive data
                        instruction_scheduler_state <= S_FETCH_WAIT;
                    end
		end

                // 2. Receive Instruction
                S_FETCH_WAIT: begin
                    if (m_axi_rvalid) begin
                        instr_reg <= m_axi_rdata;
                        m_axi_rready <= 0;
                        instruction_scheduler_state <= S_DECODE;
                    end
                end

                // 3. Decode Instruction
                S_DECODE: begin
                    // Map bits to configuration registers
                    cfg_input_addr   <= instr_reg[63:0];
                    cfg_output_addr  <= instr_reg[127:64];
                    cfg_weight_addr  <= instr_reg[191:128];
		    cfg_bias_addr    <= instr_reg[255:192];

                    cfg_img_width    <= instr_reg[271:256];
                    cfg_img_height   <= instr_reg[287:272];
                    cfg_in_channels  <= instr_reg[303:288];
                    cfg_out_channels <= instr_reg[319:304];
                    
		    cfg_quant_shift          <= instr_reg[328 +: 5];
		    cfg_input_bank  	     <= instr_reg[336 +: 2];
		    cfg_output_bank 	     <= instr_reg[338 +: 2];
		    cfg_stride      	     <= instr_reg[344 +: 2];
		    cfg_log2_mem_tile_height <= instr_reg[352 +: 3];

		    cfg_relu_en     	     <= instr_reg[360];
		    cfg_flatten              <= instr_reg[361];
		    cfg_is_sparse            <= instr_reg[362];
		    cfg_bias_en              <= instr_reg[363];

                    // Instruction specific overrides
                    if (opcode == OP_HALT) begin
			cfg_is_conv <= 0;
			cfg_is_gap <= 0;
			cfg_is_memcpy <= 0;
			cfg_quant_shift <= 0;
			cfg_is_sparse <= 0;
			cfg_ic_tile_mask <= {128{1'b1}};
			cfg_oc_tile_mask <= {128{1'b1}};
			cfg_bias_en <= 0;
			cfg_bias_addr <= 0;
                        instruction_scheduler_state <= S_DONE;
                    end else if (opcode == OP_CONV) begin
                        cfg_is_conv <= 1;
			cfg_is_gap <= 0;
			cfg_is_memcpy <= 0;
		        cfg_ic_tile_mask         <= cfg_oc_tile_mask;
		        cfg_oc_tile_mask         <= instr_reg[384 +: 128];
                        instruction_scheduler_state <= S_EXECUTE;
		    end else if (opcode == OP_GEMM) begin
			cfg_is_conv <= 0;
			cfg_is_gap <= 0;
			cfg_is_memcpy <= 0;
			cfg_img_height <= 16'd1;
			cfg_stride <= 2'b01;
		        cfg_ic_tile_mask         <= cfg_oc_tile_mask;
			// For now, no tiling support for GEMM
		        cfg_oc_tile_mask         <= {128{1'b1}};
			instruction_scheduler_state <= S_EXECUTE;
		    end else if (opcode == OP_MEMCPY) begin
			cfg_is_conv <= 0;
			cfg_is_gap <= 0;
			cfg_is_memcpy <= 1;
			cfg_is_sparse <= 0;
			cfg_in_channels <= 0;
			cfg_img_height <= 0;
			cfg_img_width <= 0;
			instruction_scheduler_state <= S_EXECUTE;
		    end else if (opcode == OP_GAP) begin 
			cfg_is_conv <= 0;
			cfg_is_memcpy <= 0;
			cfg_is_gap <= 1;
			instruction_scheduler_state <= S_EXECUTE;
		    end else begin
			$display("ERROR: Unknown Opcode %h at PC %h", opcode, pc);
			instruction_scheduler_state <= S_DONE;
		    end
                end

                // 4. Execute Layer
                S_EXECUTE: begin
                    exec_start <= 1;
                    instruction_scheduler_state <= S_WAIT_TM;
                end

                // 5. Wait for Layer Completion
                S_WAIT_TM: begin
                    if (exec_done) begin
                        instruction_scheduler_state <= S_NEXT_INSTR;
                    end
                end

                // 6. Update Program Counter
                S_NEXT_INSTR: begin
                    pc <= pc + 64; // Move 64 bytes (512 bits) forward
                    instruction_scheduler_state <= S_FETCH_REQ;
                end

                S_DONE: begin
                    done <= 1;
		    instruction_scheduler_state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
