`default_nettype none
// AXI-Lite Controller for vla accelerator
module vla_accel_control_s_axi #(
    parameter C_S_AXI_ADDR_WIDTH = 8,
    parameter C_S_AXI_DATA_WIDTH = 32,
    // Data widths for arguments controlled by this interface
    parameter NUM_SAMPLES_WIDTH = 32
    // Note: SEED_WIDTH and AXI_ADDR_WIDTH (for output base) are NOT needed here
    // as this controller doesn't manage those arguments directly anymore.
) (
    // AXI4-Lite Slave Interface
    input wire ACLK,
    input wire ARESET, // Active LOW reset expected by this module
    input wire ACLK_EN, // Optional clock enable

    // --- Write Address Channel ---
    input wire [C_S_AXI_ADDR_WIDTH-1:0] AWADDR,
    input wire                          AWVALID,
    output wire                         AWREADY,
    // --- Write Data Channel ---
    input wire [C_S_AXI_DATA_WIDTH-1:0] WDATA,
    input wire [C_S_AXI_DATA_WIDTH/8-1:0] WSTRB,
    input wire                          WVALID,
    output wire                         WREADY,
    // --- Write Response Channel ---
    output wire [1:0]                   BRESP,
    output wire                         BVALID,
    input wire                          BREADY,
    // --- Read Address Channel ---
    input wire [C_S_AXI_ADDR_WIDTH-1:0] ARADDR,
    input wire                          ARVALID,
    output wire                         ARREADY,
    // --- Read Data Channel ---
    output wire [C_S_AXI_DATA_WIDTH-1:0] RDATA,
    output wire [1:0]                   RRESP,
    output wire                         RVALID,
    input wire                          RREADY,

    output wire                          user_start,
    input  wire                          user_done,
    input  wire                          user_ready,
    input  wire                          user_idle,

    output reg [63:0] reg_heap_addr,
    output reg [63:0] reg_buff_a_addr,
    output reg [63:0] reg_buff_b_addr,
    output reg [63:0] reg_weight_input_addr
);

    //------------------------Address Info-------------------
    // Standard HLS Control Registers (Simulated)
    localparam ADDR_AP_CTRL          = 8'h00; // ap_start (W), ap_done (R), ap_idle (R), ap_ready (R)
    // Custom Registers for Arguments
    // localparam ADDR_NUM_SAMPLES_DATA = 8'h10;

    localparam ADDR_FEAT_INPUT_ADDR_LO   = 8'h10;
    localparam ADDR_FEAT_INPUT_ADDR_HI   = 8'h14;
    localparam ADDR_BUFF_A_ADDR_LO  = 8'h18;
    localparam ADDR_BUFF_A_ADDR_HI  = 8'h1C;
    localparam ADDR_BUFF_B_ADDR_LO  = 8'h20;
    localparam ADDR_BUFF_B_ADDR_HI  = 8'h24;
    localparam ADDR_WEIGHT_INPUT_ADDR_LO = 8'h28;
    localparam ADDR_WEIGHT_INPUT_ADDR_HI = 8'h2C;

    // AXI Write FSM States
    localparam WRIDLE = 2'd0, WRDATA = 2'd1, WRRESP = 2'd2, WRRESET = 2'd3;
    // AXI Read FSM States
    localparam RDIDLE = 2'd0, RDDATA = 2'd1, RDRESET = 2'd2;

    //------------------------Internal Signals-------------------
    // AXI Write FSM signals
    reg [1:0] wstate = WRRESET;
    reg [1:0] wnext;
    reg [C_S_AXI_ADDR_WIDTH-1:0] waddr;
    wire [C_S_AXI_DATA_WIDTH-1:0] wmask;
    wire aw_hs, w_hs;

    // AXI Read FSM signals
    reg [1:0] rstate = RDRESET;
    reg [1:0] rnext;
    reg [C_S_AXI_DATA_WIDTH-1:0] rdata_reg = 0; // Register to hold read data
    wire ar_hs;
    wire [C_S_AXI_ADDR_WIDTH-1:0] raddr;

    // Internal Control Registers
    reg int_ap_start = 1'b0; // Internal copy of ap_start trigger
    reg int_ap_done  = 1'b0;
    reg int_auto_restart = 1'b0;
    reg int_ap_idle; // Idle state tracker
    reg int_ap_ready; // Ready is same as idle


    //------------------------AXI Write FSM Implementation------------------
    assign AWREADY = (wstate == WRIDLE);
    assign WREADY  = (wstate == WRDATA);
    assign BRESP   = 2'b00; // OKAY response
    assign BVALID  = (wstate == WRRESP);
    assign wmask = {{8{WSTRB[3]}}, {8{WSTRB[2]}}, {8{WSTRB[1]}}, {8{WSTRB[0]}} };
    assign aw_hs = AWVALID & AWREADY;
    assign w_hs  = WVALID & WREADY;

    // Write state machine logic
    always @(posedge ACLK or negedge ARESET) begin
        if (!ARESET) wstate <= WRRESET;
        else if (ACLK_EN) wstate <= wnext;
    end

    always @(*) begin // Next state logic for write FSM
        case (wstate)
            WRIDLE: wnext = AWVALID ? WRDATA : WRIDLE;
            WRDATA: wnext = WVALID  ? WRRESP : WRDATA;
            WRRESP: wnext = BREADY  ? WRIDLE : WRRESP;
            default: wnext = WRIDLE;
        endcase
    end

    // Capture write address
    always @(posedge ACLK) begin
        if (ACLK_EN & aw_hs) waddr <= AWADDR;
    end

    //------------------------AXI Read FSM Implementation-------------------
    assign ARREADY = (rstate == RDIDLE);
    assign RDATA   = rdata_reg;
    assign RRESP   = 2'b00; // OKAY response
    assign RVALID  = (rstate == RDDATA);

    assign ar_hs = ARVALID & ARREADY;
    assign raddr = ARADDR;

    // Read state machine logic
    always @(posedge ACLK or negedge ARESET) begin
        if (!ARESET) begin // Active LOW Reset
            rstate <= RDRESET;
        end else if (ACLK_EN) begin
            rstate <= rnext;
        end
    end

    always @(*) begin // Next state logic for read FSM
        case (rstate)
            RDIDLE:  rnext = ARVALID ? RDDATA : RDIDLE;
            RDDATA:  rnext = RREADY & RVALID ? RDIDLE : RDDATA; // Stay if not ready
            default: rnext = RDIDLE;
        endcase
    end

assign user_start  = int_ap_start;
// int_ap_start
always @(posedge ACLK) begin
    if (!ARESET)
        int_ap_start <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0] && WDATA[0])
            int_ap_start <= 1'b1;
    	else if (user_idle)
	    int_ap_start <= 1'b0;
        else if (user_ready)
            int_ap_start <= int_auto_restart; // clear on handshake/auto restart
    end
end

// int_ap_done
always @(posedge ACLK) begin
    if (!ARESET)
        int_ap_done <= 1'b0;
    else if (ACLK_EN) begin
	if (user_done) 
	    int_ap_done <= 1'b1;
	else if (ar_hs && raddr == ADDR_AP_CTRL)
	    int_ap_done <= 1'b0;
    end
end

// int_ap_idle
always @(posedge ACLK) begin
    if (!ARESET)
        int_ap_idle <= 1'b0;
    else if (ACLK_EN) begin
        int_ap_idle <= user_idle;
    end
end

// int_ap_ready
always @(posedge ACLK) begin
    if (!ARESET)
        int_ap_ready <= 1'b0;
    else if (ACLK_EN) begin
            int_ap_ready <= user_ready;
    end
end

// int_auto_restart
always @(posedge ACLK) begin
    if (!ARESET)
        int_auto_restart <= 1'b0;
    else if (ACLK_EN) begin
        if (w_hs && waddr == ADDR_AP_CTRL && WSTRB[0])
            int_auto_restart <=  WDATA[7];
    end
end


    //------------------------Register Write Logic-------------------
    always @(posedge ACLK or negedge ARESET) begin
        if (!ARESET) begin // Active Low Reset
            // Reset output registers
            // reg_num_samples_out      <= 0;
	    reg_heap_addr 	     <= 0;
        end else if (ACLK_EN) begin
            if (w_hs) begin
                case (waddr)
		    ADDR_FEAT_INPUT_ADDR_LO: begin
			$display("Input Feat Addr Low Write");
			reg_heap_addr[31:0] <= WDATA;
		    end
		    ADDR_FEAT_INPUT_ADDR_HI: begin
			$display("Input Feat Addr High Write");
			reg_heap_addr[63:32] <= WDATA;
		    end
		    ADDR_WEIGHT_INPUT_ADDR_LO: begin
			$display("Input Weight Addr Low Write");
			reg_weight_input_addr[31:0] <= WDATA;
		    end
		    ADDR_WEIGHT_INPUT_ADDR_HI: begin
			$display("Input Weight Addr High Write");
			reg_weight_input_addr[63:32] <= WDATA;
		    end
		    ADDR_BUFF_A_ADDR_LO: begin
			$display("Buff A Addr Low Write");
			reg_buff_a_addr[31:0] <= WDATA;
		    end
		    ADDR_BUFF_A_ADDR_HI: begin
			$display("Buff A Addr High Write");
			reg_buff_a_addr[63:32] <= WDATA;
		    end
		    ADDR_BUFF_B_ADDR_LO: begin
			$display("Buff B Addr Low Write");
			reg_buff_b_addr[31:0] <= WDATA;
		    end
		    ADDR_BUFF_B_ADDR_HI: begin
			$display("Buff B Addr High Write");
			reg_buff_b_addr[63:32] <= WDATA;
		    end
                    default: begin
			$display("Default Write");
                    end
                endcase
            end
        end
    end

    //------------------------Register Read Logic--------------------
    always @(posedge ACLK or negedge ARESET) begin
        if (!ARESET) begin // Active Low Reset
            rdata_reg <= 0;
        end else if (ACLK_EN & ar_hs) begin
            // Read logic based on captured ARADDR
            rdata_reg <= 32'b0; // Default value for reads
            case (ARADDR)
		ADDR_AP_CTRL: begin
		    // $display("AP Control Read: %b", {int_auto_restart, int_ap_ready, int_ap_idle, int_ap_done, int_ap_start});
		    rdata_reg <= {27'b0, int_auto_restart, int_ap_ready, int_ap_idle, int_ap_done, int_ap_start};
		end
		ADDR_FEAT_INPUT_ADDR_LO: begin
		    $display("Output Addr Low Read");
		    rdata_reg <= reg_heap_addr[31:0];
		end
		ADDR_FEAT_INPUT_ADDR_HI: begin
		    $display("Output Addr High Read");
		    rdata_reg <= reg_heap_addr[63:32];
		end
		ADDR_WEIGHT_INPUT_ADDR_LO: begin
		    $display("Weight Addr Low Read");
		    rdata_reg <= reg_weight_input_addr[31:0];
		end
		ADDR_WEIGHT_INPUT_ADDR_HI: begin
		    $display("Weight Addr High Read");
		    rdata_reg <= reg_weight_input_addr[63:32];
		end
		ADDR_BUFF_A_ADDR_LO: begin
		    $display("Buff A Addr Low Read");
		    rdata_reg <= reg_buff_a_addr[31:0];
		end
		ADDR_BUFF_A_ADDR_HI: begin
		    $display("Buff A Addr High Read");
		    rdata_reg <= reg_buff_a_addr[63:32];
		end
		ADDR_BUFF_B_ADDR_LO: begin
		    $display("Buff B Addr Low Read");
		    rdata_reg <= reg_buff_b_addr[31:0];
		end
		ADDR_BUFF_B_ADDR_HI: begin
		    $display("Buff B Addr High Read");
		    rdata_reg <= reg_buff_b_addr[63:32];
		end
		default: begin
		    $display("Default Read");
		    rdata_reg <= 32'b0;
		end
            endcase
        end
    end

endmodule
`default_nettype wire
