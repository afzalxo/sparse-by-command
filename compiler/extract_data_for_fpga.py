import argparse
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import cv2
from PIL import Image
import json
from tqdm import tqdm
from models.carla_model import CarlaVLAHW_Gated as CarlaVLA
# command->branch mapping (pure string map; avoids importing the heavy `carla` module)
try:
    from carla_utils.route_utils import command_to_branch_id
except Exception:
    def command_to_branch_id(cmd):
        cmd = str(cmd).lower().strip()
        if "left" in cmd:  return 3 if "change" in cmd else 1   # 3=change-lane-left, 1=turn-left
        if "right" in cmd: return 4 if "change" in cmd else 2   # 4=change-lane-right, 2=turn-right
        if "stop" in cmd or "red light" in cmd or "brake" in cmd: return 5
        return 0  # follow lane / straight

DATASET_DIR = os.environ.get("FPGA_DATASET_DIR", "carla_ue4_dataset_dynamic")
CSV_PATH = os.path.join(DATASET_DIR, "data.csv")
IMG_DIR = os.path.join(DATASET_DIR, "images")
MODEL_PATH = os.environ.get("FPGA_MODEL_PATH", "model_checkpoints_sparse/carla_vla_furthest_driver_phase3_gated.pth")
OUTPUT_DIR = os.environ.get("FPGA_OUTPUT_DIR", "fpga_data")

n_prev_act_feat = 128
PP_PAR = 8
# Hardware Constraints
BIT_WIDTH = 8
Q_MAX = 127
Q_MIN = -128
IMG_SIZE = 256
CALIBRATION_SAMPLES = 300  # Number of samples to profile
ROUNDING_MODE = "round"

B_LIMIT = 2**27
B_MAX = B_LIMIT - 1
B_MIN = -B_LIMIT

# Preprocessing Constants
NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

OPCODES = {
    "OP_CONV": 0x01,
    "OP_GEMM": 0x02,
    "OP_MEMCPY": 0x03,
    "OP_GAP": 0x04,
    "OP_HALT": 0xFF
}

layer_specs = []

def add_layer_spec(name, in_hwc, out_c, stride, shift, relu, opcode, oc_mask_lo=None, oc_mask_hi=None, weight_f="", bias_f="", gold_f=""):
    opcode = OPCODES.get(opcode, 0x00)
    if oc_mask_lo is None: oc_mask_lo = 0xFFFFFFFFFFFFFFFF
    if oc_mask_hi is None: oc_mask_hi = 0xFFFFFFFFFFFFFFFF
    if weight_f == "": weight_f = "NULL"
    if bias_f == "": bias_f = "NULL"
    if gold_f == "": gold_f = "NULL"
    is_sparse = 1 if (oc_mask_lo != 0xFFFFFFFFFFFFFFFF or oc_mask_hi != 0xFFFFFFFFFFFFFFFF) else 0
    layer_specs.append({
        "name": name,
        "in_w": in_hwc[1], "in_h": in_hwc[0], "in_c": in_hwc[2],
        "out_c": out_c,
        "stride": stride,
        "quant_shift": shift,
        "relu": 1 if relu else 0,
        "opcode": opcode,
        "is_sparse": is_sparse,
        "oc_mask_lo": oc_mask_lo,
        "oc_mask_hi": oc_mask_hi,
        "weight_file": weight_f,
        "bias_file": bias_f,
        "golden_file": gold_f
    })


def quantize_and_save_bias(bias_tensor, cumulative_scale, filename):
    """
    bias_int = bias_float * (input_scale * weight_scale)
    Enforces 28-bit signed range: [-134217728, 134217727]
    Stored as int32 (4 bytes) for AXI/Memory alignment.
    """
    # Define 28-bit signed limits
    B_LIMIT = 2**27
    B_MAX = B_LIMIT - 1
    B_MIN = -B_LIMIT

    # Calculate scaled bias and round
    bias_hw = (bias_tensor * cumulative_scale).round()
    
    # Saturate to 28-bit range
    bias_clamped = torch.clamp(bias_hw, B_MIN, B_MAX)
    
    # Save as int32 (the higher 4 bits will be sign-extension)
    save_binary(bias_clamped, filename, dtype=torch.int32)
    return bias_clamped


def get_input_data(df, sample_idx):
    """Prepares 4-frame stack and prev_action exactly like the Eval script."""
    row = df.iloc[sample_idx]
    current_traj = row['trajectory_id']
    instruction = row['instruction']
    branch_idx = command_to_branch_id(instruction)
    task_vec = torch.zeros(1, 6, dtype=torch.float32)
    task_vec[0, branch_idx] = 1.0
    
    frames = []
    img_path = os.path.join(IMG_DIR, df.iloc[sample_idx]['image_path'])
    with Image.open(img_path) as img:
        img = img.copy()
        img_rgb = np.array(img, dtype=np.uint8)
        frames.insert(0, img_rgb)
    img_stack = np.concatenate(frames, axis=2)
    img_stack = np.transpose(img_stack, (2, 0, 1))
   
    img_stack = torch.from_numpy(img_stack).float() / 255.0
    img_stack = (img_stack.reshape(-1, 3, 256, 256) - NORM_MEAN) / NORM_STD
    input_stack = img_stack.reshape(1, 3, 256, 256)

    # 2. Prev Action
    if sample_idx > 0 and df.iloc[sample_idx-1]['trajectory_id'] == current_traj and df.iloc[sample_idx-1]['is_train_candidate'] == 1:
        p_row = df.iloc[sample_idx-1]
        prev_action = torch.tensor([
            p_row['steer'], 
            (p_row['throttle'] - p_row['brake']), 
            p_row['speed']
        ], dtype=torch.float32)
    else:
        prev_action = torch.zeros(3)

    return input_stack, prev_action.unsqueeze(0), task_vec

def save_binary(data, filename, dtype=torch.int8):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    path = os.path.join(OUTPUT_DIR, filename)
    data.to(dtype).numpy().tofile(path)

def hardware_round_shift(acc, shift):
    """Bit-accurate rounding: (acc + bias) >> shift"""
    if ROUNDING_MODE == "round":
        if shift > 0:
            bias = 1 << (shift - 1)
            return torch.floor_divide(acc + bias, 2**shift)
    elif ROUNDING_MODE == "floor":
        if shift > 0:
            return torch.floor_divide(acc, 2**shift)
    return acc * (2**abs(shift))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_idx', type=int, default=2332, help="Sample to export as golden test vector")
    parser.add_argument('--output-dir', type=str, default=None, help="Override output directory")
    parser.add_argument('--model-path', type=str, default=None, help="Override model path")
    parser.add_argument('--dataset-dir', type=str, default=None, help="Override dataset directory")
    args = parser.parse_args()

    global OUTPUT_DIR, MODEL_PATH, DATASET_DIR, CSV_PATH, IMG_DIR
    if args.output_dir: OUTPUT_DIR = args.output_dir
    if args.model_path: MODEL_PATH = args.model_path
    if args.dataset_dir:
        DATASET_DIR = args.dataset_dir
        CSV_PATH = os.path.join(DATASET_DIR, "data.csv")
        IMG_DIR = os.path.join(DATASET_DIR, "images")

    print("Loading Model and Data...")
    df = pd.read_csv(CSV_PATH)
    model = CarlaVLA(task_dim=6, oc_par=16)
    sd = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()

    df = df[df['is_train_candidate'] == 1].reset_index(drop=True)
    print(f"Dataset loaded with {len(df)} samples.")
    # =========================================================================
    # PHASE 1: GLOBAL PROFILING
    # =========================================================================
    print(f"--- Phase 1: Profiling Activations ({CALIBRATION_SAMPLES} samples) ---")
    calib_indices = df.sample(n=CALIBRATION_SAMPLES, random_state=42).index.tolist()
    
    # We track the max absolute value seen at every activation point
    act_maxima = {}

    with torch.no_grad():
        for idx in tqdm(calib_indices):
            img_stack, prev_act, task_vec = get_input_data(df, idx)
            all_pred_masks = model.gater(task_vec)
            split_masks = torch.split(all_pred_masks, model.predictable_tiles, dim=1)
            
            # Step-by-step forward pass to capture activation ranges
            curr = img_stack
            for i, layer in enumerate(model.get_conv_layers()):
                stride = 2
                curr = F.conv2d(curr, layer.weight, bias=layer.bias, stride=stride, padding=1)
                curr = model.apply_mask(curr, split_masks[i], hard_masking=True)
                curr = F.relu(curr)
                act_maxima[f'conv{i}'] = max(act_maxima.get(f'conv{i}', 0), torch.max(torch.abs(curr)).item())

            act_feat = F.relu(model.measure_proj(prev_act))
            act_maxima['measure'] = max(act_maxima.get('measure', 0), torch.max(torch.abs(act_feat)).item())

            curr = curr.view(curr.size(0), -1)
            curr = torch.cat([curr, act_feat], dim=1)

            fusion_out = F.relu(model.fusion(curr))
            act_maxima['fusion'] = max(act_maxima.get('fusion', 0), torch.max(torch.abs(fusion_out)).item())
            for b_idx, branch in enumerate(model.branches):
                b_curr = fusion_out
                # branch[0]=Linear, [1]=ReLU, [2]=Linear, [3]=ReLU, [4]=Linear
                for l_idx, name in [(0, "fc1"), (2, "fc2"), (4, "head")]:
                    b_curr = branch[l_idx](b_curr)
                    if l_idx < 4: b_curr = F.relu(b_curr)
                    
                    key = f"b{b_idx}_{name}"
                    act_maxima[key] = max(act_maxima.get(key, 0), torch.max(torch.abs(b_curr)).item())
    # =========================================================================
    # PHASE 2: CALCULATE FIXED GLOBAL SHIFTS
    # =========================================================================
    print("--- Phase 2: Freezing Global Shifts ---")
    global_shifts = {}
    
    # Track cumulative scale throughout profiling logic
    # We use a representative input scale (99.9th percentile of input)
    input_stack_example, _, task_vec = get_input_data(df, args.sample_idx)
    c_scale = Q_MAX / torch.max(torch.abs(input_stack_example)).item()
    
    for i in range(len(model.get_conv_layers())):
        w_max = torch.max(torch.abs(model.get_conv_layers()[i].weight)).item()
        w_scale = Q_MAX / w_max
        c_scale *= w_scale
        
        # Shift = ceil(log2( (Float_Max * Cumulative_Scale) / Q_MAX ))
        val_to_quantize = act_maxima[f'conv{i}'] * c_scale
        shift = int(np.ceil(np.log2(val_to_quantize / Q_MAX))) if val_to_quantize > 0 else 0
        global_shifts[f'conv{i}'] = shift
        c_scale /= (2**shift)

    # Fusion
    w_scale = Q_MAX / torch.max(torch.abs(model.fusion.weight)).item()
    c_scale *= w_scale
    val_to_quantize = act_maxima['fusion'] * c_scale
    shift = int(np.ceil(np.log2(val_to_quantize / Q_MAX))) if val_to_quantize > 0 else 0
    global_shifts['fusion'] = shift
    c_scale /= (2**shift)
    c_scale_fusion = c_scale

    for b_idx in range(len(model.branches)):
        for name, layer, use_relu in [("fc1", model.branches[b_idx][0], True), 
                                      ("fc2", model.branches[b_idx][2], True), 
                                      ("head", model.branches[b_idx][4], False)]:
            if name == "fc1":
                c_scale = c_scale_fusion
            w_scale = Q_MAX / torch.max(torch.abs(layer.weight)).item()
            c_scale *= w_scale
            val_to_quantize = act_maxima[f'b{b_idx}_{name}'] * c_scale
            shift = int(np.ceil(np.log2(val_to_quantize / Q_MAX))) if val_to_quantize > 0 else 0
            global_shifts[f'b{b_idx}_{name}'] = shift
            c_scale /= (2**shift)

    with open(os.path.join(OUTPUT_DIR, "global_shifts.json"), "w") as f:
        json.dump(global_shifts, f, indent=4)

    # =========================================================================
    # PHASE 3: EXTRACTION (SPECIFIC SAMPLE)
    # =========================================================================
    print(f"--- Phase 3: Extracting Golden Vectors for Sample {args.sample_idx} ---")
    print(df.iloc[args.sample_idx])
    # Get real sample_idx since we dropped samples which are not train candidates
    # TODO

    img_stack, prev_act, task_vec = get_input_data(df, args.sample_idx)
    all_pred_masks = model.gater(task_vec)
    split_masks = torch.split(all_pred_masks, model.predictable_tiles, dim=1)
    # [1, C, H, W]
    # Export and save input_stack_example into a txt file for debugging
    # np.savetxt("img_stack_0.txt", img_stack.flatten().numpy())
    
    current_h, current_w = img_stack.shape[2], img_stack.shape[3]
    # 1. Input Vision
    in_max = torch.max(torch.abs(img_stack)).item()
    in_scale = Q_MAX / in_max
    curr_int = (img_stack * in_scale).round().clamp(Q_MIN, Q_MAX)
    save_binary(curr_int, "input_vision.bin")
    
    # 2. Conv Layers
    cumulative_scale = in_scale
    conv_layers = model.get_conv_layers()
    for i, layer in enumerate(conv_layers):
        w_max = torch.max(torch.abs(layer.weight)).item()
        w_scale = Q_MAX / w_max
        w_int = (layer.weight * w_scale).round().clamp(Q_MIN, Q_MAX)

        save_binary(w_int, f"weights_conv{i}.bin")

        bias_scale = cumulative_scale * w_scale
        b_int = quantize_and_save_bias(layer.bias.detach(), bias_scale, f"bias_conv{i}.bin")
        
        stride = 2
        acc = F.conv2d(curr_int.float(), w_int.float(), bias=b_int.float(), stride=stride, padding=1)
        pred_mask = (split_masks[i] > 0.5).float()
        always_on = torch.ones(1, 1, device=pred_mask.device)
        full_mask = torch.cat([always_on, pred_mask], dim=1)
        mask_int = 0
        for bit_idx, bit_val in enumerate(full_mask[0]):
            if bit_val.item() > 0:
                mask_int |= (1 << bit_idx)
                
        # Reshape mask to multiply with[1, C, H, W] activation tensor
        mask_reshaped = full_mask.view(1, -1, 1, 1, 1).expand(1, -1, model.oc_par, 1, 1).reshape(1, layer.out_channels, 1, 1)
        
        acc = acc * mask_reshaped
        acc = F.relu(acc)
        
        shift = global_shifts[f'conv{i}']
        curr_int = hardware_round_shift(acc, shift).clamp(Q_MIN, Q_MAX)
        save_binary(curr_int, f"golden_conv{i}.bin")
        cumulative_scale = (cumulative_scale * w_scale) / (2**shift)

        add_layer_spec(f"conv_{i}", (current_h, current_w, layer.in_channels), layer.out_channels,
                        stride, shift, True, "OP_CONV",
                       mask_int, None,  # Hi mask all ones
                       f"fpga_data_carla/weights_conv{i}.bin",
                       f"fpga_data_carla/bias_conv{i}.bin",
                       f"fpga_data_carla/golden_conv{i}.bin")
        current_h //= stride
        current_w //= stride

    # 3. GAP
    '''
    gap_acc = torch.sum(curr_int.float(), dim=(2, 3), keepdim=True)
    curr_int = hardware_round_shift(gap_acc, global_shifts['gap']).clamp(Q_MIN, Q_MAX).flatten(1)
    save_binary(curr_int, "golden_gap.bin")
    cumulative_scale *= (8*8 / (2**global_shifts['gap']))

    # Ad-hoc crap here. Need to systematize this better
    add_layer_spec("gap", (current_h, current_w, conv_layers[-1].out_channels), conv_layers[-1].out_channels, 1, global_shifts['gap'], False, "OP_GAP", "", "", "fpga_data_carla/golden_gap.bin")
    '''

    # Memcpy for prev_action features
    add_layer_spec("memcpy", (0, 0, 0), n_prev_act_feat * (PP_PAR // 2), 0, 0, False, "OP_MEMCPY", None, None, "", "", "fpga_data_carla/golden_fused_features.bin")

    # 4. Fusion
    print(f"Prev Action: {prev_act}")
    # prev_act = torch.zeros_like(prev_act)
    act_feat_float = F.relu(model.measure_proj(prev_act))
    act_feat_int = (act_feat_float * cumulative_scale).round().clamp(Q_MIN, Q_MAX)
    print(f"Cumulative scale for prev_action features: {cumulative_scale}")
    # print(act_feat_int)
    save_binary(act_feat_int, "extra_features.bin")
    
    curr_int = curr_int.view(curr_int.size(0), -1)
    curr_int = torch.cat([curr_int, act_feat_int], dim=1)
    save_binary(curr_int, "golden_fused_features.bin")

    # Fusion Layer
    name, layer, use_relu = "fusion", model.fusion, True
    w_max = torch.max(torch.abs(layer.weight)).item()
    w_scale = Q_MAX / w_max
    w_int = (layer.weight * w_scale).round().clamp(Q_MIN, Q_MAX)
    save_binary(w_int, f"weights_{name}.bin")

    bias_scale = cumulative_scale * w_scale
    b_int = quantize_and_save_bias(layer.bias.detach(), bias_scale, f"bias_{name}.bin")
    
    acc = F.linear(curr_int.float(), w_int.float(), bias=b_int.float())
    if use_relu: acc = F.relu(acc)
    
    shift = global_shifts[name]
    curr_int = hardware_round_shift(acc, shift).clamp(Q_MIN, Q_MAX)
    save_binary(curr_int, f"golden_{name}.bin")
    cumulative_scale = (cumulative_scale * w_scale) / (2**shift)

    add_layer_spec("fusion", (1, 1, layer.in_features), layer.out_features, 1, global_shifts[name],
                    True, "OP_GEMM", None, None, f"fpga_data_carla/weights_{name}.bin",
                   f"fpga_data_carla/bias_{name}.bin",
                   f"fpga_data_carla/golden_{name}.bin")

    current_int_fusion = curr_int
    cumulative_scale_fusion = cumulative_scale
    global_scales = []
    branch_outputs = []
    # 5. Dense
    for b_idx, branch in enumerate(model.branches):
        dense_layers = [(f"b{b_idx}_fc1", branch[0], True), (f"b{b_idx}_fc2", branch[2], True), (f"b{b_idx}_head", branch[4], False)]
        curr_int = current_int_fusion
        cumulative_scale = cumulative_scale_fusion
        
        for name, layer, use_relu in dense_layers:
            w_max = torch.max(torch.abs(layer.weight)).item()
            w_scale = Q_MAX / w_max
            w_int = (layer.weight * w_scale).round().clamp(Q_MIN, Q_MAX)
            save_binary(w_int, f"weights_{name}.bin")

            bias_scale = cumulative_scale * w_scale
            b_int = quantize_and_save_bias(layer.bias.detach(), bias_scale, f"bias_{name}.bin")
            
            acc = F.linear(curr_int.float(), w_int.float(), bias=b_int.float())
            if use_relu: acc = F.relu(acc)
            
            shift = global_shifts[name]
            curr_int = hardware_round_shift(acc, shift).clamp(Q_MIN, Q_MAX)
            save_binary(curr_int, f"golden_{name}.bin")
            cumulative_scale = (cumulative_scale * w_scale) / (2**shift)

            add_layer_spec(f"{name}", (1, 1, layer.in_features), layer.out_features, 1, shift,
                            use_relu, "OP_GEMM", None, None, f"fpga_data_carla/weights_{name}.bin",
                           f"fpga_data_carla/bias_{name}.bin",
                           f"fpga_data_carla/golden_{name}.bin")

        global_scales.append(cumulative_scale)
        branch_outputs.append(curr_int.float())

    # Save model specs for host C++
    with open(os.path.join(OUTPUT_DIR, "model_spec.txt"), "w") as f:
        for spec in layer_specs:
            line = (f"{spec['name']} {spec['in_w']} {spec['in_h']} {spec['in_c']} "
                    f"{spec['out_c']} {spec['stride']} {spec['quant_shift']} {spec['relu']} "
                    f"{spec['opcode']} {spec['is_sparse']} {spec['oc_mask_lo']} {spec['oc_mask_hi']} {spec['weight_file']} {spec['bias_file']} {spec['golden_file']}\n")
            f.write(line)
    print("Exported model_spec.txt for C++ host.")


    # =========================================================================
    # GENERATE PER-TASK TILE MASKS (for dynamic FPGA sparsity)
    # =========================================================================
    print("--- Generating per-task tile masks (task_masks.txt) ---")
    task_names = ["follow_lane", "turn_left", "turn_right",
                  "change_lane_left", "change_lane_right", "stop"]
    with open(os.path.join(OUTPUT_DIR, "task_masks.txt"), "w") as f:
        for t_idx in range(6):
            tv = torch.zeros(1, 6, dtype=torch.float32)
            tv[0, t_idx] = 1.0
            with torch.no_grad():
                pred_masks = model.gater(tv)
            sm = torch.split(pred_masks, model.predictable_tiles, dim=1)
            masks_lo = []
            for layer_idx, layer in enumerate(model.get_conv_layers()):
                n_tiles = layer.out_channels // model.oc_par
                pred = sm[layer_idx][0]
                mask_int = 1  # tile 0 always ON
                for bit_idx in range(len(pred)):
                    if pred[bit_idx].item() > 0.5:
                        mask_int |= (1 << (bit_idx + 1))
                masks_lo.append(mask_int)
            line = " ".join(str(m) for m in masks_lo)
            f.write(f"{line}\n")
            active = sum(bin(m).count('1') for m in masks_lo)
            total = sum(l.out_channels // model.oc_par for l in model.get_conv_layers())
            print(f"  Task {t_idx} ({task_names[t_idx]}): {active}/{total} active tiles, "
                  f"masks=[{', '.join(hex(m) for m in masks_lo)}]")
    print(f"Saved {os.path.join(OUTPUT_DIR, 'task_masks.txt')}")

    # =========================================================================
    # FINAL VERIFICATION
    # =========================================================================
    hw_logits = branch_outputs[0].float() / global_scales[0]
    hw_logits = hw_logits.detach()
    hw_action = torch.tanh(hw_logits).numpy()[0]
    
    with torch.no_grad():
        sw_logits, _ = model(img_stack, prev_act, task_vec, hard_masking=True)
        sw_logits = sw_logits[0, 0]
        sw_action = sw_logits.numpy()

    print("\n" + "="*60)
    print(f"GLOBAL STATIC QUANTIZATION RESULTS")
    print(f"Global Scales: {global_scales}")
    print("-" * 60)
    print(f"HW Logits (No Scaling): {hw_logits.numpy()}")
    print(f"HW Action (Fixed Shifts): {hw_action}")
    print(f"SW Action (Ideal Float):  {sw_action}")
    
    err = np.abs(hw_action - sw_action)
    print(f"Max Deviation:            {np.max(err):.4f}")
    if np.max(err) < 0.1:
        print("\nSUCCESS: Global shifts provide stable, accurate results.")
    else:
        print("\nWARNING: High error. Consider increasing calibration samples.")
    print("="*60)

if __name__ == "__main__":
    main()
