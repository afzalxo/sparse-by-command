import cv2
import numpy as np


def draw_full_diagnostic(img_bgr, model, expert=None, cmd=None):
    """
    expert/model: [steer, throttle, brake]
    """
    h, w = img_bgr.shape[:2]
    overlay_h = 110
    if expert is None:
        overlay_h = overlay_h // 2
    
    # Semi-transparent background for UI
    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, overlay_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, img_bgr, 0.5, 0, img_bgr)

    # 1. Text Info
    if cmd is not None:
        cv2.putText(img_bgr, f"CMD: {cmd.upper()}", (10, 25), 0, 0.6, (0, 255, 255), 2)
    
    # 2. Steering Lines (Bottom)
    start_pt = (w // 2, h - 20)
    if expert is not None:
        exp_steer_end = (int(w // 2 + expert[0] * 150), h - 70)
    prd_steer_end = (int(w // 2 + model[0] * 150), h - 70)
    if expert is not None:
        cv2.line(img_bgr, start_pt, exp_steer_end, (0, 255, 0), 3) # Expert Green
    cv2.line(img_bgr, start_pt, prd_steer_end, (0, 255, 0), 1) # Model Red

    bar_h = 10
    bar_y_start = 20
    dist_between = 15

    # 3. Throttle/Brake Bars (Top Right)
    # Expert Bars (Green)
    if expert is not None:
        cv2.putText(img_bgr, "EXPERT", (w - 180, 45), 0, 0.4, (0, 255, 0), 1)
        cv2.rectangle(img_bgr, (w - 100, bar_y_start), (w - 100 + int(expert[1]*80), bar_y_start + bar_h), (0, 255, 0), -1)
        cv2.rectangle(img_bgr, (w - 100, bar_y_start + dist_between), (w - 100 + int(expert[2]*80), bar_y_start + dist_between + bar_h), (0, 100, 0), -1)
    if expert is not None:
        bar_y_start += bar_y_start

    # Throttle
    cv2.rectangle(img_bgr, (w - 100, bar_y_start), (w - 100 + int(model[1]*80), bar_y_start + bar_h), (0, 0, 255), -1)
    # Brake
    cv2.rectangle(img_bgr, (w - 100, bar_y_start + dist_between), (w - 100 + int(model[2]*80), bar_y_start + dist_between + bar_h), (100, 0, 100), -1)

    # Labels TODO
    if expert is not None:
        cv2.putText(img_bgr, "THR", (w - 130, 50), 0, 0.3, (255, 255, 255), 1)
        cv2.putText(img_bgr, "BRK", (w - 130, 65), 0, 0.3, (255, 255, 255), 1)
    # cv2.putText(img_bgr, "THR", (w - 130, 90), 0, 0.3, (255, 255, 255), 1)
    # cv2.putText(img_bgr, "BRK", (w - 130, 105), 0, 0.3, (255, 255, 255), 1)
    cv2.putText(img_bgr, f"{model[1]:.2f}", (w - 130, bar_y_start + 7), 0, 0.3, (255, 255, 255), 1)
    cv2.putText(img_bgr, f"{model[2]:.2f}", (w - 130, bar_y_start + dist_between + 7), 0, 0.3, (255, 255, 255), 1)

    return img_bgr


