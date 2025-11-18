import numpy as np
import cv2

def add_v_cbar(cmap, vmin, vmax, width=60, height=300, edge_thickness=0, num_ticks=5):
    '''
    Assume v_omit is applied symmetrically on both top and bottom
    '''
    values = np.linspace(0, 1, height)
    if cmap is None:
        rgb = (np.repeat(values[..., np.newaxis], 3, axis=-1)*255).astype(np.uint8)
    else:
        rgb = (cmap(values)[:, :3] * 255).astype(np.uint8) # 0-1 to 0-255
    rgb = rgb.reshape(height, 1, 3) # shape row: h x 1
    
    # Isara: add white space top and bottom
    h_edge = np.ones((edge_thickness, 1, 3), dtype=np.uint8) * 255
    rgb = cv2.vconcat([h_edge, rgb, h_edge]) #np.concatenate((h_edge, rgb, h_edge), axis=0)
    
    cbar = np.repeat(rgb, width, axis=1) # chape col: h x w
    cbar = cbar[:, :, ::-1]  # reverse rgb to bgr for cv2

    cbar = np.ascontiguousarray(cbar)

    step = (vmax - vmin) / (num_ticks - 1)
    tick_positions = np.linspace(edge_thickness, height + edge_thickness - 1, num_ticks) # shift position
    edge_buffer = -20
    for i, y in enumerate(tick_positions):
        y = int(y)
        if i==num_ticks-1:
            edge_buffer = 30
        cv2.line(cbar, (0, y), (10, y), (0, 0, 0), 2)

        value = vmax - i * step   # top = max
        text = f"{value:.2f}"
        cv2.putText(cbar, text, (15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 1, cv2.LINE_AA)
        edge_buffer = 10

    return cbar