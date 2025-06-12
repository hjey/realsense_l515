import pyrealsense2 as rs
import numpy as np
import cv2

clicked_points = []
save_point = None

def mouse_callback(event, x, y, flags, param):
    global clicked_points, save_point
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 2:
            clicked_points.append((x, y))
        elif len(clicked_points) >= 2:
            clicked_points = [(x, y)]
    elif event == cv2.EVENT_RBUTTONDOWN:
        save_point = (x, y)

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) ＃L515

def save_pointcloud(depth_frame, filename='output.ply'):
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)
    points.export_to_ply(filename)
    print(f"📦 Saved point cloud to: {filename}")

def get_roi_avg_depth(depth_frame, center, size=5):
    half = size // 2
    x, y = center
    depth_values = []

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            nx, ny = x + dx, y + dy
            if 0 <= nx < depth_frame.get_width() and 0 <= ny < depth_frame.get_height():
                d = depth_frame.get_distance(nx, ny)
                if 0 < d < 10:
                    depth_values.append(d)
    if depth_values:
        return np.mean(depth_values)
    else:
        return 0

def compute_3d_distance_roi(depth_frame, pt1, pt2):
    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    d1 = get_roi_avg_depth(depth_frame, pt1)
    d2 = get_roi_avg_depth(depth_frame, pt2)
    p1 = rs.rs2_deproject_pixel_to_point(depth_intrin, [pt1[0], pt1[1]], d1)
    p2 = rs.rs2_deproject_pixel_to_point(depth_intrin, [pt2[0], pt2[1]], d2)
    return np.linalg.norm(np.array(p1) - np.array(p2))

try:
    pipeline.start(config)
    print("✅ Pipeline started.")
    cv2.namedWindow('Depth ColorMap')
    cv2.setMouseCallback('Depth ColorMap', mouse_callback)

    show_save_distance = False
    save_distance_value = 0

    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # 좌클릭 거리 표시
        if len(clicked_points) == 2:
            pt1, pt2 = clicked_points
            cv2.circle(depth_colormap, pt1, 5, (0, 255, 0), -1)
            cv2.circle(depth_colormap, pt2, 5, (0, 255, 0), -1)
            cv2.line(depth_colormap, pt1, pt2, (255, 0, 0), 1)

            distance_m = compute_3d_distance_roi(depth_frame, pt1, pt2)
            scaled_distance = distance_m * 0.0265
            cv2.putText(depth_colormap, f"Distance: {scaled_distance:.2f} m", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # 우클릭 → 거리 표시 + PLY 저장
        if save_point is not None:
            x, y = save_point
            distance = get_roi_avg_depth(depth_frame, (x, y))

            save_distance_value = distance
            show_save_distance = True

            save_pointcloud(depth_frame)

            save_point = None

        if show_save_distance:
            scaled_save_distance = save_distance_value * 0.0265
            cv2.putText(depth_colormap, f"Distance: {scaled_save_distance:.2f} m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            show_save_distance = False

        cv2.imshow('Depth ColorMap', depth_colormap)

        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('c'):
            clicked_points = []
            print("🧹 Clicked points cleared.")

finally:
    pipeline.stop()
    print("🛑 Pipeline stopped.")
    cv2.destroyAllWindows()