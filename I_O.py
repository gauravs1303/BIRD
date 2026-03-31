import pyrealsense2 as rs
import numpy as np
import cv2

class InputOutput:
    def __init__(self):
        pass

    # Input cam stream
    @staticmethod
    def start_stream(id=0):
        # Configure pipeline
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        # config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

        # Start streaming
        pipeline.start(config)
        return pipeline
    
    @staticmethod
    def stream_frames():
        pipeline = InputOutput.start_stream()
        frame_id = 0
        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                # Convert to numpy array (OpenCV-compatible)
                img = np.asanyarray(color_frame.get_data()).copy()
                yield frame_id, img
                frame_id += 1
        finally:
            pipeline.stop()


        
    # Output the results
    def next_move(self,x,y,z):
        self.state = "move"

    def get_cam_stream_ir(id=0):
        # Configure pipeline
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

            # Start streaming
            pipeline.start(config)

            while True:
                # Wait for frames
                frames = pipeline.wait_for_frames()
                ir = frames.get_infrared_frame(1)
                if not ir:
                    continue

                # Convert to numpy arrays
                ir_image = np.asanyarray(ir.get_data())

                # Display using OpenCV
                cv2.imshow('IR left', ir_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            pipeline.stop()
            cv2.destroyAllWindows()

        
    # Output the results
    def next_move(self,x,y,z):
        self.state = "move"


if __name__ == "__main__":
    for fid, img in InputOutput.stream_frames():
        cv2.putText(
            img, f"FID {fid}", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
        cv2.imshow("Raw RGB", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
