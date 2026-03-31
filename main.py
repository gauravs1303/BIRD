from Visualizer import Visualizer
from PatchScorer import PatchScorer
from ObstacleDetection import ObstacleDetection
from I_O import InputOutput
from Terrain import Terrain
import cv2

# States are {"search", "approach", "maneuver", "commit"}
state = "search"
patchScorer = PatchScorer()
# For object detection (person, car, bicycle, bus, truck)
detector = ObstacleDetection()

frame_count = 0
for fid, img in InputOutput.stream_frames():
    cv2.putText(
        img, f"Frame {fid}", (10, 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    )
    terrain = Terrain(img)
    if terrain.ifsafe() == "safe":
            #conf_mask *= 0.9
            # if fid % 5 == 0:
            scores = patchScorer.compute_scores(img, obstacle_detector=detector)
            img = detector.obstacle_vis(img)
            # else:
            #     scores = patchScorer.compute_scores(img, obstacle_detector=None)
            print("Frame id: ", fid)
            print("scores min/max/mean", scores.min(), scores.max(), scores.mean())
            viz = Visualizer(img, scores)
            patch = viz.countour()
            viz.show()
    else:
        state = "maneuver"
        print(fid, state)

    if cv2.waitKey(1) & 0xFF == ord("q"):
         break
cv2.destroyAllWindows()