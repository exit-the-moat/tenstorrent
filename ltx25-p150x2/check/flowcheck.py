# ltx1: motion continuity across chaining joints: Farneback optical flow per frame pair (quarter res), mean (dx, dy)
# and magnitude in windows before/after each joint; also frame-diff seam metric.
import av, cv2, numpy as np, sys, json
path = sys.argv[1]; joints = [int(x) for x in sys.argv[2].split(",")]; win = 8
c = av.open(path); fr = [cv2.cvtColor(f.to_ndarray(format="rgb24")[::4, ::4], cv2.COLOR_RGB2GRAY) for f in c.decode(video=0)]
flows = []
for i in range(len(fr) - 1):
    f = cv2.calcOpticalFlowFarneback(fr[i], fr[i + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flows.append((float(f[..., 0].mean()), float(f[..., 1].mean()), float(np.linalg.norm(f, axis=2).mean())))
fl = np.array(flows)
d = np.array([np.abs(fr[i + 1].astype(np.float32) - fr[i]).mean() for i in range(len(fr) - 1)])
# typical frame-to-frame change of the mean flow vector
dv = np.linalg.norm(np.diff(fl[:, :2], axis=0), axis=1)
out = {"file": path.split("/")[-1], "frames": len(fr), "median_mag": round(float(np.median(fl[:, 2])), 3),
       "median_flow_vec_change": round(float(np.median(dv)), 3), "joints": []}
for k in joints:
    b, a = fl[k - 1 - win:k - 1].mean(0), fl[k:k + win].mean(0)
    cosv = float(np.dot(b[:2], a[:2]) / (np.linalg.norm(b[:2]) * np.linalg.norm(a[:2]) + 1e-9))
    out["joints"].append({"at": k, "before_vec": [round(b[0], 3), round(b[1], 3)], "after_vec": [round(a[0], 3), round(a[1], 3)],
                          "mag_before": round(b[2], 3), "mag_after": round(a[2], 3), "direction_cos": round(cosv, 3),
                          "flow_vec_jump_at_joint": round(float(np.linalg.norm(fl[k, :2] - fl[k - 2, :2])), 3),
                          "frame_diff_at_joint": round(float(d[k - 1]), 2), "frame_diff_median": round(float(np.median(d)), 2)})
print(json.dumps(out))
