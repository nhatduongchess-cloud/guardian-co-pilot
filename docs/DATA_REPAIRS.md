# Data repairs

Documented modifications made to my local copy of the source dataset.
Listed here for full transparency and reproducibility.

## 1. `T08d/kitti/image_2/001615.jpg` — corrupt in the distributed archive

**Problem.** Extracting `Hackathon_Dataset_Redacted.zip` fails a CRC-32 check on
this single file (1 of 93,620 files, 0.001%). The bytes in the archive are
damaged, so the original pixels are unrecoverable locally.

**Repair.** We copied the preceding frame `001614.jpg` to `001615.jpg`.
At 20 FPS the two frames are 0.05 s apart and visually near-identical.

**Effect on results.** One frame out of 18,000 scored frames (0.006%) in
Challenge 1 carries a 50 ms-stale left image. The right image, calibration,
driver frame and kinematics for that frame are intact.

**Reproduce:**
```bash
cp <DATA_ROOT>/T08d/kitti/image_2/001614.jpg <DATA_ROOT>/T08d/kitti/image_2/001615.jpg
```

**Alternative considered.** Re-downloading the archive would be the clean fix;
we kept the local repair because it is deterministic, documented, and affects a
single frame.
