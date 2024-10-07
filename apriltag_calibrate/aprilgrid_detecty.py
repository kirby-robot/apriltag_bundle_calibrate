import aprilgrid
import cv2
import numpy as np
import os
import sys
import argparse

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="path to the image")
    ap.add_argument("--family", required=False, help="april tag family")
    ap.add_argument("--output", required=True, help="path to save the result")
    args = ap.parse_args()

    img =cv2.imread(args.image)
    max_size = np.max(img.shape)
    if max_size > 3840:
        new_size_ratio = 3840.0 / max_size
        img = cv2.resize(
            img, None, None, new_size_ratio, new_size_ratio)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    at_detector = aprilgrid.Detector(
        tag_family_name=args.family if args.family else os.path.basename(args.image).split('.')[0]
    )
    print(f"gray.. {gray.shape}")
    tags = at_detector.detect(gray)

    print("%d apriltags have been detected."%len(tags))
    for tag in tags:
        for idx, corner in enumerate(tag.corners):
            cv2.circle(img, tuple(*corner.astype(int)), 12, (255, 0, 0), 6)
            cv2.putText(img, str(idx), tuple(*corner.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 2)
        center = np.sum(tag.corners, axis=0) / 4
        cv2.putText(img,
                    f"{tag.tag_id}",
                    tuple(*center.astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    3.0,
                    (0, 0, 255),
                    3
                )

    cv2.imwrite(os.path.join(args.output, os.path.basename(args.image).split('.')[0] + '_det.jpg'), img)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.waitKey()
