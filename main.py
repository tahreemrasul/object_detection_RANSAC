# coding=utf-8
import numpy as np
import data_utils as data_ut
import ransac_utils as algo_ut
import model

def main():
    print("=== START ===")

    threshold = 0.005
    example = 1

    # load image:
    cloud_image = data_ut.load_cloud_image(1)
    data_ut.show_image (cloud_image, "cloud image")

    # # filter image (e.g. mean filter)
    # median_filtered = median_filter(cloud_image, 3)
    # show_image(median_filtered, "median filtered image")
    # mean_filtered = mean_filter(cloud_image, 3)
    # show_image(mean_filtered, "mean filtered image")

    # cloud_image = median_filtered

    # use RANSAC to compute a floor model
    ransac_model = model.Ransac (threshold=threshold,
                                 max_iterations=500,
                                 min_inliers=90000,
                                 samples_to_fit=3)
    floor_model = ransac_model.model_calculation (cloud_image)

    # visualize inliers and use morphological operators to improve mask
    floor_mask = algo_ut.get_plane_mask (cloud_image, floor_model, threshold)
    opened_mask = algo_ut.opening (floor_mask, 3)
    closed_mask = algo_ut.closing (floor_mask, 3)
    data_ut.show_images (floor_mask, opened_mask, closed_mask, "floor mask")



    print("=== END ===")

    #new commit
    return 0


if __name__ == "__main__":
    main()
