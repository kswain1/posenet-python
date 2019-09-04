import tensorflow as tf
import cv2
import time
import argparse
import os
from pandas import DataFrame

import posenet

posenetObject = {
    'leftHip': [],
    'rightHip': [],
    'leftKnee': [],
    'rightKnee': [],
    'leftAnkle': [],
    'rightAnkle': []
}

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--video_dir', type=str, default='./video_dir')
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--output_csv_dir', type=str, default='./outputcsv')
parser.add_argument('--output_video', type=str, default='./outputVideo' )
parser.add_argument('--output_name', type=str, default='./test')
args = parser.parse_args()


def main():

    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        if args.output_csv_dir:
            if not os.path.exists(args.output_csv_dir):
                os.makedirs(args.output_csv_dir)

        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        start = time.time()
        for f in filenames:
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

            keypoint_coords *= output_scale

            if args.output_dir:
                draw_image = posenet.draw_skel_and_kp(
                    draw_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.25, min_part_score=0.25)

                cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

            if args.output_csv_dir:
                script_dir = os.path.dirname(__file__)
                filenameUpdate = os.path.relpath(f, args.image_dir)
                filenameUpdate = args.output_csv_dir +'/' + filenameUpdate.strip('jpg') + 'txt'
                #filenameUpdate = filenameUpdate.strip('jpg') + 'txt'
                f = open(filenameUpdate, "w")
                f.write("Results for image: %s" % f.name + '\n')
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    f.writelines(" Pose #%d, score = %f" % (pi, pose_scores[pi]) + '\n')
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        f.write("keypoint %s, score =%f , coordindates = %s" % (posenet.PART_NAMES[ki], s, c) + '\n')
                f.close()

            if args.output_video:
                fileName = args.output_video + '/' + args.output_name + '.csv'
                # f = open(fileName, "a")
                # f.write("Results for image: %s " %f.name + '\n')
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0:
                        break
                    # f.writelines(" Pose #%d, score = %f" % (pi, pose_scores[pi]) + '\n')
                    # f.writelines("leftHip, rightHip, leftKnee, rightKnee, leftAnkle, rightAnkle")
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        if posenet.PART_NAMES[ki]== "leftHip":
                            posenetObject["leftHip"].append(c)
                        elif posenet.PART_NAMES[ki] == "rightHip":
                            posenetObject["rightHip"].append(c)
                        elif posenet.PART_NAMES[ki] == "leftKnee":
                            posenetObject["leftKnee"].append(c)
                        elif posenet.PART_NAMES[ki] == "rightKnee":
                            posenetObject["rightKnee"].append(c)
                        elif posenet.PART_NAMES[ki] == "leftAnkle":
                            posenetObject["leftAnkle"].append(c)
                        elif posenet.PART_NAMES[ki] == "rightAnkle":
                            posenetObject["rightAnkle"].append(c)
                        # f.write("keypoint %s, score =%f , coordindates = %s" % (posenet.PART_NAMES[ki], s, c) + '\n')
                # f.close()


        df = DataFrame(posenetObject, columns=["leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"])
        export_csv = df.to_csv(fileName,index=None, header=True)


            #Check for filenames length has been reached
            #print iterate to the first file name
            #print the pose instance in the file (for multiple people)
            #iterate through each key point in the file, and print them line by line
            ## issue: will print one file then the next file. Won't concatenate all the key points samples into one blog



            if not args.notxt:
                print()
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

        print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
