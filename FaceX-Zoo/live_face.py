import cv2
import numpy

import sys
sys.path.append('.')
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('api')

import yaml
import cv2
import numpy as np
import torch
from core.model_loader.face_detection.FaceDetModelLoader import FaceDetModelLoader
from core.model_handler.face_detection.FaceDetModelHandler import FaceDetModelHandler
from core.model_loader.face_alignment.FaceAlignModelLoader import FaceAlignModelLoader
from core.model_handler.face_alignment.FaceAlignModelHandler import FaceAlignModelHandler
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
from core.model_loader.face_recognition.FaceRecModelLoader import FaceRecModelLoader
from core.model_handler.face_recognition.FaceRecModelHandler import FaceRecModelHandler

with open('config/model_conf.yaml') as f:
    model_conf = yaml.load(f, Loader=yaml.FullLoader)


# Open the default camera
# cam = cv2.VideoCapture(0)

# Get the default frame width and height
# frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))




# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))


class LiveFace:

    def __init__(self, video_source=None):
        self.source = video_source
    

    def draw_rectangle_on_face(self, dets, frame):
            box = dets[0]
            box = list(map(int, box))
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)


    def run(self):
        """
            Facex detection
        """

        # common setting for all models, need not modify.
        model_path = 'models'

        # setting device on GPU if available, else CPU
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        device ='cpu'

        # face detection model setting.
        scene = 'non-mask'
        model_category = 'face_detection'
        model_name =  model_conf[scene][model_category]
        logger.info('Start to load the face detection model...')
        try:
            faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
            model, cfg = faceDetModelLoader.load_model()

            faceDetModelHandler = FaceDetModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Falied to load face detection Model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        # face landmark model setting.
        model_category = 'face_alignment'
        model_name =  model_conf[scene][model_category]
        logger.info('Start to load the face landmark model...')
        try:
            faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)

            model, cfg = faceAlignModelLoader.load_model()

            faceAlignModelHandler = FaceAlignModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Failed to load face landmark model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        # face recognition model setting.
        model_category = 'face_recognition'
        model_name =  model_conf[scene][model_category]    
        logger.info('Start to load the face recognition model...')
        try:
            faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)


            model, cfg = faceRecModelLoader.load_model()
            model = model.module.cpu() # added

            faceRecModelHandler = FaceRecModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Failed to load face recognition model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        face_cropper = FaceRecImageCropper()




        """
            Pętla wideo
        """


        # video = cv2.VideoCapture('output2.mp4')
        video = cv2.VideoCapture(self.source)
        cv2.namedWindow("video", cv2.WINDOW_NORMAL)

        counter = 0

        dets = numpy.array([])

        score = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (00, 185)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2

        while video.isOpened():
            ret, frame = video.read()

            if not ret:
                break

            try:
                dets = faceDetModelHandler.inference_on_image(frame)
                dets = numpy.append(dets, faceDetModelHandler.inference_on_image(cv2.imread('imagesDB/dawid0.jpg')))

                cv2.imshow('video', frame)

                if dets.shape[0] == 10:
                    dets = dets.reshape(2, 5)
                    self.draw_rectangle_on_face(dets, frame)
            except Exception as e:
                    logger.error('Face detection failed!')
                    logger.error(e)


            # frame rate - 10 fps - prawie...
            # if counter % 20 != 0:
            """
                pipeline
            """

            try:
                if dets.shape[0] == 2:
                    face_nums = dets.shape[0]
                    # face_nums = []
                    if face_nums != 2:
                        logger.info('Input image should contain two faces to compute similarity!')
                    feature_list = []
                    for i in range(face_nums):
                        landmarks = faceAlignModelHandler.inference_on_image(frame, dets[i])
                        landmarks_list = []
                        for (x, y) in landmarks.astype(np.int32):
                            landmarks_list.extend((x, y))
                            if i == 0:
                                cv2.circle(frame, (x, y), 2, (0, 255, 0),-1)
                        cropped_image = face_cropper.crop_image_by_mat(frame, landmarks_list)
                        feature = faceRecModelHandler.inference_on_image(cropped_image)
                        feature_list.append(feature)
                    score = np.dot(feature_list[0], feature_list[1])
                    logger.info('The similarity score of two faces: %f' % score)
            except Exception as e:
                logger.error('Pipeline failed!')
                logger.error(e)
                sys.exit(-1)
            else:
                logger.info('Success!')


                # Write the frame to the output file
                # out.write(frame)

                # Display the captured frame
                # cv2.imshow('Camera', frame)
            cv2.imshow('video', frame)

            counter += 1
            # try:
            #     dets = faceDetModelHandler.inference_on_image(frame)
            # except Exception as e:
            #    logger.error('Face detection failed!')
            #    logger.error(e)
            #    sys.exit(-1)
            # else:
            #    logger.info('Successful face detection!')

            # # typ to nd.array
            # # print(type(frame))
            # # print(frame.shape)

            # bboxs = dets
            # for box in bboxs:
            #     box = list(map(int, box))
            #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)


            # """
            #     pipeline
            # """

            
            # try:
            #     dets = faceDetModelHandler.inference_on_image(frame)
            #     dets = numpy.append(dets, faceDetModelHandler.inference_on_image(cv2.imread('imagesDB/dawid0.jpg')))
            #     dets = dets.reshape(2, 5)
            #     face_nums = dets.shape[0]
            #     # face_nums = []
            #     if face_nums != 2:
            #         logger.info('Input image should contain two faces to compute similarity!')
            #     feature_list = []
            #     for i in range(face_nums):
            #         landmarks = faceAlignModelHandler.inference_on_image(frame, dets[i])
            #         landmarks_list = []
            #         for (x, y) in landmarks.astype(np.int32):
            #             landmarks_list.extend((x, y))
            #         cropped_image = face_cropper.crop_image_by_mat(frame, landmarks_list)
            #         feature = faceRecModelHandler.inference_on_image(cropped_image)
            #         feature_list.append(feature)
            #     score = np.dot(feature_list[0], feature_list[1])
            #     logger.info('The similarity score of two faces: %f' % score)
            # except Exception as e:
            #     logger.error('Pipeline failed!')
            #     logger.error(e)
            #     sys.exit(-1)
            # else:
            #     logger.info('Success!')



            # Write the frame to the output file
            # out.write(frame)

            # Display the captured frame
            # cv2.imshow('Camera', frame)

            frame = cv2.putText(frame, str(score), org, font, fontScale, 
                                color, thickness, cv2.LINE_AA, False)
            cv2.imshow('video', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the capture and writer objects
        # cam.release()
        video.release()
        # out.release()
        cv2.destroyAllWindows()


    def run_live(self):
        """
            Facex detection
        """

        # common setting for all models, need not modify.
        model_path = 'models'

        # setting device on GPU if available, else CPU
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')

        device ='cpu'

        # face detection model setting.
        scene = 'non-mask'
        model_category = 'face_detection'
        model_name =  model_conf[scene][model_category]
        logger.info('Start to load the face detection model...')
        try:
            faceDetModelLoader = FaceDetModelLoader(model_path, model_category, model_name)
            model, cfg = faceDetModelLoader.load_model()
            faceDetModelHandler = FaceDetModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Falied to load face detection Model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        # face landmark model setting.
        model_category = 'face_alignment'
        model_name =  model_conf[scene][model_category]
        logger.info('Start to load the face landmark model...')
        try:
            faceAlignModelLoader = FaceAlignModelLoader(model_path, model_category, model_name)
            model, cfg = faceAlignModelLoader.load_model()
            faceAlignModelHandler = FaceAlignModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Failed to load face landmark model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        # face recognition model setting.
        model_category = 'face_recognition'
        model_name =  model_conf[scene][model_category]    
        logger.info('Start to load the face recognition model...')
        try:
            faceRecModelLoader = FaceRecModelLoader(model_path, model_category, model_name)
            model, cfg = faceRecModelLoader.load_model()
            model = model.module.cpu() # added
            faceRecModelHandler = FaceRecModelHandler(model, device, cfg)
        except Exception as e:
            logger.error('Failed to load face recognition model.')
            logger.error(e)
            sys.exit(-1)
        else:
            logger.info('Success!')

        face_cropper = FaceRecImageCropper()




        """
            Pętla wideo
        """


        # video = cv2.VideoCapture('output2.mp4')
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("live", cv2.WINDOW_NORMAL)

        counter = 0

        dets = numpy.array([])

        score = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (00, 185)
        fontScale = 1
        color = (0, 0, 255)
        thickness = 2

        while True:
            ret, frame = cam.read()

            if not ret:
                break

            try:
                dets = faceDetModelHandler.inference_on_image(frame)
                dets = numpy.append(dets, faceDetModelHandler.inference_on_image(cv2.imread('imagesDB/dawid0.jpg')))

                cv2.imshow('video', frame)

                if dets.shape[0] == 10:
                    dets = dets.reshape(2, 5)
                    self.draw_rectangle_on_face(dets, frame)
            except Exception as e:
                    logger.error('Face detection failed!')
                    logger.error(e)


            # frame rate - 10 fps - prawie...
            # if counter % 20 != 0:
            """
                pipeline
            """

            try:
                if dets.shape[0] == 2:
                    face_nums = dets.shape[0]
                    # face_nums = []
                    if face_nums != 2:
                        logger.info('Input image should contain two faces to compute similarity!')
                    feature_list = []
                    for i in range(face_nums):
                        landmarks = faceAlignModelHandler.inference_on_image(frame, dets[i])
                        landmarks_list = []
                        for (x, y) in landmarks.astype(np.int32):
                            landmarks_list.extend((x, y))
                            if i == 0:
                                cv2.circle(frame, (x, y), 2, (0, 255, 0),-1)
                        cropped_image = face_cropper.crop_image_by_mat(frame, landmarks_list)
                        feature = faceRecModelHandler.inference_on_image(cropped_image)
                        feature_list.append(feature)
                    score = np.dot(feature_list[0], feature_list[1])
                    logger.info('The similarity score of two faces: %f' % score)
            except Exception as e:
                logger.error('Pipeline failed!')
                logger.error(e)
                sys.exit(-1)
            else:
                logger.info('Success!')


                # Write the frame to the output file
                # out.write(frame)

                # Display the captured frame
                # cv2.imshow('Camera', frame)

            frame = cv2.putText(frame, str(score), org, font, fontScale, 
                                color, thickness, cv2.LINE_AA, False)
            cv2.imshow('video', frame)

            counter += 1

            # bboxs = dets
            # for box in bboxs:
            #     box = list(map(int, box))
            #     cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)



            # Write the frame to the output file
            # out.write(frame)

            # Display the captured frame
            # cv2.imshow('Camera', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) == ord('q'):
                break

        # Release the capture and writer objects
        # cam.release()
        cam.release()
        # out.release()
        cv2.destroyAllWindows()