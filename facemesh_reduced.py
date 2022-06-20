import cv2
import mediapipe as mp

RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
draw = False

class FaceMeshDetector():

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 min_detection_confidence = self.minDetectionCon, min_tracking_confidence = self.minTrackCon)
        self.drawSpec_landmark = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1, color=RED_COLOR)
        self.drawSpec_connection = self.mpDraw.DrawingSpec(thickness=2, circle_radius=1, color=GREEN_COLOR)


    def findFaceMesh(self, img, draw=draw):

        dont_draw = False
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                                               landmark_drawing_spec = self.drawSpec_landmark,
                                               connection_drawing_spec = self.drawSpec_connection)
                    dont_draw = True
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)

                    # print(id,x,y)
                    face.append([x, y])
                faces.append(face)
        return img, faces, dont_draw


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=2)

    while True:
        success, img = cap.read()
        img, faces, dont_draw = detector.findFaceMesh(img, draw=draw)

        if dont_draw == False:
            if faces:
                for found_face in faces:
                    face = found_face
                    left_lip = face[61]
                    right_lip = face[291]
                    top_lip = face[0]
                    bottom_lip = face[17]
                    left_eye = face[145]
                    right_eye = face[374]

                    # first considering distance of mouth to find the focal length
                    # when dm(distance of mouth from lens) = 42, H(space between the upper and lower lip) = 4,
                    # we get h(space in pixels captured by lens) = 54
                    # f = (h * d)/H; f(focal length of lens) = 567
                    # if we consider distance of eyes from lens = distance of mouth from lens
                    w = right_eye[0] - left_eye[0]
                    f = 567
                    W = 6.3
                    d = (f * W) / w
                    h = bottom_lip[1] - top_lip[1]
                    H = (h * d) / f
                    if H > 3:
                        cv2.circle(img, top_lip, 3, GREEN_COLOR, cv2.FILLED)
                        cv2.circle(img, bottom_lip, 3, GREEN_COLOR, cv2.FILLED)
                        cv2.putText(img, "Mouth Width > 3cm", (right_eye[0], (right_eye[1] - 50)), cv2.FONT_HERSHEY_COMPLEX,
                                    0.8, (255, 0, 0), thickness=2)
                    else:
                        #cv2.line(img, left_lip, right_lip, GREEN_COLOR, 3)
                        cv2.circle(img, left_lip, 3, RED_COLOR, cv2.FILLED)
                        cv2.circle(img, right_lip, 3, RED_COLOR, cv2.FILLED)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
