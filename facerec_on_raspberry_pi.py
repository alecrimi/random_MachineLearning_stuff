# This is a demo of running face recognition on a Raspberry Pi.
# This program will print out the names of anyone it recognizes to the console.

# To run this, you need a Raspberry Pi 2 (or greater) with face_recognition and
# the picamera[array] module installed.
# You can follow this installation instructions to get your RPi set up:
# https://gist.github.com/ageitgey/1ac8dbe8572f3f533df6269dab35df65

import face_recognition
import picamera
import numpy as np
#from PIL import Image, ImageDraw
#import io
import time
from datetime import date

#img = Image.new('RGB',(2000,700), color =(73,109,137))


# Get a reference to the Raspberry Pi camera.
# If this fails, make sure you have a camera connected to the RPi and that you
# enabled your camera in raspi-config and rebooted first.
camera = picamera.PiCamera()
camera.resolution = (320, 240)
output = np.empty((240, 320, 3), dtype=np.uint8)
#output = io.BytesIO()

# Load a sample picture and learn how to recognize it.
print("Loading known face image(s)")
#obama_image = face_recognition.load_image_file("obama_small.jpg")
#obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
alex_image = face_recognition.load_image_file("cams.jpg")
alex_face_encoding = face_recognition.face_encodings(alex_image)[0]

# Initialize some variables
face_locations = []
face_encodings = []

while True:
    print("Capturing image.")
    # Grab a single frame of video from the RPi camera as a numpy array
    camera.capture(output, format="rgb")
#    pil_image = Image.open(output)
#    draw = ImageDraw.Draw(pil_image)
#    pil_image.show()

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(output)
    print("Found {} faces in image.".format(len(face_locations)))
    face_encodings = face_recognition.face_encodings(output, face_locations)



    # Loop over each face found in the frame to see if it's someone we know.
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces([alex_face_encoding], face_encoding)
        name = "<Unknown Person>"

        print(match[0])
        if match[0]:
            name = "Alessandro"
            print("I see {}!".format(name)) 
            print("Hi Ale, how are you?" ) 

            today = date.today()

            camera.start_preview(fullscreen=False,window=(100,200,600,800))
            time.sleep(2)
            camera.capture('/home/pi/Documents/intrusions/'+today.strftime("%m-%d-%Y")+'.jpg')
            camera.stop_preview()

          #  import pygame
          #  pygame.mixer.init()
          #  pygame.mixer.music.load("not.mp3")
          #  pygame.mixer.music.play()
          #  while pygame.mixer.music.get_busy() == True:
          #    continue


#            d = ImageDraw.Draw(img)
#            d.text((10,10),"You are NOT Alessandro",fill=(255,255,0))
#            img.show()
        else: 
            print("You are not Alessandro")
            today = date.today()
            camera.start_preview(fullscreen=False,window=(100,200,600,800))
            time.sleep(2)
            camera.capture('/home/pi/Documents/intrusions/'+today.strftime("%m-%d-%Y")+'.jpg')

            camera.stop_preview()
