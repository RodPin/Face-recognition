import face_recognition
import cv2
from os import listdir
from os.path import isfile, join
import sys
import numpy

numpy.set_printoptions(threshold=sys.maxsize)


myPath="./People I know/"
persons = [f for f in listdir(myPath) if isfile(join(myPath, f))]
#Encode Main PHOTO
image = cv2.imread("./Pictures/selecao2002.jpg")
# image = cv2.resize(image2, (0,0), fx=0.3, fy=0.3) 
unknown_face_encoding = face_recognition.face_encodings(image)

personsFaceEncoding=[]
personsName=[]
print('People i know:')
for person in persons:
    personName=person.split('.')[0]
    print(' - '+personName)
    personIMG= face_recognition.load_image_file(myPath+person)
    person_face_encoding = face_recognition.face_encodings(personIMG)[0]
    personsFaceEncoding.append(person_face_encoding) 
    personsName.append(personName)
face_locations = face_recognition.face_locations(image)
print()

def writeSquare(eachFaceEncode,name,recognized):
    if name != 'Unknown':
        print('Person found: '+name)

    COLOR=getColor(recognized)
    cv2.rectangle(image, (eachFaceEncode[3],eachFaceEncode[2]),(eachFaceEncode[1],eachFaceEncode[0]),COLOR , 2)
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (eachFaceEncode[3],eachFaceEncode[2]+15)
    fontScale              = 0.6
    fontColor              = COLOR
    lineType               = 2

    cv2.putText(image,
        name,
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

def getColor(boolean):  
    GREEN= (0,255,0)
    RED=(0,0,255)
    if boolean:
        return GREEN
    else:
        return RED


aux=[]
for i, face in enumerate(unknown_face_encoding):
    #Compare the faces in the folder of people we know with the ones in the image we're looking for 
    aux=face_recognition.compare_faces(personsFaceEncoding, unknown_face_encoding[i])
    if True in aux:
        writeSquare(face_locations[i],personsName[aux.index(True)],True)
    else:
        writeSquare(face_locations[i],"Unknown",False)
            
    
cv2.imwrite(join('./', 'processed.jpg'), image)
print()
print('End...')

