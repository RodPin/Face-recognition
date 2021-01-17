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

 
#rodrigoIMG = face_recognition.load_image_file("./People I know/rodrigo.jpg")
# unknown_image = face_recognition.load_image_file("unknown.jpg")

# biden_encoding = face_recognition.face_encodings(known_image)[0]
# unknown_encoding = face_recognition.face_encodings(image)[0]

# try:
#     rodrigo_face_encoding = face_recognition.face_encodings(rodrigoIMG)[0]
#     # obama_sface_encoding = face_recognition.face_encodings(obama_image)[0]
# except IndexError:
#     print("I wasn't able to locate any faces in at least one of the images. Check the image files. Aborting...")
#     quit()


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
    #COMPARA TODAS A FACES DADAS, COM CADA FACE RECONHECIDA NA FOTO PRINCIPAL
    aux=face_recognition.compare_faces(personsFaceEncoding, unknown_face_encoding[i])
    if True in aux:
        writeSquare(face_locations[i],personsName[aux.index(True)],True)
    else:
        writeSquare(face_locations[i],"Unknown",False)
            
    
# print(face_locations)
cv2.imshow("Minha Imagem", image)
cv2.imwrite(join('./', 'processed.jpg'), image)
k = cv2.waitKey(0)
print()
print('End...')

