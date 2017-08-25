import cv2
import sys
import os

zeroCount = 0
bigCount = 0

def get_tuple(rect):
	return (-1 * rect[2] * rect[3], rect[1])

def find_actual_face(face_rects):
	#we want largest, highest face detected
	return sorted(face_rects, key=lambda rect: get_tuple(rect))[0]

def identify_face(img, ldap):
	print ldap
	global zeroCount, bigCount
	classifier = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt0.xml")
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	face_rects = classifier.detectMultiScale(gray, 1.05, 4)
	rect = None
	if len(face_rects) < 1:
		zeroCount += 1
		print 'No faces found:', ldap
	elif len(face_rects) > 1:
		rect = find_actual_face(face_rects)
	else:
		rect = face_rects[0]

	if rect is not None:
		(x,y,w,h) = rect
		cropped = img[y:y+h, x:x+w]
		grayCropped = gray[y:y+h, x:x+w]
		grayCroppedEqualized = cv2.equalizeHist(grayCropped)
		return grayCroppedEqualized
	else:
		return None
		# cv2.imshow('just a face', cropped)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

def crop_all_faces(indir, outdir):
	global zeroCount, bigCount
	for filename in os.listdir(indir):
		filepath = indir + "/" + filename
		img = cv2.imread(filepath)
		if img is not None:
			processed_face = identify_face(img, filename[:-4])
			if processed_face is not None:
				cv2.imwrite(outdir + "/" + filename, processed_face)
		else:
			print filepath
	print 'zeros:', zeroCount
	print 'tooMany:', bigCount

def main():
	# print cv2.__version__
	indir = sys.argv[1]
	outdir = sys.argv[2]
	crop_all_faces(indir, outdir)


if __name__ == '__main__':
	main()
