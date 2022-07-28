import cv2
import uuid
import os

# define a video capture object
vid = cv2.VideoCapture(0)

# define anchor and positive path
anchor_path = os.path.join('data', 'anchor')
positive_path = os.path.join('data', 'positive')
negative_path = os.path.join('data', 'negative')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    # time is in milliseconds
    # Cut down frame to 250x250px
    frame = frame[120:120 + 250, 200:200 + 250, :]

    # Collect anchors
    if cv2.waitKey(1) & 0XFF == ord('a'):
        # Create the unique file path
        img_name = os.path.join(anchor_path, '{}.jpg'.format(uuid.uuid1()))
        # Write out anchor image
        cv2.imwrite(img_name, frame)

        # Collect negatives
        if cv2.waitKey(1) & 0XFF == ord('n'):
            # Create the unique file path
            img_name = os.path.join(negative_path, '{}.jpg'.format(uuid.uuid1()))
            # Write out positive image
            cv2.imwrite(img_name, frame)

    # Collect positives
    if cv2.waitKey(1) & 0XFF == ord('p'):
        # Create the unique file path
        img_name = os.path.join(positive_path, '{}.jpg'.format(uuid.uuid1()))
        # Write out positive image
        cv2.imwrite(img_name, frame)

    # Show image back to screen
    cv2.imshow('Image Collection', frame)

    # Breaking gracefully
    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

# Release the webcam
cap.release()
# Close the image show frame
cv2.destroyAllWindows()
