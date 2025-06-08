import cv2
import numpy as np
import matplotlib.pyplot as plt

# OpenCV test
img = np.zeros((400, 600, 3), dtype=np.uint8)
cv2.putText(img, "OpenCV in Docker", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
cv2.imshow("OpenCV Window", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Matplotlib test
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Matplotlib Window")
plt.axis('off')
plt.show()
