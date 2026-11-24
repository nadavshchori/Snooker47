import cv2

img = cv2.imread("eyes.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 100)

contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

result = img.copy()
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:  # Skip tiny noise contours
        cv2.drawContours(result, [cnt], -1, (0, 255, 0), 2)
        M = cv2.moments(cnt)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(result, f"{area:.0f}px", (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
