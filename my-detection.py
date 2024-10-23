import jetson.inference
import jetson.utils
image_path="/home/nvidia/jetson-inference/data/images/airplane_0.jpg"
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)   
img = jetson.utils.loadImage(image_path) 
detections = net.Detect(img)
for detection in detections:
    class_id=detection.ClassID
    confidence=detection.Confidence
    left =detection.Left
    top=detection.Top
    right=detection.Right
    bottom=detection.Bottom
    width=right-left
    height=bottom-top
    area=width*height
    center_x=(left+right)/2
    center_y=(top+bottom)/2
    print(f"-- ClassID: {class_id}")
    print(f"-- Confidence: {confidence:.6f}")
    print(f"-- Left: {left:.5f}")
    print(f"-- Top: {top:.5f}")
    print(f"-- Right: {right:.5f}")
    print(f"-- Bottom: {bottom:.5f}")
    print(f"-- Width: {width:.5f}")
    print(f"-- Height: {height:.5f}")
    print(f"-- area: {area:.5f}")
    print(f"-- Left: {left:.5f}")
    print(f"-- Center: ({center_x:.5f},{center_y:.5f})")
jetson.utils.saveImage("/home/nvidia/jetson-inference/examples/myimage.jpg",img)
print("photo has saved")
