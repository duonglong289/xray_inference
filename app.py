from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64

from io import BytesIO
from PIL import Image
import numpy as np

from main_function import predict_faster_rcnn, predict_resnet

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Tạo một model để nhận dữ liệu từ client
class ImageData(BaseModel):
    image_base64: str

# Hàm chuyển NumPy array thành Base64
def numpy_to_base64(image_np: np.ndarray) -> str:
    # Chuyển mảng NumPy thành ảnh Pillow
    image = Image.fromarray(image_np.astype("uint8"), "RGB")
    
    # Mã hóa ảnh thành Base64
    buffered = BytesIO()
    image.save(buffered, format="jpeg")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/call-resnet")
async def call_resnet(image_data: ImageData):
    try:
        # Tách phần header nếu base64 có dạng "data:image/png;base64,...."
        if "," in image_data.image_base64:
            header, encoded = image_data.image_base64.split(",", 1)
        else:
            encoded = image_data.image_base64
        
        # Giải mã chuỗi Base64
        image_data_decoded = base64.b64decode(encoded)


        # Đọc dữ liệu ảnh từ chuỗi nhị phân và chuyển thành ảnh RGB
        image = Image.open(BytesIO(image_data_decoded)).convert("RGB")

        # Chuyển đổi ảnh thành mảng NumPy
        image_np = np.array(image)
        
        predicted_label, true_label = predict_resnet(image_np)
        
        return {
            'predicted_label': predicted_label,
            'true_label': true_label
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload image: {str(e)}")


@app.post("/call-faster")
async def call_faster(image_data: ImageData):
    try:
        # Tách phần header nếu base64 có dạng "data:image/png;base64,...."
        if "," in image_data.image_base64:
            header, encoded = image_data.image_base64.split(",", 1)
        else:
            encoded = image_data.image_base64
        
        # Giải mã chuỗi Base64
        image_data_decoded = base64.b64decode(encoded)

         # Giải mã chuỗi Base6

        # Đọc dữ liệu ảnh từ chuỗi nhị phân và chuyển thành ảnh RGB
        image = Image.open(BytesIO(image_data_decoded)).convert("RGB")

        # Chuyển đổi ảnh thành mảng NumPy
        image_np = np.array(image)
        
        predicted_label_np, true_label_np = predict_faster_rcnn(image_np)
        
        predicted_b64 = numpy_to_base64(predicted_label_np)
        true_b64 = numpy_to_base64(true_label_np)
        
        return {
            'predicted_label': predicted_b64,
            'true_label': true_b64
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8289)
