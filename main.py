from fastapi import FastAPI, Depends, status, HTTPException,Query
from sqlalchemy.orm import Session
import models
import schemas
from database import engine, get_db
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
# from helper_functions import set_seeds

import torch
from pathlib import Path


app = FastAPI()
models.Base.metadata.create_all(bind=engine)


origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# @app.get("/")
# async def root():
#     return {"message": "Backend Gutter API!!"}

# device = "cuda" if torch.cuda.is_available() else "cpu"

# pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT

# pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# class_names = ['broken','cracked','proper']

# set_seeds()
# pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)


# def load_model(model: torch.nn.Module,
#                target_dir: str,
#                model_name: str):
#     model_load_path = Path(target_dir) / model_name

#     print(f"[INFO] Loading model from: {model_load_path}")
#     model.load_state_dict(torch.load(model_load_path,map_location ='cpu'))

#     return model

# loaded_model = load_model(model=pretrained_vit,  
#                           target_dir="models",
#                           model_name="08_pretrained_vit_feature_extractor_gutter.pth")


# loaded_model.eval() 

# def pred_and_print_text(
#     model: torch.nn.Module,
#     class_names: List[str],
#     image_path: str,
#     image_size: Tuple[int, int] = (224, 224),
#     transform: torchvision.transforms = None,
#     device: torch.device = device,
# ):

#     img = Image.open(image_path)

#     if transform is not None:
#         image_transform = transform
#     else:
#         image_transform = transforms.Compose(
#             [
#                 transforms.Resize(image_size),
#                 transforms.ToTensor(),
#                 transforms.Normalize(
#                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                 ),
#             ]
#         )

#     model.to(device)

#     model.eval()
#     with torch.inference_mode():
#         transformed_image = image_transform(img).unsqueeze(dim=0)

#         target_image_pred = model(transformed_image.to(device))

#     target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

#     target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)
    
#     return {
#         "prediction": class_names[target_image_pred_label.item()]    }


# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         # Open the uploaded image

#         # Make prediction
#         prediction_result = pred_and_print_text(model=pretrained_vit,
#                             image_path=file.file,
#                             class_names=class_names)

#         return JSONResponse(content=prediction_result)

#     except Exception as e:
#         return JSONResponse(content={"error": str(e)})

     
# Get all Users
@app.get("/users")
def get_all_users(db: Session = Depends(get_db)):
    return db.query(models.User).all()

# @app.get("/admins")
# def get_all_admins(db: Session = Depends(get_db)):
#     return db.query(models.Admin).all()

# Register User
@app.post("/users")
def create_user(payload: schemas.CreateUser, db: Session = Depends(get_db)):
    
    
    existing_user = db.query(models.User).filter(models.User.email == payload.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already exists. Please choose a different email."
        )

    new_user = models.User(**payload.model_dump())
    if new_user.password=="":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Password can't be blank")
        
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    message = "Admin Added successfully"
    return  {message}

# @app.post("/admins")
# def create_admin(payload: schemas.CreateAdmin, db: Session = Depends(get_db)):
    
#     existing_admin = db.query(models.Admin).filter(
#         (models.Admin.email == payload.email)
#     ).first()
    

#     if existing_admin:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Admin with the same email already exists"
#         )

#     new_admin = models.Admin(**payload.model_dump())
#     if new_admin.password=="":
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,detail="Password can't be blank")
        
#     db.add(new_admin)
#     db.commit()
#     db.refresh(new_admin)
#     message = "Admin Added successfully"
#     return  {message}


# Get User by its ID
# @app.get("/users/{email_id}")
# def get_username_by_email(email: str, db: Session = Depends(get_db)):
#     user = db.query(models.User).filter(models.User.email == email).first()

#     if user is None:
#         raise HTTPException(status_code=404, detail="User not found")

#     return {"username": user.name}

# Get User by its ID
@app.get("/admins/{admin_id}", response_model=schemas.CreateUser)
def get_user_by_id(user_id: int, db: Session = Depends(get_db)):
    admin = db.query(models.User).filter_by(id=user_id, is_active=True).first()
    if not admin:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found")
    
    return admin

# Delete User by its ID
# @app.delete("/admins/{admins_id}")
# def delete_admin_by_id(admin_id: int, db: Session = Depends(get_db)):
#     admin = db.query(models.Admin).filter_by(id=admin_id, is_active=True).first()
#     if not admin:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Admin not found")

#     admin.is_active = False
#     db.commit()

#     message="Admin Deleted Successfully"
#     return {message} 

# Delete User by its ID
@app.delete("/users/{users_id}")
def delete_user_by_id(user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter_by(id=user_id, is_active=True).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.is_active = False
    db.commit()

    message="Admin Deleted Successfully"
    return {message}   

# Update User by its ID



@app.put("/users/{user_id}", response_model=schemas.UpdateUser)
def update_user_by_id(user_id: int, payload: schemas.UpdateUser, db: Session = Depends(get_db)):
    
    user = db.query(models.User).filter_by(id=user_id, is_active=True).first()

    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    # if not payload.password:
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password cannot be empty")
    # if len(payload.password) < 6:
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password must be 6 characters long")
    # if not any(char.isdigit() for char in payload.password):
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password requires a number")
    # if not any(char.isupper() for char in payload.password):
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password requires an uppercase letter")
    # if not any(char.islower() for char in payload.password):
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password requires a lowercase letter")
    # if not re.search(r'[^\w]', payload.password):
    #     raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Password requires a symbol")

    user.password=payload.password

    db.commit()
    db.refresh(user)
   
  
    return user



    



# @app.put("/admins/{admin_id}", response_model=schemas.CreateAdmin)
# def update_admin_by_id(admin_id: int, payload: schemas.CreateAdmin, db: Session = Depends(get_db)):
    
#     admin = db.query(models.Admin).filter_by(id=admin_id, is_active=True).first()

#     if not admin:
#         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    
    


#     admin.email = payload.email
#     admin.password=payload.password

#     db.commit()
#     db.refresh(admin)
   
#     return admin

# Login API
@app.post("/login")
def login(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == username).first()

    if not user or user.password != password or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {"message": "Login successful"}
# raise HTTPException(status_code=status.HTTP_200_OK, detail="Login successful")
