from pydantic import BaseModel, EmailStr

class CreateUser(BaseModel):
    email: EmailStr
    password: str
    name:str
    
    class Config:
        from_attributes = True
        
        
class CreateAdmin(BaseModel):
    email: EmailStr
    password: str
    name:str
    
    class Config:
        from_attributes = True
        
                
class UpdateUser(BaseModel):
    password: str
    class Config:
        from_attributes = True
        
class GetUser(BaseModel):
    
    email: str
    
    
    class Config:
        from_attributes = True

