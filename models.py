from sqlalchemy import Column, Integer, String, Boolean

from database import Base


class User(Base):
    __tablename__ ="users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), index=True)
    password = Column(String(255),)
    name=Column(String(255),)
    is_active = Column(Boolean, default=True)
    
    
    
# class Admin(Base):
#     __tablename__ ="admins"

#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String(255), unique=True, index=True)
#     password = Column(String(255))
#     name=Column(String(255),)
#     is_active = Column(Boolean, default=True)
    
    

